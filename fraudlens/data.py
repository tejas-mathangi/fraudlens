"""Load the Elliptic Bitcoin dataset and build a PyTorch Geometric graph.

The raw feature CSV is ~658 MB, so parsing it takes ~30 s and dominates the runtime of
every pipeline stage. :func:`build_graph` therefore caches the assembled tensors and
reuses them whenever the source CSVs have not changed, which brings warm loads under
two seconds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from fraudlens.config import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    N_LOCAL_FEATURES,
    PATHS,
    Paths,
)

log = logging.getLogger(__name__)

CACHE_VERSION = 2


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Elliptic CSVs are not where the configuration expects them."""


@dataclass
class Graph:
    """The assembled transaction graph.

    Attributes:
        data: PyG ``Data`` with ``x``, ``edge_index``, ``y`` and ``time_step``.
        label_mask: True for nodes carrying a known licit/illicit label.
        node_to_idx: original Elliptic ``txId`` -> contiguous node index.
    """

    data: Data
    label_mask: torch.Tensor
    node_to_idx: dict[int, int]

    @property
    def idx_to_node(self) -> np.ndarray:
        """Node index -> original ``txId``, as an array for O(1) reverse lookup."""
        if not hasattr(self, "_idx_to_node"):
            arr = np.empty(len(self.node_to_idx), dtype=np.int64)
            for tx_id, idx in self.node_to_idx.items():
                arr[idx] = tx_id
            object.__setattr__(self, "_idx_to_node", arr)
        return self._idx_to_node

    # Convenience passthroughs so callers rarely need ``.data``
    @property
    def num_nodes(self) -> int:
        return int(self.data.num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self.data.num_edges)

    @property
    def num_features(self) -> int:
        return int(self.data.num_node_features)

    def unpack(self) -> tuple[Data, torch.Tensor, dict[int, int]]:
        """Return the legacy 3-tuple shape used by the original scripts."""
        return self.data, self.label_mask, self.node_to_idx


def encode_label(raw: object) -> int:
    """Map an Elliptic class value to our label encoding.

    Elliptic ships ``'1'`` for illicit, ``'2'`` for licit and ``'unknown'`` otherwise.
    We re-encode to illicit=1, licit=0, unknown=-1 so that ``y >= 0`` selects the
    supervised subset.
    """
    text = str(raw)
    if text == "1":
        return LABEL_ILLICIT
    if text == "2":
        return LABEL_LICIT
    return LABEL_UNKNOWN


def feature_names(n_features: int) -> list[str]:
    """Human-readable names for the anonymised Elliptic features.

    The dataset documents the first 93 columns as properties of the transaction itself
    and the remaining 72 as aggregations over its neighbours, but gives no semantics
    beyond that. We surface the distinction, which is the part that matters when
    reading an explanation.
    """
    names = []
    for i in range(n_features):
        if i < N_LOCAL_FEATURES:
            names.append(f"local_{i + 1:03d}")
        else:
            names.append(f"agg_{i - N_LOCAL_FEATURES + 1:03d}")
    return names


def feature_group(index: int) -> str:
    """``'local'`` for transaction-intrinsic features, ``'aggregated'`` otherwise."""
    return "local" if index < N_LOCAL_FEATURES else "aggregated"


def _cache_signature(paths: Paths) -> dict[str, object]:
    """Fingerprint the source CSVs so a changed dataset invalidates the cache."""
    sig: dict[str, object] = {"version": CACHE_VERSION}
    for name, path in (
        ("features", paths.features_csv),
        ("edges", paths.edges_csv),
        ("classes", paths.classes_csv),
    ):
        stat = path.stat()
        sig[name] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    return sig


def _load_from_cache(paths: Paths, signature: dict[str, object]) -> Graph | None:
    if not paths.graph_cache.is_file():
        return None
    try:
        blob = torch.load(paths.graph_cache, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - corrupt cache is best-effort
        log.warning("Ignoring unreadable graph cache (%s)", exc)
        return None
    if blob.get("signature") != signature:
        log.info("Graph cache is stale; rebuilding from CSVs")
        return None

    data = Data(
        x=blob["x"], edge_index=blob["edge_index"], y=blob["y"], time_step=blob["time_step"]
    )
    log.info("Loaded graph from cache: %s", paths.graph_cache)
    return Graph(data=data, label_mask=blob["label_mask"], node_to_idx=blob["node_to_idx"])


def _write_cache(paths: Paths, graph: Graph, signature: dict[str, object]) -> None:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "x": graph.data.x,
            "edge_index": graph.data.edge_index,
            "y": graph.data.y,
            "time_step": graph.data.time_step,
            "label_mask": graph.label_mask,
            "node_to_idx": graph.node_to_idx,
        },
        paths.graph_cache,
    )
    log.info("Wrote graph cache: %s", paths.graph_cache)


def build_graph(paths: Paths | None = None, use_cache: bool = True) -> Graph:
    """Load the Elliptic CSVs and assemble the transaction graph.

    Args:
        paths: path configuration; defaults to the module-level :data:`PATHS`.
        use_cache: read and write ``.cache/graph.pt``. Disable for a forced rebuild.

    Raises:
        DatasetNotFoundError: if any of the three CSVs is missing.
    """
    paths = paths or PATHS

    if not paths.dataset_available():
        raise DatasetNotFoundError(
            f"Elliptic CSVs not found under {paths.data_dir}.\n"
            "Download the dataset (see scripts/download_data.py) or point "
            "FRAUDLENS_DATA_DIR at an existing copy."
        )

    signature = _cache_signature(paths)
    if use_cache:
        cached = _load_from_cache(paths, signature)
        if cached is not None:
            return cached

    log.info("Parsing Elliptic CSVs from %s", paths.data_dir)

    # The features file has no header: col 0 is txId, col 1 is the time step, the rest
    # are the anonymised features.
    features_df = pd.read_csv(paths.features_csv, header=None)
    edges_df = pd.read_csv(paths.edges_csv)
    classes_df = pd.read_csv(paths.classes_csv)

    n_features = features_df.shape[1] - 2
    features_df.columns = ["txId", "time_step"] + [
        f"f{i}" for i in range(1, n_features + 1)
    ]

    # ── Node index mapping ───────────────────────────────────────────────────
    all_tx_ids = features_df["txId"].to_numpy()
    node_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(all_tx_ids)}
    n_nodes = len(node_to_idx)

    # ── Feature matrix ───────────────────────────────────────────────────────
    feature_cols = [f"f{i}" for i in range(1, n_features + 1)]
    x = torch.from_numpy(features_df[feature_cols].to_numpy(dtype=np.float32))

    # ── Edge index ───────────────────────────────────────────────────────────
    src_col, dst_col = edges_df.columns[0], edges_df.columns[1]
    valid = edges_df[src_col].isin(node_to_idx) & edges_df[dst_col].isin(node_to_idx)
    dropped = int((~valid).sum())
    if dropped:
        log.warning("Dropped %d edges referencing unknown nodes", dropped)
    edges_clean = edges_df[valid]

    edge_index = torch.from_numpy(
        np.stack(
            [
                edges_clean[src_col].map(node_to_idx).to_numpy(dtype=np.int64),
                edges_clean[dst_col].map(node_to_idx).to_numpy(dtype=np.int64),
            ]
        )
    )

    # ── Labels, aligned to the node ordering ─────────────────────────────────
    merged = pd.DataFrame({"txId": all_tx_ids}).merge(classes_df, on="txId", how="left")
    y = torch.from_numpy(
        merged["class"].map(encode_label).to_numpy(dtype=np.int64)
    )
    label_mask = y != LABEL_UNKNOWN

    time_step = torch.from_numpy(
        features_df["time_step"].to_numpy(dtype=np.int64)
    )

    data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)

    # ── Sanity checks ────────────────────────────────────────────────────────
    assert data.x.shape[0] == n_nodes, "feature/node count mismatch"
    assert data.y.shape[0] == n_nodes, "label/node count mismatch"
    if edge_index.numel():
        assert int(edge_index.max()) < n_nodes, "edge index out of bounds"
    assert not torch.isnan(data.x).any(), "NaN in feature matrix"

    graph = Graph(data=data, label_mask=label_mask, node_to_idx=node_to_idx)

    log.info(
        "Graph built: %d nodes, %d edges, %d features, %d labelled",
        graph.num_nodes,
        graph.num_edges,
        graph.num_features,
        int(label_mask.sum()),
    )

    if use_cache:
        _write_cache(paths, graph, signature)

    return graph


def dataset_summary(graph: Graph) -> dict[str, int]:
    """Counts used by the overview endpoints and the README results table."""
    y = graph.data.y
    return {
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "features": graph.num_features,
        "time_steps": int(graph.data.time_step.max()),
        "illicit": int((y == LABEL_ILLICIT).sum()),
        "licit": int((y == LABEL_LICIT).sum()),
        "unknown": int((y == LABEL_UNKNOWN).sum()),
        "labeled": int(graph.label_mask.sum()),
    }


def write_feature_metadata(graph_features: int, path) -> None:
    """Dump feature index -> name/group metadata for the frontend."""
    names = feature_names(graph_features)
    payload = {
        "n_features": graph_features,
        "n_local": N_LOCAL_FEATURES,
        "features": [
            {"index": i, "name": names[i], "group": feature_group(i)}
            for i in range(graph_features)
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
