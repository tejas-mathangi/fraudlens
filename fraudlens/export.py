"""Export compact JSON artifacts for the web console.

The Elliptic dataset is 697 MB and cannot be committed, so a clone of this repository
has nothing to show. This module distils the pipeline's output into a few hundred
kilobytes of JSON that ships with the repo, which lets the frontend render a complete,
honest demo with no backend and no dataset -- and lets the API answer requests in
"artifact mode" for the same reason.

Every file carries ``"synthetic": false`` when generated from the real dataset so the
UI can be explicit about what the viewer is looking at.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from fraudlens.config import (
    DECISION_THRESHOLD,
    EXPORT,
    HIGH_RISK_THRESHOLD,
    N_LOCAL_FEATURES,
    PATHS,
    ExportConfig,
    Paths,
)
from fraudlens.data import Graph, build_graph, dataset_summary, feature_group, feature_names
from fraudlens.metrics import comparison_table, load_metrics
from fraudlens.model import fraud_scores, load_model
from fraudlens.pipeline.explain import explain_node, pick_interesting_nodes
from fraudlens.pipeline.rings import Ring, detect_rings, ring_subgraph

log = logging.getLogger(__name__)

ARTIFACT_FILES = (
    "overview.json",
    "graph_sample.json",
    "rings.json",
    "explanations.json",
    "nodes_index.json",
    "features.json",
)


def _write(path: Path, payload: dict[str, Any]) -> int:
    """Write JSON compactly and return the byte size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    path.write_text(text)
    return len(text.encode())


def _label_name(value: int) -> str:
    return {1: "illicit", 0: "licit", -1: "unknown"}[int(value)]


def _adjacency(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style adjacency (indptr, indices) over the undirected graph.

    Built once and reused by every exporter; scanning the 234k edge list per node
    would otherwise dominate the runtime.
    """
    src, dst = graph.data.edge_index.numpy()
    n = graph.num_nodes
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    order = np.argsort(both_src, kind="stable")
    sorted_src, sorted_dst = both_src[order], both_dst[order]
    indptr = np.searchsorted(sorted_src, np.arange(n + 1))
    return indptr, sorted_dst


def neighbours(indptr: np.ndarray, indices: np.ndarray, node: int) -> np.ndarray:
    """Unique neighbours of ``node``."""
    return np.unique(indices[indptr[node] : indptr[node + 1]])


def export_overview(
    graph: Graph,
    scores: np.ndarray,
    rings_stats: dict[str, Any],
    metrics: dict[str, Any],
    cfg: ExportConfig,
) -> dict[str, Any]:
    """Headline stats, distributions and the model comparison table."""
    y = graph.data.y.numpy()
    steps = graph.data.time_step.numpy()

    # Score distribution, split by true label so the separation is visible.
    bins = np.linspace(0, 1, cfg.score_histogram_bins + 1)
    centers = ((bins[:-1] + bins[1:]) / 2).round(4)
    hist = {
        "bin_centers": centers.tolist(),
        "illicit": np.histogram(scores[y == 1], bins=bins)[0].tolist(),
        "licit": np.histogram(scores[y == 0], bins=bins)[0].tolist(),
        "unknown": np.histogram(scores[y == -1], bins=bins)[0].tolist(),
    }

    timeline = []
    for step in range(1, int(steps.max()) + 1):
        in_step = steps == step
        if not in_step.any():
            continue
        timeline.append(
            {
                "time_step": step,
                "transactions": int(in_step.sum()),
                "illicit": int((y[in_step] == 1).sum()),
                "licit": int((y[in_step] == 0).sum()),
                "avg_score": round(float(scores[in_step].mean()), 4),
            }
        )

    return {
        "synthetic": False,
        "dataset": dataset_summary(graph),
        "scoring": {
            "high_risk": int((scores > HIGH_RISK_THRESHOLD).sum()),
            "flagged": int((scores > DECISION_THRESHOLD).sum()),
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "decision_threshold": DECISION_THRESHOLD,
            "mean_score": round(float(scores.mean()), 4),
        },
        "score_histogram": hist,
        "timeline": timeline,
        "rings": rings_stats,
        "comparison": comparison_table(metrics),
        "models": metrics.get("models", {}),
    }


def sample_connected_subgraph(
    graph: Graph,
    scores: np.ndarray,
    rings: list[Ring],
    n_nodes: int,
) -> list[int]:
    """Pick a node sample that actually looks like a transaction network.

    Elliptic averages ~1.15 edges per node — globally it is closer to a forest of
    chains than a dense graph — so neither a uniform sample nor a breadth-first walk
    produces a picture with any structure in it: both return ~1 induced edge per node
    and draw as scattered dust.

    The dense structure is inside the detected rings, so the sample is built from
    *whole* ring communities first. Every intra-ring edge then survives into the
    induced subgraph and each ring reads as a visible cluster. Remaining budget goes
    to the highest-scoring illicit nodes and their neighbours, and isolates are
    dropped at the end.
    """
    indptr, indices = _adjacency(graph)
    y = graph.data.y.numpy()
    selected: set[int] = set()

    # Whole communities, largest-first, while they fit.
    for ring in sorted(rings, key=lambda r: -r.size):
        if len(selected) >= n_nodes:
            break
        room = n_nodes - len(selected)
        members = ring.members
        if len(members) > room:
            # Keep the most suspicious slice of an oversized ring rather than a
            # random one, so what is shown is the part worth looking at.
            members = sorted(members, key=lambda m: -scores[m])[:room]
        selected.update(int(m) for m in members)

    # Then the highest-scoring confirmed fraud outside those rings, with neighbours,
    # so the picture is not only rings.
    if len(selected) < n_nodes:
        illicit = np.where(y == 1)[0]
        ranked = illicit[np.argsort(scores[illicit])[::-1]]
        queue: deque[int] = deque(int(i) for i in ranked if int(i) not in selected)
        while queue and len(selected) < n_nodes:
            node = queue.popleft()
            selected.add(node)
            for nb in neighbours(indptr, indices, node):
                nb = int(nb)
                if nb not in selected and len(selected) < n_nodes:
                    selected.add(nb)

    # Drop isolates: a node with no selected neighbour adds nothing to the picture.
    keep_set = selected
    connected = [
        node
        for node in sorted(keep_set)
        if any(int(nb) in keep_set for nb in neighbours(indptr, indices, node))
    ]
    return connected[:n_nodes]


def induced_edges(graph: Graph, nodes: list[int]) -> list[list[int]]:
    """Edges of the subgraph induced on ``nodes``, deduplicated and undirected."""
    node_set = set(nodes)
    src, dst = graph.data.edge_index.numpy()
    mask = np.isin(src, nodes) & np.isin(dst, nodes)
    seen = set()
    out = []
    for a, b in zip(src[mask].tolist(), dst[mask].tolist()):
        if a == b or a not in node_set or b not in node_set:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])
    return out


def node_record(
    graph: Graph,
    scores: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    node: int,
    with_neighbours: bool = False,
    neighbour_cap: int = 20,
) -> dict[str, Any]:
    """The per-node payload shared by every endpoint and artifact."""
    y = graph.data.y.numpy()
    nbrs = neighbours(indptr, indices, node)
    record: dict[str, Any] = {
        "idx": int(node),
        "tx_id": int(graph.idx_to_node[node]),
        "score": round(float(scores[node]), 4),
        "label": _label_name(y[node]),
        "prediction": "illicit" if scores[node] >= DECISION_THRESHOLD else "licit",
        "time_step": int(graph.data.time_step[node]),
        "degree": int(nbrs.size),
    }
    if with_neighbours:
        # Highest-risk neighbours first: if the list has to be truncated, those are
        # the ones that explain the verdict.
        ranked = nbrs[np.argsort(scores[nbrs])[::-1]][:neighbour_cap] if nbrs.size else nbrs
        record["neighbors"] = [
            {
                "idx": int(nb),
                "score": round(float(scores[nb]), 4),
                "label": _label_name(y[nb]),
            }
            for nb in ranked
        ]
    return record


def export_all(
    graph: Graph | None = None,
    paths: Paths = PATHS,
    cfg: ExportConfig = EXPORT,
    out_dir: Path | None = None,
) -> dict[str, int]:
    """Run every exporter and return ``{filename: bytes}``."""
    paths.ensure_dirs()
    out_dir = Path(out_dir) if out_dir else paths.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = graph or build_graph(paths)
    model = load_model(graph.num_features, paths.checkpoint)
    scores = fraud_scores(model, graph.data).numpy()
    indptr, indices = _adjacency(graph)
    y = graph.data.y.numpy()

    rings, _, _ = detect_rings(graph, fraud_prob=scores, paths=paths)
    metrics = load_metrics(paths.metrics_json)
    sizes: dict[str, int] = {}

    # ── overview ─────────────────────────────────────────────────────────────
    sizes["overview.json"] = _write(
        out_dir / "overview.json",
        export_overview(graph, scores, metrics.get("rings", {}), metrics, cfg),
    )

    # ── features ─────────────────────────────────────────────────────────────
    names = feature_names(graph.num_features)
    sizes["features.json"] = _write(
        out_dir / "features.json",
        {
            "synthetic": False,
            "n_features": graph.num_features,
            "n_local": N_LOCAL_FEATURES,
            "features": [
                {"index": i, "name": names[i], "group": feature_group(i)}
                for i in range(graph.num_features)
            ],
        },
    )

    # ── graph sample ─────────────────────────────────────────────────────────
    sample = sample_connected_subgraph(graph, scores, rings, cfg.graph_sample_nodes)
    sizes["graph_sample.json"] = _write(
        out_dir / "graph_sample.json",
        {
            "synthetic": False,
            "nodes": [
                node_record(graph, scores, indptr, indices, n) for n in sample
            ],
            "edges": induced_edges(graph, sample),
            "note": "Fraud-rich connected sample seeded from detected rings.",
        },
    )

    # ── rings ────────────────────────────────────────────────────────────────
    ring_payload = []
    for ring in rings[: cfg.max_rings]:
        members, edges = ring_subgraph(ring, graph.data.edge_index)
        ring_payload.append(
            {
                **ring.summary(),
                "members": members,
                "edges": [[a, b] for a, b in edges],
                "member_detail": [
                    node_record(graph, scores, indptr, indices, m) for m in members
                ],
            }
        )
    sizes["rings.json"] = _write(
        out_dir / "rings.json",
        {"synthetic": False, "rings": ring_payload, "stats": metrics.get("rings", {})},
    )

    # ── explanations ─────────────────────────────────────────────────────────
    explanations = []
    for node in pick_interesting_nodes(graph, scores, n=cfg.n_explanations):
        exp = explain_node(model, graph.data, node, fraud_prob=scores)
        subset = exp.subset.tolist()
        explanations.append(
            {
                "node": node_record(graph, scores, indptr, indices, node),
                "predicted_class": exp.predicted_class,
                "probability": exp.probability,
                "driver": exp.driver(N_LOCAL_FEATURES),
                "top_features": exp.top_features(cfg.top_features, graph.num_features),
                "subgraph": {
                    "nodes": [
                        node_record(graph, scores, indptr, indices, n) for n in subset
                    ],
                    "edges": [
                        {
                            "source": int(subset[a]),
                            "target": int(subset[b]),
                            "weight": round(float(w), 4),
                        }
                        for a, b, w in zip(
                            exp.sub_edge_index[0],
                            exp.sub_edge_index[1],
                            exp.edge_mask,
                        )
                    ][:400],
                },
            }
        )
    sizes["explanations.json"] = _write(
        out_dir / "explanations.json",
        {"synthetic": False, "explanations": explanations},
    )

    # ── searchable node index ────────────────────────────────────────────────
    # Every confirmed-illicit node, the highest-scoring unlabelled ones, and a licit
    # sample for contrast. The high-risk set is capped by score rather than taken
    # whole: 33,696 nodes clear the threshold on the real dataset, which produced a
    # 7.4 MB index that is both unusable in a browser and too big to commit.
    illicit = np.where(y == 1)[0]
    high_pool = np.where((scores > HIGH_RISK_THRESHOLD) & (y != 1))[0]
    high = high_pool[np.argsort(scores[high_pool])[::-1]][: cfg.nodes_index_high_risk]

    rng = np.random.default_rng(42)
    licit_pool = np.where(y == 0)[0]
    licit_sample = rng.choice(
        licit_pool,
        size=min(cfg.nodes_index_licit_sample, licit_pool.size),
        replace=False,
    )
    index_nodes = sorted(
        set(illicit.tolist()) | set(high.tolist()) | set(licit_sample.tolist())
    )
    sizes["nodes_index.json"] = _write(
        out_dir / "nodes_index.json",
        {
            "synthetic": False,
            "nodes": [
                node_record(
                    graph,
                    scores,
                    indptr,
                    indices,
                    n,
                    with_neighbours=True,
                    neighbour_cap=cfg.nodes_index_neighbours,
                )
                for n in index_nodes
            ],
            "note": (
                f"All {illicit.size} confirmed-illicit nodes, the top "
                f"{high.size} unlabelled by score, and a {licit_sample.size}-node "
                "licit sample."
            ),
        },
    )

    total = sum(sizes.values())
    for name, size in sizes.items():
        log.info("%-22s %8.1f KB", name, size / 1024)
    log.info("%-22s %8.1f KB total", "", total / 1024)
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export web artifacts")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_all(out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
