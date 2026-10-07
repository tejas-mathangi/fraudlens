"""Fraud ring detection via Louvain community detection.

Communities are found on the raw transaction topology rather than in the model's
embedding space, so a "ring" reflects actual money movement between wallets rather
than learned feature similarity. The GNN's fraud scores are then used to decide which
of those real communities are worth reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass

import community as community_louvain
import networkx as nx
import numpy as np
import torch

from fraudlens.config import PATHS, RINGS, Paths, RingConfig
from fraudlens.data import Graph, build_graph
from fraudlens.metrics import update_metrics
from fraudlens.model import fraud_scores, load_model

log = logging.getLogger(__name__)


@dataclass
class Ring:
    """A community the thresholds promoted to a reported fraud ring."""

    community_id: int
    size: int
    labeled: int
    illicit_count: int
    licit_count: int
    illicit_ratio: float
    avg_fraud_score: float
    max_fraud_score: float
    members: list[int]

    def summary(self) -> dict[str, object]:
        """The ring without its member list, for table views."""
        d = asdict(self)
        d.pop("members")
        return d


def is_fraud_ring(
    illicit_ratio: float,
    avg_fraud_score: float,
    illicit_count: int,
    size: int,
    cfg: RingConfig = RINGS,
) -> bool:
    """The reporting rule, isolated so it can be tested without a graph.

    A community is reported when it is big enough to be a group, concentrated enough
    in confirmed fraud, and scored highly by the model overall. All three matter: the
    ratio alone promotes a 2-node pair, and the score alone promotes anything the
    model happens to dislike.
    """
    return (
        size >= cfg.min_size
        and illicit_ratio > cfg.min_illicit_ratio
        and avg_fraud_score > cfg.min_avg_score
        and illicit_count >= cfg.min_illicit
    )


def build_nx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Undirected NetworkX view of the transaction graph, for Louvain."""
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    edges = edge_index.numpy()
    g.add_edges_from(zip(edges[0].tolist(), edges[1].tolist()))
    return g


def louvain_partition(
    g: nx.Graph, paths: Paths = PATHS, use_cache: bool = True, seed: int = 42
) -> np.ndarray:
    """Community label per node index.

    Louvain on the full graph takes 1-2 minutes, which is far too slow to run per web
    request, so the partition is cached on disk.
    """
    if use_cache and paths.partition_cache.is_file():
        try:
            blob = json.loads(paths.partition_cache.read_text())
            if blob.get("num_nodes") == g.number_of_nodes():
                log.info("Loaded Louvain partition from cache")
                return np.asarray(blob["labels"], dtype=np.int64)
        except Exception as exc:  # pragma: no cover
            log.warning("Ignoring unreadable partition cache (%s)", exc)

    log.info("Running Louvain on %d nodes (1-2 minutes)", g.number_of_nodes())
    partition = community_louvain.best_partition(g, random_state=seed)
    labels = np.array([partition[i] for i in range(g.number_of_nodes())], dtype=np.int64)

    if use_cache:
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        paths.partition_cache.write_text(
            json.dumps({"num_nodes": g.number_of_nodes(), "labels": labels.tolist()})
        )
    return labels


def detect_rings(
    graph: Graph | None = None,
    fraud_prob: np.ndarray | None = None,
    paths: Paths = PATHS,
    cfg: RingConfig = RINGS,
    use_cache: bool = True,
    write_metrics: bool = True,
) -> tuple[list[Ring], np.ndarray, np.ndarray]:
    """Find fraud rings.

    Returns the ranked rings, the community label per node, and the fraud scores used
    (recomputed from the checkpoint when not supplied).

    ``write_metrics`` exists because the API calls this on startup: a GET request has
    no business rewriting a committed file, and leaving it on meant simply browsing
    the console left ``artifacts/metrics.json`` modified in git.
    """
    paths.ensure_dirs()
    graph = graph or build_graph(paths)

    if fraud_prob is None:
        model = load_model(graph.num_features, paths.checkpoint)
        fraud_prob = fraud_scores(model, graph.data).numpy()

    g = build_nx_graph(graph.data.edge_index, graph.num_nodes)
    labels = louvain_partition(g, paths, use_cache=use_cache)

    y = graph.data.y.numpy()
    rings: list[Ring] = []

    # Group node indices by community in one pass rather than scanning per community.
    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    for members in np.split(order, boundaries):
        if members.size < cfg.min_size:
            continue
        member_labels = y[members]
        labeled = members[member_labels != -1]
        if labeled.size == 0:
            continue

        n_illicit = int((y[labeled] == 1).sum())
        n_licit = int((y[labeled] == 0).sum())
        illicit_ratio = n_illicit / labeled.size
        scores = fraud_prob[members]
        avg_score = float(scores.mean())

        if not is_fraud_ring(illicit_ratio, avg_score, n_illicit, members.size, cfg):
            continue

        rings.append(
            Ring(
                community_id=int(labels[members[0]]),
                size=int(members.size),
                labeled=int(labeled.size),
                illicit_count=n_illicit,
                licit_count=n_licit,
                illicit_ratio=round(illicit_ratio, 4),
                avg_fraud_score=round(avg_score, 4),
                max_fraud_score=round(float(scores.max()), 4),
                members=[int(m) for m in members],
            )
        )

    rings.sort(key=lambda r: (r.illicit_ratio, r.avg_fraud_score), reverse=True)

    total_illicit = int((y == 1).sum())
    in_rings = sum(r.illicit_count for r in rings)
    stats = {
        "count": len(rings),
        "illicit_in_rings": in_rings,
        "pct_of_illicit": round(100 * in_rings / total_illicit, 2) if total_illicit else 0,
        "largest": max((r.size for r in rings), default=0),
        "communities_total": int(labels.max()) + 1 if labels.size else 0,
        "thresholds": asdict(cfg),
    }
    if write_metrics:
        update_metrics(paths.metrics_json, rings=stats)

    log.info(
        "Detected %d fraud rings covering %d illicit nodes (%.1f%% of all illicit)",
        stats["count"],
        in_rings,
        stats["pct_of_illicit"],
    )
    return rings, labels, fraud_prob


def ring_subgraph(
    ring: Ring, edge_index: torch.Tensor, max_nodes: int = 80, seed: int = 42
) -> tuple[list[int], list[tuple[int, int]]]:
    """Induced subgraph of a ring, down-sampled deterministically for display.

    The original code used an unseeded ``np.random.choice`` here, so the same ring
    rendered differently on every run.
    """
    members = ring.members
    if len(members) > max_nodes:
        rng = np.random.default_rng(seed)
        members = sorted(rng.choice(members, size=max_nodes, replace=False).tolist())

    keep = set(members)
    src, dst = edge_index.numpy()
    mask = np.isin(src, members) & np.isin(dst, members)
    edges = {
        (int(a), int(b))
        for a, b in zip(src[mask], dst[mask])
        if int(a) in keep and int(b) in keep
    }
    return members, sorted(edges)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect fraud rings via Louvain")
    parser.add_argument("--no-cache", action="store_true", help="recompute the partition")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rings, _, _ = detect_rings(use_cache=not args.no_cache)
    for i, ring in enumerate(rings[:10], 1):
        log.info(
            "%2d. community %-6d size %-5d illicit %-4d (%.1f%%)  avg score %.4f",
            i,
            ring.community_id,
            ring.size,
            ring.illicit_count,
            ring.illicit_ratio * 100,
            ring.avg_fraud_score,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
