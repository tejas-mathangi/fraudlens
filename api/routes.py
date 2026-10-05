"""HTTP routes.

Every handler works in both backend modes: in live mode it computes from the in-memory
graph, in artifact mode it reads the precomputed JSON. The response shapes are
identical so the frontend needs no branching.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.state import Backend, get_backend
from fraudlens.config import N_LOCAL_FEATURES

router = APIRouter(prefix="/api")


def _require_ready(backend: Backend) -> None:
    if backend.status == "warming":
        raise HTTPException(
            status_code=503,
            detail={"status": "warming", "message": "Loading graph and model"},
        )


# ── Health ───────────────────────────────────────────────────────────────────
@router.get("/health")
def health(backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    """Liveness plus which mode the service is serving from.

    Never returns 503: the frontend polls this to decide what to render.
    """
    return backend.health()


# ── Overview ─────────────────────────────────────────────────────────────────
@router.get("/overview")
def overview(backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    _require_ready(backend)
    if backend.mode == "artifact":
        return backend.artifact("overview")

    from fraudlens.config import EXPORT
    from fraudlens.export import export_overview
    from fraudlens.metrics import load_metrics

    metrics = load_metrics(backend.paths.metrics_json)
    return export_overview(
        backend.graph, backend.scores, metrics.get("rings", {}), metrics, EXPORT
    )


@router.get("/features")
def features(backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    _require_ready(backend)
    if backend.mode == "artifact":
        return backend.artifact("features")

    from fraudlens.data import feature_group, feature_names

    n = backend.graph.num_features
    names = feature_names(n)
    return {
        "synthetic": False,
        "n_features": n,
        "n_local": N_LOCAL_FEATURES,
        "features": [
            {"index": i, "name": names[i], "group": feature_group(i)} for i in range(n)
        ],
    }


# ── Nodes ────────────────────────────────────────────────────────────────────
@router.get("/nodes/{idx}")
def node_detail(idx: int, backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    _require_ready(backend)

    if backend.mode == "artifact":
        record = backend._node_by_idx.get(idx)
        if record is None:
            raise HTTPException(404, detail=f"Node {idx} is not in the demo subset")
        return record

    if not 0 <= idx < backend.graph.num_nodes:
        raise HTTPException(404, detail=f"Node {idx} out of range")

    from fraudlens.export import node_record

    return node_record(
        backend.graph,
        backend.scores,
        backend.indptr,
        backend.indices,
        idx,
        with_neighbours=True,
    )


@router.get("/nodes/{idx}/subgraph")
def node_subgraph(
    idx: int,
    hops: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(120, ge=10, le=600),
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """The node's k-hop neighbourhood, capped for legibility."""
    _require_ready(backend)

    if backend.mode == "artifact":
        exp = backend._exp_by_idx.get(idx)
        if exp:
            return exp["subgraph"]
        record = backend._node_by_idx.get(idx)
        if record is None:
            raise HTTPException(404, detail=f"Node {idx} is not in the demo subset")
        nbrs = record.get("neighbors", [])
        return {
            "nodes": [record] + nbrs,
            "edges": [
                {"source": idx, "target": n["idx"], "weight": n.get("score", 0.5)}
                for n in nbrs
            ],
        }

    if not 0 <= idx < backend.graph.num_nodes:
        raise HTTPException(404, detail=f"Node {idx} out of range")

    from torch_geometric.utils import k_hop_subgraph

    from fraudlens.export import induced_edges, node_record

    subset, _, _, _ = k_hop_subgraph(
        node_idx=idx,
        num_hops=hops,
        edge_index=backend.graph.data.edge_index,
        relabel_nodes=False,
        num_nodes=backend.graph.num_nodes,
    )
    nodes = subset.tolist()
    if len(nodes) > max_nodes:
        # Keep the target plus its highest-risk neighbours: a random cut tends to
        # discard exactly the nodes that explain the verdict.
        others = [n for n in nodes if n != idx]
        ranked = sorted(others, key=lambda n: -backend.scores[n])[: max_nodes - 1]
        nodes = [idx] + sorted(ranked)

    return {
        "nodes": [
            node_record(backend.graph, backend.scores, backend.indptr, backend.indices, n)
            for n in nodes
        ],
        "edges": [
            {"source": a, "target": b, "weight": round(float(backend.scores[a]), 4)}
            for a, b in induced_edges(backend.graph, nodes)
        ],
        "target": idx,
    }


@router.get("/search")
def search(
    q: str = Query(..., min_length=1),
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """Resolve a node index or an Elliptic txId."""
    _require_ready(backend)
    idx = backend.resolve(q)
    if idx is None:
        return {"query": q, "match": None}
    return {"query": q, "match": node_detail(idx, backend)}


# ── Graph sample ─────────────────────────────────────────────────────────────
@router.get("/graph/sample")
def graph_sample(
    n: int = Query(1500, ge=100, le=3000),
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """A connected, fraud-rich sample for the graph explorer."""
    _require_ready(backend)
    if backend.mode == "artifact":
        payload = backend.artifact("graph_sample")
        nodes = payload.get("nodes", [])[:n]
        keep = {node["idx"] for node in nodes}
        return {
            "synthetic": payload.get("synthetic", False),
            "nodes": nodes,
            "edges": [e for e in payload.get("edges", []) if e[0] in keep and e[1] in keep],
        }

    from fraudlens.export import induced_edges, node_record, sample_connected_subgraph

    sample = sample_connected_subgraph(backend.graph, backend.scores, backend.rings, n)
    return {
        "synthetic": False,
        "nodes": [
            node_record(backend.graph, backend.scores, backend.indptr, backend.indices, s)
            for s in sample
        ],
        "edges": induced_edges(backend.graph, sample),
    }


# ── Fraud rings ──────────────────────────────────────────────────────────────
@router.get("/rings")
def rings(
    limit: int = Query(25, ge=1, le=100),
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    _require_ready(backend)
    if backend.mode == "artifact":
        payload = backend.artifact("rings")
        return {
            "synthetic": payload.get("synthetic", False),
            "stats": payload.get("stats", {}),
            "rings": [
                {k: v for k, v in ring.items() if k not in ("edges", "member_detail")}
                for ring in payload.get("rings", [])[:limit]
            ],
        }

    from fraudlens.metrics import load_metrics

    return {
        "synthetic": False,
        "stats": load_metrics(backend.paths.metrics_json).get("rings", {}),
        "rings": [r.summary() for r in backend.rings[:limit]],
    }


@router.get("/rings/{ring_id}")
def ring_detail(ring_id: int, backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    """One ring plus the induced subgraph needed to draw it."""
    _require_ready(backend)

    if backend.mode == "artifact":
        for ring in backend.artifact("rings").get("rings", []):
            if ring.get("community_id") == ring_id:
                return ring
        raise HTTPException(404, detail=f"Ring {ring_id} not found")

    from fraudlens.export import node_record
    from fraudlens.pipeline.rings import ring_subgraph

    for ring in backend.rings:
        if ring.community_id != ring_id:
            continue
        members, edges = ring_subgraph(ring, backend.graph.data.edge_index)
        return {
            **ring.summary(),
            "members": members,
            "edges": [[a, b] for a, b in edges],
            "member_detail": [
                node_record(
                    backend.graph, backend.scores, backend.indptr, backend.indices, m
                )
                for m in members
            ],
        }
    raise HTTPException(404, detail=f"Ring {ring_id} not found")


# ── Explainability ───────────────────────────────────────────────────────────
@router.get("/explain/candidates")
def explain_candidates(backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    """Nodes worth explaining, used to seed the explainability page."""
    _require_ready(backend)
    if backend.mode == "artifact":
        return {
            "candidates": [
                e["node"] for e in backend.artifact("explanations").get("explanations", [])
            ]
        }

    from fraudlens.export import node_record
    from fraudlens.pipeline.explain import pick_interesting_nodes

    picks = pick_interesting_nodes(backend.graph, backend.scores, n=20)
    return {
        "candidates": [
            node_record(backend.graph, backend.scores, backend.indptr, backend.indices, p)
            for p in picks
        ]
    }


@router.get("/explain/{idx}")
def explain(idx: int, backend: Backend = Depends(get_backend)) -> dict[str, Any]:
    """Feature and edge attributions for one node's prediction."""
    _require_ready(backend)

    if backend.mode == "artifact":
        exp = backend._exp_by_idx.get(idx)
        if exp is None:
            raise HTTPException(
                404,
                detail=(
                    f"No precomputed explanation for node {idx}. "
                    "The demo ships explanations for a fixed set of nodes; run the "
                    "API with the dataset present to explain any node."
                ),
            )
        return exp

    if not 0 <= idx < backend.graph.num_nodes:
        raise HTTPException(404, detail=f"Node {idx} out of range")

    from fraudlens.config import EXPORT
    from fraudlens.export import node_record
    from fraudlens.pipeline.explain import explain_node

    exp = explain_node(backend.model, backend.graph.data, idx, fraud_prob=backend.scores)
    subset = exp.subset.tolist()
    return {
        "node": node_record(
            backend.graph, backend.scores, backend.indptr, backend.indices, idx
        ),
        "predicted_class": exp.predicted_class,
        "probability": exp.probability,
        "driver": exp.driver(N_LOCAL_FEATURES),
        "top_features": exp.top_features(EXPORT.top_features, backend.graph.num_features),
        "subgraph": {
            "nodes": [
                node_record(
                    backend.graph, backend.scores, backend.indptr, backend.indices, n
                )
                for n in subset
            ],
            "edges": [
                {"source": int(subset[a]), "target": int(subset[b]), "weight": round(float(w), 4)}
                for a, b, w in zip(
                    exp.sub_edge_index[0], exp.sub_edge_index[1], exp.edge_mask
                )
            ][:400],
        },
    }
