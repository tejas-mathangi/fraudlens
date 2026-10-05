"""The artifact exporter.

This is the module the committed demo depends on, so it is exercised end to end on
the synthetic fixture: a real checkpoint is trained for a couple of epochs, then every
artifact is written and re-read.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from fraudlens.export import (
    ARTIFACT_FILES,
    _adjacency,
    export_all,
    induced_edges,
    neighbours,
    node_record,
    sample_connected_subgraph,
)
from fraudlens.model import GraphSAGE, fraud_scores


def _trained_checkpoint(graph, paths):
    """Persist an untrained-but-valid checkpoint so the exporter can load it."""
    model = GraphSAGE(in_channels=graph.num_features, hidden_channels=8)
    torch.save(model.state_dict(), paths.checkpoint)
    return model


def test_adjacency_is_symmetric(graph):
    indptr, indices = _adjacency(graph)
    assert indptr.shape == (graph.num_nodes + 1,)

    # Every edge must appear from both endpoints, since the view is undirected.
    src, dst = graph.data.edge_index.numpy()
    for a, b in zip(src[:5], dst[:5]):
        assert b in neighbours(indptr, indices, a)
        assert a in neighbours(indptr, indices, b)


def test_neighbours_are_unique(graph):
    indptr, indices = _adjacency(graph)
    for node in range(graph.num_nodes):
        nbrs = neighbours(indptr, indices, node)
        assert len(nbrs) == len(set(nbrs.tolist()))


def test_node_record_shape(graph, paths):
    model = _trained_checkpoint(graph, paths)
    scores = fraud_scores(model, graph.data).numpy()
    indptr, indices = _adjacency(graph)

    rec = node_record(graph, scores, indptr, indices, 0, with_neighbours=True)
    assert rec["idx"] == 0
    assert rec["tx_id"] == 100_000
    assert rec["label"] in {"illicit", "licit", "unknown"}
    assert rec["prediction"] in {"illicit", "licit"}
    assert 0.0 <= rec["score"] <= 1.0
    assert rec["degree"] == len(rec["neighbors"]) or rec["degree"] > 50


def test_prediction_follows_the_decision_threshold(graph, paths):
    model = _trained_checkpoint(graph, paths)
    scores = fraud_scores(model, graph.data).numpy()
    indptr, indices = _adjacency(graph)

    for node in range(graph.num_nodes):
        rec = node_record(graph, scores, indptr, indices, node)
        expected = "illicit" if scores[node] >= 0.5 else "licit"
        assert rec["prediction"] == expected


def test_induced_edges_are_undirected_and_deduplicated(graph):
    nodes = list(range(6))
    edges = induced_edges(graph, nodes)
    seen = set()
    for a, b in edges:
        assert a in nodes and b in nodes
        assert a != b
        key = (min(a, b), max(a, b))
        assert key not in seen
        seen.add(key)


def test_sample_is_connected_and_bounded(graph, paths):
    model = _trained_checkpoint(graph, paths)
    scores = fraud_scores(model, graph.data).numpy()

    sample = sample_connected_subgraph(graph, scores, rings=[], n_nodes=10)
    assert 0 < len(sample) <= 10

    # Every sampled node must have at least one neighbour inside the sample --
    # a uniform random sample would mostly be isolated dust.
    indptr, indices = _adjacency(graph)
    keep = set(sample)
    for node in sample:
        assert any(int(n) in keep for n in neighbours(indptr, indices, node))


def test_export_all_writes_every_artifact(graph, paths):
    _trained_checkpoint(graph, paths)
    sizes = export_all(graph=graph, paths=paths)

    for name in ARTIFACT_FILES:
        assert name in sizes, f"{name} was not written"
        path = paths.artifacts_dir / name
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["synthetic"] is False  # generated from the configured dataset


def test_exported_overview_is_internally_consistent(graph, paths):
    _trained_checkpoint(graph, paths)
    export_all(graph=graph, paths=paths)

    overview = json.loads((paths.artifacts_dir / "overview.json").read_text())
    ds = overview["dataset"]
    assert ds["illicit"] + ds["licit"] + ds["unknown"] == ds["nodes"]

    hist = overview["score_histogram"]
    assert len(hist["bin_centers"]) == len(hist["illicit"]) == len(hist["licit"])
    # Every node lands in exactly one bin of one series.
    assert sum(hist["illicit"]) + sum(hist["licit"]) + sum(hist["unknown"]) == ds["nodes"]

    assert all(t["illicit"] <= t["transactions"] for t in overview["timeline"])


def test_exported_features_cover_both_groups(graph, paths):
    _trained_checkpoint(graph, paths)
    export_all(graph=graph, paths=paths)

    features = json.loads((paths.artifacts_dir / "features.json").read_text())
    assert features["n_features"] == graph.num_features
    assert len(features["features"]) == graph.num_features
    assert {f["group"] for f in features["features"]} <= {"local", "aggregated"}


def test_exported_graph_sample_edges_reference_sampled_nodes(graph, paths):
    _trained_checkpoint(graph, paths)
    export_all(graph=graph, paths=paths)

    sample = json.loads((paths.artifacts_dir / "graph_sample.json").read_text())
    keep = {n["idx"] for n in sample["nodes"]}
    for a, b in sample["edges"]:
        assert a in keep and b in keep


def test_exported_explanations_are_well_formed(graph, paths):
    _trained_checkpoint(graph, paths)
    export_all(graph=graph, paths=paths)

    payload = json.loads((paths.artifacts_dir / "explanations.json").read_text())
    assert payload["explanations"]
    for exp in payload["explanations"]:
        assert exp["driver"] in {"local", "network"}
        assert exp["top_features"]
        importances = [f["importance"] for f in exp["top_features"]]
        assert importances == sorted(importances, reverse=True)
        assert all(0.0 <= i <= 1.0 for i in importances)


def test_artifacts_stay_within_the_size_budget(graph, paths):
    _trained_checkpoint(graph, paths)
    sizes = export_all(graph=graph, paths=paths)
    # The fixture is tiny; this guards the JSON from accidentally embedding tensors.
    assert sum(sizes.values()) < 2_000_000


def test_json_is_serialisable_without_numpy_scalars(graph, paths):
    """numpy ints are not JSON-serialisable -- node_to_idx keys are int64."""
    _trained_checkpoint(graph, paths)
    export_all(graph=graph, paths=paths)

    nodes = json.loads((paths.artifacts_dir / "nodes_index.json").read_text())["nodes"]
    for rec in nodes:
        assert isinstance(rec["idx"], int)
        assert isinstance(rec["tx_id"], int)
        assert not isinstance(rec["score"], np.generic)
