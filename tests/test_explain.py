"""Gradient-saliency explanations on a toy graph."""

from __future__ import annotations

import numpy as np

from fraudlens.model import GraphSAGE, fraud_scores
from fraudlens.pipeline.explain import explain_node, pick_interesting_nodes


def _model(graph):
    m = GraphSAGE(in_channels=graph.num_features, hidden_channels=8)
    m.eval()
    return m


def test_explanation_shapes(graph):
    model = _model(graph)
    scores = fraud_scores(model, graph.data).numpy()
    exp = explain_node(model, graph.data, node_idx=3, fraud_prob=scores)

    assert exp.node_idx == 3
    assert exp.feature_importance.shape == (graph.num_features,)
    assert exp.edge_mask.shape[0] == exp.sub_edge_index.shape[1]
    assert exp.predicted_class in (0, 1)
    assert 0.0 <= exp.probability <= 1.0


def test_masks_are_normalised(graph):
    model = _model(graph)
    exp = explain_node(model, graph.data, node_idx=3)
    assert exp.feature_importance.min() >= 0.0
    assert exp.feature_importance.max() <= 1.0 + 1e-6
    if exp.edge_mask.size:
        assert exp.edge_mask.max() <= 1.0 + 1e-6


def test_target_node_is_inside_its_own_subgraph(graph):
    model = _model(graph)
    exp = explain_node(model, graph.data, node_idx=5)
    assert exp.subset[exp.target_local_idx] == 5


def test_driver_reports_local_vs_network(graph):
    model = _model(graph)
    exp = explain_node(model, graph.data, node_idx=4)

    # Force the top feature into the local block, then the aggregated block.
    exp.feature_importance = np.zeros(graph.num_features)
    exp.feature_importance[0] = 1.0
    assert exp.driver(n_local=7) == "local"

    exp.feature_importance = np.zeros(graph.num_features)
    exp.feature_importance[9] = 1.0
    assert exp.driver(n_local=7) == "network"


def test_works_without_precomputed_scores(graph):
    model = _model(graph)
    exp = explain_node(model, graph.data, node_idx=2, fraud_prob=None)
    assert exp.feature_importance.shape == (graph.num_features,)


def test_candidate_picks_prefer_confirmed_illicit(graph):
    model = _model(graph)
    scores = fraud_scores(model, graph.data).numpy()
    picks = pick_interesting_nodes(graph, scores, n=4)

    y = graph.data.y.numpy()
    assert len(picks) <= 4
    # All but the trailing contrast node should be confirmed illicit.
    assert all(y[p] == 1 for p in picks[:-1])
