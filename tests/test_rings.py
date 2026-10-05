"""The fraud-ring promotion rule, as a pure function."""

from __future__ import annotations

from fraudlens.config import RingConfig
from fraudlens.pipeline.rings import build_nx_graph, is_fraud_ring, louvain_partition

CFG = RingConfig()  # ratio > 0.4, score > 0.4, illicit >= 2, size >= 3


def test_a_clear_ring_is_promoted():
    assert is_fraud_ring(0.9, 0.85, illicit_count=12, size=20)


def test_thresholds_are_strict_not_inclusive():
    # 0.4 exactly must fail: the documented rule is "> 40%".
    assert not is_fraud_ring(0.4, 0.85, illicit_count=5, size=20)
    assert not is_fraud_ring(0.9, 0.4, illicit_count=5, size=20)
    assert is_fraud_ring(0.41, 0.41, illicit_count=5, size=20)


def test_a_single_illicit_node_is_not_a_ring():
    # Ratio and score can both look damning for one node in a pair; a ring needs >= 2.
    assert not is_fraud_ring(1.0, 0.99, illicit_count=1, size=5)


def test_tiny_communities_are_rejected():
    assert not is_fraud_ring(1.0, 0.99, illicit_count=2, size=2)
    assert is_fraud_ring(1.0, 0.99, illicit_count=2, size=3)


def test_custom_config_is_honoured():
    strict = RingConfig(min_illicit_ratio=0.95, min_avg_score=0.9, min_illicit=10, min_size=10)
    assert not is_fraud_ring(0.9, 0.85, 12, 20, cfg=strict)
    assert is_fraud_ring(0.99, 0.95, 15, 20, cfg=strict)


def test_louvain_partition_labels_every_node(graph, paths):
    g = build_nx_graph(graph.data.edge_index, graph.num_nodes)
    labels = louvain_partition(g, paths, use_cache=False)
    assert labels.shape == (graph.num_nodes,)
    assert labels.min() >= 0


def test_louvain_partition_is_cached(graph, paths):
    g = build_nx_graph(graph.data.edge_index, graph.num_nodes)
    first = louvain_partition(g, paths, use_cache=True)
    assert paths.partition_cache.is_file()
    second = louvain_partition(g, paths, use_cache=True)
    assert (first == second).all()
