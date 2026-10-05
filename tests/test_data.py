"""Graph construction: shapes, label encoding, edge hygiene and caching."""

from __future__ import annotations

import pytest
import torch

from fraudlens.data import (
    DatasetNotFoundError,
    build_graph,
    dataset_summary,
    encode_label,
    feature_group,
    feature_names,
)
from tests.conftest import N_FEATURES, N_LOCAL, N_NODES


def test_shapes(graph):
    assert graph.num_nodes == N_NODES
    assert graph.num_features == N_FEATURES
    assert graph.data.x.shape == (N_NODES, N_FEATURES)
    assert graph.data.edge_index.shape[0] == 2
    assert graph.data.y.shape == (N_NODES,)
    assert graph.data.time_step.shape == (N_NODES,)


def test_label_encoding_maps_elliptic_classes():
    # Elliptic's '1' is illicit and '2' is licit -- the inversion is easy to get wrong.
    assert encode_label("1") == 1
    assert encode_label("2") == 0
    assert encode_label("unknown") == -1
    assert encode_label(None) == -1


def test_label_mask_selects_only_known_labels(graph):
    assert graph.label_mask.sum() == (graph.data.y != -1).sum()
    assert not graph.label_mask[graph.data.y == -1].any()
    # The fixture labels every 4th node illicit and marks every 4th+1 unknown.
    assert int((graph.data.y == 1).sum()) == 6
    assert int((graph.data.y == -1).sum()) == 6


def test_edges_referencing_unknown_nodes_are_dropped(graph):
    # The fixture includes an edge to txId 999999, which is not in the feature file.
    assert int(graph.data.edge_index.max()) < graph.num_nodes
    assert graph.data.edge_index.shape[1] == 26


def test_node_id_mapping_round_trips(graph):
    for tx_id, idx in list(graph.node_to_idx.items())[:5]:
        assert int(graph.idx_to_node[idx]) == tx_id


def test_no_nan_in_features(graph):
    assert not torch.isnan(graph.data.x).any()


def test_cache_round_trip(paths):
    first = build_graph(paths, use_cache=True)
    assert paths.graph_cache.is_file()

    second = build_graph(paths, use_cache=True)
    assert torch.equal(first.data.x, second.data.x)
    assert torch.equal(first.data.edge_index, second.data.edge_index)
    assert torch.equal(first.label_mask, second.label_mask)
    assert first.node_to_idx == second.node_to_idx


def test_cache_invalidated_when_csv_changes(paths):
    build_graph(paths, use_cache=True)
    # Touching the features file changes its mtime, so the fingerprint must miss.
    features = paths.features_csv
    features.write_text(features.read_text() + "\n")
    rebuilt = build_graph(paths, use_cache=True)
    assert rebuilt.num_nodes == N_NODES


def test_missing_dataset_raises_actionable_error(tmp_path):
    from fraudlens.config import Paths

    empty = Paths(data_dir=tmp_path / "nope")
    with pytest.raises(DatasetNotFoundError) as exc:
        build_graph(empty)
    assert "FRAUDLENS_DATA_DIR" in str(exc.value)


def test_feature_names_split_local_from_aggregated():
    names = feature_names(165)
    assert len(names) == 165
    assert names[0].startswith("local_")
    assert names[92].startswith("local_")
    assert names[93].startswith("agg_")
    assert feature_group(0) == "local"
    assert feature_group(92) == "local"
    assert feature_group(93) == "aggregated"


def test_dataset_summary_counts_add_up(graph):
    s = dataset_summary(graph)
    assert s["illicit"] + s["licit"] + s["unknown"] == s["nodes"]
    assert s["labeled"] == s["illicit"] + s["licit"]
    assert N_LOCAL < N_FEATURES  # fixture sanity
