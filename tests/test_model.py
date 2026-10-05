"""Model shape and -- critically -- checkpoint compatibility.

``models/best_model.pt`` is a bare ``state_dict``, so the module attribute names are
an on-disk contract. This test pins the exact key set: if a refactor renames a layer,
every existing checkpoint silently stops loading, and this fails instead.
"""

from __future__ import annotations

import torch

from fraudlens.model import GraphSAGE, fraud_scores, read_state_dict

EXPECTED_KEYS = {
    "conv1.lin_l.weight",
    "conv1.lin_l.bias",
    "conv1.lin_r.weight",
    "conv2.lin_l.weight",
    "conv2.lin_l.bias",
    "conv2.lin_r.weight",
    "conv3.lin_l.weight",
    "conv3.lin_l.bias",
    "conv3.lin_r.weight",
    "bn1.weight",
    "bn1.bias",
    "bn1.running_mean",
    "bn1.running_var",
    "bn1.num_batches_tracked",
    "bn2.weight",
    "bn2.bias",
    "bn2.running_mean",
    "bn2.running_var",
    "bn2.num_batches_tracked",
    "bn3.weight",
    "bn3.bias",
    "bn3.running_mean",
    "bn3.running_var",
    "bn3.num_batches_tracked",
    "classifier.weight",
    "classifier.bias",
}


def test_state_dict_keys_are_stable():
    model = GraphSAGE(in_channels=165)
    assert set(model.state_dict()) == EXPECTED_KEYS


def test_forward_returns_two_logits_per_node(graph):
    model = GraphSAGE(in_channels=graph.num_features, hidden_channels=16)
    model.eval()
    with torch.no_grad():
        out = model(graph.data.x, graph.data.edge_index)
    assert out.shape == (graph.num_nodes, 2)


def test_hidden_width_halves_in_the_third_layer():
    model = GraphSAGE(in_channels=165, hidden_channels=128)
    assert model.classifier.in_features == 64
    assert model.bn3.num_features == 64


def test_fraud_scores_are_probabilities(graph):
    model = GraphSAGE(in_channels=graph.num_features, hidden_channels=16)
    scores = fraud_scores(model, graph.data)
    assert scores.shape == (graph.num_nodes,)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_read_state_dict_accepts_both_checkpoint_layouts(tmp_path):
    model = GraphSAGE(in_channels=8, hidden_channels=4)

    bare = tmp_path / "bare.pt"
    torch.save(model.state_dict(), bare)
    assert set(read_state_dict(bare)) == set(model.state_dict())

    wrapped = tmp_path / "wrapped.pt"
    torch.save({"state_dict": model.state_dict(), "best_f1": 0.88, "epoch": 40}, wrapped)
    assert set(read_state_dict(wrapped)) == set(model.state_dict())


def test_infer_architecture_reads_widths_from_tensors():
    from fraudlens.model import infer_architecture

    model = GraphSAGE(in_channels=165, hidden_channels=128, out_channels=2)
    arch = infer_architecture(model.state_dict())
    assert arch == {"in_channels": 165, "hidden_channels": 128, "out_channels": 2}


def test_load_model_sizes_itself_from_the_checkpoint(tmp_path):
    from fraudlens.model import load_model

    # A non-default width must load without the caller knowing it.
    saved = GraphSAGE(in_channels=12, hidden_channels=8)
    path = tmp_path / "narrow.pt"
    torch.save(saved.state_dict(), path)

    loaded = load_model(checkpoint=path)
    assert loaded.bn1.num_features == 8
    assert loaded.classifier.in_features == 4
    for key, tensor in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], tensor)


def test_load_model_rejects_a_feature_count_mismatch(tmp_path):
    import pytest

    from fraudlens.model import load_model

    path = tmp_path / "m.pt"
    torch.save(GraphSAGE(in_channels=12, hidden_channels=8).state_dict(), path)
    with pytest.raises(ValueError, match="expects 12 input features"):
        load_model(in_channels=165, checkpoint=path)
