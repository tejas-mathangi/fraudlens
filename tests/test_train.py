"""Splitting and class weighting."""

from __future__ import annotations

import pytest
import torch

from fraudlens.pipeline.train import compute_class_weights, get_train_test_masks


def test_masks_are_disjoint_and_cover_only_labelled_nodes(graph):
    train, test = get_train_test_masks(graph.label_mask, graph.data.y, test_size=0.25)
    assert not (train & test).any()
    assert (train | test).sum() == graph.label_mask.sum()
    # Unknown-label nodes must appear in neither split.
    assert not train[graph.data.y == -1].any()
    assert not test[graph.data.y == -1].any()


def test_split_is_deterministic(graph):
    a, _ = get_train_test_masks(graph.label_mask, graph.data.y, random_state=7)
    b, _ = get_train_test_masks(graph.label_mask, graph.data.y, random_state=7)
    assert torch.equal(a, b)


def test_class_weights_upweight_the_minority_class():
    y = torch.tensor([0] * 90 + [1] * 10)
    mask = torch.ones(100, dtype=torch.bool)
    w = compute_class_weights(y, mask)
    # weight = total / (n_classes * count)
    assert w[0] == pytest.approx(100 / (2 * 90))
    assert w[1] == pytest.approx(100 / (2 * 10))
    assert w[1] > w[0]


def test_class_weights_reject_a_single_class_split():
    y = torch.zeros(10, dtype=torch.long)
    with pytest.raises(ValueError):
        compute_class_weights(y, torch.ones(10, dtype=torch.bool))
