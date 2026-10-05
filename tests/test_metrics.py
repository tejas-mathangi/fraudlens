"""Metrics computation and the metrics.json contract."""

from __future__ import annotations

import numpy as np

from fraudlens.metrics import (
    classification_metrics,
    comparison_table,
    load_metrics,
    update_metrics,
)


def test_perfect_separation_scores_one():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    m = classification_metrics(y_true, y_prob)
    assert m["auc"] == 1.0
    assert m["f1_illicit"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m["support"] == {"licit": 2, "illicit": 2}


def test_pr_curve_is_downsampled_but_present():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 5000)
    y_prob = rng.random(5000)
    m = classification_metrics(y_true, y_prob)
    assert 0 < len(m["pr_curve"]["x"]) <= 200
    assert len(m["pr_curve"]["x"]) == len(m["pr_curve"]["y"])


def test_missing_metrics_file_returns_skeleton(tmp_path):
    payload = load_metrics(tmp_path / "absent.json")
    assert payload["models"] == {}


def test_stages_merge_without_clobbering_each_other(tmp_path):
    path = tmp_path / "metrics.json"
    update_metrics(path, models={"random_forest": {"auc": 0.99}})
    update_metrics(path, models={"graphsage": {"auc": 0.98}})
    update_metrics(path, rings={"count": 10})

    payload = load_metrics(path)
    # The baseline section must survive the GNN run and the ring run.
    assert payload["models"]["random_forest"]["auc"] == 0.99
    assert payload["models"]["graphsage"]["auc"] == 0.98
    assert payload["rings"]["count"] == 10
    assert "generated_at" in payload


def test_comparison_table_marks_graph_only_capabilities(tmp_path):
    path = tmp_path / "metrics.json"
    update_metrics(
        path,
        models={
            "random_forest": {"auc": 0.9958, "f1_illicit": 0.9316, "avg_precision": 0.9807},
            "graphsage": {"auc": 0.9885, "f1_illicit": 0.8833, "avg_precision": 0.9566},
        },
    )
    rows = comparison_table(load_metrics(path))
    assert [r["key"] for r in rows] == ["random_forest", "graphsage"]
    rf, gnn = rows
    # The point of the comparison: the forest can score, but cannot do these.
    assert rf["ring_detection"] is False and rf["explainability"] is False
    assert gnn["ring_detection"] is True and gnn["explainability"] is True


def test_comparison_table_skips_models_that_have_not_run(tmp_path):
    path = tmp_path / "metrics.json"
    update_metrics(path, models={"graphsage": {"auc": 0.98}})
    assert [r["key"] for r in comparison_table(load_metrics(path))] == ["graphsage"]
