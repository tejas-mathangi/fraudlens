"""Evaluation helpers and the single source of truth for reported numbers.

Before this module the headline metrics were hardcoded string literals duplicated
across four files, so the README, the dashboard and the training script could -- and
did -- disagree. Everything now reads ``artifacts/metrics.json``, which only the
pipeline writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

SCHEMA_VERSION = 1


def _downsample(xs: np.ndarray, ys: np.ndarray, n: int = 200) -> dict[str, list[float]]:
    """Thin a curve to ``n`` points so it stays small in JSON but keeps its shape."""
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return {
        "x": [round(float(v), 5) for v in xs[idx]],
        "y": [round(float(v), 5) for v in ys[idx]],
    }


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """AUC / F1 / average precision plus a confusion matrix and a PR curve.

    F1 is reported for the illicit class specifically. Accuracy is deliberately not
    reported: predicting "licit" for everything scores ~90% on this dataset.
    """
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1_illicit": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "threshold": threshold,
        "support": {"licit": int((y_true == 0).sum()), "illicit": int((y_true == 1).sum())},
        "confusion_matrix": [[int(v) for v in row] for row in cm],
        "pr_curve": _downsample(recall, precision),
    }


def load_metrics(path: Path) -> dict[str, Any]:
    """Read ``metrics.json``, or return an empty skeleton when it does not exist."""
    if not Path(path).is_file():
        return {"schema_version": SCHEMA_VERSION, "models": {}}
    return json.loads(Path(path).read_text())


def update_metrics(path: Path, **sections: Any) -> dict[str, Any]:
    """Merge ``sections`` into ``metrics.json`` and write it back.

    Each pipeline stage owns its own section, so stages can run independently without
    clobbering each other's results. ``models`` merges one level deeper so the
    baseline and the GNN can be trained separately.
    """
    path = Path(path)
    payload = load_metrics(path)
    payload["schema_version"] = SCHEMA_VERSION
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for key, value in sections.items():
        if key == "models" and isinstance(value, dict):
            payload.setdefault("models", {}).update(value)
        else:
            payload[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def comparison_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the RF-vs-GraphSAGE table shown in the UI and the README.

    The capability columns are the point of the comparison: a tabular model cannot
    produce them at all, whatever its F1.
    """
    models = metrics.get("models", {})
    rows = []
    for key, label, caps in (
        ("random_forest", "Random Forest (baseline)", False),
        ("graphsage", "GraphSAGE (FraudLens)", True),
    ):
        m = models.get(key)
        if not m:
            continue
        rows.append(
            {
                "key": key,
                "model": label,
                "auc": m.get("auc"),
                "f1_illicit": m.get("f1_illicit"),
                "avg_precision": m.get("avg_precision"),
                "ring_detection": caps,
                "explainability": caps,
                "inductive": caps,
            }
        )
    return rows
