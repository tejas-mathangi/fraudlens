"""Random Forest baseline.

The point of the baseline is the honest comparison: Elliptic ships 72 pre-engineered
neighbourhood-aggregate features, so a tabular model already sees a hand-crafted
summary of each transaction's graph context and scores very well. What it cannot do --
at any F1 -- is detect a coordinated ring or explain a flag structurally.

Unlike the original script this one persists the fitted forest, so the API and the
dashboard can reuse it instead of retraining.
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fraudlens.config import PATHS, TRAIN, Paths
from fraudlens.data import Graph, build_graph, feature_group, feature_names
from fraudlens.metrics import classification_metrics, update_metrics

log = logging.getLogger(__name__)


def run_baseline(
    graph: Graph | None = None, paths: Paths = PATHS, n_estimators: int = 100
) -> dict[str, object]:
    """Fit the forest on labelled nodes' raw features and record its metrics."""
    paths.ensure_dirs()
    graph = graph or build_graph(paths)

    labeled = np.where(graph.label_mask.numpy())[0]
    x = graph.data.x.numpy()[labeled]
    y = graph.data.y.numpy()[labeled]
    log.info("Labelled samples %d (illicit %d)", len(x), int((y == 1).sum()))

    # Same split parameters as the GNN so the two numbers are comparable.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TRAIN.test_size, random_state=TRAIN.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=TRAIN.seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    y_prob = clf.predict_proba(x_test)[:, 1]
    payload = classification_metrics(y_test, y_prob)

    names = feature_names(graph.num_features)
    order = np.argsort(clf.feature_importances_)[::-1][:15]
    payload["top_features"] = [
        {
            "index": int(i),
            "name": names[i],
            "group": feature_group(int(i)),
            "importance": round(float(clf.feature_importances_[i]), 5),
        }
        for i in order
    ]
    payload["n_estimators"] = n_estimators

    joblib.dump(clf, paths.rf_checkpoint)
    log.info("Saved forest to %s", paths.rf_checkpoint)

    update_metrics(paths.metrics_json, models={"random_forest": payload})
    log.info(
        "Baseline AUC %.4f  F1 %.4f  AP %.4f",
        payload["auc"],
        payload["f1_illicit"],
        payload["avg_precision"],
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Random Forest baseline")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_baseline(n_estimators=args.n_estimators)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
