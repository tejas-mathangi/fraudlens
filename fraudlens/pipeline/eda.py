"""Exploratory data analysis.

Previously this ran as bare top-level script code with no ``__main__`` guard, so
importing the module executed a 30-second analysis as a side effect.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this runs in CI and containers
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from fraudlens.config import PATHS, Paths  # noqa: E402

log = logging.getLogger(__name__)

ILLICIT_COLOR = "#e74c3c"
LICIT_COLOR = "#2ecc71"
UNKNOWN_COLOR = "#95a5a6"


def run_eda(paths: Paths = PATHS, n_corr_features: int = 20) -> dict[str, object]:
    """Print dataset statistics and write ``notebooks/eda_plots.png``."""
    paths.ensure_dirs()

    features = pd.read_csv(paths.features_csv, header=None)
    edges = pd.read_csv(paths.edges_csv)
    classes = pd.read_csv(paths.classes_csv)

    n_features = features.shape[1] - 2
    features.columns = ["txId", "time_step"] + [
        f"f{i}" for i in range(1, n_features + 1)
    ]
    feature_cols = [c for c in features.columns if c.startswith("f")]

    counts = classes["class"].value_counts()
    summary = {
        "transactions": int(len(features)),
        "edges": int(len(edges)),
        "features": n_features,
        "time_steps": int(features["time_step"].nunique()),
        "illicit": int(counts.get("1", 0)),
        "licit": int(counts.get("2", 0)),
        "unknown": int(counts.get("unknown", 0)),
    }
    for key, value in summary.items():
        log.info("%-14s %s", key, f"{value:,}" if isinstance(value, int) else value)

    # ── Plots ────────────────────────────────────────────────────────────────
    merged = features[["txId", "time_step"]].merge(classes, on="txId", how="left")
    per_step = (
        merged.assign(illicit=merged["class"].eq("1"), licit=merged["class"].eq("2"))
        .groupby("time_step")[["illicit", "licit"]]
        .sum()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FraudLens — Elliptic dataset overview", fontsize=14, fontweight="bold")

    labels = ["illicit", "licit", "unknown"]
    values = [summary["illicit"], summary["licit"], summary["unknown"]]
    bars = axes[0].bar(labels, values, color=[ILLICIT_COLOR, LICIT_COLOR, UNKNOWN_COLOR])
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("transactions")
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].plot(per_step.index, per_step["illicit"], color=ILLICIT_COLOR, label="illicit")
    axes[1].plot(per_step.index, per_step["licit"], color=LICIT_COLOR, label="licit")
    axes[1].set_title("Labelled transactions per time step")
    axes[1].set_xlabel("time step")
    axes[1].legend()

    corr = features[feature_cols[:n_corr_features]].sample(1000, random_state=42).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[2], cbar_kws={"shrink": 0.8})
    axes[2].set_title(f"Feature correlation (first {n_corr_features})")

    fig.tight_layout()
    out = paths.plots_dir / "eda_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore the Elliptic dataset")
    parser.add_argument("--corr-features", type=int, default=20)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_eda(n_corr_features=args.corr_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
