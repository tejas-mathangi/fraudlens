"""Central configuration: paths and hyperparameters.

Every path is overridable through an environment variable so the pipeline can run
against a dataset that lives outside the repository -- useful in git worktrees, CI and
containers, where ``data/`` and ``models/`` are not checked in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# config.py lives at <root>/fraudlens/config.py
ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    """Read a path from the environment, falling back to a repo-relative default."""
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Override any entry via the matching ``FRAUDLENS_*`` env var."""

    data_dir: Path = field(
        default_factory=lambda: _env_path(
            "FRAUDLENS_DATA_DIR", ROOT / "data" / "elliptic_bitcoin_dataset"
        )
    )
    models_dir: Path = field(
        default_factory=lambda: _env_path("FRAUDLENS_MODELS_DIR", ROOT / "models")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("FRAUDLENS_ARTIFACTS_DIR", ROOT / "artifacts")
    )
    cache_dir: Path = field(
        default_factory=lambda: _env_path("FRAUDLENS_CACHE_DIR", ROOT / ".cache")
    )
    plots_dir: Path = field(
        default_factory=lambda: _env_path("FRAUDLENS_PLOTS_DIR", ROOT / "notebooks")
    )

    # ── Dataset files ────────────────────────────────────────────────────────
    @property
    def features_csv(self) -> Path:
        return self.data_dir / "elliptic_txs_features.csv"

    @property
    def edges_csv(self) -> Path:
        return self.data_dir / "elliptic_txs_edgelist.csv"

    @property
    def classes_csv(self) -> Path:
        return self.data_dir / "elliptic_txs_classes.csv"

    # ── Produced files ───────────────────────────────────────────────────────
    @property
    def checkpoint(self) -> Path:
        return self.models_dir / "best_model.pt"

    @property
    def rf_checkpoint(self) -> Path:
        return self.models_dir / "random_forest.joblib"

    @property
    def metrics_json(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def graph_cache(self) -> Path:
        return self.cache_dir / "graph.pt"

    @property
    def partition_cache(self) -> Path:
        return self.cache_dir / "partition.json"

    # ── Availability probes ──────────────────────────────────────────────────
    def dataset_available(self) -> bool:
        """True when all three Elliptic CSVs are present."""
        return all(
            p.is_file() for p in (self.features_csv, self.edges_csv, self.classes_csv)
        )

    def checkpoint_available(self) -> bool:
        return self.checkpoint.is_file()

    def ensure_dirs(self) -> None:
        """Create every writable output directory.

        Without this a fresh clone crashes on the first ``torch.save``.
        """
        for d in (self.models_dir, self.artifacts_dir, self.cache_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    """GraphSAGE training hyperparameters."""

    hidden_channels: int = 128
    out_channels: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    resume_lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    eval_every: int = 5
    patience: int = 30
    test_size: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class RingConfig:
    """Thresholds that promote a Louvain community to a reported fraud ring."""

    min_size: int = 3
    min_illicit_ratio: float = 0.4
    min_avg_score: float = 0.4
    min_illicit: int = 2


@dataclass(frozen=True)
class ExportConfig:
    """Budget for the committed demo artifacts.

    These exist to keep ``artifacts/`` committable (~1.5 MB) and the static demo quick
    to load. The caps matter more than they look: on the real dataset the model scores
    33,696 nodes above the high-risk threshold, almost all of them unlabelled, so an
    uncapped searchable index came out at 7.4 MB.
    """

    graph_sample_nodes: int = 1500
    max_rings: int = 25
    n_explanations: int = 20
    #: Confirmed-illicit nodes are always included; these cap the rest.
    nodes_index_high_risk: int = 1500
    nodes_index_licit_sample: int = 1000
    #: Neighbours stored per indexed node.
    nodes_index_neighbours: int = 20
    score_histogram_bins: int = 50
    top_features: int = 25


PATHS = Paths()
TRAIN = TrainConfig()
RINGS = RingConfig()
EXPORT = ExportConfig()

# Elliptic features: the first 93 describe the transaction itself; the remaining 72
# are aggregations over its one-hop neighbourhood.
N_LOCAL_FEATURES = 93
N_FEATURES = 165

HIGH_RISK_THRESHOLD = 0.7
DECISION_THRESHOLD = 0.5

LABEL_ILLICIT = 1
LABEL_LICIT = 0
LABEL_UNKNOWN = -1
