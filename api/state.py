"""Backend state and the live/artifact mode split.

The service must boot and answer requests in two very different situations:

* **live** -- the 697 MB dataset and a trained checkpoint are present, so we hold the
  real graph in memory and can score, cluster and explain any of the 203,769 nodes on
  demand.
* **artifact** -- neither is present (a fresh clone, a reviewer's laptop, a free-tier
  container), so we serve the precomputed JSON in ``artifacts/``.

The HTTP contract is identical in both modes, which is what lets the frontend be
unaware of the difference beyond a badge in the top bar.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Literal

import numpy as np

from fraudlens.config import (
    DECISION_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    N_LOCAL_FEATURES,
    PATHS,
    Paths,
)

log = logging.getLogger("fraudlens.api")

Mode = Literal["live", "artifact"]
Status = Literal["warming", "ready", "error"]


class Backend:
    """Holds whatever data the service managed to load.

    Loading is deliberately lazy and happens on a worker thread: parsing the CSVs and
    running a forward pass over 203k nodes takes ~40 s, and a web server that blocks
    that long on startup looks broken.
    """

    def __init__(self, paths: Paths = PATHS) -> None:
        self.paths = paths
        self.mode: Mode = "live" if self._can_go_live() else "artifact"
        self.status: Status = "warming" if self.mode == "live" else "ready"
        self.error: str | None = None

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._artifacts: dict[str, Any] = {}

        # Live-mode state
        self.graph = None
        self.model = None
        self.scores: np.ndarray | None = None
        self.indptr: np.ndarray | None = None
        self.indices: np.ndarray | None = None
        self.rings: list = []
        self._tx_to_idx: dict[int, int] = {}

        if self.mode == "artifact":
            self._load_artifacts()

    # ── Mode detection ───────────────────────────────────────────────────────
    def _can_go_live(self) -> bool:
        return self.paths.dataset_available() and self.paths.checkpoint_available()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    # ── Artifact mode ────────────────────────────────────────────────────────
    def _load_artifacts(self) -> None:
        """Read every artifact JSON that exists; missing ones degrade gracefully."""
        for name in (
            "overview",
            "graph_sample",
            "rings",
            "explanations",
            "nodes_index",
            "features",
        ):
            path = self.paths.artifacts_dir / f"{name}.json"
            if path.is_file():
                try:
                    self._artifacts[name] = json.loads(path.read_text())
                except json.JSONDecodeError as exc:
                    log.warning("Bad artifact %s: %s", path, exc)
        if not self._artifacts:
            log.warning(
                "No artifacts in %s and no dataset available -- the API will return "
                "empty payloads. Run `fraudlens export` to generate them.",
                self.paths.artifacts_dir,
            )
        self._node_by_idx = {
            n["idx"]: n
            for n in self._artifacts.get("nodes_index", {}).get("nodes", [])
        }
        self._exp_by_idx = {
            e["node"]["idx"]: e
            for e in self._artifacts.get("explanations", {}).get("explanations", [])
        }

    def artifact(self, name: str, default: Any = None) -> Any:
        return self._artifacts.get(name, default if default is not None else {})

    # ── Live mode loading ────────────────────────────────────────────────────
    def start_loading(self) -> None:
        """Kick off background loading if we are in live mode."""
        if self.mode != "live" or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._load_live, daemon=True)
        self._thread.start()

    def _load_live(self) -> None:
        try:
            from fraudlens.data import build_graph
            from fraudlens.export import _adjacency
            from fraudlens.model import fraud_scores, load_model
            from fraudlens.pipeline.rings import detect_rings

            log.info("Loading graph...")
            graph = build_graph(self.paths)
            log.info("Loading checkpoint...")
            model = load_model(graph.num_features, self.paths.checkpoint)
            log.info("Scoring %d nodes...", graph.num_nodes)
            scores = fraud_scores(model, graph.data).numpy()
            indptr, indices = _adjacency(graph)
            log.info("Detecting fraud rings...")
            rings, _, _ = detect_rings(graph, fraud_prob=scores, paths=self.paths)

            with self._lock:
                self.graph = graph
                self.model = model
                self.scores = scores
                self.indptr = indptr
                self.indices = indices
                self.rings = rings
                self._tx_to_idx = graph.node_to_idx
                self.status = "ready"
            log.info("Backend ready in live mode")
        except Exception as exc:  # pragma: no cover - surfaced through /api/health
            log.exception("Live loading failed; falling back to artifact mode")
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.mode = "artifact"
                self.status = "ready"
            self._load_artifacts()

    # ── Shared helpers ───────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        overview = self.artifact("overview", {})
        dataset = overview.get("dataset", {}) if self.mode == "artifact" else {}
        return {
            "status": self.status,
            "mode": self.mode,
            "model_loaded": self.model is not None,
            "graph_loaded": self.graph is not None,
            "nodes": self.graph.num_nodes if self.graph else dataset.get("nodes", 0),
            "edges": self.graph.num_edges if self.graph else dataset.get("edges", 0),
            "artifacts": sorted(self._artifacts),
            "error": self.error,
            "thresholds": {
                "decision": DECISION_THRESHOLD,
                "high_risk": HIGH_RISK_THRESHOLD,
            },
            "n_local_features": N_LOCAL_FEATURES,
        }

    def resolve(self, query: str) -> int | None:
        """Resolve a node index or an Elliptic ``txId`` to a node index.

        The original dashboard accepted only integer indices even though the txId map
        was already in memory, which made it impossible to look up a transaction you
        had an actual id for.
        """
        query = query.strip()
        if not query.isdigit():
            return None
        value = int(query)

        if self.mode == "live" and self.graph is not None:
            if value < self.graph.num_nodes:
                return value
            return self._tx_to_idx.get(value)

        nodes = self._node_by_idx
        if value in nodes:
            return value
        for idx, record in nodes.items():
            if record.get("tx_id") == value:
                return idx
        return None


_backend: Backend | None = None


def get_backend() -> Backend:
    """FastAPI dependency: the process-wide backend singleton."""
    global _backend
    if _backend is None:
        _backend = Backend()
    return _backend


def reset_backend(paths: Paths | None = None) -> Backend:
    """Replace the singleton. Used by the tests to point at fixture artifacts."""
    global _backend
    _backend = Backend(paths or PATHS)
    return _backend
