"""Shared fixtures.

Every test here runs without the real 697 MB Elliptic dataset: a tiny
Elliptic-shaped CSV triple is written to a temp directory and the configuration is
pointed at it. That is what lets CI exercise the pipeline at all.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

N_NODES = 24
N_FEATURES = 12
# Mirror the real layout proportionally: a local block followed by an aggregated one.
N_LOCAL = 7


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a synthetic three-CSV Elliptic dataset and return its directory."""
    d = tmp_path / "elliptic_bitcoin_dataset"
    d.mkdir(parents=True)

    tx_ids = [100_000 + i for i in range(N_NODES)]

    # features: no header, col0=txId, col1=time_step, then the feature columns
    with (d / "elliptic_txs_features.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        for i, tx in enumerate(tx_ids):
            time_step = (i % 4) + 1
            feats = [round(0.1 * ((i + j) % 10), 3) for j in range(N_FEATURES)]
            w.writerow([tx, time_step, *feats])

    # edges: a connected chain plus a few cross links, so k-hop subgraphs are non-trivial
    edges = [(tx_ids[i], tx_ids[i + 1]) for i in range(N_NODES - 1)]
    edges += [(tx_ids[0], tx_ids[5]), (tx_ids[2], tx_ids[8]), (tx_ids[3], tx_ids[9])]
    # An edge referencing a txId that does not exist -- must be dropped, not crash.
    edges.append((tx_ids[1], 999_999))
    with (d / "elliptic_txs_edgelist.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["txId1", "txId2"])
        w.writerows(edges)

    # classes: a mix of all three label values
    with (d / "elliptic_txs_classes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["txId", "class"])
        for i, tx in enumerate(tx_ids):
            if i % 4 == 0:
                label = "1"  # illicit
            elif i % 4 == 1:
                label = "unknown"
            else:
                label = "2"  # licit
            w.writerow([tx, label])

    return d


@pytest.fixture
def paths(tmp_path: Path, dataset_dir: Path):
    """A :class:`Paths` pointed entirely inside ``tmp_path``."""
    from fraudlens.config import Paths

    p = Paths(
        data_dir=dataset_dir,
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        plots_dir=tmp_path / "plots",
    )
    p.ensure_dirs()
    return p


@pytest.fixture
def graph(paths):
    from fraudlens.data import build_graph

    return build_graph(paths, use_cache=False)


@pytest.fixture
def artifact_paths(tmp_path: Path):
    """Paths with no dataset and no checkpoint, but with minimal artifacts present.

    This is the configuration a reviewer who clones the repo actually gets, so the
    API tests run against it.
    """
    from fraudlens.config import Paths

    p = Paths(
        data_dir=tmp_path / "missing",
        models_dir=tmp_path / "no-models",
        artifacts_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        plots_dir=tmp_path / "plots",
    )
    p.ensure_dirs()

    (p.artifacts_dir / "overview.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "dataset": {"nodes": 24, "edges": 26, "features": 12, "illicit": 6},
                "scoring": {"high_risk": 3, "flagged": 5},
                "score_histogram": {"bin_centers": [0.25, 0.75], "illicit": [1, 5], "licit": [10, 2]},
                "timeline": [{"time_step": 1, "transactions": 6, "illicit": 2, "avg_score": 0.3}],
                "rings": {"count": 1},
                "comparison": [],
            }
        )
    )
    (p.artifacts_dir / "nodes_index.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "nodes": [
                    {
                        "idx": 0,
                        "tx_id": 100_000,
                        "score": 0.91,
                        "label": "illicit",
                        "prediction": "illicit",
                        "time_step": 1,
                        "degree": 2,
                        "neighbors": [{"idx": 1, "score": 0.4, "label": "unknown"}],
                    }
                ],
            }
        )
    )
    (p.artifacts_dir / "rings.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "stats": {"count": 1},
                "rings": [
                    {
                        "community_id": 3,
                        "size": 5,
                        "illicit_count": 4,
                        "illicit_ratio": 0.8,
                        "avg_fraud_score": 0.77,
                        "members": [0, 1, 2],
                        "edges": [[0, 1], [1, 2]],
                        "member_detail": [],
                    }
                ],
            }
        )
    )
    (p.artifacts_dir / "explanations.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "explanations": [
                    {
                        "node": {"idx": 0, "tx_id": 100_000, "score": 0.91, "label": "illicit"},
                        "predicted_class": 1,
                        "probability": 0.91,
                        "driver": "network",
                        "top_features": [
                            {"index": 9, "name": "agg_003", "group": "aggregated", "importance": 1.0}
                        ],
                        "subgraph": {"nodes": [], "edges": []},
                    }
                ],
            }
        )
    )
    (p.artifacts_dir / "graph_sample.json").write_text(
        json.dumps({"synthetic": True, "nodes": [{"idx": 0, "score": 0.9}], "edges": []})
    )
    return p
