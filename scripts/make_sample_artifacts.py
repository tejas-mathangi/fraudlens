#!/usr/bin/env python3
"""Generate *synthetic* demo artifacts so the console renders on a fresh clone.

The real Elliptic dataset is 697 MB and Kaggle-licensed, so it cannot be committed,
and without it `artifacts/` is empty and every page of the web console shows an empty
state. This script builds a synthetic Elliptic-shaped graph, trains a small model on
it for a few epochs, and runs the ordinary export pipeline over the result.

Every file it writes is tagged ``"synthetic": true``, which makes the console display a
persistent "Sample data" banner. Running `fraudlens export` against the real dataset
overwrites these with ``"synthetic": false`` and the banner disappears.

    python scripts/make_sample_artifacts.py --out web/public/demo
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_NODES = 2600
N_FEATURES = 165
N_LOCAL = 93
N_TIME_STEPS = 49
N_RINGS = 14


def write_synthetic_csvs(target: Path, rng: np.random.Generator) -> None:
    """Write a three-CSV dataset with planted, densely connected fraud rings."""
    target.mkdir(parents=True, exist_ok=True)
    tx_ids = 10_000_000 + np.arange(N_NODES)

    # Assign ring membership first; ring members get shifted features and a dense
    # internal edge structure, which is what makes the planted rings detectable.
    ring_of = np.full(N_NODES, -1, dtype=int)
    cursor = 120
    for ring in range(N_RINGS):
        size = int(rng.integers(12, 55))
        if cursor + size >= N_NODES:
            break
        ring_of[cursor : cursor + size] = ring
        cursor += size + int(rng.integers(10, 40))

    in_ring = ring_of >= 0
    # 72% of ring members are confirmed illicit; outside rings, a 1.5% base rate.
    illicit = np.where(
        in_ring, rng.random(N_NODES) < 0.72, rng.random(N_NODES) < 0.015
    )
    # ~23% of nodes carry no label at all, as in the real dataset.
    unlabelled = rng.random(N_NODES) < 0.23

    features = rng.normal(0, 1, size=(N_NODES, N_FEATURES)).astype(np.float32)
    # Make a handful of features genuinely predictive, split across the local and
    # aggregated blocks so explanations have both kinds to surface.
    for col, shift in ((7, 1.5), (31, -1.2), (88, 1.1), (97, 2.1), (140, -1.8)):
        features[illicit, col] += shift
    # Ring members share a neighbourhood signature in the aggregated block.
    for ring in range(N_RINGS):
        members = ring_of == ring
        if members.any():
            features[members, 110:120] += rng.normal(0, 1, size=10) * 1.4

    time_steps = rng.integers(1, N_TIME_STEPS + 1, size=N_NODES)

    with (target / "elliptic_txs_features.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        for i in range(N_NODES):
            writer.writerow(
                [tx_ids[i], int(time_steps[i]), *np.round(features[i], 4).tolist()]
            )

    edges: set[tuple[int, int]] = set()

    # Dense intra-ring edges: this is the structure Louvain has to find.
    for ring in range(N_RINGS):
        members = np.where(ring_of == ring)[0]
        for a in members:
            for b in rng.choice(members, size=min(5, members.size), replace=False):
                if a != b:
                    edges.add((int(min(a, b)), int(max(a, b))))

    # Background structure, kept deliberately sparse. An earlier version wired two
    # random long-range edges per node, which merged everything into a handful of
    # giant communities and left Louvain with no rings to find at all. Non-ring
    # nodes get local chain edges plus small cliques, so the graph is connected but
    # modular.
    non_ring = np.where(~in_ring)[0]
    for i in range(len(non_ring) - 1):
        a, b = int(non_ring[i]), int(non_ring[i + 1])
        edges.add((min(a, b), max(a, b)))
    for start in range(0, len(non_ring) - 6, 6):
        block = non_ring[start : start + 6]
        for a in block:
            for b in rng.choice(block, size=2, replace=False):
                if a != b:
                    edges.add((int(min(a, b)), int(max(a, b))))

    # A few bridges tie each ring to the surrounding graph without dissolving it.
    for ring in range(N_RINGS):
        members = np.where(ring_of == ring)[0]
        if not members.size or not non_ring.size:
            continue
        for _ in range(2):
            a = int(rng.choice(members))
            b = int(rng.choice(non_ring))
            edges.add((min(a, b), max(a, b)))

    with (target / "elliptic_txs_edgelist.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["txId1", "txId2"])
        for a, b in sorted(edges):
            writer.writerow([tx_ids[a], tx_ids[b]])

    with (target / "elliptic_txs_classes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["txId", "class"])
        for i in range(N_NODES):
            if unlabelled[i]:
                label = "unknown"
            else:
                label = "1" if illicit[i] else "2"
            writer.writerow([tx_ids[i], label])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("web/public/demo"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    import logging

    import torch

    from fraudlens.config import Paths, TrainConfig
    from fraudlens.data import build_graph
    from fraudlens.export import export_all
    from fraudlens.pipeline.baseline import run_baseline
    from fraudlens.pipeline.train import train

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    workdir = Path(tempfile.mkdtemp(prefix="fraudlens-sample-"))
    try:
        data_dir = workdir / "elliptic_bitcoin_dataset"
        print(f"Generating a synthetic {N_NODES}-node graph...")
        write_synthetic_csvs(data_dir, rng)

        paths = Paths(
            data_dir=data_dir,
            models_dir=workdir / "models",
            artifacts_dir=workdir / "artifacts",
            cache_dir=workdir / "cache",
            plots_dir=workdir / "plots",
        )
        paths.ensure_dirs()

        graph = build_graph(paths, use_cache=False)
        print("Training the baseline...")
        run_baseline(graph=graph, paths=paths, n_estimators=60)
        print(f"Training GraphSAGE for {args.epochs} epochs...")
        train(
            graph=graph,
            cfg=TrainConfig(hidden_channels=64, epochs=args.epochs, eval_every=5),
            paths=paths,
        )
        print("Exporting artifacts...")
        export_all(graph=graph, paths=paths)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        written = []
        for src in sorted(paths.artifacts_dir.glob("*.json")):
            payload = json.loads(src.read_text())
            # The banner in the UI keys off this flag.
            payload["synthetic"] = True
            (out / src.name).write_text(json.dumps(payload, separators=(",", ":")))
            written.append((src.name, (out / src.name).stat().st_size))

        total = sum(size for _, size in written)
        for name, size in written:
            print(f"  {name:<24} {size / 1024:8.1f} KB")
        print(f"  {'total':<24} {total / 1024:8.1f} KB -> {out}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
