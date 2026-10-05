"""``fraudlens`` command line entry point.

One reproducible pipeline, in dependency order::

    fraudlens eda        # dataset statistics and plots
    fraudlens baseline   # Random Forest comparison
    fraudlens train      # GraphSAGE  (--resume to continue from a checkpoint)
    fraudlens rings      # Louvain fraud ring detection
    fraudlens explain    # gradient saliency explanations
    fraudlens export     # JSON artifacts for the web console
    fraudlens all        # everything above, skipping train unless --train
"""

from __future__ import annotations

import argparse
import logging
import sys

from fraudlens import __version__
from fraudlens.config import PATHS

log = logging.getLogger("fraudlens")


def _check_dataset() -> bool:
    if PATHS.dataset_available():
        return True
    log.error(
        "Elliptic dataset not found under %s\n"
        "  Download it:  python scripts/download_data.py\n"
        "  Or point FRAUDLENS_DATA_DIR at an existing copy.",
        PATHS.data_dir,
    )
    return False


def _check_checkpoint() -> bool:
    if PATHS.checkpoint_available():
        return True
    log.error(
        "No trained model at %s\n  Train one:  fraudlens train", PATHS.checkpoint
    )
    return False


def cmd_eda(args: argparse.Namespace) -> int:
    from fraudlens.pipeline.eda import run_eda

    run_eda()
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from fraudlens.pipeline.baseline import run_baseline

    run_baseline(n_estimators=args.n_estimators)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from fraudlens.pipeline.train import train

    train(resume=args.resume, epochs=args.epochs)
    return 0


def cmd_rings(args: argparse.Namespace) -> int:
    from fraudlens.pipeline.rings import detect_rings

    if not _check_checkpoint():
        return 1
    rings, _, _ = detect_rings(use_cache=not args.no_cache)
    log.info("Detected %d fraud rings", len(rings))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from fraudlens.pipeline.explain import run_explainer

    if not _check_checkpoint():
        return 1
    run_explainer(n=args.n)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from fraudlens.export import export_all

    if not _check_checkpoint():
        return 1
    export_all(out_dir=args.out)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run the whole pipeline. Training is opt-in because it takes a long time."""
    from fraudlens.data import build_graph

    graph = build_graph()

    from fraudlens.pipeline.baseline import run_baseline
    from fraudlens.pipeline.eda import run_eda

    run_eda()
    run_baseline(graph=graph)

    if args.train:
        from fraudlens.pipeline.train import train

        train(graph=graph)
    elif not _check_checkpoint():
        log.error("Pass --train to train a model as part of `all`.")
        return 1

    from fraudlens.export import export_all

    export_all(graph=graph)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudlens",
        description="Graph neural network fraud detection on the Elliptic dataset.",
    )
    parser.add_argument("--version", action="version", version=f"fraudlens {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug-level logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="dataset statistics and plots").set_defaults(func=cmd_eda)

    p = sub.add_parser("baseline", help="train the Random Forest baseline")
    p.add_argument("--n-estimators", type=int, default=100)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("train", help="train GraphSAGE")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("rings", help="detect fraud rings")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_rings)

    p = sub.add_parser("explain", help="explain fraud predictions")
    p.add_argument("--n", type=int, default=4)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("export", help="write JSON artifacts for the web console")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("all", help="run the full pipeline")
    p.add_argument("--train", action="store_true", help="also train from scratch")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    # Every stage reads the raw CSVs, so fail fast with actionable guidance.
    if not _check_dataset():
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
