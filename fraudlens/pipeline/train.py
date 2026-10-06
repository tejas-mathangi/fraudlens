"""Train the GraphSAGE classifier.

Supports a fresh run and resuming from the saved checkpoint. The resume path used to
live in a separate ``train_more.py`` that reset the best-F1 tracker to zero, so the
first evaluation always "improved" and overwrote a better checkpoint with a worse one.
Resuming now seeds the tracker from the checkpoint's own recorded score.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from fraudlens.config import PATHS, TRAIN, Paths, TrainConfig
from fraudlens.data import Graph, build_graph
from fraudlens.metrics import classification_metrics, update_metrics
from fraudlens.model import GraphSAGE, load_model

log = logging.getLogger(__name__)


def get_train_test_masks(
    label_mask: torch.Tensor,
    y: torch.Tensor,
    test_size: float = TRAIN.test_size,
    random_state: int = TRAIN.seed,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified split over labelled nodes only, returned as boolean node masks."""
    labeled_indices = torch.where(label_mask)[0].numpy()
    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y[label_mask].numpy(),
    )

    n_nodes = label_mask.shape[0]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def compute_class_weights(y: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights for cross-entropy.

    The labelled set is roughly 1:9 illicit:licit. Unweighted, the model reaches ~90%
    accuracy by calling everything licit, which is useless for fraud detection. The
    weight ``total / (n_classes * count)`` makes a missed fraud ~5x costlier than a
    false alarm.
    """
    y_train = y[train_mask]
    n_licit = int((y_train == 0).sum())
    n_illicit = int((y_train == 1).sum())
    total = n_licit + n_illicit
    if not n_licit or not n_illicit:
        raise ValueError("training split must contain both classes")
    return torch.tensor(
        [total / (2 * n_licit), total / (2 * n_illicit)], dtype=torch.float
    )


@torch.no_grad()
def evaluate(
    model: GraphSAGE,
    data,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> dict[str, object]:
    """Evaluate on ``mask``, returning both scalars and the raw arrays."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask], weight=class_weights)

    probs = F.softmax(logits[mask], dim=1)[:, 1].numpy()
    y_true = data.y[mask].numpy()
    scores = classification_metrics(y_true, probs)

    return {
        "loss": float(loss),
        "auc": scores["auc"],
        "f1": scores["f1_illicit"],
        "avg_precision": scores["avg_precision"],
        "y_true": y_true,
        "y_prob": probs,
        "metrics": scores,
    }


def train(
    graph: Graph | None = None,
    cfg: TrainConfig = TRAIN,
    paths: Paths = PATHS,
    resume: bool = False,
    epochs: int | None = None,
) -> dict[str, object]:
    """Run the training loop and persist the best checkpoint plus its metrics.

    Returns the final test metrics. The checkpoint stores the score it achieved, so a
    later ``--resume`` can only overwrite it by genuinely beating it.
    """
    paths.ensure_dirs()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    graph = graph or build_graph(paths)
    data, label_mask = graph.data, graph.label_mask

    train_mask, test_mask = get_train_test_masks(
        label_mask, data.y, cfg.test_size, cfg.seed
    )
    class_weights = compute_class_weights(data.y, train_mask)
    log.info(
        "Class weights -> licit %.3f, illicit %.3f",
        class_weights[0],
        class_weights[1],
    )

    model = GraphSAGE(
        in_channels=graph.num_features,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.out_channels,
        dropout=cfg.dropout,
    )

    best_f1 = 0.0
    start_epoch = 1
    lr = cfg.lr

    if resume:
        if not paths.checkpoint_available():
            raise FileNotFoundError(f"no checkpoint to resume from at {paths.checkpoint}")
        blob = torch.load(paths.checkpoint, map_location="cpu", weights_only=True)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state)
        # Older checkpoints are a bare state_dict with no recorded score; fall back to
        # measuring the loaded model so we never overwrite it with something worse.
        if isinstance(blob, dict) and "best_f1" in blob:
            best_f1 = float(blob["best_f1"])
        else:
            best_f1 = float(evaluate(model, data, test_mask, class_weights)["f1"])
        start_epoch = int(blob.get("epoch", 0)) + 1 if isinstance(blob, dict) else 1
        lr = cfg.resume_lr
        log.info("Resuming from F1 %.4f at lr %g", best_f1, lr)

    total_epochs = epochs or cfg.epochs
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
        "val_auc": [],
    }
    stale_evals = 0
    max_stale = max(1, cfg.patience // cfg.eval_every)
    best_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + total_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[train_mask], data.y[train_mask], weight=class_weights
        )
        loss.backward()
        optimizer.step()

        is_eval_epoch = epoch % cfg.eval_every == 0 or epoch == start_epoch
        if not is_eval_epoch:
            continue

        tr = evaluate(model, data, train_mask, class_weights)
        va = evaluate(model, data, test_mask, class_weights)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(tr["loss"], 4))
        history["train_f1"].append(tr["f1"])
        history["val_loss"].append(round(va["loss"], 4))
        history["val_f1"].append(va["f1"])
        history["val_auc"].append(va["auc"])
        scheduler.step(va["f1"])

        log.info(
            "epoch %4d  train_loss %.4f  train_f1 %.4f  val_f1 %.4f  val_auc %.4f",
            epoch,
            tr["loss"],
            tr["f1"],
            va["f1"],
            va["auc"],
        )

        if va["f1"] > best_f1:
            best_f1, best_epoch, stale_evals = va["f1"], epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "epoch": epoch,
                    "config": vars(cfg),
                },
                paths.checkpoint,
            )
        else:
            stale_evals += 1
            if stale_evals >= max_stale:
                log.info(
                    "Early stop at epoch %d (%d evals without improvement)",
                    epoch,
                    stale_evals,
                )
                break

    log.info("Best val F1 %.4f at epoch %d", best_f1, best_epoch)

    # Reload the best weights before reporting final numbers.
    blob = torch.load(paths.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(blob.get("state_dict", blob))
    final = evaluate(model, data, test_mask, class_weights)

    payload = dict(final["metrics"])
    payload["history"] = history
    payload["best_epoch"] = best_epoch
    payload["architecture"] = {
        "layers": 3,
        "hidden_channels": cfg.hidden_channels,
        "dropout": cfg.dropout,
        "in_channels": graph.num_features,
    }
    update_metrics(paths.metrics_json, models={"graphsage": payload})

    log.info(
        "Test AUC %.4f  F1 %.4f  AP %.4f",
        final["auc"],
        final["f1"],
        final["avg_precision"],
    )
    return final


def evaluate_checkpoint(
    graph: Graph | None = None,
    cfg: TrainConfig = TRAIN,
    paths: Paths = PATHS,
) -> dict[str, object]:
    """Score the saved checkpoint and record its metrics, without training.

    Useful when a trained model already exists: retraining to refresh
    ``metrics.json`` would risk replacing a good checkpoint with a worse one, and on
    this graph costs 10+ minutes for numbers we already have the weights for.
    """
    paths.ensure_dirs()
    if not paths.checkpoint_available():
        raise FileNotFoundError(f"no checkpoint at {paths.checkpoint}")

    graph = graph or build_graph(paths)
    data, label_mask = graph.data, graph.label_mask
    train_mask, test_mask = get_train_test_masks(
        label_mask, data.y, cfg.test_size, cfg.seed
    )
    class_weights = compute_class_weights(data.y, train_mask)

    model = load_model(in_channels=graph.num_features, checkpoint=paths.checkpoint)
    blob = torch.load(paths.checkpoint, map_location="cpu", weights_only=True)

    result = evaluate(model, data, test_mask, class_weights)
    payload = dict(result["metrics"])
    payload["architecture"] = {
        "layers": 3,
        "hidden_channels": model.bn1.num_features,
        "dropout": cfg.dropout,
        "in_channels": graph.num_features,
    }
    if isinstance(blob, dict) and "epoch" in blob:
        payload["best_epoch"] = int(blob["epoch"])
    update_metrics(paths.metrics_json, models={"graphsage": payload})

    log.info(
        "Checkpoint on the held-out test split: AUC %.4f  F1 %.4f  AP %.4f",
        result["auc"],
        result["f1"],
        result["avg_precision"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the FraudLens GraphSAGE model")
    parser.add_argument("--epochs", type=int, default=None, help="epochs to run")
    parser.add_argument(
        "--resume", action="store_true", help="continue from models/best_model.pt"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(resume=args.resume, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
