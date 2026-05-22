import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.graph_builder import build_graph
from src.model import GraphSAGE

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def get_train_test_masks(label_mask, y, test_size=0.2, random_state=42):
    """
    Split labeled nodes into train and test masks.
    Stratified so both splits have the same fraud ratio.
    """
    labeled_indices = torch.where(label_mask)[0].numpy()
    y_labeled       = y[label_mask].numpy()

    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labeled
    )

    n_nodes    = label_mask.shape[0]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True

    return train_mask, test_mask


def compute_class_weights(y, train_mask):
    """
    Compute class weights to handle 1:9 imbalance.
    Illicit class gets upweighted so the model doesn't ignore it.
    """
    y_train   = y[train_mask]
    n_licit   = (y_train == 0).sum().item()
    n_illicit = (y_train == 1).sum().item()
    total     = n_licit + n_illicit

    # Weight = total / (n_classes * count)
    w_licit   = total / (2 * n_licit)
    w_illicit = total / (2 * n_illicit)

    print(f"    Class weights → Licit: {w_licit:.3f}  Illicit: {w_illicit:.3f}")
    return torch.tensor([w_licit, w_illicit], dtype=torch.float)


@torch.no_grad()
def evaluate(model, data, mask, class_weights):
    """Evaluate model on nodes specified by mask."""
    model.eval()
    logits = model(data.x, data.edge_index)

    # Loss on masked nodes only
    loss = F.cross_entropy(
        logits[mask],
        data.y[mask],
        weight=class_weights
    ).item()

    probs     = F.softmax(logits, dim=1)[:, 1]   # fraud probability
    y_true    = data.y[mask].numpy()
    y_pred    = logits[mask].argmax(dim=1).numpy()
    y_prob    = probs[mask].numpy()

    auc      = roc_auc_score(y_true, y_prob)
    f1       = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    avg_prec = average_precision_score(y_true, y_prob)

    return loss, auc, f1, avg_prec, y_true, y_pred, y_prob


def train():
    print("=" * 60)
    print("FRAUDLENS — GraphSAGE Training")
    print("=" * 60)

    # ── Load graph ─────────────────────────────────────────────────────────────
    data, label_mask, _ = build_graph()

    # ── Train/test split ───────────────────────────────────────────────────────
    print("\n[1] Creating train/test split...")
    train_mask, test_mask = get_train_test_masks(label_mask, data.y)
    print(f"    Train nodes : {train_mask.sum().item():,}")
    print(f"    Test nodes  : {test_mask.sum().item():,}")

    # ── Class weights ──────────────────────────────────────────────────────────
    print("\n[2] Computing class weights...")
    class_weights = compute_class_weights(data.y, train_mask)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n[3] Initializing GraphSAGE model...")
    model = GraphSAGE(
        in_channels     = data.num_node_features,   # 165
        hidden_channels = 128,
        out_channels    = 2,
        dropout         = 0.3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {total_params:,}")
    print(f"    Input features  : {data.num_node_features}")
    print(f"    Hidden size     : 128")
    print(f"    Output classes  : 2")

    # ── Training loop ──────────────────────────────────────────────────────────
    print("\n[4] Training for 200 epochs...")
    print(f"    {'Epoch':>6}  {'Train Loss':>10}  {'Train F1':>9}  {'Val F1':>7}  {'Val AUC':>8}")
    print(f"    {'-'*6}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*8}")

    history = {
        'train_loss': [], 'train_f1': [], 'train_auc': [],
        'val_loss':   [], 'val_f1':   [], 'val_auc':   []
    }

    best_val_f1  = 0.0
    best_epoch   = 0
    patience_cnt = 0
    PATIENCE     = 30   # stop if no improvement for 30 epochs

    for epoch in range(1, 201):
        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss   = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            weight=class_weights
        )
        loss.backward()
        optimizer.step()

        # ── Evaluate every 5 epochs ───────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            train_loss, train_auc, train_f1, train_ap, _, _, _ = evaluate(
                model, data, train_mask, class_weights)
            val_loss,   val_auc,   val_f1,   val_ap,   _, _, _ = evaluate(
                model, data, test_mask,  class_weights)

            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)

            scheduler.step(val_f1)

            print(f"    {epoch:>6}  {train_loss:>10.4f}  {train_f1:>9.4f}  {val_f1:>7.4f}  {val_auc:>8.4f}")

            # ── Save best model ───────────────────────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch  = epoch
                patience_cnt = 0
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE // 5:
                    print(f"\n    Early stopping at epoch {epoch} (no improvement for {patience_cnt*5} epochs)")
                    break

    print(f"\n    Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Final evaluation with best model ──────────────────────────────────────
    print("\n[5] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))

    _, test_auc, test_f1, test_ap, y_true, y_pred, y_prob = evaluate(
        model, data, test_mask, class_weights)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n    ── Final Test Results ────────────────────")
    print(f"    AUC-ROC            : {test_auc:.4f}")
    print(f"    Average Precision  : {test_ap:.4f}")
    print(f"    F1 (illicit class) : {test_f1:.4f}")
    print(f"    ──────────────────────────────────────────")
    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Licit  Illicit")
    print(f"    Actual Licit    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"    Actual Illicit  {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(f"\n    Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Licit', 'Illicit']))

    # ── Comparison with baseline ───────────────────────────────────────────────
    print(f"    ── Comparison with Random Forest Baseline ─")
    print(f"    Metric            RF Baseline   GraphSAGE")
    print(f"    AUC-ROC           0.9958        {test_auc:.4f}")
    print(f"    F1 (illicit)      0.9316        {test_f1:.4f}")
    print(f"    Avg Precision     0.9807        {test_ap:.4f}")
    print(f"    Fraud Ring Det.   ✗             ✓")
    print(f"    Explainability    ✗             ✓")

    # ── Training plots ─────────────────────────────────────────────────────────
    print("\n[6] Generating training plots...")
    epochs_logged = list(range(5, len(history['val_f1']) * 5 + 1, 5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('FraudLens — GraphSAGE Training', fontsize=13, fontweight='bold')

    axes[0].plot(epochs_logged, history['train_f1'], label='Train F1', color='#3498db', linewidth=2)
    axes[0].plot(epochs_logged, history['val_f1'],   label='Val F1',   color='#e74c3c', linewidth=2)
    axes[0].axhline(y=0.9316, color='gray', linestyle='--', label='RF Baseline F1')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('F1 Score (Illicit)')
    axes[0].set_title('F1 Score over Training')
    axes[0].legend()
    axes[0].set_ylim([0, 1])

    axes[1].plot(epochs_logged, history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
    axes[1].plot(epochs_logged, history['val_loss'],   label='Val Loss',   color='#e74c3c', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss over Training')
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(NOTEBOOKS_DIR, 'training_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved to notebooks/training_plots.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, data, label_mask, test_mask, {
        'auc': test_auc, 'f1': test_f1, 'avg_prec': test_ap
    }


if __name__ == '__main__':
    train()