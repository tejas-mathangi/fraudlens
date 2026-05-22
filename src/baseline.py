import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.graph_builder import build_graph

# ── Output dir ─────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def run_baseline():
    print("=" * 60)
    print("FRAUDLENS — Random Forest Baseline")
    print("=" * 60)

    # ── Load graph data ────────────────────────────────────────────────────────
    data, label_mask, _ = build_graph()

    # ── Extract labeled nodes only ─────────────────────────────────────────────
    print("\n[1] Extracting labeled nodes...")
    X_all = data.x.numpy()               # [203769, 165]
    y_all = data.y.numpy()               # [203769]

    # Keep only nodes with known labels (illicit=1, licit=0)
    labeled_indices = np.where(label_mask.numpy())[0]
    X = X_all[labeled_indices]           # [46564, 165]
    y = y_all[labeled_indices]           # [46564]

    print(f"    Labeled samples : {len(X):,}")
    print(f"    Illicit         : {(y==1).sum():,}")
    print(f"    Licit           : {(y==0).sum():,}")

    # ── Train/Test split ───────────────────────────────────────────────────────
    # Stratified so both splits have the same fraud ratio
    print("\n[2] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    Train: {len(X_train):,} samples")
    print(f"    Test : {len(X_test):,}  samples")

    # ── Class weights ──────────────────────────────────────────────────────────
    # Tell the model to penalize missing fraud more than missing licit
    print("\n[3] Training Random Forest (this takes ~30 seconds)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',   # handles 1:9 imbalance automatically
        random_state=42,
        n_jobs=-1                  # use all CPU cores
    )
    rf.fit(X_train, y_train)
    print("    Training complete ✓")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[4] Evaluating...")
    y_pred      = rf.predict(X_test)
    y_prob      = rf.predict_proba(X_test)[:, 1]   # fraud probability

    auc         = roc_auc_score(y_test, y_prob)
    f1_illicit  = f1_score(y_test, y_pred, pos_label=1)
    f1_macro    = f1_score(y_test, y_pred, average='macro')
    avg_prec    = average_precision_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)

    print(f"\n    ── Baseline Results ──────────────────────")
    print(f"    AUC-ROC              : {auc:.4f}")
    print(f"    Average Precision    : {avg_prec:.4f}")
    print(f"    F1 (illicit class)   : {f1_illicit:.4f}")
    print(f"    F1 (macro)           : {f1_macro:.4f}")
    print(f"    ──────────────────────────────────────────")

    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Licit  Illicit")
    print(f"    Actual Licit    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"    Actual Illicit  {cm[1][0]:5d}   {cm[1][1]:5d}")

    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Licit', 'Illicit']))

    # ── Feature Importance ────────────────────────────────────────────────────
    print("\n[5] Top 10 most important features:")
    importances = rf.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. f{idx+1:<4d}  importance: {importances[idx]:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[6] Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('FraudLens — Random Forest Baseline', fontsize=13, fontweight='bold')

    # Plot 1: Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[0].plot(recall, precision, color='#e74c3c', linewidth=2)
    axes[0].fill_between(recall, precision, alpha=0.15, color='#e74c3c')
    axes[0].axhline(y=(y_test==1).mean(), color='gray',
                    linestyle='--', label=f'Random baseline ({(y_test==1).mean():.2f})')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve\n(AP = {avg_prec:.4f})')
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # Plot 2: Feature importances (top 20)
    top20_idx  = np.argsort(importances)[::-1][:20]
    top20_vals = importances[top20_idx]
    top20_lbls = [f'f{i+1}' for i in top20_idx]
    axes[1].barh(range(20), top20_vals[::-1], color='#3498db')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels(top20_lbls[::-1], fontsize=8)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 20 Feature Importances')

    plt.tight_layout()
    plot_path = os.path.join(NOTEBOOKS_DIR, 'baseline_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved to notebooks/baseline_plots.png")

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE — record these numbers")
    print("=" * 60)
    print(f"\n  AUC-ROC            : {auc:.4f}")
    print(f"  F1 (illicit)       : {f1_illicit:.4f}")
    print(f"  Average Precision  : {avg_prec:.4f}")
    print(f"\n  → GNN must beat these to justify the approach")

    return {
        'auc':       auc,
        'f1':        f1_illicit,
        'avg_prec':  avg_prec,
    }


if __name__ == '__main__':
    run_baseline()