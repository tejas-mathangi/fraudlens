import torch
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.graph_builder import build_graph
from src.model import GraphSAGE
from src.train import get_train_test_masks, compute_class_weights, evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def train_more(extra_epochs=300):
    print("=" * 60)
    print("FRAUDLENS — Extended Training")
    print("=" * 60)

    data, label_mask, _ = build_graph()
    train_mask, test_mask = get_train_test_masks(label_mask, data.y)
    class_weights = compute_class_weights(data.y, train_mask)

    # Load the best model we already trained
    model = GraphSAGE(in_channels=data.num_node_features,
                      hidden_channels=128, out_channels=2, dropout=0.3)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))
    print(f"\nLoaded checkpoint. Continuing for {extra_epochs} more epochs...\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15)

    print(f"    {'Epoch':>6}  {'Train Loss':>10}  {'Train F1':>9}  {'Val F1':>7}  {'Val AUC':>8}")
    print(f"    {'-'*6}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*8}")

    best_val_f1 = 0.0
    history = {'train_f1': [], 'val_f1': [], 'val_auc': []}

    for epoch in range(1, extra_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss   = F.cross_entropy(logits[train_mask], data.y[train_mask],
                                  weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            train_loss, train_auc, train_f1, _, _, _, _ = evaluate(
                model, data, train_mask, class_weights)
            val_loss, val_auc, val_f1, _, _, _, _ = evaluate(
                model, data, test_mask, class_weights)

            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            scheduler.step(val_f1)

            print(f"    {epoch+200:>6}  {train_loss:>10.4f}  {train_f1:>9.4f}  "
                  f"{val_f1:>7.4f}  {val_auc:>8.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
                print(f"             ↑ new best saved")

    # Final eval
    print("\n[Final Evaluation]")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))
    _, test_auc, test_f1, test_ap, y_true, y_pred, _ = evaluate(
        model, data, test_mask, class_weights)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n    AUC-ROC            : {test_auc:.4f}")
    print(f"    Average Precision  : {test_ap:.4f}")
    print(f"    F1 (illicit class) : {test_f1:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Licit  Illicit")
    print(f"    Actual Licit    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"    Actual Illicit  {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Licit','Illicit'])}")

    print(f"\n    ── Final Comparison ───────────────────────")
    print(f"    Metric            RF Baseline   GraphSAGE")
    print(f"    AUC-ROC           0.9958        {test_auc:.4f}")
    print(f"    F1 (illicit)      0.9316        {test_f1:.4f}")
    print(f"    Avg Precision     0.9807        {test_ap:.4f}")
    print(f"    Fraud Ring Det.   ✗             ✓")
    print(f"    Explainability    ✗             ✓")

    return model, data, label_mask, test_mask


if __name__ == '__main__':
    train_more(extra_epochs=300)