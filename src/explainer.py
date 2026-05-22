import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.graph_builder import build_graph
from src.model import GraphSAGE
from src.train import get_train_test_masks, compute_class_weights

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def explain_node(model, data, node_idx, n_epochs=200, lr=0.01):
    """
    Gradient-based explanation for a single node.
    Computes which input features most influence the fraud prediction.
    """
    model.eval()

    from torch_geometric.utils import k_hop_subgraph
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx      = int(node_idx),
        num_hops      = 2,
        edge_index    = data.edge_index,
        relabel_nodes = True,
        num_nodes     = data.num_nodes
    )

    target_node_idx = mapping.item()

    # Get prediction info
    with torch.no_grad():
        full_logits  = model(data.x, data.edge_index)
        target_class = full_logits[node_idx].argmax().item()
        target_prob  = F.softmax(full_logits[node_idx], dim=0)[target_class].item()

    # Gradient-based feature importance
    # Enable gradients on input features for the subgraph
    sub_x = data.x[subset].clone().requires_grad_(True)

    logits = model(sub_x, sub_edge_index)
    score  = F.softmax(logits[target_node_idx], dim=0)[target_class]
    score.backward()

    # Feature importance = |gradient| of the target node's features
    feature_mask = sub_x.grad[target_node_idx].abs().detach().numpy()
    # Normalize to [0, 1]
    if feature_mask.max() > 0:
        feature_mask = feature_mask / feature_mask.max()

    # Edge importance: how much does each neighbor's presence matter?
    # Approximate: use fraud score difference between node and neighbors
    edge_mask = np.ones(sub_edge_index.shape[1])
    with torch.no_grad():
        node_scores = F.softmax(model(data.x[subset], sub_edge_index), dim=1)[:, 1].numpy()
        for e_idx in range(sub_edge_index.shape[1]):
            src = sub_edge_index[0, e_idx].item()
            dst = sub_edge_index[1, e_idx].item()
            # Edge importance = how fraudulent is the source neighbor
            edge_mask[e_idx] = float(node_scores[src])

    # Normalize edge mask
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()

    return (edge_mask, feature_mask,
            sub_edge_index, subset, target_node_idx,
            target_class, target_prob)


def run_explainer():
    print("=" * 60)
    print("FRAUDLENS — GNNExplainer")
    print("=" * 60)

    # ── Load everything ────────────────────────────────────────────────────────
    data, label_mask, _ = build_graph()
    train_mask, test_mask = get_train_test_masks(label_mask, data.y)
    class_weights = compute_class_weights(data.y, train_mask)

    model = GraphSAGE(in_channels=data.num_node_features,
                      hidden_channels=128, out_channels=2, dropout=0.3)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))
    model.eval()
    print("\nModel loaded ✓")

    # ── Pick nodes to explain ──────────────────────────────────────────────────
    # Find high-confidence illicit nodes from the test set
    print("\n[1] Finding nodes to explain...")
    with torch.no_grad():
        logits    = model(data.x, data.edge_index)
        probs     = F.softmax(logits, dim=1)
        fraud_prob = probs[:, 1].numpy()

    y_np = data.y.numpy()
    test_indices = torch.where(test_mask)[0].numpy()

    # True illicit nodes in test set, sorted by confidence
    illicit_test = [i for i in test_indices if y_np[i] == 1]
    illicit_test.sort(key=lambda i: fraud_prob[i], reverse=True)

    # True licit nodes in test set, sorted by confidence
    licit_test = [i for i in test_indices if y_np[i] == 0]
    licit_test.sort(key=lambda i: fraud_prob[i])

    # Pick top 3 illicit and 1 licit for explanation
    nodes_to_explain = illicit_test[:3] + licit_test[:1]
    labels_to_explain = ['Illicit #1', 'Illicit #2', 'Illicit #3', 'Licit (contrast)']

    print(f"    Explaining {len(nodes_to_explain)} nodes")
    for name, idx in zip(labels_to_explain, nodes_to_explain):
        print(f"    {name}: node {idx}, fraud score {fraud_prob[idx]:.4f}")

    # ── Run GNNExplainer on each node ─────────────────────────────────────────
    print("\n[2] Running GNNExplainer (this takes ~1-2 minutes)...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('FraudLens — GNNExplainer: Why Was This Node Flagged?',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for plot_idx, (node_idx, label) in enumerate(zip(nodes_to_explain, labels_to_explain)):
        print(f"    Explaining {label} (node {node_idx})...")

        result = explain_node(model, data, node_idx, n_epochs=150)
        (edge_mask, feature_mask, sub_edge_index,
         subset, target_local_idx, pred_class, pred_prob) = result

        pred_label = 'ILLICIT' if pred_class == 1 else 'LICIT'
        color_pred = '#e74c3c' if pred_class == 1 else '#2ecc71'

        # ── Build subgraph for visualization ──────────────────────────────────
        ax = axes[plot_idx]
        G_sub = nx.DiGraph()

        subset_np = subset.numpy()
        G_sub.add_nodes_from(range(len(subset_np)))

        # Add edges weighted by explanation mask
        edge_arr = sub_edge_index.numpy()
        for e_idx in range(edge_arr.shape[1]):
            src, dst = edge_arr[0, e_idx], edge_arr[1, e_idx]
            G_sub.add_edge(src, dst, weight=float(edge_mask[e_idx]))

        # Limit to top edges for clarity
        top_edges = sorted(G_sub.edges(data=True),
                           key=lambda x: x[2]['weight'], reverse=True)[:40]
        G_display = nx.DiGraph()
        G_display.add_nodes_from(G_sub.nodes())
        G_display.add_edges_from([(u, v) for u, v, _ in top_edges])

        pos = nx.spring_layout(G_display, seed=42, k=1.2)

        # Node colors
        node_colors, node_sizes = [], []
        for n in G_display.nodes():
            global_idx = subset_np[n]
            true_label = y_np[global_idx]
            if n == target_local_idx:
                node_colors.append('#f39c12')   # orange = target node
                node_sizes.append(500)
            elif true_label == 1:
                node_colors.append('#e74c3c')   # red = illicit neighbor
                node_sizes.append(200)
            elif true_label == 0:
                node_colors.append('#2ecc71')   # green = licit neighbor
                node_sizes.append(150)
            else:
                node_colors.append('#bdc3c7')   # gray = unknown
                node_sizes.append(100)

        # Edge colors by importance
        edge_weights = [G_display[u][v].get('weight', 0.1)
                        for u, v in G_display.edges()]
        edge_colors  = [plt.cm.Reds(0.3 + 0.7 * w) for w in edge_weights]
        edge_widths  = [0.5 + 3.0 * w for w in edge_weights]

        nx.draw_networkx_edges(G_display, pos, ax=ax,
                               edge_color=edge_colors,
                               width=edge_widths, alpha=0.7,
                               arrows=True, arrowsize=10)
        nx.draw_networkx_nodes(G_display, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes, alpha=0.95)

        # Highlight target node label
        if target_local_idx in pos:
            tx, ty = pos[target_local_idx]
            ax.annotate('TARGET', xy=(tx, ty),
                        xytext=(tx + 0.1, ty + 0.15),
                        fontsize=7, color='#f39c12', fontweight='bold')

        # Legend
        legend_elements = [
            mpatches.Patch(color='#f39c12', label='Target node'),
            mpatches.Patch(color='#e74c3c', label='Illicit neighbor'),
            mpatches.Patch(color='#2ecc71', label='Licit neighbor'),
            mpatches.Patch(color='#bdc3c7', label='Unknown neighbor'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)
        ax.set_title(
            f'{label} | Prediction: {pred_label} ({pred_prob:.3f})\n'
            f'Node {node_idx} | Fraud score: {fraud_prob[node_idx]:.4f}',
            color=color_pred, fontweight='bold', fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(NOTEBOOKS_DIR, 'explanations.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n    Saved to notebooks/explanations.png")

    # ── Feature importance summary ─────────────────────────────────────────────
    print("\n[3] Feature importance for top illicit node...")
    result = explain_node(model, data, nodes_to_explain[0], n_epochs=150)
    _, feature_mask, _, _, _, _, _ = result

    top10_feat = np.argsort(feature_mask)[::-1][:10]
    print(f"    Top 10 features driving fraud prediction:")
    for rank, f_idx in enumerate(top10_feat, 1):
        print(f"    {rank:2d}. Feature f{f_idx+1:<4d}  importance: {feature_mask[f_idx]:.4f}")

    print("\n" + "=" * 60)
    print("EXPLAINER COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_explainer()