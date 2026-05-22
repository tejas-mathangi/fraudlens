import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.graph_builder import build_graph
from src.model import GraphSAGE
from src.train import get_train_test_masks, compute_class_weights

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def detect_fraud_rings():
    print("=" * 60)
    print("FRAUDLENS — Fraud Ring Detection")
    print("=" * 60)

    # ── Load graph and model ───────────────────────────────────────────────────
    data, label_mask, _ = build_graph()
    train_mask, test_mask = get_train_test_masks(label_mask, data.y)
    class_weights = compute_class_weights(data.y, train_mask)

    model = GraphSAGE(in_channels=data.num_node_features,
                      hidden_channels=128, out_channels=2, dropout=0.3)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pt')))
    model.eval()
    print("\nModel loaded ✓")

    # ── Step 1: Get fraud probability scores for all nodes ────────────────────
    print("\n[1] Computing fraud scores for all nodes...")
    with torch.no_grad():
        logits     = model(data.x, data.edge_index)
        probs      = F.softmax(logits, dim=1)
        fraud_prob = probs[:, 1].numpy()   # probability of being illicit

    print(f"    Fraud scores computed for {len(fraud_prob):,} nodes")
    print(f"    High-risk nodes (score > 0.7): {(fraud_prob > 0.7).sum():,}")
    print(f"    High-risk nodes (score > 0.5): {(fraud_prob > 0.5).sum():,}")

    # ── Step 2: Build NetworkX graph for Louvain ──────────────────────────────
    print("\n[2] Building NetworkX graph...")
    edge_index_np = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    # Add edges in batches for speed
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    G.add_edges_from(edges)

    print(f"    NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── Step 3: Run Louvain community detection ───────────────────────────────
    print("\n[3] Running Louvain community detection...")
    print("    (this may take 1-2 minutes on the full graph)")
    partition = community_louvain.best_partition(G, random_state=42)

    n_communities = len(set(partition.values()))
    print(f"    Communities detected: {n_communities:,}")

    # Map each node to its community
    community_labels = np.array([partition[i] for i in range(data.num_nodes)])

    # ── Step 4: Identify fraud rings ──────────────────────────────────────────
    print("\n[4] Identifying fraud rings...")

    y_np = data.y.numpy()

    fraud_rings    = []
    community_stats = []

    for comm_id in range(n_communities):
        member_mask   = community_labels == comm_id
        member_indices = np.where(member_mask)[0]
        n_members      = len(member_indices)

        if n_members < 3:   # skip tiny communities
            continue

        # Among labeled members only
        labeled_members = member_indices[y_np[member_indices] != -1]
        if len(labeled_members) == 0:
            continue

        n_illicit  = (y_np[labeled_members] == 1).sum()
        n_licit    = (y_np[labeled_members] == 0).sum()
        illicit_rt = n_illicit / len(labeled_members) if len(labeled_members) > 0 else 0
        avg_fraud_score = fraud_prob[member_indices].mean()

        community_stats.append({
            'community_id':    comm_id,
            'size':            n_members,
            'labeled':         len(labeled_members),
            'illicit':         int(n_illicit),
            'licit':           int(n_licit),
            'illicit_ratio':   illicit_rt,
            'avg_fraud_score': avg_fraud_score,
            'members':         member_indices
        })

        # A fraud ring = community where >40% labeled nodes are illicit
        # AND average fraud score > 0.4
        if illicit_rt > 0.4 and avg_fraud_score > 0.4 and n_illicit >= 2:
            fraud_rings.append({
                'community_id':    comm_id,
                'size':            n_members,
                'illicit_count':   int(n_illicit),
                'illicit_ratio':   illicit_rt,
                'avg_fraud_score': avg_fraud_score,
                'members':         member_indices
            })

    fraud_rings.sort(key=lambda x: x['illicit_ratio'], reverse=True)

    print(f"    Total communities analyzed : {len(community_stats):,}")
    print(f"    Fraud rings detected       : {len(fraud_rings):,}")

    print(f"\n    Top 10 Fraud Rings:")
    print(f"    {'Rank':>4}  {'Comm ID':>7}  {'Size':>6}  {'Illicit':>7}  "
          f"{'Illicit%':>8}  {'Avg Score':>9}")
    print(f"    {'-'*4}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*9}")
    for i, ring in enumerate(fraud_rings[:10], 1):
        print(f"    {i:>4}  {ring['community_id']:>7}  {ring['size']:>6}  "
              f"{ring['illicit_count']:>7}  {ring['illicit_ratio']*100:>7.1f}%  "
              f"{ring['avg_fraud_score']:>9.4f}")

    # ── Step 5: Visualize the top fraud ring ──────────────────────────────────
    print("\n[5] Visualizing top fraud rings...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('FraudLens — Fraud Ring Detection', fontsize=14, fontweight='bold')

    # Plot 1: Community fraud ratio distribution
    ax1 = axes[0]
    stats_df = pd.DataFrame(community_stats)
    ax1.hist(stats_df['illicit_ratio'], bins=50,
             color='#3498db', edgecolor='black', linewidth=0.3)
    ax1.axvline(x=0.4, color='#e74c3c', linestyle='--',
                linewidth=2, label='Fraud ring threshold (40%)')
    ax1.set_xlabel('Illicit Node Ratio in Community')
    ax1.set_ylabel('Number of Communities')
    ax1.set_title('Community Fraud Concentration')
    ax1.legend()

    # Plot 2: Subgraph of the top fraud ring
    ax2 = axes[1]
    if fraud_rings:
        top_ring    = fraud_rings[0]
        ring_nodes  = top_ring['members']

        # Sample up to 60 nodes for visibility
        if len(ring_nodes) > 60:
            ring_nodes = np.random.choice(ring_nodes, 60, replace=False)

        subgraph    = G.subgraph(ring_nodes.tolist())
        pos         = nx.spring_layout(subgraph, seed=42, k=0.8)

        # Color nodes by true label
        node_colors = []
        node_sizes  = []
        for node in subgraph.nodes():
            label = y_np[node]
            score = fraud_prob[node]
            if label == 1:
                node_colors.append('#e74c3c')    # red = confirmed illicit
                node_sizes.append(200)
            elif label == 0:
                node_colors.append('#2ecc71')    # green = confirmed licit
                node_sizes.append(120)
            else:
                # Unknown — color by fraud score
                intensity = score
                node_colors.append(plt.cm.Oranges(0.3 + 0.7 * intensity))
                node_sizes.append(80)

        nx.draw_networkx_edges(subgraph, pos, ax=ax2,
                               alpha=0.3, edge_color='gray', width=0.8)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax2,
                               node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label=f'Illicit ({top_ring["illicit_count"]})'),
            Patch(facecolor='#2ecc71', label='Licit'),
            Patch(facecolor='#f39c12', label='Unknown (scored)')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
        ax2.set_title(
            f'Top Fraud Ring (Community {top_ring["community_id"]})\n'
            f'{top_ring["illicit_ratio"]*100:.1f}% illicit | '
            f'Avg fraud score: {top_ring["avg_fraud_score"]:.3f}'
        )
        ax2.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(NOTEBOOKS_DIR, 'fraud_rings.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    Saved to notebooks/fraud_rings.png")

    # ── Step 6: Summary stats ─────────────────────────────────────────────────
    total_illicit_in_rings = sum(r['illicit_count'] for r in fraud_rings)
    total_illicit = (y_np == 1).sum()

    print(f"\n[6] Fraud Ring Summary:")
    print(f"    Total fraud rings found      : {len(fraud_rings):,}")
    print(f"    Total illicit nodes in rings : {total_illicit_in_rings:,}")
    print(f"    % of all illicit in rings    : {100*total_illicit_in_rings/total_illicit:.1f}%")
    print(f"    Avg ring size                : {np.mean([r['size'] for r in fraud_rings]):.1f} nodes")
    print(f"    Largest ring size            : {max(r['size'] for r in fraud_rings):,} nodes")

    print("\n" + "=" * 60)
    print("FRAUD RING DETECTION COMPLETE")
    print("=" * 60)

    return fraud_rings, community_labels, fraud_prob, G


if __name__ == '__main__':
    detect_fraud_rings()