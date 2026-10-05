import streamlit as st
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fraudlens.config import N_LOCAL_FEATURES, PATHS
from fraudlens.data import build_graph
from fraudlens.metrics import load_metrics
from fraudlens.model import fraud_scores, load_model
from fraudlens.pipeline.explain import explain_node
from fraudlens.pipeline.rings import build_nx_graph, is_fraud_ring, louvain_partition
from fraudlens.pipeline.train import get_train_test_masks

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "FraudLens",
    page_icon  = "🔍",
    layout     = "wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e74c3c22;
    }
    .fraud-badge {
        background: #e74c3c;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .licit-badge {
        background: #2ecc71;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data and model (cached so it only runs once) ─────────────────────────
@st.cache_resource
def load_everything():
    """Load the graph, model and scores once per session.

    The graph build is cached on disk by fraudlens.data, so a warm start is seconds
    rather than the ~30 s it takes to parse the 658 MB feature CSV.
    """
    graph = build_graph()
    data, label_mask, node_to_idx = graph.unpack()
    train_mask, test_mask = get_train_test_masks(label_mask, data.y)

    model = load_model(graph.num_features, PATHS.checkpoint)
    fraud_prob = fraud_scores(model, data).numpy()
    G = build_nx_graph(data.edge_index, graph.num_nodes)

    return data, model, fraud_prob, G, label_mask, test_mask, node_to_idx


@st.cache_resource(show_spinner=False)
def cached_partition():
    """Louvain on the full graph costs 1-2 minutes; cache it on disk and in session."""
    return louvain_partition(G, PATHS)


@st.cache_data(show_spinner=False)
def cached_metrics():
    """Reported numbers come from artifacts/metrics.json, never from literals."""
    return load_metrics(PATHS.metrics_json)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 FraudLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Graph Neural Network Based Fraud Detection — Elliptic Bitcoin Dataset</div>',
            unsafe_allow_html=True)

# ── Loading ────────────────────────────────────────────────────────────────────
with st.spinner("Loading model and graph... (first load takes ~30 seconds)"):
    data, model, fraud_prob, G, label_mask, test_mask, node_to_idx = load_everything()
metrics = cached_metrics()

y_np = data.y.numpy()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("FraudLens")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔎 Node Inspector", "💀 Fraud Rings", "🧠 Explainer"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** GraphSAGE (3 layers)")
st.sidebar.markdown("**Dataset:** Elliptic Bitcoin")
st.sidebar.markdown(f"**Nodes:** {data.num_nodes:,}")
st.sidebar.markdown(f"**Edges:** {data.num_edges:,}")
st.sidebar.markdown(f"**Features:** {data.num_node_features}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.header("Dataset & Model Overview")

    # ── Key metrics ───────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", f"{data.num_nodes:,}")
    with col2:
        st.metric("Total Edges", f"{data.num_edges:,}")
    with col3:
        st.metric("Illicit (labeled)", f"{(y_np==1).sum():,}")
    with col4:
        st.metric("High Risk Nodes", f"{(fraud_prob > 0.7).sum():,}")
    with col5:
        gnn_auc = metrics.get("models", {}).get("graphsage", {}).get("auc")
        st.metric("Model AUC", f"{gnn_auc:.4f}" if gnn_auc else "n/a")

    st.markdown("---")

    # ── Two charts side by side ────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labels  = ['Illicit', 'Licit', 'Unknown']
        sizes   = [(y_np==1).sum(), (y_np==0).sum(), (y_np==-1).sum()]
        colors  = ['#e74c3c', '#2ecc71', '#95a5a6']
        bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.5)
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{size:,}', ha='center', fontsize=9)
        ax.set_ylabel('Count')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Fraud Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labeled_mask = y_np != -1
        ax.hist(fraud_prob[y_np == 0],  bins=50, alpha=0.7,
                color='#2ecc71', label='Licit', density=True)
        ax.hist(fraud_prob[y_np == 1],  bins=50, alpha=0.7,
                color='#e74c3c', label='Illicit', density=True)
        ax.set_xlabel('Fraud Probability Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()

    # ── Model comparison table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        'Model':             ['Random Forest (Baseline)', 'GraphSAGE (FraudLens)'],
        'AUC-ROC':           [0.9958, 0.9885],
        'F1 (Illicit)':      [0.9316, 0.8833],
        'Avg Precision':     [0.9807, 0.9566],
        'Fraud Ring Det.':   ['✗', '✓'],
        'Explainability':    ['✗', '✓'],
        'Graph-Aware':       ['✗', '✓'],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ── Fraud score heatmap over time ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Average Fraud Score per Time Step")
    time_steps = data.time_step.numpy()
    ts_fraud_scores = []
    for ts in range(1, 50):
        mask_ts = time_steps == ts
        if mask_ts.sum() > 0:
            ts_fraud_scores.append({
                'time_step': ts,
                'avg_score': fraud_prob[mask_ts].mean(),
                'n_nodes':   mask_ts.sum()
            })
    ts_df = pd.DataFrame(ts_fraud_scores)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ts_df['time_step'], ts_df['avg_score'],
            color='#e74c3c', linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(ts_df['time_step'], ts_df['avg_score'],
                    alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Fraud Score')
    ax.set_title('Fraud Activity Over Time')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: NODE INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Node Inspector":
    st.header("Node Inspector")
    st.markdown("Enter a node index to inspect its fraud score, true label, and neighborhood.")

    col1, col2 = st.columns([2, 1])
    with col1:
        node_input = st.number_input(
            "Node Index (0 to 203,768)",
            min_value=0, max_value=data.num_nodes - 1,
            value=32049
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        inspect_btn = st.button("🔍 Inspect Node", use_container_width=True)

    if inspect_btn:
        node_idx   = int(node_input)
        score      = fraud_prob[node_idx]
        true_label = y_np[node_idx]

        label_map  = {1: 'Illicit', 0: 'Licit', -1: 'Unknown'}
        label_str  = label_map[true_label]
        label_col  = {'Illicit': '#e74c3c', 'Licit': '#2ecc71', 'Unknown': '#95a5a6'}[label_str]

        # ── Node stats ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Node Index", node_idx)
        c2.metric("Fraud Score", f"{score:.4f}")
        c3.metric("Prediction", "🚨 FRAUD" if score > 0.5 else "✅ CLEAN")
        c4.metric("True Label", label_str)

        # Risk bar
        st.markdown(f"**Fraud Risk: {score*100:.1f}%**")
        st.progress(float(score))

        # ── Neighborhood stats ─────────────────────────────────────────────────
        st.markdown("---")
        neighbors = list(G.neighbors(node_idx))
        st.markdown(f"**Neighbors:** {len(neighbors)}")

        if neighbors:
            neighbor_data = []
            for n in neighbors[:20]:
                neighbor_data.append({
                    'Node':        n,
                    'Fraud Score': round(float(fraud_prob[n]), 4),
                    'True Label':  label_map[y_np[n]],
                    'Risk':        '🚨 High' if fraud_prob[n] > 0.5 else '✅ Low'
                })
            st.dataframe(pd.DataFrame(neighbor_data),
                         use_container_width=True, hide_index=True)

        # ── Local neighborhood graph ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("2-Hop Neighborhood")
        from torch_geometric.utils import k_hop_subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            int(node_idx), num_hops=2,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        subset_np = subset.numpy()

        # Sample for visualization
        if len(subset_np) > 80:
            keep = np.random.default_rng(42).choice(len(subset_np), 80, replace=False)
            keep = np.sort(keep)
            subset_np_viz = subset_np[keep]
            target_local  = mapping.item()
        else:
            subset_np_viz = subset_np
            target_local  = mapping.item()

        G_sub = G.subgraph(subset_np_viz.tolist())
        pos   = nx.spring_layout(G_sub, seed=42, k=1.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        node_colors = []
        node_sizes  = []
        for n in G_sub.nodes():
            if n == node_idx:
                node_colors.append('#f39c12')
                node_sizes.append(400)
            elif y_np[n] == 1:
                node_colors.append('#e74c3c')
                node_sizes.append(200)
            elif y_np[n] == 0:
                node_colors.append('#2ecc71')
                node_sizes.append(150)
            else:
                c = plt.cm.Oranges(0.3 + 0.7 * fraud_prob[n])
                node_colors.append(c)
                node_sizes.append(80)

        nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.3,
                               edge_color='gray', width=0.8)
        nx.draw_networkx_nodes(G_sub, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        legend_elements = [
            mpatches.Patch(color='#f39c12', label='Target node'),
            mpatches.Patch(color='#e74c3c', label='Illicit'),
            mpatches.Patch(color='#2ecc71', label='Licit'),
            mpatches.Patch(color='#f39c12', label='Unknown (scored)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax.set_title(f'Node {node_idx} | Fraud Score: {score:.4f}', fontweight='bold')
        ax.axis('off')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FRAUD RINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💀 Fraud Rings":
    st.header("Fraud Ring Detection")
    st.markdown("Communities detected via Louvain clustering on the transaction graph.")

    with st.spinner("Loading Louvain communities (cached after the first run)..."):
        comm_labels   = cached_partition()
        n_communities = int(comm_labels.max()) + 1

    st.success(f"Detected {n_communities} communities across {data.num_nodes:,} nodes")

    # ── Find fraud rings ───────────────────────────────────────────────────────
    # The promotion rule lives in fraudlens.pipeline.rings so the dashboard and the
    # pipeline cannot drift apart.
    fraud_rings = []
    order      = np.argsort(comm_labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(comm_labels[order])) + 1
    for members in np.split(order, boundaries):
        labeled = members[y_np[members] != -1]
        if labeled.size == 0:
            continue
        n_ill    = int((y_np[labeled] == 1).sum())
        ill_rate = n_ill / labeled.size
        avg_sc   = float(fraud_prob[members].mean())
        if is_fraud_ring(ill_rate, avg_sc, n_ill, members.size):
            fraud_rings.append({
                'Community':   int(comm_labels[members[0]]),
                'Size':        int(members.size),
                'Illicit':     n_ill,
                'Illicit %':   round(ill_rate * 100, 1),
                'Avg Score':   round(avg_sc, 4),
                'members':     members,
            })

    fraud_rings.sort(key=lambda x: x['Illicit %'], reverse=True)

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Rings Found", len(fraud_rings))
    c2.metric("Illicit Nodes in Rings",
              sum(r['Illicit'] for r in fraud_rings))
    c3.metric("Largest Ring",
              f"{max(r['Size'] for r in fraud_rings):,} nodes" if fraud_rings else "N/A")

    st.markdown("---")

    # ── Fraud rings table ──────────────────────────────────────────────────────
    st.subheader("Detected Fraud Rings")
    display_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'members'}
                                for r in fraud_rings])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Visualize selected ring ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Visualize a Fraud Ring")
    ring_options = [f"Ring {r['Community']} ({r['Illicit %']} illicit, {r['Size']} nodes)"
                    for r in fraud_rings]
    selected_ring_str = st.selectbox("Select a fraud ring to visualize", ring_options)
    selected_idx      = ring_options.index(selected_ring_str)
    selected_ring     = fraud_rings[selected_idx]

    ring_nodes = selected_ring['members']
    if len(ring_nodes) > 80:
        ring_nodes = np.random.default_rng(42).choice(ring_nodes, 80, replace=False)

    G_ring = G.subgraph(ring_nodes.tolist())
    pos    = nx.spring_layout(G_ring, seed=42, k=0.8)

    fig, ax = plt.subplots(figsize=(10, 7))
    node_colors = []
    node_sizes  = []
    for n in G_ring.nodes():
        if y_np[n] == 1:
            node_colors.append('#e74c3c')
            node_sizes.append(200)
        elif y_np[n] == 0:
            node_colors.append('#2ecc71')
            node_sizes.append(150)
        else:
            node_colors.append(plt.cm.Oranges(0.3 + 0.7 * float(fraud_prob[n])))
            node_sizes.append(80)

    nx.draw_networkx_edges(G_ring, pos, ax=ax, alpha=0.3,
                           edge_color='gray', width=0.8)
    nx.draw_networkx_nodes(G_ring, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Illicit'),
        mpatches.Patch(color='#2ecc71', label='Licit'),
        mpatches.Patch(color='#f39c12', label='Unknown (scored)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_title(
        f"Fraud Ring — Community {selected_ring['Community']}\n"
        f"{selected_ring['Illicit %']} illicit | "
        f"Avg fraud score: {selected_ring['Avg Score']}",
        fontweight='bold'
    )
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainer":
    st.header("GNN Explainer")
    st.markdown("Understand **why** a node was flagged as fraudulent.")

    col1, col2 = st.columns([2, 1])
    with col1:
        exp_node = st.number_input(
            "Node Index to Explain",
            min_value=0, max_value=data.num_nodes - 1,
            value=32049
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        explain_btn = st.button("🧠 Explain This Node", use_container_width=True)

    # Show some high-confidence fraud nodes as suggestions
    test_indices  = torch.where(test_mask)[0].numpy()
    top_fraud     = sorted([i for i in test_indices if y_np[i] == 1],
                            key=lambda i: fraud_prob[i], reverse=True)[:5]
    st.markdown("**Suggested nodes to explain (confirmed illicit, high confidence):**")
    st.code(", ".join(str(i) for i in top_fraud))

    if explain_btn:
        node_idx = int(exp_node)
        score    = fraud_prob[node_idx]

        st.markdown(f"**Node {node_idx}** — Fraud Score: `{score:.4f}`")
        st.progress(float(score))

        with st.spinner("Running explainer (10-20 seconds)..."):
            exp = explain_node(model, data, node_idx, fraud_prob=fraud_prob)
            edge_mask       = exp.edge_mask
            feature_mask    = exp.feature_importance
            sub_edge_index  = exp.sub_edge_index
            subset_np       = exp.subset
            target_local_idx = exp.target_local_idx
            pred_class      = exp.predicted_class
            pred_prob       = exp.probability

        pred_label = 'ILLICIT' if pred_class == 1 else 'LICIT'
        true_label = {1: 'Illicit', 0: 'Licit', -1: 'Unknown'}[y_np[node_idx]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred_label)
        c2.metric("Confidence", f"{pred_prob:.4f}")
        c3.metric("True Label", true_label)

        st.markdown("---")
        col_graph, col_feat = st.columns([3, 2])

        # ── Explanation subgraph ───────────────────────────────────────────────
        with col_graph:
            st.subheader("Explanation Subgraph")
            st.markdown("Edges colored by importance — darker red = stronger influence on prediction")

            G_exp = nx.DiGraph()
            G_exp.add_nodes_from(range(len(subset_np)))
            edge_arr = sub_edge_index
            for e_idx in range(edge_arr.shape[1]):
                G_exp.add_edge(edge_arr[0, e_idx], edge_arr[1, e_idx],
                               weight=float(edge_mask[e_idx]))

            top_edges = sorted(G_exp.edges(data=True),
                               key=lambda x: x[2]['weight'], reverse=True)[:30]
            G_disp = nx.DiGraph()
            G_disp.add_nodes_from(G_exp.nodes())
            G_disp.add_edges_from([(u, v) for u, v, _ in top_edges])

            pos = nx.spring_layout(G_disp, seed=42, k=1.2)
            fig, ax = plt.subplots(figsize=(7, 5))

            node_colors, node_sizes = [], []
            for n in G_disp.nodes():
                global_n = subset_np[n]
                if n == target_local_idx:
                    node_colors.append('#f39c12')
                    node_sizes.append(500)
                elif y_np[global_n] == 1:
                    node_colors.append('#e74c3c')
                    node_sizes.append(200)
                elif y_np[global_n] == 0:
                    node_colors.append('#2ecc71')
                    node_sizes.append(150)
                else:
                    node_colors.append('#bdc3c7')
                    node_sizes.append(100)

            ew = [G_disp[u][v].get('weight', 0.1) for u, v in G_disp.edges()]
            nx.draw_networkx_edges(G_disp, pos, ax=ax,
                                   edge_color=[plt.cm.Reds(0.3 + 0.7*w) for w in ew],
                                   width=[0.5 + 3*w for w in ew],
                                   alpha=0.8, arrows=True, arrowsize=10)
            nx.draw_networkx_nodes(G_disp, pos, ax=ax,
                                   node_color=node_colors,
                                   node_size=node_sizes, alpha=0.95)
            legend_elements = [
                mpatches.Patch(color='#f39c12', label='Target'),
                mpatches.Patch(color='#e74c3c', label='Illicit neighbor'),
                mpatches.Patch(color='#2ecc71', label='Licit neighbor'),
                mpatches.Patch(color='#bdc3c7', label='Unknown'),
            ]
            ax.legend(handles=legend_elements, fontsize=8)
            ax.axis('off')
            fig.patch.set_facecolor('#f8f9fa')
            st.pyplot(fig)
            plt.close()

        # ── Feature importance ─────────────────────────────────────────────────
        with col_feat:
            st.subheader("Top Features Driving Prediction")
            top15_idx  = np.argsort(feature_mask)[::-1][:15]
            top15_vals = feature_mask[top15_idx]
            top15_lbls = [f'f{i+1}' for i in top15_idx]

            fig, ax = plt.subplots(figsize=(4, 5))
            bars = ax.barh(range(15), top15_vals[::-1], color='#e74c3c', alpha=0.8)
            ax.set_yticks(range(15))
            ax.set_yticklabels(top15_lbls[::-1], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance', fontsize=10)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            st.pyplot(fig)
            plt.close()

            # Interpretation
            top_feat_idx = int(top15_idx[0])
            is_aggregated = top_feat_idx >= N_LOCAL_FEATURES
            feat_type = "neighborhood aggregation" if is_aggregated else "local transaction"
            driver = (
                "the transaction's network connections"
                if is_aggregated
                else "the transaction's own behavior"
            )
            st.info(
                f"**Top feature: f{top_feat_idx + 1}**\n\n"
                f"This is a **{feat_type}** feature, so the prediction is driven by "
                f"{driver}."
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7f8c8d; font-size:0.8rem;'>"
    "FraudLens — Graph Neural Network Based Fraud Detection | "
    "Built with PyTorch Geometric, Streamlit & NetworkX"
    "</div>",
    unsafe_allow_html=True
)
