import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'elliptic_bitcoin_dataset')

FEATURES_PATH = os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
EDGES_PATH    = os.path.join(DATA_DIR, 'elliptic_txs_edgelist.csv')
CLASSES_PATH  = os.path.join(DATA_DIR, 'elliptic_txs_classes.csv')


def build_graph():
    """
    Loads the Elliptic dataset and constructs a PyTorch Geometric Data object.

    Returns:
        data        : PyG Data object (the full graph)
        label_mask  : boolean tensor — True for nodes that have a known label
        node_to_idx : dict mapping original txId → integer node index
    """

    print("=" * 60)
    print("FRAUDLENS — Graph Construction")
    print("=" * 60)

    # ── Step 1: Load raw CSVs ──────────────────────────────────────────────────
    print("\n[1] Loading CSVs...")
    features_df = pd.read_csv(FEATURES_PATH, header=None)
    edges_df    = pd.read_csv(EDGES_PATH)
    classes_df  = pd.read_csv(CLASSES_PATH)

    # Name the columns properly
    # Column 0 = txId, Column 1 = time_step, Columns 2-166 = features
    n_features = features_df.shape[1] - 2   # subtract txId and time_step
    features_df.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(1, n_features + 1)]

    print(f"    Loaded {len(features_df):,} transactions")
    print(f"    Loaded {len(edges_df):,} edges")
    print(f"    Feature dimensions: {n_features}")

    # ── Step 2: Build node index mapping ──────────────────────────────────────
    # GNNs need integer indices 0, 1, 2... not raw txIds
    # We create a dictionary: txId (string/int) → integer index
    print("\n[2] Building node index mapping...")
    all_tx_ids  = features_df['txId'].values
    node_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}
    n_nodes     = len(node_to_idx)
    print(f"    Total nodes: {n_nodes:,}")

    # ── Step 3: Build node feature matrix X ───────────────────────────────────
    # Shape: [n_nodes, n_features]
    # We drop txId and time_step — time_step is stored separately
    print("\n[3] Building node feature matrix...")
    feature_cols = [f'f{i}' for i in range(1, n_features + 1)]
    X = features_df[feature_cols].values.astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float)
    print(f"    Feature matrix shape: {X_tensor.shape}")

    # ── Step 4: Build edge index ───────────────────────────────────────────────
    # PyG expects edge_index of shape [2, n_edges]
    # Row 0 = source nodes, Row 1 = target nodes
    print("\n[4] Building edge index...")
    src_col = edges_df.columns[0]
    dst_col = edges_df.columns[1]

    # Filter out edges where either node isn't in our node set
    valid_mask = (
        edges_df[src_col].isin(node_to_idx) &
        edges_df[dst_col].isin(node_to_idx)
    )
    edges_clean = edges_df[valid_mask]
    dropped = len(edges_df) - len(edges_clean)
    if dropped > 0:
        print(f"    Dropped {dropped} edges with unknown nodes")

    src_indices = edges_clean[src_col].map(node_to_idx).values
    dst_indices = edges_clean[dst_col].map(node_to_idx).values

    edge_index = torch.tensor(
        np.array([src_indices, dst_indices]),
        dtype=torch.long
    )
    print(f"    Edge index shape: {edge_index.shape}")

    # ── Step 5: Build labels ───────────────────────────────────────────────────
    # Label encoding: illicit=1, licit=0, unknown=-1
    # We only train/evaluate on labeled nodes
    print("\n[5] Building node labels...")

    # Merge labels onto the ordered node list
    tx_id_df = pd.DataFrame({'txId': all_tx_ids})
    merged   = tx_id_df.merge(classes_df, on='txId', how='left')

    def encode_label(c):
        c = str(c)
        if c == '1':   return 1    # illicit / fraud
        if c == '2':   return 0    # licit / clean
        return -1                  # unknown

    labels = merged['class'].apply(encode_label).values
    y      = torch.tensor(labels, dtype=torch.long)

    # label_mask: True for nodes we can actually train/evaluate on
    label_mask = (y != -1)

    n_illicit = (y == 1).sum().item()
    n_licit   = (y == 0).sum().item()
    n_unknown = (y == -1).sum().item()

    print(f"    Illicit (1) : {n_illicit:,}")
    print(f"    Licit   (0) : {n_licit:,}")
    print(f"    Unknown (-1): {n_unknown:,}")
    print(f"    Labeled nodes for training/eval: {label_mask.sum().item():,}")

    # ── Step 6: Store time step per node ──────────────────────────────────────
    time_steps = torch.tensor(features_df['time_step'].values, dtype=torch.long)

    # ── Step 7: Assemble PyG Data object ──────────────────────────────────────
    print("\n[6] Assembling PyG Data object...")
    data = Data(
        x          = X_tensor,      # node features [n_nodes, n_features]
        edge_index = edge_index,    # edges         [2, n_edges]
        y          = y,             # labels        [n_nodes]
        time_step  = time_steps,    # time steps    [n_nodes]
    )

    print(f"\n    ✓ Graph built successfully")
    print(f"    Nodes     : {data.num_nodes:,}")
    print(f"    Edges     : {data.num_edges:,}")
    print(f"    Features  : {data.num_node_features}")
    print(f"    Labeled   : {label_mask.sum().item():,}")

    # ── Step 8: Sanity checks ──────────────────────────────────────────────────
    print("\n[7] Sanity checks...")
    assert data.x.shape[0] == n_nodes,        "Node count mismatch in features"
    assert data.y.shape[0] == n_nodes,        "Node count mismatch in labels"
    assert edge_index.max().item() < n_nodes, "Edge index out of bounds"
    assert not torch.isnan(data.x).any(),     "NaN values in features"
    print("    All checks passed ✓")

    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)

    return data, label_mask, node_to_idx


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data, label_mask, node_to_idx = build_graph()

    # Extra info when run directly
    print(f"\nData object contents:")
    print(f"    data.x          : {data.x.shape}")
    print(f"    data.edge_index : {data.edge_index.shape}")
    print(f"    data.y          : {data.y.shape}")
    print(f"    data.time_step  : {data.time_step.shape}")
    print(f"    label_mask sum  : {label_mask.sum().item():,}")