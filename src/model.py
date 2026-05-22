import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for fraud detection on the Elliptic dataset.

    Architecture:
        Input (165 features)
            ↓
        SAGEConv Layer 1  →  BatchNorm  →  ReLU  →  Dropout
            ↓
        SAGEConv Layer 2  →  BatchNorm  →  ReLU  →  Dropout
            ↓
        SAGEConv Layer 3  →  BatchNorm  →  ReLU  →  Dropout
            ↓
        Linear Classifier
            ↓
        Output (2 classes: licit / illicit)

    Three SAGE layers means each node aggregates information from
    its 3-hop neighborhood — i.e. it can see fraud patterns up to
    3 transactions away. This is what captures fraud ring structure.
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=2, dropout=0.3):
        super(GraphSAGE, self).__init__()

        # ── Three GraphSAGE convolution layers ────────────────────────────────
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)

        # ── Batch normalization after each conv ───────────────────────────────
        # Stabilizes training, especially important with class imbalance
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)

        # ── Final linear classifier ───────────────────────────────────────────
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)

        # ── Dropout for regularization ────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x          : node feature matrix [n_nodes, in_channels]
            edge_index : graph connectivity   [2, n_edges]

        Returns:
            logits     : raw class scores     [n_nodes, 2]
        """

        # Layer 1: aggregate from 1-hop neighbors
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: aggregate from 2-hop neighbors
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3: aggregate from 3-hop neighbors
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Classifier head
        logits = self.classifier(x)
        return logits

    def get_embeddings(self, x, edge_index):
        """
        Returns node embeddings BEFORE the classifier head.
        Used for Louvain fraud ring detection later.

        Args:
            x          : node feature matrix [n_nodes, in_channels]
            edge_index : graph connectivity   [2, n_edges]

        Returns:
            embeddings : learned node representations [n_nodes, hidden//2]
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        return x