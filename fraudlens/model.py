"""The GraphSAGE classifier.

IMPORTANT: the submodule attribute names (``conv1``-``conv3``, ``bn1``-``bn3``,
``classifier``) are part of the on-disk contract -- ``models/best_model.pt`` is a bare
``state_dict`` keyed by them. Renaming any attribute silently breaks every existing
checkpoint, so ``tests/test_model.py`` pins the key set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from fraudlens.config import TRAIN


class GraphSAGE(nn.Module):
    """Three-layer GraphSAGE node classifier.

    Three convolutions mean each node aggregates over its 3-hop neighbourhood, which is
    the depth at which fraud-ring structure becomes visible: a wallet one hop from a
    ring looks ordinary, but its neighbourhood does not.

    Architecture::

        in_channels -> SAGEConv -> BatchNorm -> ReLU -> Dropout
                    -> SAGEConv -> BatchNorm -> ReLU -> Dropout
                    -> SAGEConv -> BatchNorm -> ReLU -> Dropout
                    -> Linear -> [licit_logit, illicit_logit]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = TRAIN.hidden_channels,
        out_channels: int = TRAIN.out_channels,
        dropout: float = TRAIN.dropout,
    ) -> None:
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)

        # BatchNorm stabilises training under the ~1:9 class imbalance.
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)

        self.classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[n_nodes, out_channels]``."""
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x, edge_index))))
        return self.classifier(x)


def read_state_dict(checkpoint) -> dict:
    """Pull the tensor state dict out of either checkpoint layout.

    Accepts the original bare ``state_dict`` and the richer
    ``{"state_dict": ..., "best_f1": ..., "epoch": ...}`` dict that training now
    writes. ``weights_only=True`` is passed explicitly -- we only ever persist tensors
    and plain scalars, and it avoids the unpickling warning on torch >= 2.4.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def infer_architecture(state: dict) -> dict[str, int]:
    """Recover the layer widths from a ``state_dict``.

    A bare state dict carries no metadata, so callers used to have to know the
    hyperparameters that produced it and silently got a shape-mismatch crash when they
    guessed wrong. The shapes themselves are unambiguous, so read them instead:

    * ``conv1.lin_l.weight`` is ``[hidden, in_channels]``
    * ``bn1.weight``         is ``[hidden]``
    * ``classifier.weight``  is ``[out_channels, hidden // 2]``
    """
    try:
        in_channels = int(state["conv1.lin_l.weight"].shape[1])
        hidden_channels = int(state["bn1.weight"].shape[0])
        out_channels = int(state["classifier.weight"].shape[0])
    except KeyError as exc:  # pragma: no cover - corrupt or foreign checkpoint
        raise ValueError(f"not a FraudLens checkpoint: missing {exc}") from exc
    return {
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
    }


def load_model(
    in_channels: int | None = None,
    checkpoint=None,
    hidden_channels: int | None = None,
    out_channels: int | None = None,
    dropout: float = TRAIN.dropout,
) -> GraphSAGE:
    """Build a :class:`GraphSAGE` sized to match ``checkpoint`` and load it.

    Layer widths are inferred from the saved tensors, so this works for any checkpoint
    this project has ever written. Explicit arguments override the inference; a
    mismatch against the checkpoint raises rather than loading a half-initialised
    model.
    """
    if checkpoint is None:
        raise ValueError("checkpoint is required")

    state = read_state_dict(checkpoint)
    arch = infer_architecture(state)

    if in_channels is not None and in_channels != arch["in_channels"]:
        raise ValueError(
            f"checkpoint expects {arch['in_channels']} input features, got {in_channels}"
        )

    model = GraphSAGE(
        in_channels=arch["in_channels"],
        hidden_channels=hidden_channels or arch["hidden_channels"],
        out_channels=out_channels or arch["out_channels"],
        dropout=dropout,
    )
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def fraud_scores(model: GraphSAGE, data) -> torch.Tensor:
    """P(illicit) for every node, as a 1-D tensor in ``[0, 1]``.

    One full forward pass over 203k nodes costs a few seconds, so callers should
    compute this once and pass the result around rather than recomputing per request.
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    return F.softmax(logits, dim=1)[:, 1]
