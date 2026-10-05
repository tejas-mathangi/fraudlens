"""Per-node explanations via input-gradient saliency.

This is gradient saliency, not Ying et al.'s GNNExplainer: we backpropagate the
predicted-class probability into the input features of the node's 2-hop subgraph and
read off ``|dp/dx|``. The explanation is therefore faithful to the actual computation
rather than a learned post-hoc approximation -- the trade-off being that it reports
sensitivity, not a minimal sufficient subgraph.

The original implementation ran a full 203k-node forward pass inside every call just
to read one node's prediction, which made each explanation cost seconds. Callers now
pass in precomputed scores.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from fraudlens.config import PATHS, Paths
from fraudlens.data import Graph, build_graph, feature_group, feature_names
from fraudlens.model import GraphSAGE, fraud_scores, load_model

log = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Why the model scored one node the way it did."""

    node_idx: int
    predicted_class: int
    probability: float
    feature_importance: np.ndarray
    subset: np.ndarray
    sub_edge_index: np.ndarray
    edge_mask: np.ndarray
    target_local_idx: int

    def top_features(self, k: int, n_features: int) -> list[dict[str, object]]:
        """The ``k`` most influential input features, named and grouped."""
        names = feature_names(n_features)
        order = np.argsort(self.feature_importance)[::-1][:k]
        return [
            {
                "index": int(i),
                "name": names[i],
                "group": feature_group(int(i)),
                "importance": round(float(self.feature_importance[i]), 5),
            }
            for i in order
        ]

    def driver(self, n_local: int) -> str:
        """Whether the verdict rests on the node itself or on its neighbourhood.

        This is the sentence the old dashboard tried to render before crashing on a
        misspelled variable.
        """
        top = int(np.argmax(self.feature_importance))
        return "network" if top >= n_local else "local"


def explain_node(
    model: GraphSAGE,
    data,
    node_idx: int,
    fraud_prob: np.ndarray | torch.Tensor | None = None,
    num_hops: int = 2,
) -> Explanation:
    """Explain one node's prediction.

    Args:
        model: the trained classifier, in eval mode.
        data: the full PyG graph.
        node_idx: global index of the node to explain.
        fraud_prob: precomputed P(illicit) for every node. When omitted the node's
            own prediction is recomputed from its subgraph, which is cheap; passing
            the array in is still preferred for consistency with the rest of the UI.
        num_hops: receptive field to explain over. The model has three layers, but two
            hops already covers the neighbourhood that drives most predictions and
            keeps the subgraph small enough to draw.
    """
    model.eval()
    node_idx = int(node_idx)

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    target_local = int(mapping.item())

    # ── Feature saliency ─────────────────────────────────────────────────────
    sub_x = data.x[subset].clone().requires_grad_(True)
    logits = model(sub_x, sub_edge_index)
    target_class = int(logits[target_local].argmax())
    prob = float(F.softmax(logits[target_local], dim=0)[target_class])
    F.softmax(logits[target_local], dim=0)[target_class].backward()

    grad = sub_x.grad
    assert grad is not None  # backward() above guarantees this
    importance = grad[target_local].abs().detach().numpy()
    if importance.max() > 0:
        importance = importance / importance.max()

    # If the caller gave us scores from the full graph, prefer them: a subgraph
    # forward pass sees fewer neighbours and can disagree slightly.
    if fraud_prob is not None:
        scores = (
            fraud_prob.numpy() if isinstance(fraud_prob, torch.Tensor) else fraud_prob
        )
        prob_illicit = float(scores[node_idx])
        target_class = int(prob_illicit >= 0.5)
        prob = prob_illicit if target_class == 1 else 1.0 - prob_illicit
        sub_scores = scores[subset.numpy()]
    else:
        with torch.no_grad():
            sub_scores = (
                F.softmax(model(data.x[subset], sub_edge_index), dim=1)[:, 1].numpy()
            )

    # ── Edge influence ───────────────────────────────────────────────────────
    # Weight each edge by how fraudulent its source is: the signal flowing along the
    # edge into the target is what the aggregation actually consumes.
    edges = sub_edge_index.numpy()
    edge_mask = sub_scores[edges[0]].astype(np.float64)
    if edge_mask.size and edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()

    return Explanation(
        node_idx=node_idx,
        predicted_class=target_class,
        probability=round(prob, 4),
        feature_importance=importance,
        subset=subset.numpy(),
        sub_edge_index=edges,
        edge_mask=edge_mask,
        target_local_idx=target_local,
    )


def pick_interesting_nodes(
    graph: Graph, fraud_prob: np.ndarray, n: int = 20
) -> list[int]:
    """Nodes worth explaining: confirmed fraud the model is most confident about.

    One confidently-licit node is appended as a contrast case, so the explainability
    view can show what a clean verdict looks like.
    """
    y = graph.data.y.numpy()
    illicit = np.where(y == 1)[0]
    ranked = illicit[np.argsort(fraud_prob[illicit])[::-1]]
    picks = [int(i) for i in ranked[: max(1, n - 1)]]

    licit = np.where(y == 0)[0]
    if licit.size:
        picks.append(int(licit[np.argmin(fraud_prob[licit])]))
    return picks


def run_explainer(
    graph: Graph | None = None, paths: Paths = PATHS, n: int = 4
) -> list[Explanation]:
    """Explain a handful of high-confidence fraud nodes and log the drivers."""
    graph = graph or build_graph(paths)
    model = load_model(graph.num_features, paths.checkpoint)
    scores = fraud_scores(model, graph.data).numpy()

    out = []
    for node_idx in pick_interesting_nodes(graph, scores, n=n):
        exp = explain_node(model, graph.data, node_idx, fraud_prob=scores)
        out.append(exp)
        top = exp.top_features(3, graph.num_features)
        log.info(
            "node %-7d score %.4f  driver=%s  top=%s",
            node_idx,
            scores[node_idx],
            exp.driver(93),
            ", ".join(f["name"] for f in top),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain fraud predictions")
    parser.add_argument("--n", type=int, default=4, help="nodes to explain")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_explainer(n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
