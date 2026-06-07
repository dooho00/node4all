import torch
from typing import Optional, Tuple

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


def _feature_batch_chunks(
    num_nodes: int,
    num_feats: int,
    budget: Optional[int],
):
    """
    Yield (start, end) slices over the feature dimension based on a per-graph
    row budget to keep memory usage manageable (batch over features, not nodes).
    """
    max_rows = budget if budget and budget > 0 else None
    if max_rows is None or num_nodes * num_feats <= max_rows:
        yield 0, num_feats
        return

    chunk_width = max(1, min(num_feats, max_rows // max(1, num_nodes)))
    for start in range(0, num_feats, chunk_width):
        end = min(num_feats, start + chunk_width)
        yield start, end


def _build_subgraph(batch: Data, args) -> Data:
    """
    Build a PyG subgraph object from a batch.

    This is meant to work directly with GraphSAINT batches, which are already
    valid Data objects. We just clone the relevant fields and keep masks and
    auxiliary attributes.
    """
    subgraph = Data(
        x=batch.x,
        edge_index=batch.edge_index,
        y=getattr(batch, "y", None),
    )

    # Propagate masks if present
    for key in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(batch, key):
            setattr(subgraph, key, getattr(batch, key))

    # Propagate common GraphSAINT fields if present
    for key in ["n_id", "batch", "node_norm", "edge_norm"]:
        if hasattr(batch, key):
            setattr(subgraph, key, getattr(batch, key))

    return subgraph


def _negative_sampling(graph: Data, num_neg: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform negative sampling for a PyG graph.

    Args:
        graph: PyG Data with edge_index and num_nodes.
        num_neg: number of negative edges to sample.

    Returns:
        (neg_src, neg_dst) as 1D tensors of length num_neg.
    """
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse",
    )
    # neg_edge_index has shape [2, num_neg]
    return neg_edge_index[0], neg_edge_index[1]


@torch.no_grad()
def _compute_hits(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    k = min(k, scores.numel())
    topk_idx = torch.topk(scores, k=k, dim=0).indices
    hits = labels[topk_idx].sum().float() / max(1.0, float(k))
    return float(hits.item())


@torch.no_grad()
def _pairwise_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    compare = (pos_scores.view(-1, 1) > neg_scores.view(1, -1)).float()
    return float(compare.mean().item())


__all__ = [
    "_feature_batch_chunks",
    "_build_subgraph",
    "_negative_sampling",
    "_compute_hits",
    "_pairwise_auc",
]
