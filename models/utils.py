from functools import partial
import torch
import torch.nn as nn

import dgl


def create_activation(name: str | None):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    if name == "elu":
        return nn.ELU()
    if name is None:
        return nn.Identity()
    raise NotImplementedError(f"Activation '{name}' is not implemented.")


def create_norm(name: str | None):
    if name == "layernorm":
        return nn.LayerNorm
    if name == "batchnorm":
        return nn.BatchNorm1d
    if name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    return nn.Identity


class NormLayer(nn.Module):
    def __init__(self, hidden_dim: int, norm_type: str = "batchnorm"):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "groupnorm":
            self.norm = nn.GroupNorm(1, hidden_dim)
        else:
            raise NotImplementedError(f"Norm type '{norm_type}' is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def mask_edge(graph: dgl.DGLGraph, mask_prob: float) -> torch.Tensor:
    num_edges = graph.num_edges()
    mask_rates = torch.full((num_edges,), float(mask_prob), device=graph.device, dtype=torch.float32)
    masks = torch.bernoulli(1 - mask_rates).to(dtype=torch.bool)
    return masks


def drop_edge(graph: dgl.DGLGraph, drop_rate: float, return_edges: bool = False):
    if drop_rate <= 0:
        return (graph, None) if return_edges else graph

    num_nodes = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    kept_src = src[edge_mask]
    kept_dst = dst[edge_mask]

    new_graph = dgl.graph((kept_src, kept_dst), num_nodes=num_nodes)
    new_graph = new_graph.add_self_loop()

    dropped_src = src[~edge_mask]
    dropped_dst = dst[~edge_mask]

    if return_edges:
        return new_graph, (dropped_src, dropped_dst)
    return new_graph

