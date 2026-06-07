import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .channel_base import ChannelGraphBase, ScalarFFN


# ---------------------------------------------------------------------------
# GCN-style neighborhood aggregation over k-dim tokens
# ---------------------------------------------------------------------------

class ChannelGCNAggregation(nn.Module):
    """
    GCN-style aggregation in k-dim token space.

    For adjacency A with added self loops, we compute:

        out_j[f] = sum_i ( norm_ij * h_i[f] )

    where norm_ij = 1 / sqrt(deg(i) * deg(j)), with deg from A + I.

    Implementation:
      - h: (N, F, k)
      - edge_index: LongTensor[2, E] with [0]=src, [1]=dst
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        edge_index: Tensor,   # (2, E)
        h: Tensor,            # (N, F, k)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        N, F, K = h.shape
        if num_nodes is None:
            num_nodes = N

        device = h.device

        src, dst = edge_index
        src = src.to(device=device, dtype=torch.long)
        dst = dst.to(device=device, dtype=torch.long)

        # Add self loops
        '''
        self_idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        src = torch.cat([src, self_idx], dim=0)
        dst = torch.cat([dst, self_idx], dim=0)'''
        E = src.numel()

        if E == 0:
            return h

        # Degree of A + I, using dst (like GCNConv)
        one = h.new_ones(E)
        deg = h.new_zeros(num_nodes)
        deg.scatter_add_(0, dst, one)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]   # (E,)

        # Messages: scaled h[src]
        h_src = h[src]                     # (E, F, K)
        msg = h_src * norm.view(E, 1, 1)   # (E, F, K)

        out = h.new_zeros(num_nodes, F, K)
        out.index_add_(0, dst, msg)        # (N, F, K)

        return out


# ---------------------------------------------------------------------------
# GCN-style channel-wise layer (agg + FFN with residuals)
# ---------------------------------------------------------------------------

class ChannelGCNLayer(nn.Module):
    """
    One channel-wise GCN-style layer in k-dim token space:

      h -> ChannelGCNAggregation -> residual
        -> ScalarFFN -> residual
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_hidden_dim: int = 4,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.agg = ChannelGCNAggregation()
        self.ffn = ScalarFFN(
            embed_dim=embed_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=ffn_dropout,
        )
        self.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,  # (2, E)
        h: Tensor,           # (N, F, k)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        h_agg = self.agg(edge_index, h, num_nodes=num_nodes)
        h = h_agg

        h_ffn = self.ffn(h)
        h = h_ffn

        return h

    def reset_parameters(self):
        # ChannelGCNAggregation has no parameters
        self.ffn.reset_parameters()


# ---------------------------------------------------------------------------
# GCN-style channel-wise encoder (channel_gcn)
# ---------------------------------------------------------------------------

class ChannelGCNEncoder(ChannelGraphBase):
    """
    Channel-wise encoder using GCN-style aggregation (channel_gcn).

    API:
      forward(x, edge_index, num_nodes=None)
        - x: (N, F)
        - edge_index: (2, E)
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int = 16,
        ffn_dropout: float = 0.5,
        ffn_hidden_mul: int = 2,
        summary_attn_dropout: Optional[float] = None,
    ):
        layer_kwargs = dict(
            ffn_hidden_dim=embed_dim * ffn_hidden_mul,
            ffn_dropout=ffn_dropout,
        )
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_dropout=ffn_dropout,
            layer_cls=ChannelGCNLayer,
            layer_kwargs=layer_kwargs,
            summary_attn_dropout=summary_attn_dropout,
        )
