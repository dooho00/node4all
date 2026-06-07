import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .channel_base import ChannelGraphBase, ScalarFFN


# ---------------------------------------------------------------------------
# Neighborhood self-attention over k-dim tokens (GAT-style)
# ---------------------------------------------------------------------------
class ChannelGATNeighborAttention(nn.Module):
    """
    Multi-head self-attention over one-hop neighbors + self, in k-dim token space.

    For each node j, feature f and head h, we consider:
        { h_i[f, h] | i in N(j) ∪ {j} }

    Input:
      - h: (N, F, K) where K == embed_dim
      - edge_index: LongTensor[2, E] with [0]=src, [1]=dst

    Internals:
      - K = num_heads * head_dim
      - Segment softmax over (dst, feature, head) triples.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # standard multihead style: one big linear per Q/K/V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.reset_parameters()

    def reset_parameters(self):
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)

    def forward(
        self,
        edge_index: Tensor,   # (2, E)
        h: Tensor,            # (N, F, K)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        N, F, K = h.shape
        assert K == self.embed_dim
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
        dst = torch.cat([dst, self_idx], dim=0)
        '''
        E = src.numel()

        if E == 0:
            return h

        # ------------------------------------------------------------
        # Linear projections and reshape to heads
        # q_all, k_all, v_all: (N, F, num_heads, head_dim)
        # ------------------------------------------------------------
        def proj(x: Tensor, layer: nn.Linear) -> Tensor:
            out = layer(x)  # (N, F, K)
            return out.view(N, F, self.num_heads, self.head_dim)

        q_all = proj(h, self.q_proj)
        k_all = proj(h, self.k_proj)
        v_all = proj(h, self.v_proj)

        # Gather per edge: query from dst, key/value from src
        # shapes: (E, F, H, Dh)
        q_j = q_all[dst]
        k_i = k_all[src]
        v_i = v_all[src]

        # ------------------------------------------------------------
        # Dot-product attention per head
        # logits: (E, F, H)
        # ------------------------------------------------------------
        logits = (q_j * k_i).sum(dim=-1) * self.scale

        # ------------------------------------------------------------
        # Segment softmax over (dst, feature, head)
        # ------------------------------------------------------------
        H = self.num_heads
        num_segments = num_nodes * F * H

        feat_idx = torch.arange(F, device=device)
        head_idx = torch.arange(H, device=device)

        dst_grid = dst.view(E, 1, 1).expand(E, F, H)
        feat_grid = feat_idx.view(1, F, 1).expand(E, F, H)
        head_grid = head_idx.view(1, 1, H).expand(E, F, H)

        seg_idx = dst_grid * (F * H) + feat_grid * H + head_grid  # (E, F, H)
        seg_idx_flat = seg_idx.view(-1)                            # (E*F*H,)
        logits_flat = logits.view(-1)                              # (E*F*H,)

        # 1) max per segment
        max_per_seg = h.new_full((num_segments,), float("-inf"))
        max_per_seg.scatter_reduce_(
            0,
            seg_idx_flat,
            logits_flat,
            reduce="amax",
            include_self=True,
        )

        # 2) exponentiate and sum per segment
        logits_exp = torch.exp(logits_flat - max_per_seg[seg_idx_flat])
        denom = h.new_zeros(num_segments)
        denom.scatter_add_(0, seg_idx_flat, logits_exp)

        alpha_flat = logits_exp / (denom[seg_idx_flat] + 1e-16)
        alpha = self.dropout(alpha_flat).view(E, F, H)  # (E, F, H)

        # ------------------------------------------------------------
        # Aggregate values
        # ------------------------------------------------------------
        msg = alpha.unsqueeze(-1) * v_i   # (E, F, H, Dh)

        out = h.new_zeros(num_nodes, F, H, self.head_dim)
        out.index_add_(0, dst, msg)      # (N, F, H, Dh)

        # concat heads and output projection
        out = out.view(num_nodes, F, self.embed_dim)  # (N, F, K)
        out = self.out_proj(out)

        return out


# ---------------------------------------------------------------------------
# GAT-style channel-wise layer (attn + FFN with residuals)
# ---------------------------------------------------------------------------

class ChannelGATLayer(nn.Module):
    """
    One channel-wise GAT-style layer in k-dim token space:

      h -> ChannelGATNeighborAttention -> residual
        -> ScalarFFN -> residual
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = ChannelGATNeighborAttention(
            embed_dim=embed_dim,
            dropout=attn_dropout,
        )
        self.ffn = ScalarFFN(
            embed_dim=embed_dim,
            hidden_dim= embed_dim * 4,
            dropout=ffn_dropout,
        )
        self.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,  # (2, E)
        h: Tensor,           # (N, F, k)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        h_attn = self.attn(edge_index, h, num_nodes=num_nodes)
        h = h_attn

        h_ffn = self.ffn(h)
        h = h_ffn

        return h

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ffn.reset_parameters()


# ---------------------------------------------------------------------------
# GAT-style channel-wise encoder (channel_gat)
# ---------------------------------------------------------------------------

class ChannelGATEncoder(ChannelGraphBase):
    """
    Channel-wise encoder using attention-based aggregation (channel_gat).

    API:
      forward(x, edge_index, num_nodes=None)
        - x: (N, F)
        - edge_index: (2, E)
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int = 16,
        attn_dropout: float = 0.2,
        ffn_dropout: float = 0.5,
        ffn_hidden_mul: int = 2,
        summary_attn_dropout: Optional[float] = None,
    ):
        layer_kwargs = dict(
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_dropout=ffn_dropout,
            layer_cls=ChannelGATLayer,
            layer_kwargs=layer_kwargs,
            summary_attn_dropout=summary_attn_dropout,
        )
