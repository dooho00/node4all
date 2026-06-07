from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .channel_base import ChannelGraphBase, ScalarFFN


class CGTAggregation(nn.Module):
    """
    CGT multi-hop tokenization for each scalar channel graph.

    Let K be the token dimension.
    We split K into n_filters * n_hops heads, each with hop_dim dims.

      - hop 0: identity
      - hop 1: 1-hop GCN-style aggregation
      - hop 2: 2-hop aggregation (apply 1-hop twice)
      - ...

    Each head operates on its own slice of the last dimension of h: (N, F, hop_dim).
    K must equal n_hops * hop_dim * n_filters.
    """

    def __init__(
        self,
        n_hops: int,
        hop_dim: int,
        n_filters: int,
        filter_norms: Optional[Sequence[str]] = None,
        use_self_loops: bool = True,
    ):
        super().__init__()
        self.n_hops = int(n_hops)
        self.hop_dim = int(hop_dim)
        self.n_filters = int(n_filters)
        self.use_self_loops = bool(use_self_loops)

        if self.n_hops < 1 or self.hop_dim < 1 or self.n_filters < 1:
            raise ValueError("n_hops, hop_dim, and n_filters must be >= 1.")

        self.filter_norms: List[str] = self._normalize_filter_norms(filter_norms)
        if len(self.filter_norms) == 1 and self.n_filters > 1:
            self.filter_norms = self.filter_norms * self.n_filters
        if len(self.filter_norms) != self.n_filters:
            raise ValueError(
                f"n_filters={self.n_filters} must match len(filter_norms)={len(self.filter_norms)}."
            )

    @staticmethod
    def _normalize_filter_norms(filter_norms: Optional[Sequence[str]]) -> List[str]:
        if filter_norms is None:
            return ["sym"]
        if isinstance(filter_norms, str):
            norms = [s.strip() for s in filter_norms.split(",") if s.strip()]
        else:
            norms = [str(s).strip() for s in filter_norms if str(s).strip()]
        norms = [n.lower() for n in norms]
        if not norms:
            norms = ["sym"]
        if "all" in norms:
            return ["sym", "rw", "none"]
        allowed = {"sym", "rw", "none"}
        bad = [n for n in norms if n not in allowed]
        if bad:
            raise ValueError(f"Unknown filter_norms={bad}. Allowed: {sorted(allowed)}")
        return norms

    @staticmethod
    def _one_hop_aggregate(
        src: Tensor,
        dst: Tensor,
        norm: Tensor,          # (E,)
        h: Tensor,             # (N, F, k_per_head)
        num_nodes: int,
    ) -> Tensor:
        """
        Single GCN-style aggregation step:

            out_j = sum_{i in N(j)} norm_{ij} * h_i
        """
        E = src.numel()
        if E == 0:
            return h

        # Messages
        h_src = h[src]                        # (E, F, k_per_head)
        msg = h_src * norm.view(E, 1, 1)      # (E, F, k_per_head)

        # Aggregate at destinations
        out = h.new_zeros(num_nodes, h.size(1), h.size(2))
        out.index_add_(0, dst, msg)           # (N, F, k_per_head)
        return out

    def _compute_norm(self, src: Tensor, dst: Tensor, num_nodes: int, norm_type: str, dtype: torch.dtype) -> Tensor:
        E = int(src.numel())
        if E == 0:
            return src.new_zeros((0,), dtype=dtype)

        one = src.new_ones(E, dtype=dtype)
        if norm_type == "none":
            return one

        if norm_type == "rw":
            # Random-walk (row) normalization: D^{-1} A, where D is out-degree of src nodes.
            deg = src.new_zeros(num_nodes, dtype=dtype)
            deg.scatter_add_(0, src, one)
            deg_inv = deg.pow(-1.0)
            deg_inv[torch.isinf(deg_inv)] = 0
            return deg_inv[src]

        # "sym": GCN symmetric normalization: D^{-1/2} A D^{-1/2}.
        deg = src.new_zeros(num_nodes, dtype=dtype)
        deg.scatter_add_(0, dst, one)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        return deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    def forward(
        self,
        edge_index: Tensor,   # (2, E)
        h: Tensor,            # (N, F, K)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        N, F, K = h.shape
        if num_nodes is None:
            num_nodes = N

        expected_k = self.n_hops * self.hop_dim * self.n_filters
        if K != expected_k:
            raise ValueError(
                f"K={K} must equal n_hops*hop_dim*n_filters={expected_k}."
            )
        k_per_head = self.hop_dim

        device = h.device

        src, dst = edge_index
        src = src.to(device=device, dtype=torch.long)
        dst = dst.to(device=device, dtype=torch.long)

        if not self.use_self_loops:
            keep = src != dst
            if keep.any():
                src = src[keep]
                dst = dst[keep]
            else:
                src = src[:0]
                dst = dst[:0]

        E = src.numel()
        if E == 0:
            # no edges: every head is just identity
            return h

        # Precompute per-edge norms for requested normalization modes.
        norms_by_type = {
            n: self._compute_norm(src, dst, num_nodes, n, dtype=h.dtype)
            for n in set(self.filter_norms)
        }

        # Process each filter group: reuse intermediate hops for all heads.
        head_outputs = []
        for group_idx in range(self.n_filters):
            group_start = group_idx * self.n_hops * k_per_head
            group_end = group_start + (self.n_hops * k_per_head)
            h_group = h[:, :, group_start:group_end]  # (N, F, n_hops * hop_dim)
            norm = norms_by_type[self.filter_norms[group_idx]]

            tmp = h_group
            for hop_depth in range(self.n_hops):
                hop_start = hop_depth * k_per_head
                hop_end = hop_start + k_per_head
                head_outputs.append(tmp[:, :, hop_start:hop_end])
                if hop_depth < self.n_hops - 1:
                    tmp = self._one_hop_aggregate(src, dst, norm, tmp, num_nodes)

        # Concatenate all heads back along K dimension
        out = torch.cat(head_outputs, dim=2)  # (N, F, K)
        return out

class CGTLayer(nn.Module):
    """
    One CGT layer in k-dim token space.

    The layer applies CGT multi-hop tokenization independently to
    each feature channel and updates the resulting tokens with a
    shared scalar FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dropout: float = 0.0,
        is_linear: bool = False,
        n_hops: int = 1,
        hop_dim: int = 1,
        n_filters: int = 1,
        filter_norms: Optional[Sequence[str]] = None,
        use_self_loops: bool = True,
    ):
        super().__init__()
        self.agg = CGTAggregation(
            n_hops=n_hops,
            hop_dim=hop_dim,
            n_filters=n_filters,
            filter_norms=filter_norms,
            use_self_loops=use_self_loops,
        )
        self.ffn = ScalarFFN(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            is_linear=is_linear,
            dropout=ffn_dropout,
        )
        self.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,  # (2, E)
        h: Tensor,           # (N, F, k)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        h_state = h
        h_obs = self._observe(edge_index, h_state, num_nodes=num_nodes)
        h_write = self._write(h_obs)
        return h_write

    def _observe(
        self,
        edge_index: Tensor,
        h: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        return self.agg(edge_index, h, num_nodes=num_nodes)

    def _write(self, h_obs: Tensor) -> Tensor:
        return self.ffn(h_obs)

    def reset_parameters(self):
        # CGTAggregation has no parameters
        self.ffn.reset_parameters()

class CGTEncoder(ChannelGraphBase):
    """
    Channel Graph Transformer (CGT) encoder.

    CGT maps each channel graph to scalar node embeddings using
    coordinate-wise multi-hop tokenization followed by Transformer-based
    token readout in ChannelGraphBase.

    API:
      forward(x, edge_index, num_nodes=None)
        - x: (N, F)
        - edge_index: (2, E)

    embed_dim is derived as n_hops * n_filters (hop_dim is fixed to 1).
    """

    def __init__(
        self,
        num_layers: int,
        n_hops: int,
        n_filters: int,
        ffn_dropout: float = 0.5,
        is_linear: bool = False,
        filter_norms: Optional[Sequence[str]] = None,
        use_self_loops: bool = True,
        summary_attn_dropout: Optional[float] = None,
    ):
        n_hops = int(n_hops)
        hop_dim = 1
        n_filters = int(n_filters)
        if n_hops < 1 or n_filters < 1:
            raise ValueError("n_hops and n_filters must be >= 1.")
        embed_dim = n_hops * n_filters
        layer_kwargs = dict(
            ffn_dropout=ffn_dropout,
            is_linear=is_linear,
            n_hops=n_hops,
            hop_dim=hop_dim,
            n_filters=n_filters,
            filter_norms=filter_norms,
            use_self_loops=use_self_loops,
        )
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_dropout=ffn_dropout,
            layer_cls=CGTLayer,
            layer_kwargs=layer_kwargs,
            summary_attn_dropout=summary_attn_dropout,
        )
