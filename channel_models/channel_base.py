import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Type, Dict

from models.mlp import MLP


# ---------------------------------------------------------------------------
# Scalar <-> k-dimensional token projectors
# ---------------------------------------------------------------------------

class ScalarToKProjector(nn.Module):
    """
    Project each scalar feature to a k-dimensional token.

    Input:  x: (N, F)
    Output: y: (N, F, k)
    """

    def __init__(
        self,
        k: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.net = MLP(
            in_dim=1,
            hidden_dim=hidden_dim,
            out_dim=k,
            num_layers=1,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        if x.dim() != 2:
            raise ValueError("x must have shape (N, F)")
        N, F = x.shape
        x_flat = x.reshape(N * F, 1)
        y = self.net(x_flat)             # (N*F, k)
        return y.view(N, F, self.k)

    def reset_parameters(self):
        self.net.reset_parameters()


class KToScalarProjector(nn.Module):
    """
    Compress each k-dimensional token back to a scalar feature.

    Input:  x: (N, F, k) or (..., k)
    Output: y: (N, F) or (...) (same leading shape, last dim removed)
    """

    def __init__(self, k: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.net = MLP(
            in_dim=k,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=1,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        *leading, k = x.shape
        assert k == self.k
        x_flat = x.reshape(-1, k)
        y = self.net(x_flat)        # (prod(leading), 1)
        return y.reshape(*leading)

    def reset_parameters(self):
        self.net.reset_parameters()


# ---------------------------------------------------------------------------
# Token-wise FFN in k-dim space, feature-agnostic
# ---------------------------------------------------------------------------

class ScalarFFN(nn.Module):
    """
    Token-wise MLP applied to each (node, feature) token independently in k-dim.

    Input:
      h: (N, F, k)
    Output:
      out: (N, F, k)

    Parameters are shared across all nodes and features.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, is_linear: bool = False, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers= 1 if is_linear else 2,
            dropout=dropout,
        )

    def forward(self, h: Tensor) -> Tensor:
        N, F, K = h.shape
        assert K == self.embed_dim
        x = h.view(N * F, K)
        y = self.net(x)                # (N*F, K)
        return y.view(N, F, K)

    def reset_parameters(self):
        self.net.reset_parameters()


# ---------------------------------------------------------------------------
# Transformer-based token readout with a summary token
# ---------------------------------------------------------------------------
class _FixedHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, attn_dropout: float):
        super().__init__()
        attn_dim = num_heads * head_dim
        self.q_proj = nn.Linear(embed_dim, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.v_proj = nn.Linear(embed_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attn_dropout = float(attn_dropout)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        seq_len, batch, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        x = x.transpose(0, 1)  # (B, S, D)
        q = self.q_proj(x).view(batch, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, h, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, h * d)
        out = self.out_proj(out)
        return out.transpose(0, 1)

    def reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            if proj is self.out_proj:
                proj.weight.data.mul_(0.1)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)


class LayerwiseTransformerSummary(nn.Module):
    """
    Read out layer-wise CGT tokens with a Transformer over token depth,
    with pre-norm LayerNorms for stability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: int = 4,
    ):
        super().__init__()
        head_dim = embed_dim
        self.attn_dropout = float(attn_dropout)
        self.ffn_dropout = float(ffn_dropout)
        self.attn = nn.ModuleList(
            [
                _FixedHeadSelfAttention(
                    embed_dim=embed_dim,
                    num_heads=int(nhead),
                    head_dim=int(head_dim),
                    attn_dropout=self.attn_dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.attn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(int(num_layers))])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * ff_mult),
                    nn.GELU(),
                    nn.Dropout(self.ffn_dropout),
                    nn.Linear(embed_dim * ff_mult, embed_dim),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(int(num_layers))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for attn in self.attn:
            attn.reset_parameters()
        for norm in self.attn_norm:
            norm.reset_parameters()
        for ffn in self.ffn:
            for module in ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for norm in self.ffn_norm:
            norm.reset_parameters()

    def forward(
        self,
        h_seq: Tensor,
        return_all_tokens: bool = False,
    ) -> Tensor:
        """
        h_seq: (N, F, L+1, k)

        If return_all_tokens == False:
            returns: (N, F, k)          # CLS only
        If return_all_tokens == True:
            returns: (N, F, L+1, k)     # all layer tokens after transformer
        """
        N, F, Lp, K = h_seq.shape
        B = N * F

        x = h_seq.view(B, Lp, K)      # (B, L+1, K)

        x = x.transpose(0, 1)          # (L+1, B, K)

        cls = self.cls_token.expand(1, B, K)   # (1, B, K)
        x = torch.cat([cls, x], dim=0)         # (L+2, B, K)
        y = x
        for idx, ffn in enumerate(self.ffn):
            y = y + torch.tanh(self.attn[idx](y))
            #y = y + self.attn_norm[idx](self.attn[idx](y))
            y = y + torch.tanh(ffn(y))
            #y = y + self.ffn_norm[idx](ffn(y))

        cls_out = y[0]               # (B, K)

        if not return_all_tokens:
            # Just CLS
            return cls_out.view(N, F, K)

        # All non-CLS tokens: (L+1, B, K) -> (B, L+1, K) -> (N, F, L+1, K)
        #layer_tokens = y[1:].transpose(0, 1)   # (B, L+1, K)
        #Exclude 2nd token
        layer_tokens = y.transpose(0, 1)   # (B, L+1, K)
        return layer_tokens.view(N, F, Lp + 1, K)


# ---------------------------------------------------------------------------
# Channel-wise base encoder (scalar -> k -> transformer summary -> scalar)
# ---------------------------------------------------------------------------

class ChannelGraphBase(nn.Module):
    """
    Base encoder for channel-wise graph processing:

      - Handles scalar -> k-dim token projection (shared over features).
      - Stacks 'num_layers' of a given layer class operating in k-dim token space.
      - Reads out layer-wise tokens with a Transformer + summary token.
      - Projects summarized k-dim back to scalar per feature.

    Subclasses choose the inner layer type (e.g. attention, GCN) by
    passing 'layer_cls' and 'layer_kwargs'.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int = 16,     # k
        ffn_dropout: float = 0.0,
        layer_cls: Optional[Type[nn.Module]] = None,
        layer_kwargs: Optional[Dict[str, Any]] = None,
        # transformer for layer-wise summary
        summary_num_layers: int = 2,
        summary_nhead: int = 4,
        summary_ff_mult: int = 4,
        summary_attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        if layer_cls is None:
            raise ValueError("layer_cls must be provided for ChannelGraphBase")

        if layer_kwargs is None:
            layer_kwargs = {}

        self.num_layers = num_layers
        self.embed_dim = embed_dim

        proj_hidden_dim = embed_dim * 2

        # Scalar <-> k-dim projectors
        self.in_proj = ScalarToKProjector(
            k=embed_dim,
            hidden_dim=proj_hidden_dim,
            dropout=ffn_dropout,
        )
        self.out_proj = KToScalarProjector(
            k=embed_dim,
            hidden_dim=proj_hidden_dim,
            dropout=ffn_dropout,
        )

        # Stacked inner layers in k-dim token space
        self.layers = nn.ModuleList(
            layer_cls(embed_dim=embed_dim, **layer_kwargs)
            for _ in range(num_layers)
        )

        # Transformer that reads out a summary over layer-wise states
        self.layerwise_summary = LayerwiseTransformerSummary(
            embed_dim=embed_dim,
            num_layers=summary_num_layers,
            nhead=summary_nhead,
            attn_dropout=ffn_dropout if summary_attn_dropout is None else summary_attn_dropout,
            ffn_dropout=ffn_dropout,
            ff_mult=summary_ff_mult,
        )

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,           # (N, F)
        edge_index: Tensor,  # (2, E)
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        N, F = x.shape
        if num_nodes is None:
            num_nodes = N

        # Project features -> initial state in k-dim token space
        h = self.in_proj(x)   # (N, F, k)

        # Collect layer-wise states (state-space style evolution)
        states = [h]#[h]
        for i, layer in enumerate(self.layers):
            h_write = layer(edge_index, h, num_nodes=num_nodes)   # (N, G, k)
            states.append(h_write)
            h = h_write

        # Stack along layer dimension: (N, F, L+1, k)
        h_seq = torch.stack(states, dim=2)

        # Read out a summary token from the layer-wise state sequence (N, F, k)
        h_cls = self.layerwise_summary(h_seq, return_all_tokens=False)

        # Compress back to scalar features
        node_repr = self.out_proj(h_cls)                  # (N, F)
        return node_repr

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.out_proj.reset_parameters()
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        if hasattr(self.layerwise_summary, "reset_parameters"):
            self.layerwise_summary.reset_parameters()
