import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = False,  # memoryless BN flag
    ):
        super().__init__()

        if layer_norm is True:
            batch_norm = False  # layer norm takes precedence
        self.input_norm = None
        if layer_norm:
            self.input_norm = nn.LayerNorm(in_dim)
        elif batch_norm:
            self.input_norm = nn.BatchNorm1d(
                in_dim,
                affine=True,
                track_running_stats=False,  # memoryless
            )

        if num_layers == 1:
            self.lins = nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=bias)
            ])
            self.norms = None
        else:
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            self.lins = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(num_layers)]
            )

            if layer_norm:
                self.norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
                )
            elif batch_norm:
                self.norms = nn.ModuleList(
                    [
                        nn.BatchNorm1d(
                            hidden_dim,
                            affine=True,
                            track_running_stats=False,  # memoryless
                        )
                        for _ in range(num_layers - 1)
                    ]
                )
            else:
                self.norms = None

        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()
        if self.input_norm is not None:
            self.input_norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        '''
        if self.input_norm is not None:
            x = self.input_norm(x)'''

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)

            if self.norms is not None:
                x = self.norms[i](x)

            x = F.gelu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return x