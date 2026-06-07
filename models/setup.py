from models.gat import GAT
from models.gcn import GCN
from models.gin import GIN
from models.mlp import MLP
import torch.nn as nn

from models.utils import create_norm, create_activation


def setup_model(
    in_dim,
    args,
    purpose='enc',
    num_layers_override=None,
    out_dim=None,
    sat_embed_dim=None,
    model_type_override=None,
) -> nn.Module:
    if purpose == 'enc':
        m_type = model_type_override if model_type_override is not None else args.encoder
        num_layers = args.enc_num_layers
    elif purpose == 'dec':
        m_type = model_type_override if model_type_override is not None else args.decoder
        num_layers = args.dec_num_layers
    elif purpose == 'pred':
        m_type = model_type_override if model_type_override is not None else args.predictor_type
        num_layers = 2
    else:
        raise ValueError(f"Unknown purpose: {purpose}")

    if num_layers_override is not None:
        num_layers = num_layers_override

    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=args.nhead,
            activation=create_activation(args.activation),
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=create_norm(args.norm),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=args.feat_drop,
            activation=create_activation(args.activation),
            residual=args.residual,
            norm=create_norm(args.norm),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=args.feat_drop,
            activation=create_activation(args.activation),
            residual=args.residual,
            norm=create_norm(args.norm),
        )
    elif m_type == "mlp":
        mod = MLP(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=args.feat_drop,
            layer_norm=True,
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    return mod
