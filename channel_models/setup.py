import torch.nn as nn

from channel_models.gat import ChannelGATEncoder
from channel_models.cgt import CGTEncoder
from channel_models.gcn import ChannelGCNEncoder


def normalize_arch_type(model_type: str) -> str:
    """Normalize architecture names to the paper-aligned public names."""
    canonical = {
        "cgt": "cgt",
        "channel_gat": "channel_gat",
        "channel_gcn": "channel_gcn",
    }
    former_prefix = "".join(("s", "c"))
    compatibility_aliases = {
        f"{former_prefix}_cgt": "cgt",
        f"{former_prefix}_gat": "channel_gat",
        f"{former_prefix}_gcn": "channel_gcn",
    }
    name = str(model_type).strip().lower()
    return canonical.get(name, compatibility_aliases.get(name, name))


def setup_model(
    is_enc: bool,
    args,
    purpose: str = "enc",
    is_linear: bool = False,
) -> nn.Module:
    """
    Set up a channel-wise encoder/decoder used by Node4All.

    Supported types (for encoder / decoder / predictor):
      - "cgt": Channel Graph Transformer (CGTEncoder)
      - "channel_gat": attention-based channel-wise encoder (ChannelGATEncoder)
      - "channel_gcn": GCN-style channel-wise encoder (ChannelGCNEncoder)
    """

    if not is_enc:
        raise ValueError("Encoders without features are no longer supported.")

    # ------------------------------------------------------------------
    # 1. Resolve model type and number of layers
    # ------------------------------------------------------------------
    if purpose == "enc":
        m_type = args.arch_type
        num_layers = args.enc_num_layers
    elif purpose == "dec":
        m_type = args.arch_type
        num_layers = args.dec_num_layers
        #embed_dim = 2
    elif purpose == "pred":
        m_type = args.predictor_type
        num_layers = getattr(args, "pred_num_layers", 2)
    else:
        raise ValueError(f"Unknown purpose: {purpose}")
    m_type = normalize_arch_type(m_type)

    # ------------------------------------------------------------------
    # 2. Common hyperparameters for channel-wise graph encoders
    # ------------------------------------------------------------------
    # embed_dim (k)
    embed_dim = getattr(args, "embed_dim", 16)
    # dropouts
    mlp_dropout = getattr(args, "feat_drop", 0.5)
    attn_dropout = getattr(args, "attn_drop", 0.2)

    # ------------------------------------------------------------------
    # 3. Instantiate the requested channel-wise graph model
    # ------------------------------------------------------------------
    def _parse_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "y"):
                return True
            if lowered in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"{name} must be a boolean string, got {value!r}")
        return bool(value)

    if m_type == "channel_gat":
        mod = ChannelGATEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attn_dropout=attn_dropout,
            ffn_dropout=mlp_dropout,
            summary_attn_dropout=attn_dropout,
        )

    elif m_type == "channel_gcn":
        mod = ChannelGCNEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_dropout=mlp_dropout,
            summary_attn_dropout=attn_dropout,
        )
    
    elif m_type == "cgt":
        cgt_hops = getattr(args, "cgt_hops", None)
        cgt_filters = getattr(args, "cgt_filters", None)
        if cgt_hops is None or cgt_filters is None:
            raise ValueError("Missing CGT layout args: cgt_hops, cgt_filters.")
        cgt_hops = int(cgt_hops)
        cgt_filters = int(cgt_filters)
        cgt_norms = getattr(args, "cgt_norms", None)
        cgt_use_self_loops = getattr(args, "cgt_use_self_loops", "true")
        cgt_use_self_loops = _parse_bool(cgt_use_self_loops, name="cgt_use_self_loops")
        if cgt_hops < 1 or cgt_filters < 1:
            raise ValueError("cgt_hops and cgt_filters must be >= 1.")
        mod = CGTEncoder(
            num_layers=num_layers,
            n_hops=cgt_hops,
            n_filters=cgt_filters,
            ffn_dropout=mlp_dropout,
            is_linear=is_linear,
            filter_norms=cgt_norms,
            use_self_loops=cgt_use_self_loops,
            summary_attn_dropout=attn_dropout,
        )

    else:
        raise NotImplementedError(f"Unsupported model type: {m_type}. "
                                  f"Use 'channel_gat', 'channel_gcn', or 'cgt'.")

    return mod
