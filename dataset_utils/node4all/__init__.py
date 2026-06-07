import random
from typing import Dict, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected, remove_self_loops

from .features import generate_node_features
from .graph import build_graph_structure

_DEFAULT_CFG: Dict[str, object] = {
    "num_nodes_range": (128, 4096),
    "num_nodes_bias": 1.0,
    "feature_dim_range": (32, 32),
    "avg_degree_range": (2.0, 16.0), # average degree range
    "avg_degree_bias": 1.0,
    "degree_sigma_range": (0.5, 1.5), # lognormal degree variation strength
    "degree_sigma_bias": 1.0,
    
    "semantic_dim_scale_range": (0.4, 0.8),
    "hidden_dim_scale_range": (0.5, 2.0),

    "propagate_alpha": (0.0, 1.0),
    "propagate_position": "output",
    "propagate_scope": "graph",
    "propagate_mode_probs": {
        "anti": 0.5,
        "highpass": 0.4,
        "identity": 0.1,
    },
    "propagate_steps_probs": {
        1: 0.5,
        2: 0.3,
        3: 0.2,
    },
    "seed": None,
}

def _merge_cfg(cfg: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not cfg:
        return dict(_DEFAULT_CFG)
    merged = dict(_DEFAULT_CFG)
    merged.update(cfg)
    return merged

def _clamp_range(lo, hi):
    lo_v = lo if lo <= hi else hi
    hi_v = hi if lo <= hi else lo
    return lo_v, hi_v


def _sample_int(rng: random.Random, lo: int, hi: int) -> int:
    lo, hi = _clamp_range(int(lo), int(hi))
    return rng.randint(lo, hi)


def _sample_int_powerbias(rng: random.Random, lo: int, hi: int, *, bias: float) -> int:
    lo, hi = _clamp_range(int(lo), int(hi))
    if lo == hi:
        return int(lo)
    b = float(bias)
    if b <= 0.0:
        b = 1.0
    u = rng.random() ** b
    val = lo + (hi - lo) * u
    return int(min(hi, max(lo, round(val))))


def _sample_float_powerbias(rng: random.Random, lo: float, hi: float, *, bias: float) -> float:
    lo, hi = _clamp_range(float(lo), float(hi))
    if lo == hi:
        return float(lo)
    b = float(bias)
    if b <= 0.0:
        b = 1.0
    u = rng.random() ** b
    return float(lo + (hi - lo) * u)


def _sample_int_range_value(
    rng: random.Random,
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a (lo, hi) pair")
        lo, hi = _clamp_range(int(value[0]), int(value[1]))
        return _sample_int(rng, int(lo), int(hi))
    return int(value)


def _weighted_choice(rng: random.Random, items, weights):
    if not items:
        raise ValueError("items must be non-empty")
    total = float(sum(weights))
    if total <= 0.0:
        return rng.choice(list(items))
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return item
    return items[-1]


def _parse_feature_dim_range(cfg: Dict[str, object]) -> Optional[Tuple[int, int]]:
    f_range = cfg.get("feature_dim_range", None)
    if f_range is None:
        return None
    if not (isinstance(f_range, (list, tuple)) and len(f_range) == 2):
        raise ValueError("feature_dim_range must be a (lo, hi) pair")
    lo, hi = _clamp_range(int(f_range[0]), int(f_range[1]))
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    return int(lo), int(hi)


def _parse_num_nodes_range(cfg: Dict[str, object]) -> Optional[Tuple[int, int]]:
    n_range = cfg.get("num_nodes_range", None)
    if n_range is None:
        return None
    if not (isinstance(n_range, (list, tuple)) and len(n_range) == 2):
        raise ValueError("num_nodes_range must be a (lo, hi) pair")
    lo, hi = _clamp_range(int(n_range[0]), int(n_range[1]))
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    return int(lo), int(hi)


def _resolve_num_nodes(cfg: Dict[str, object], rng: random.Random) -> int:
    n = cfg.get("num_nodes", None)
    if n is not None:
        return int(n)
    n_range = _parse_num_nodes_range(cfg)
    if n_range is None:
        raise ValueError("num_nodes or num_nodes_range must be set")
    bias = float(cfg.get("num_nodes_bias", 1.0))
    return _sample_int_powerbias(rng, n_range[0], n_range[1], bias=bias)


def _parse_range(value: object, *, name: str) -> Tuple[float, float]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{name} must be a (lo, hi) pair")
    lo, hi = _clamp_range(float(value[0]), float(value[1]))
    return float(lo), float(hi)


def _sample_range_value(
    rng: random.Random,
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = _parse_range(value, name=name)
        return rng.uniform(float(lo), float(hi))
    return float(value)


def _parse_bool(value: object, *, name: str) -> bool:
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


def _parse_float_pair(value: object, *, name: str) -> Tuple[float, float]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{name} must be a (lo, hi) pair")
    lo, hi = _clamp_range(float(value[0]), float(value[1]))
    return float(lo), float(hi)


_FEATURE_CFG_KEYS = (
    "num_layers_mix",
    "num_layers_post_causal",
    "semantic_scale",
    "semantic_dim_scale_range",
    "hidden_dim_scale_range",
    "propagate_steps",
    "propagate_alpha",
    "propagate_mode",
    "propagate_position",
    "propagate_scope",
    "propagate_mode_probs",
    "propagate_steps_probs",
)


def _resolve_feature_cfg(rng: random.Random, cfg: Dict[str, object]) -> Dict[str, object]:
    resolved: Dict[str, object] = {}
    propagate_mode_probs = cfg.get("propagate_mode_probs", None)
    propagate_steps_probs = cfg.get("propagate_steps_probs", None)
    scope = str(cfg.get("propagate_scope", "graph")).lower().strip()
    if scope == "graph" and "propagate_steps" not in cfg and isinstance(propagate_steps_probs, dict):
        items = [int(k) for k in propagate_steps_probs.keys()]
        weights = list(propagate_steps_probs.values())
        resolved["propagate_steps"] = int(_weighted_choice(rng, items, weights))
    if scope == "graph" and "propagate_mode" not in cfg and isinstance(propagate_mode_probs, dict):
        items = list(propagate_mode_probs.keys())
        weights = list(propagate_mode_probs.values())
        resolved["propagate_mode"] = _weighted_choice(rng, items, weights)
    for key in _FEATURE_CFG_KEYS:
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "propagate_mode":
            if scope == "graph" and isinstance(propagate_mode_probs, dict):
                items = list(propagate_mode_probs.keys())
                weights = list(propagate_mode_probs.values())
                value = _weighted_choice(rng, items, weights)
            elif isinstance(value, (list, tuple)) and scope == "graph":
                if len(value) == 0:
                    continue
                value = rng.choice(list(value))
        elif key == "propagate_steps":
            if scope == "graph" and isinstance(propagate_steps_probs, dict):
                items = [int(k) for k in propagate_steps_probs.keys()]
                weights = list(propagate_steps_probs.values())
                value = int(_weighted_choice(rng, items, weights))
            elif isinstance(value, (list, tuple)) and scope == "graph":
                if len(value) == 2:
                    value = _sample_range_value(rng, value, name=key)
        elif isinstance(value, (list, tuple)):
            if key in ("propagate_position", "propagate_scope"):
                if len(value) == 0:
                    continue
                value = rng.choice(list(value))
            elif len(value) == 2:
                value = _sample_range_value(rng, value, name=key)
        resolved[key] = value
    return resolved


def _compute_feature_homophily(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    if edge_index is None or edge_index.numel() == 0:
        return 0.0
    row, col = edge_index
    if row.numel() == 0:
        return 0.0
    mask = row != col
    if not torch.any(mask):
        return 0.0
    row = row[mask]
    col = col[mask]
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = (x_norm[row] * x_norm[col]).sum(dim=1)
    if sim.numel() == 0:
        return 0.0
    sim01 = (sim + 1.0) * 0.5
    sim01 = torch.clamp(sim01, 0.0, 1.0)
    return float(sim01.mean().item())


def sample_node4all(
    cfg: Optional[Dict[str, object]] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> Data:
    cfg = _merge_cfg(cfg)
    rng = random.Random(cfg.get("seed", None))

    num_nodes = _resolve_num_nodes(cfg, rng)
    feature_dim_range = _parse_feature_dim_range(cfg)
    if feature_dim_range is None:
        raise ValueError("feature_dim_range must be set")
    feature_dim = _sample_int(rng, feature_dim_range[0], feature_dim_range[1])

    avg_degree_range = cfg.get("avg_degree_range")
    if avg_degree_range is None:
        raise ValueError("avg_degree_range must be set")
    d_lo, d_hi = _parse_range(avg_degree_range, name="avg_degree_range")
    avg_degree_bias = float(cfg.get("avg_degree_bias", 1.0))
    avg_degree = _sample_float_powerbias(
        rng,
        d_lo,
        d_hi,
        bias=float(avg_degree_bias),
    )

    degree_sigma_range = cfg.get("degree_sigma_range")
    if degree_sigma_range is None:
        raise ValueError("degree_sigma_range must be set")
    s_lo, s_hi = _parse_range(degree_sigma_range, name="degree_sigma_range")
    degree_sigma = _sample_float_powerbias(
        rng,
        s_lo,
        s_hi,
        bias=float(cfg.get("degree_sigma_bias", 1.0)),
    )
    _, edge_index, _, _ = build_graph_structure(
        rng=rng,
        num_nodes=num_nodes,
        avg_degree=avg_degree,
        degree_sigma=degree_sigma,
    )
    labels = torch.zeros((num_nodes,), dtype=torch.long)
    feature_cfg_used = _resolve_feature_cfg(rng, cfg)
    features = generate_node_features(
        labels,
        feature_dim,
        feature_cfg=feature_cfg_used,
        generator=generator,
        edge_index=edge_index,
    )
    
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
   
    data = Data(
        x=features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=labels,
    )

    return data


__all__ = ["sample_node4all"]
