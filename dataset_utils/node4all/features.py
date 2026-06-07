from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

def _prepare_gcn_message_passing(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        weight = torch.empty((0,), dtype=torch.float, device=device)
        return empty, empty, weight
    edge_index = edge_index.to(device)
    row, col = edge_index
    deg = torch.bincount(row, minlength=int(num_nodes)).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return row, col, weight

def _propagate(
    x: torch.Tensor,
    *,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if row.numel() == 0:
        return x
    out = x.new_zeros((int(num_nodes), x.size(1)))
    out.index_add_(0, row, x[col] * weight.unsqueeze(1))
    return out

def _apply_propagation(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    mode: str,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    steps = int(steps)
    if steps <= 0:
        return h
    mode = str(mode).lower().strip()
    if mode == "identity":
        return h
    alpha = float(alpha)
    if alpha == 0.0:
        return h
    row, col, weight = _prepare_gcn_message_passing(edge_index, num_nodes=num_nodes, device=h.device)
    if row.numel() == 0:
        return h
    for _ in range(steps):
        neigh = _propagate(h, row=row, col=col, weight=weight, num_nodes=num_nodes)
        if mode == "anti":
            h = (1.0 + alpha) * h - alpha * neigh
        if mode == "highpass":
            h = h - alpha * neigh
    return h

def _rng_device(
    generator: Optional[torch.Generator],
    fallback: torch.device,
) -> torch.device:
    if generator is None:
        return fallback
    try:
        return generator.device
    except Exception:
        return fallback

def _weighted_choice(
    items,
    weights,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
):
    if not items:
        raise ValueError("items must be non-empty")
    rng_device = _rng_device(generator, device)
    total = float(sum(weights))
    if total <= 0.0:
        idx = torch.randint(0, len(items), (1,), generator=generator, device=rng_device).item()
        return items[idx]
    r = torch.rand((), generator=generator, device=rng_device).item() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return item
    return items[-1]

def _sample_choice(
    items,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
):
    if not items:
        raise ValueError("items must be non-empty")
    rng_device = _rng_device(generator, device)
    idx = torch.randint(0, len(items), (1,), generator=generator, device=rng_device).item()
    return items[idx]

def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _sample_mode_value(
    mode_cfg: object,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> str:
    if isinstance(mode_cfg, dict):
        items = list(mode_cfg.keys())
        weights = list(mode_cfg.values())
        return str(_weighted_choice(items, weights, generator=generator, device=device))
    if isinstance(mode_cfg, (list, tuple)):
        if len(mode_cfg) == 0:
            return "identity"
        return str(_sample_choice(list(mode_cfg), generator=generator, device=device))
    if mode_cfg is None:
        return "identity"
    return str(mode_cfg)

def _sample_steps_value(
    steps_cfg: object,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> int:
    if isinstance(steps_cfg, dict):
        items = [int(k) for k in steps_cfg.keys()]
        weights = list(steps_cfg.values())
        return int(_weighted_choice(items, weights, generator=generator, device=device))
    if isinstance(steps_cfg, (list, tuple)):
        if len(steps_cfg) == 0:
            return 0
        if len(steps_cfg) == 2 and _is_number(steps_cfg[0]) and _is_number(steps_cfg[1]):
            lo = int(min(steps_cfg[0], steps_cfg[1]))
            hi = int(max(steps_cfg[0], steps_cfg[1]))
            if lo == hi:
                return int(lo)
            rng_device = _rng_device(generator, device)
            return int(torch.randint(lo, hi + 1, (1,), generator=generator, device=rng_device).item())
        return int(_sample_choice([int(v) for v in steps_cfg], generator=generator, device=device))
    if steps_cfg is None:
        return 0
    return int(steps_cfg)

def _sample_vector(
    feature_dim: int,
    num_samples: int = 1,
    *,
    device: torch.device,
    generator: Optional[torch.Generator],
    dtype: torch.dtype,
) -> torch.Tensor:
    mean = torch.normal(
        mean=0.0,
        std=1.0,
        size=(),
        device=device,
        generator=generator,
        dtype=dtype,
    )
    std = torch.abs(
        torch.normal(
            mean=0.0,
            std=1.0,
            size=(),
            device=device,
            generator=generator,
            dtype=dtype,
        )
        * mean
    )
    return torch.normal(
        mean=mean,
        std=std,
        size=(int(num_samples), int(feature_dim)),
        device=device,
        generator=generator,
        dtype=dtype,
    )

def generate_node_features(
    labels: torch.Tensor,
    feature_dim: int,
    *,
    feature_cfg: Optional[Dict[str, object]] = None,
    generator: Optional[torch.Generator] = None,
    edge_index: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate Node4All features from sampled semantic factors, hidden
    dimensions, and stochastic graph propagation.
    """
    output_device = device or labels.device or torch.device("cpu")
    num_nodes = int(labels.numel())
    feature_dim = max(0, int(feature_dim))
    if num_nodes == 0 or feature_dim == 0:
        return torch.empty((num_nodes, feature_dim), device=output_device)
    if edge_index is None:
        raise ValueError("edge_index is required for feature propagation")

    fc = feature_cfg or {}
    semantic_dim_scale_range = fc.get("semantic_dim_scale_range", (0.0, 0.8))
    hidden_dim_scale_range = fc.get("hidden_dim_scale_range", (2.0, 2.0))
    propagate_steps_cfg = fc.get("propagate_steps", None)
    propagate_alpha = float(fc.get("propagate_alpha", 0.1))
    propagate_mode_cfg = fc.get("propagate_mode", None)
    propagate_position = str(fc.get("propagate_position", "root")).lower().strip()
    propagate_mode_probs = fc.get("propagate_mode_probs", None)
    propagate_steps_probs = fc.get("propagate_steps_probs", None)

    if isinstance(semantic_dim_scale_range, (list, tuple)) and len(semantic_dim_scale_range) == 2:
        lo = float(semantic_dim_scale_range[0])
        hi = float(semantic_dim_scale_range[1])
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        scale = lo + (hi - lo) * torch.rand((), device=output_device, generator=generator).item()
    else:
        scale = float(semantic_dim_scale_range)
    scale = max(0.0, min(1.0, float(scale)))
    semantic_dim = max(1, min(feature_dim, int(scale * feature_dim)))

    root = _sample_vector(
        semantic_dim,
        num_samples=num_nodes,
        device=output_device,
        generator=generator,
        dtype=torch.get_default_dtype(),
    )

    lo, hi = 1.0, 1.0
    if isinstance(hidden_dim_scale_range, (list, tuple)) and len(hidden_dim_scale_range) == 2:
        lo = float(hidden_dim_scale_range[0])
        hi = float(hidden_dim_scale_range[1])

    hidden_frac = torch.rand((), device=output_device, generator=generator).item()
    delta_dim = feature_dim * hi - feature_dim * lo
    hidden_dim = int(feature_dim * lo + hidden_frac * delta_dim)

    def _apply_scoped_propagation(current: torch.Tensor) -> torch.Tensor:
        mode_source = propagate_mode_cfg if propagate_mode_cfg is not None else propagate_mode_probs
        steps_source = propagate_steps_cfg if propagate_steps_cfg is not None else propagate_steps_probs
        if mode_source is None:
            mode_source = "smooth"
        if steps_source is None:
            steps_source = 0

        mode_val = _sample_mode_value(mode_source, generator=generator, device=output_device)
        steps_val = _sample_steps_value(steps_source, generator=generator, device=output_device)
        if steps_val <= 0:
            return current
        return _apply_propagation(
            current,
            edge_index,
            num_nodes=num_nodes,
            mode=mode_val,
            alpha=propagate_alpha,
            steps=steps_val,
        )

    # orthogonal projection to hidden_dim
    proj = torch.nn.init.orthogonal_(
        torch.empty((semantic_dim, hidden_dim), device=output_device)
    )
    h = root @ proj

    # orthogonal projection to feature_dim
    proj = torch.nn.init.orthogonal_(
        torch.empty((hidden_dim, feature_dim), device=output_device)
    )
    h = h @ proj

    if propagate_position == "output":
        h_mean = h.mean(dim=0, keepdim=True)
        h_std = h.std(dim=0, keepdim=True) + 1e-6
        h = (h - h_mean) / h_std    
        h = _apply_scoped_propagation(h)
        # Variation Recovery
        h = h * h_std + h_mean

    return h


__all__ = ["generate_node_features"]
