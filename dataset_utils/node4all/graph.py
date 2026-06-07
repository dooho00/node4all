from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch_geometric.utils import to_undirected, coalesce


def _np_rng_from_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randrange(1 << 32))


def _weighted_choice(
    rng_np: np.random.Generator, items: np.ndarray, weights: np.ndarray, size: int
) -> np.ndarray:
    w = weights.astype(np.float64, copy=False)
    s = float(w.sum())
    if s <= 1e-12:
        return rng_np.choice(items, size=int(size), replace=True)
    p = w / s
    return rng_np.choice(items, size=int(size), replace=True, p=p)


def build_graph_structure(
    *,
    rng: random.Random,
    num_nodes: int,
    avg_degree: float = 4.0,
    degree_sigma: float = 0.8,
    oversample_factor: float = 1.3,
    max_rounds: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, int, np.ndarray]:
    """
    Chung–Lu style expected-degree sampler (fast, sparse-regime).
    - Sample expected degrees w_u from a log-normal with E[w_u] = avg_degree
    - Target expected number of undirected edges is m_target = 0.5 * sum_u w_u
    - Sample candidate endpoints proportional to w_u, deduplicate, repeat until enough unique edges

    Returns: labels, edge_index(undirected), N, M (empty)
    """
    rng_np = _np_rng_from_rng(rng)

    n = int(num_nodes)
    if n <= 0:
        raise ValueError("num_nodes must be > 0")
    if avg_degree < 0.0:
        raise ValueError("avg_degree must be >= 0")

    labels = torch.arange(n, dtype=torch.long)

    sigma = float(max(degree_sigma, 0.0))
    z = rng_np.normal(loc=-0.5 * sigma * sigma, scale=sigma, size=n).astype(np.float64)
    w = float(avg_degree) * np.exp(z)  # expected-degree parameters, shape (n,)

    w = np.clip(w, 0.0, float(max(n - 1, 0))).astype(np.float64, copy=False)
    W = float(w.sum())
    if W <= 1e-12 or n <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return labels, edge_index, n, np.empty((0, 0), dtype=np.float32)

    # Target number of undirected edges in expectation
    m_target = int(max(0, round(0.5 * W)))
    if m_target <= 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return labels, edge_index, n, np.empty((0, 0), dtype=np.float32)

    nodes = np.arange(n, dtype=np.int64)
    weights = w

    # We build a set of unique undirected edges (u < v) using a vectorized encoding
    # key = u * n + v
    edge_keys = np.empty((0,), dtype=np.int64)

    # Oversample to reduce loss from self-loops and duplicates, repeat a few rounds if needed
    for _ in range(int(max_rounds)):
        need = m_target - edge_keys.size
        if need <= 0:
            break

        batch = int(np.ceil(float(need) * float(max(1.0, oversample_factor))))
        src = _weighted_choice(rng_np, nodes, weights, batch)
        dst = _weighted_choice(rng_np, nodes, weights, batch)

        mask = (src != dst)
        if not np.any(mask):
            continue
        src = src[mask]
        dst = dst[mask]

        u = np.minimum(src, dst)
        v = np.maximum(src, dst)

        keys = (u.astype(np.int64) * np.int64(n)) + v.astype(np.int64)
        if edge_keys.size == 0:
            edge_keys = np.unique(keys)
        else:
            edge_keys = np.unique(np.concatenate([edge_keys, keys], axis=0))

        if edge_keys.size >= m_target:
            edge_keys = edge_keys[:m_target]
            break

    if edge_keys.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return labels, edge_index, n, np.empty((0, 0), dtype=np.float32)

    u = (edge_keys // np.int64(n)).astype(np.int64)
    v = (edge_keys % np.int64(n)).astype(np.int64)

    edge_index = torch.from_numpy(np.stack([u, v], axis=0)).long().contiguous()

    # Coalesce, remove any self-loops (should already be none), and symmetrize
    edge_index, _ = coalesce(edge_index, None, n, n)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_index = to_undirected(edge_index, num_nodes=n)

    return labels, edge_index, n, np.empty((0, 0), dtype=np.float32)


__all__ = ["build_graph_structure"]