from __future__ import annotations

from typing import Optional, Mapping, Tuple, Dict

import torch
from torch_geometric.utils import degree as pyg_degree

from .cfg import PLOT_CFG


def _normalize_label_sets(labels, label_sets):
    if label_sets is not None:
        if isinstance(label_sets, Mapping):
            items = list(label_sets.items())
        else:
            items = list(label_sets)
        return [(str(name), values) for name, values in items if values is not None]
    if labels is None:
        return []
    return [("labels", labels)]


def _label_style(
    labels_np,
    np,
    label_name: Optional[str] = None,
) -> Tuple[str, str, Optional[float], Optional[float]]:
    if labels_np.dtype.kind in "iu":
        unique = np.unique(labels_np)
        if unique.size <= 20:
            cmap = "tab20" if unique.size > 10 else "tab10"
            return "categorical", cmap, None, None
    vmin = None
    vmax = None
    q_cfg = PLOT_CFG.get("color_quantile", None)
    if q_cfg is not None:
        try:
            if isinstance(q_cfg, (list, tuple)) and len(q_cfg) == 2:
                qmin = float(q_cfg[0])
                qmax = float(q_cfg[1])
            else:
                qmin = None
                qmax = None
        except Exception:
            qmin = None
            qmax = None
        if qmin is not None and qmax is not None and 0.0 <= qmin < qmax <= 1.0:
            vals = labels_np[np.isfinite(labels_np)]
            if vals.size:
                vmin = float(np.nanpercentile(vals, qmin * 100.0))
                vmax = float(np.nanpercentile(vals, qmax * 100.0))
    if vmin is None or vmax is None:
        vmin = float(np.nanmin(labels_np)) if labels_np.size else 0.0
        vmax = float(np.nanmax(labels_np)) if labels_np.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    lname = str(label_name or "").strip().lower()
    if lname == "degree":
        return "continuous", "magma", vmin, vmax
    return "continuous", "viridis", vmin, vmax


def _coerce_edge_index(edge_index):
    if edge_index is None:
        return None
    if torch.is_tensor(edge_index):
        edge_index = edge_index
    else:
        try:
            row, col, _ = edge_index.coo()
            edge_index = torch.stack((row, col), dim=0)
        except Exception:
            try:
                coo = edge_index.to_torch_sparse_coo_tensor()
                edge_index = coo.indices()
            except Exception:
                return None
    if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    return edge_index


def _add_feature_labels(
    label_sets: Dict[str, torch.Tensor],
    features_cpu: torch.Tensor,
) -> None:
    num_nodes, num_feats = features_cpu.shape
    if num_nodes <= 0 or num_feats <= 0:
        return

    mean_vec = features_cpu.mean(dim=0)
    centered = features_cpu - mean_vec
    pc1_scores = None
    if num_nodes >= 2 and num_feats >= 1:
        try:
            _, _, v = torch.pca_lowrank(centered, q=1)
            pc1_scores = centered @ v[:, 0]
            if v[:, 0].sum() < 0:
                pc1_scores = -pc1_scores
        except Exception:
            try:
                _, _, v = torch.linalg.svd(centered, full_matrices=False)
                pc1_scores = centered @ v[0]
                if v[0].sum() < 0:
                    pc1_scores = -pc1_scores
            except Exception:
                pc1_scores = None
    if pc1_scores is not None:
        pc1_scores = pc1_scores - pc1_scores.mean()
        std = pc1_scores.std(unbiased=False)
        if std > 0:
            pc1_scores = pc1_scores / std
        label_sets["feature_pc1"] = pc1_scores
    norm_mean = mean_vec.norm()
    dot = (features_cpu * mean_vec).sum(dim=1)
    norm_x = (features_cpu * features_cpu).sum(dim=1).sqrt()
    denom = norm_x * norm_mean
    cos = torch.zeros_like(dot)
    mask = denom > 0
    if mask.any():
        cos[mask] = dot[mask] / denom[mask]
    label_sets["global_mean_cosine"] = cos


def build_graph_label_sets(
    graph,
    *,
    features: Optional[torch.Tensor] = None,
    include_community: bool = True,
) -> Dict[str, torch.Tensor]:
    community_labels: Dict[str, torch.Tensor] = {}
    feature_labels: Dict[str, torch.Tensor] = {}
    structure_labels: Dict[str, torch.Tensor] = {}
    labels = getattr(graph, "y", None)
    labels_cpu = None
    if labels is not None:
        labels_cpu = labels.detach().cpu()
        if labels_cpu.ndim != 1:
            labels_cpu = labels_cpu.view(-1)
        if include_community:
            community_labels["community"] = labels_cpu

    edge_index = _coerce_edge_index(getattr(graph, "edge_index", None))
    if edge_index is None:
        edge_index = _coerce_edge_index(getattr(graph, "adj_t", None))

    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None:
        if labels_cpu is not None:
            num_nodes = int(labels_cpu.numel())
        elif features is not None and torch.is_tensor(features):
            num_nodes = int(features.size(0))
        elif edge_index is not None and edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
    if num_nodes is None:
        return label_sets
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return community_labels

    features_cpu = None
    if features is not None and torch.is_tensor(features):
        features_cpu = features.detach().cpu()
        if features_cpu.dim() != 2 or features_cpu.size(0) != num_nodes:
            features_cpu = None
    if features_cpu is not None:
        _add_feature_labels(feature_labels, features_cpu)

    if edge_index is None or edge_index.numel() == 0:
        deg = torch.zeros(num_nodes, dtype=torch.float)
        structure_labels["degree"] = torch.log1p(deg)
        if include_community and labels_cpu is not None:
            community_labels["community_homophily"] = torch.zeros(num_nodes, dtype=torch.float)
        label_sets: Dict[str, torch.Tensor] = {}
        label_sets.update(community_labels)
        label_sets.update(feature_labels)
        label_sets.update(structure_labels)
        return label_sets

    edge_index_cpu = edge_index.detach().cpu()
    if edge_index_cpu.dim() == 2 and edge_index_cpu.size(0) != 2 and edge_index_cpu.size(1) == 2:
        edge_index_cpu = edge_index_cpu.t().contiguous()

    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]
    mask = src != dst
    if mask.any():
        src = src[mask]
        dst = dst[mask]
    else:
        src = src[:0]
        dst = dst[:0]

    src_u = torch.cat((src, dst), dim=0)
    dst_u = torch.cat((dst, src), dim=0)

    deg = pyg_degree(src_u, num_nodes=num_nodes, dtype=torch.float)
    structure_labels["degree"] = torch.log1p(deg)
    if features_cpu is not None:
        diff = features_cpu[src] - features_cpu[dst]
        sq = (diff * diff).sum(dim=1)
        sq_u = torch.cat((sq, sq), dim=0)
        smooth_sum = torch.zeros(num_nodes, dtype=torch.float)
        smooth_sum.index_add_(0, src_u, sq_u)
        smooth = torch.zeros_like(smooth_sum)
        mask = deg > 0
        if mask.any():
            smooth[mask] = smooth_sum[mask] / deg[mask]
        structure_labels["smoothness"] = smooth
    if include_community and labels_cpu is not None and labels_cpu.numel() == num_nodes:
        homophily = torch.zeros(num_nodes, dtype=torch.float)
        if src_u.numel() > 0:
            same = (labels_cpu[src_u] == labels_cpu[dst_u]).to(torch.float)
            homophily.index_add_(0, src_u, same)
        mask = deg > 0
        if mask.any():
            homophily[mask] = homophily[mask] / deg[mask]
        community_labels["community_homophily"] = homophily
    label_sets: Dict[str, torch.Tensor] = {}
    label_sets.update(community_labels)
    label_sets.update(feature_labels)
    label_sets.update(structure_labels)
    return label_sets
