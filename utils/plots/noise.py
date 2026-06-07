from __future__ import annotations

import os
from typing import Optional, Mapping

from .labels import _normalize_label_sets, _label_style


def save_feature_noise_pair_scatter(
    *,
    out_path: str,
    values_pre,
    values_post,
    noise_pre,
    noise_post,
    labels=None,
    label_sets: Optional[Mapping[str, object]] = None,
    max_features: int = 8,
    max_nodes: int = 2000,
    seed: int = 42,
    title: str = "",
    outlier_keep_quantile: float = 0.99,
) -> bool:
    """
    Save one plot with pre/post rows per label set to compare noise scatter.
    """
    if not out_path:
        return False

    try:
        import numpy as np
    except Exception:
        return False

    if values_pre is None or values_post is None or noise_pre is None or noise_post is None:
        return False

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    values_pre = _to_numpy(values_pre)
    noise_pre = _to_numpy(noise_pre)
    values_post = _to_numpy(values_post)
    noise_post = _to_numpy(noise_post)

    if values_pre.ndim != 2 or noise_pre.ndim != 2:
        return False
    if values_post.ndim != 2 or noise_post.ndim != 2:
        return False
    if values_pre.shape != noise_pre.shape:
        return False
    if values_post.shape != noise_post.shape:
        return False
    if values_pre.shape != values_post.shape:
        return False

    num_nodes, num_feats = values_pre.shape
    if num_nodes == 0 or num_feats == 0:
        return False

    label_items = _normalize_label_sets(labels, label_sets)
    if not label_items:
        return False
    label_info = []
    for name, vals in label_items:
        lbl = _to_numpy(vals)
        if lbl.ndim == 1 and lbl.shape[0] == num_nodes:
            label_info.append((name, lbl, False))
            continue
        if lbl.ndim == 2 and lbl.shape[0] == num_nodes and lbl.shape[1] == num_feats:
            label_info.append((name, lbl, True))
            continue
        return False

    max_features = max(1, int(max_features))
    max_nodes = max(1, int(max_nodes))
    outlier_keep_quantile = float(outlier_keep_quantile)
    if outlier_keep_quantile <= 0.0 or outlier_keep_quantile > 1.0:
        outlier_keep_quantile = 1.0

    if num_feats <= max_features:
        feat_idx = list(range(num_feats))
    else:
        step = max(1, num_feats // max_features)
        feat_idx = list(range(0, num_feats, step))[:max_features]

    rng = np.random.default_rng(int(seed))
    sample_idx = {}
    if num_nodes > max_nodes:
        for f_idx in feat_idx:
            sample_idx[f_idx] = rng.choice(num_nodes, size=max_nodes, replace=False)
    else:
        for f_idx in feat_idx:
            sample_idx[f_idx] = None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    ncols = max(1, len(feat_idx))
    total_rows = len(label_info) * 2
    fig, axes = plt.subplots(
        total_rows,
        ncols,
        figsize=(3.2 * ncols, 2.4 * total_rows),
        dpi=300,
    )
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    axes = np.array(axes).reshape(-1)

    used = set()
    for label_idx, (label_name, label_vals, per_feature) in enumerate(label_info):
        for i, f_idx in enumerate(feat_idx):
            for row_offset, tag, values, noise in (
                (0, "input", values_pre, noise_pre),
                (1, "encoder", values_post, noise_post),
            ):
                row = label_idx * 2 + row_offset
                col = i
                ax_idx = row * ncols + col
                ax = axes[ax_idx]
                used.add(ax_idx)
                vals = values[:, f_idx]
                noi = noise[:, f_idx]
                sel = sample_idx[f_idx]
                lbl_full = label_vals[:, f_idx] if per_feature else label_vals
                kind, cmap, vmin, vmax = _label_style(lbl_full, np, label_name=label_name)
                if sel is not None:
                    vals = vals[sel]
                    noi = noi[sel]
                    lbl = lbl_full[sel]
                else:
                    lbl = lbl_full

                if outlier_keep_quantile < 1.0 and vals.size > 8:
                    tail = (1.0 - outlier_keep_quantile) * 0.5
                    v_lo, v_hi = np.quantile(vals, [tail, 1.0 - tail])
                    n_lo, n_hi = np.quantile(noi, [tail, 1.0 - tail])
                    inlier = (
                        np.isfinite(vals)
                        & np.isfinite(noi)
                        & (vals >= v_lo)
                        & (vals <= v_hi)
                        & (noi >= n_lo)
                        & (noi <= n_hi)
                    )
                    if bool(np.any(inlier)):
                        vals = vals[inlier]
                        noi = noi[inlier]
                        lbl = lbl[inlier]

                if kind == "categorical":
                    scatter = ax.scatter(vals, noi, c=lbl, s=8, alpha=0.6, cmap=cmap)
                else:
                    scatter = ax.scatter(vals, noi, c=lbl, s=8, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                if len(label_info) == 1:
                    ax.set_title(f"{tag} feature {f_idx}")
                else:
                    ax.set_title(f"{label_name} - {tag} feature {f_idx}")
                ax.set_xlabel("value")
                ax.set_ylabel("noise")

    for j in range(len(axes)):
        if j not in used:
            axes[j].axis("off")

    if title:
        fig.suptitle(title)

    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return True
