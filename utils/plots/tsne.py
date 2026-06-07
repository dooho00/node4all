from __future__ import annotations

import os
from typing import Optional, Mapping

from .labels import _normalize_label_sets, _label_style


def save_tsne_comparison(
    *,
    out_path: str,
    features,
    embeddings,
    labels=None,
    label_sets: Optional[Mapping[str, object]] = None,
    max_nodes: int = 2000,
    max_iter: int = 1000,
    pca_enabled: bool = False,
    pca_dim: int = 50,
    seed: int = 42,
    perplexity: float = 30.0,
    title: str = "",
) -> bool:
    """
    Save side-by-side t-SNE plots for input features and representations.
    If label_sets is provided, it renders one row per label set in the same image.
    """
    if not out_path:
        return False

    try:
        import numpy as np
    except Exception:
        return False

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    features = _to_numpy(features)
    embeddings = _to_numpy(embeddings)

    if features.ndim != 2 or embeddings.ndim != 2:
        return False
    if features.shape[0] != embeddings.shape[0]:
        return False

    num_nodes = features.shape[0]
    if num_nodes == 0:
        return False

    label_items = _normalize_label_sets(labels, label_sets)
    if not label_items:
        return False
    label_info = []
    for name, vals in label_items:
        lbl = _to_numpy(vals)
        if lbl.ndim != 1 or lbl.shape[0] != num_nodes:
            return False
        label_info.append((name, lbl))

    max_nodes = max(1, int(max_nodes))
    if num_nodes > max_nodes:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(num_nodes, size=max_nodes, replace=False)
        features = features[sel]
        embeddings = embeddings[sel]
        label_info = [(name, lbl[sel]) for name, lbl in label_info]

    try:
        from sklearn.manifold import TSNE
    except Exception:
        return False

    if pca_enabled:
        try:
            from sklearn.decomposition import PCA
        except Exception:
            return False
        pca_dim = int(pca_dim)
        if pca_dim >= 2:
            if features.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=int(seed), svd_solver="auto")
                features = pca.fit_transform(features)
            if embeddings.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=int(seed), svd_solver="auto")
                embeddings = pca.fit_transform(embeddings)

    perplexity = float(perplexity)
    max_perp = max(5.0, (features.shape[0] - 1) / 3.0)
    if perplexity >= max_perp:
        perplexity = max(5.0, max_perp)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": int(seed),
    }
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["max_iter"] = int(max_iter)
    else:
        tsne_kwargs["n_iter"] = int(max_iter)
    tsne = TSNE(**tsne_kwargs)
    emb_input = tsne.fit_transform(features)
    tsne = TSNE(**tsne_kwargs)
    emb_repr = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(2, len(label_info), figsize=(4.25 * len(label_info), 7.6), dpi=160)
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for col_idx, (label_name, label_vals) in enumerate(label_info):
        kind, cmap, vmin, vmax = _label_style(label_vals, np, label_name=label_name)
        if kind == "continuous" and str(label_name).strip().lower() == "feature_pc1":
            cmap = "coolwarm"
        if kind == "categorical":
            scatter_in = axes[0, col_idx].scatter(
                emb_input[:, 0],
                emb_input[:, 1],
                c=label_vals,
                s=8,
                alpha=0.7,
                cmap=cmap,
            )
            scatter_out = axes[1, col_idx].scatter(
                emb_repr[:, 0],
                emb_repr[:, 1],
                c=label_vals,
                s=8,
                alpha=0.7,
                cmap=cmap,
            )
        else:
            scatter_in = axes[0, col_idx].scatter(
                emb_input[:, 0],
                emb_input[:, 1],
                c=label_vals,
                s=8,
                alpha=0.7,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            scatter_out = axes[1, col_idx].scatter(
                emb_repr[:, 0],
                emb_repr[:, 1],
                c=label_vals,
                s=8,
                alpha=0.7,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(scatter_in, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
            fig.colorbar(scatter_out, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

        if len(label_info) == 1:
            axes[0, col_idx].set_title("Input features (t-SNE)")
            axes[1, col_idx].set_title("Encoder output (t-SNE)")
        else:
            axes[0, col_idx].set_title(f"{label_name} - input (t-SNE)")
            axes[1, col_idx].set_title(f"{label_name} - encoder (t-SNE)")
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])

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


def save_tsne_multi(
    *,
    out_path: str,
    embeddings: Mapping[str, object],
    labels=None,
    label_sets: Optional[Mapping[str, object]] = None,
    max_nodes: int = 2000,
    max_iter: int = 1000,
    pca_enabled: bool = False,
    pca_dim: int = 50,
    seed: int = 42,
    perplexity: float = 30.0,
    title: str = "",
) -> bool:
    """
    Save a multi-row t-SNE plot for multiple embedding variants.
    Each embedding gets its own row, and each label set is a column.
    """
    if not out_path:
        return False
    if not embeddings:
        return False

    try:
        import numpy as np
    except Exception:
        return False

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    emb_items = []
    for name, emb in embeddings.items():
        if emb is None:
            continue
        arr = _to_numpy(emb)
        if arr.ndim != 2:
            return False
        emb_items.append((str(name), arr))
    if not emb_items:
        return False

    num_nodes = emb_items[0][1].shape[0]
    if num_nodes == 0:
        return False
    for _, arr in emb_items:
        if arr.shape[0] != num_nodes:
            return False

    label_items = _normalize_label_sets(labels, label_sets)
    if not label_items:
        return False
    label_info = []
    for name, vals in label_items:
        lbl = _to_numpy(vals)
        if lbl.ndim != 1 or lbl.shape[0] != num_nodes:
            return False
        label_info.append((name, lbl))

    max_nodes = max(1, int(max_nodes))
    if num_nodes > max_nodes:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(num_nodes, size=max_nodes, replace=False)
        emb_items = [(name, arr[sel]) for name, arr in emb_items]
        label_info = [(name, lbl[sel]) for name, lbl in label_info]

    try:
        from sklearn.manifold import TSNE
    except Exception:
        return False

    if pca_enabled:
        try:
            from sklearn.decomposition import PCA
        except Exception:
            return False
        pca_dim = int(pca_dim)
        if pca_dim >= 2:
            next_items = []
            for name, arr in emb_items:
                if arr.shape[1] > pca_dim:
                    pca = PCA(n_components=pca_dim, random_state=int(seed), svd_solver="auto")
                    arr = pca.fit_transform(arr)
                next_items.append((name, arr))
            emb_items = next_items

    perplexity = float(perplexity)
    max_perp = max(5.0, (emb_items[0][1].shape[0] - 1) / 3.0)
    if perplexity >= max_perp:
        perplexity = max(5.0, max_perp)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": int(seed),
    }
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["max_iter"] = int(max_iter)
    else:
        tsne_kwargs["n_iter"] = int(max_iter)

    tsne_embs = []
    for name, arr in emb_items:
        tsne = TSNE(**tsne_kwargs)
        tsne_embs.append((name, tsne.fit_transform(arr)))

    nrows = len(tsne_embs)
    ncols = len(label_info)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.8 * nrows), dpi=160)
    axes = np.array(axes)
    if axes.ndim == 1:
        if nrows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for row_idx, (emb_name, emb_xy) in enumerate(tsne_embs):
        for col_idx, (label_name, label_vals) in enumerate(label_info):
            ax = axes[row_idx, col_idx]
            kind, cmap, vmin, vmax = _label_style(label_vals, np, label_name=label_name)
            if kind == "continuous" and str(label_name).strip().lower() == "feature_pc1":
                cmap = "coolwarm"
            if kind == "categorical":
                scatter = ax.scatter(
                    emb_xy[:, 0],
                    emb_xy[:, 1],
                    c=label_vals,
                    s=8,
                    alpha=0.7,
                    cmap=cmap,
                )
            else:
                scatter = ax.scatter(
                    emb_xy[:, 0],
                    emb_xy[:, 1],
                    c=label_vals,
                    s=8,
                    alpha=0.7,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            if ncols == 1:
                ax.set_title(f"{emb_name} (t-SNE)")
            else:
                ax.set_title(f"{emb_name} - {label_name}")
            ax.set_xticks([])
            ax.set_yticks([])

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
