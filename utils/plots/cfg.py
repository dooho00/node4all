from __future__ import annotations

from typing import Optional, Dict


PLOT_CFG = {
    "plot_dir": "plots",
    "color_quantile": (0.03, 0.97),
    "tsne": {
        "enabled": True,
        "max_nodes": 2000,
        "max_iter": 3000,
        "seed": 42,
        "perplexity": 50.0,
        "pca": {
            "enabled": False,
            "dim": 50,
        },
    },
    "token_noise": {
        "enabled": True,
        "std": 0.1,
        "seed": 42,
        "max_features": 4,
        "max_nodes": 200,
    },
}


def get_plot_dir() -> Optional[str]:
    plot_dir = PLOT_CFG.get("plot_dir", None)
    if plot_dir:
        return str(plot_dir)
    return None


def get_tsne_cfg() -> Dict[str, object]:
    cfg = PLOT_CFG.get("tsne", {})
    pca_cfg = cfg.get("pca", {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "max_nodes": int(cfg.get("max_nodes", 5000)),
        "max_iter": int(cfg.get("max_iter", 5000)),
        "seed": int(cfg.get("seed", 42)),
        "perplexity": float(cfg.get("perplexity", 5.0)),
        "pca_enabled": bool(pca_cfg.get("enabled", False)),
        "pca_dim": int(pca_cfg.get("dim", 50)),
    }


def get_token_noise_cfg() -> Dict[str, object]:
    cfg = PLOT_CFG.get("token_noise", {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "std": float(cfg.get("std", 0.0)),
        "seed": int(cfg.get("seed", 42)),
        "max_features": int(cfg.get("max_features", 9)),
        "max_nodes": int(cfg.get("max_nodes", 200)),
    }
