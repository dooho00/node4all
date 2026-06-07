from __future__ import annotations

from typing import Optional

import torch

from .cfg import get_tsne_cfg, get_token_noise_cfg
from .labels import build_graph_label_sets
from .noise import save_feature_noise_pair_scatter
from .tsne import save_tsne_comparison


def visualize_graph_embeddings(
    *,
    graph,
    features: Optional[torch.Tensor],
    embeddings: Optional[torch.Tensor],
    out_base: Optional[str],
    tsne_title: str = "t-SNE: input vs encoder",
    token_noise_pre_title: str = "Token noise distribution (input)",
    token_noise_post_title: str = "Token noise distribution (encoder output)",
) -> None:
    if not out_base:
        return
    if features is None and embeddings is None:
        return

    base = str(out_base)
    tsne_cfg = get_tsne_cfg()
    if tsne_cfg["enabled"] and features is not None and embeddings is not None:
        label_sets = build_graph_label_sets(graph, features=features)
        if label_sets:
            save_tsne_comparison(
                out_path=base + "_tsne_compare.png",
                features=features,
                embeddings=embeddings,
                labels=label_sets.get("community", None),
                label_sets=label_sets,
                max_nodes=tsne_cfg["max_nodes"],
                max_iter=tsne_cfg["max_iter"],
                pca_enabled=tsne_cfg["pca_enabled"],
                pca_dim=tsne_cfg["pca_dim"],
                seed=tsne_cfg["seed"],
                perplexity=tsne_cfg["perplexity"],
                title=tsne_title,
            )

    token_cfg = get_token_noise_cfg()
    if not token_cfg["enabled"] or token_cfg["std"] <= 0.0 or features is None:
        return

    label_sets_pre = build_graph_label_sets(graph, features=features, include_community=False)
    if not label_sets_pre:
        return

    seed = token_cfg["seed"]
    max_features = token_cfg["max_features"]
    max_nodes = token_cfg["max_nodes"]
    gen = torch.Generator(device=features.device)
    gen.manual_seed(seed)
    noise = torch.randn(
        features.size(),
        device=features.device,
        dtype=features.dtype,
        generator=gen,
    ) * float(token_cfg["std"])
    if embeddings is None:
        return

    if embeddings.shape == noise.shape and embeddings.device == noise.device and embeddings.dtype == noise.dtype:
        post_noise = noise
    else:
        gen = torch.Generator(device=embeddings.device)
        gen.manual_seed(seed)
        post_noise = torch.randn(
            embeddings.size(),
            device=embeddings.device,
            dtype=embeddings.dtype,
            generator=gen,
        ) * float(token_cfg["std"])
    label_sets = {"feature": features, **label_sets_pre}
    save_feature_noise_pair_scatter(
        out_path=base + "_token_noise.png",
        values_pre=features,
        noise_pre=noise,
        values_post=embeddings,
        noise_post=post_noise,
        labels=None,
        label_sets=label_sets,
        max_features=max_features,
        max_nodes=max_nodes,
        seed=seed,
        title="",
    )
