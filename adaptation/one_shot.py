import datetime
import os
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from dataset_utils.load_dataset import load_dataset
from utils.utils import set_seed, derive_split_seed
from .node_level import (
    NodeLevelAdaptationTrainer,
    _aggregate_split_results,
    _format_args,
)

# One-shot is intentionally fixed to ridge + TSVD.
ONE_SHOT_CFG = {
    "predictor_type": "ridge",
    "dim_reduction": "tsvd",
    "ridge_alpha": 10.0,
    "seed_count": 5,
    "samples_per_seed": 100,
    "test_samples": 100,
    "tsvd_dim": 16,
}


def _build_mask(num_nodes: int, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if indices.size > 0:
        mask[torch.as_tensor(indices, device=device)] = True
    return mask


def _sample_one_shot_indices(
    labels_np: np.ndarray,
    unique_labels: np.ndarray,
    rng: np.random.RandomState,
) -> Optional[np.ndarray]:
    train_indices = []
    for label in unique_labels:
        candidates = np.where(labels_np == label)[0]
        if candidates.size == 0:
            return None
        train_indices.append(int(rng.choice(candidates, size=1)[0]))
    return np.asarray(train_indices, dtype=np.int64)


def _predict_with_ridge(
    features: torch.Tensor,
    train_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> torch.Tensor:
    if train_mask is None or not bool(train_mask.any()):
        raise ValueError("Empty training set for one-shot ridge.")
    y_one_hot = torch.zeros((train_labels.size(0), num_classes), dtype=features.dtype)
    y_one_hot.scatter_(1, train_labels.view(-1, 1).cpu(), 1.0)

    ref_nodes = train_mask.nonzero(as_tuple=False).view(-1).cpu()
    x_ref = features.detach().cpu()[ref_nodes]
    reg = float(max(alpha, 0.0)) ** 0.5
    eye = torch.eye(x_ref.size(1), dtype=x_ref.dtype, device=x_ref.device)
    x_aug = torch.cat([x_ref, reg * eye], dim=0)
    y_aug = torch.cat(
        [y_one_hot, torch.zeros((x_ref.size(1), num_classes), dtype=y_one_hot.dtype, device=y_one_hot.device)],
        dim=0,
    )
    w = torch.linalg.lstsq(x_aug, y_aug, driver="gelss")[0]
    preds = features.detach().cpu() @ w
    return preds


def _accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if mask is None or not bool(mask.any()):
        return 0.0
    preds = logits.argmax(dim=1)
    labels_cpu = labels.detach().cpu()
    mask_cpu = mask.detach().cpu()
    correct = (preds[mask_cpu] == labels_cpu[mask_cpu]).float().mean()
    return float(correct.item())


def evaluate_node_level_one_shot_stage(dataset_name, args, results_dir: Optional[str] = None):
    print(f"\n=== One-Shot Evaluating Dataset: {dataset_name} ===")
    device = args.device
    tuner = NodeLevelAdaptationTrainer(args)

    seed_count = max(1, int(ONE_SHOT_CFG["seed_count"]))
    samples_per_seed = max(1, int(ONE_SHOT_CFG["samples_per_seed"]))
    predictor_type = str(ONE_SHOT_CFG["predictor_type"])
    test_samples = int(ONE_SHOT_CFG["test_samples"])
    ridge_alpha = float(ONE_SHOT_CFG["ridge_alpha"])
    tsvd_dim = max(1, int(ONE_SHOT_CFG["tsvd_dim"]))

    shared_random_feature_dim = int(getattr(args, "adapt_shared_random_feature_dim", 0) or 0)
    base_graph = load_dataset(
        dataset_name,
        split_index=0,
        seed=args.seed,
        shared_random_feature_dim=shared_random_feature_dim,
    )

    if hasattr(base_graph, "x") and base_graph.x is not None:
        base_graph.x = base_graph.x.to(torch.float32)
    base_graph = base_graph.to(device)

    labels = base_graph.y
    if labels is None:
        raise ValueError(f"Dataset {dataset_name} has no labels; cannot run one-shot evaluation.")
    if labels.dim() > 1 and labels.size(1) == 1:
        labels = labels.view(-1)
        base_graph.y = labels

    labels_np = labels.detach().cpu().numpy()
    valid_mask = labels_np >= 0
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        raise ValueError(f"Dataset {dataset_name} has no valid labeled nodes.")
    labels_valid = labels_np[valid_mask]
    unique_labels = np.unique(labels_valid)

    print(
        f"[One-Shot] Seeds: {seed_count}, samples/seed: {samples_per_seed}, "
        f"total runs: {seed_count * samples_per_seed}"
    )
    print(f"[One-Shot] Predictor: {predictor_type}")
    print(f"[One-Shot] Ridge alpha: {ridge_alpha}")
    print(f"[One-Shot] Test samples: {test_samples}")
    print(f"[One-Shot] Using Stage 2 representation: {getattr(args, 'stage2_representation', 'enc')}")
    print(f"[One-Shot] Dim reduction: {ONE_SHOT_CFG['dim_reduction']} (dim={tsvd_dim})")

    set_seed(int(args.seed))
    encoder_bundle = tuner._load_encoder_bundle()
    base_embeddings = tuner._build_embeddings(base_graph, base_graph.x, encoder_bundle)
    base_embeddings = base_embeddings.detach()
    base_embeddings.requires_grad_(False)
    
    if base_embeddings.dim() != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(base_embeddings.shape)}")
    emb_np = base_embeddings.detach().cpu().numpy()
    if emb_np.shape[1] > tsvd_dim:
        reducer = TruncatedSVD(n_components=tsvd_dim)
        emb_np = reducer.fit_transform(emb_np)
    base_embeddings = torch.as_tensor(emb_np, device=device, dtype=base_embeddings.dtype)
    del encoder_bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_results: List[Dict[str, float]] = []
    seed_summaries: List[Dict[str, Any]] = []
    num_nodes = int(base_graph.num_nodes)

    label_to_class = {int(lbl): idx for idx, lbl in enumerate(unique_labels.tolist())}
    labels_tensor = base_graph.y
    labels_mapped_np = np.full_like(labels_np, -1)
    for lbl, idx in label_to_class.items():
        labels_mapped_np[labels_np == lbl] = idx
    labels_mapped = torch.as_tensor(labels_mapped_np, device=device, dtype=torch.long)

    for seed_idx in range(seed_count):
        base_seed = derive_split_seed(args.seed, seed_idx)
        seed_results: List[Dict[str, float]] = []
        sample_iter = tqdm(
            range(samples_per_seed),
            desc=f"[One-Shot] Seed {seed_idx + 1}/{seed_count}",
            leave=True,
        )
        for sample_idx in sample_iter:
            sample_seed = derive_split_seed(base_seed, sample_idx)
            rng = np.random.RandomState(sample_seed)

            train_idx_local = _sample_one_shot_indices(labels_valid, unique_labels, rng)
            if train_idx_local is None:
                continue
            train_idx = valid_indices[train_idx_local]

            test_count = min(test_samples, num_nodes)
            test_start = max(0, num_nodes - test_count)
            test_idx = np.arange(test_start, num_nodes, dtype=np.int64)

            train_mask = _build_mask(num_nodes, np.asarray(train_idx), device)
            test_mask = _build_mask(num_nodes, np.asarray(test_idx), device)

            train_labels = labels_tensor[train_mask]
            train_labels = torch.tensor(
                [label_to_class[int(lbl.item())] for lbl in train_labels],
                device=device,
                dtype=torch.long,
            )

            preds = _predict_with_ridge(
                features=base_embeddings,
                train_mask=train_mask,
                train_labels=train_labels,
                num_classes=len(unique_labels),
                alpha=ridge_alpha,
            )
            adapt_result = {
                "train": _accuracy_from_logits(preds, labels_mapped, train_mask),
                "test": _accuracy_from_logits(preds, labels_mapped, test_mask),
                "h": str(predictor_type),
            }
            adapt_result["seed"] = base_seed
            adapt_result["sample_seed"] = sample_seed
            seed_results.append(adapt_result)
            total_results.append(adapt_result)

        seed_stats = _aggregate_split_results(seed_results)
        seed_summaries.append(
            {
                "seed": base_seed,
                "num_runs": len(seed_results),
                "stats": seed_stats,
            }
        )

    stats = _aggregate_split_results(total_results)
    agg_train = stats.get("train", {}).get("mean", 0.0)
    agg_val = stats.get("val", {}).get("mean", 0.0)
    agg_test = stats.get("test", {}).get("mean", 0.0)
    agg_train_std = stats.get("train", {}).get("std", 0.0)
    agg_val_std = stats.get("val", {}).get("std", 0.0)
    agg_test_std = stats.get("test", {}).get("std", 0.0)

    print_node_level_one_shot_summary(
        dataset_name=dataset_name,
        results_dir=results_dir,
        args=args,
        seed_summaries=seed_summaries,
        total_results=total_results,
        predictor_type=str(predictor_type),
        test_ratio=None,
    )

    return {
        "Agg_best_train": agg_train,
        "Agg_best_val": agg_val,
        "Agg_best_test": agg_test,
        "Agg_best_train_std": agg_train_std,
        "Agg_best_val_std": agg_val_std,
        "Agg_best_test_std": agg_test_std,
        "best_agg_h": str(predictor_type),
        "num_splits": len(total_results),
        "split_source": "one_shot",
    }


def print_node_level_one_shot_summary(
    dataset_name: str,
    results_dir: Optional[str],
    args: Optional[Any],
    seed_summaries: List[Dict[str, Any]],
    total_results: List[Dict[str, float]],
    predictor_type: str,
    test_ratio: Optional[float],
):
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{dataset_name}_one_shot_summary.txt"

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"ONE-SHOT RESULTS SUMMARY FOR {dataset_name.upper()}")
    output_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args_lines = _format_args(args)
    if args_lines:
        output_lines.extend(args_lines)
    output_lines.append("=" * 80)
    output_lines.append(f"Predictor: {predictor_type}")
    if test_ratio is not None:
        output_lines.append(f"Test ratio: {test_ratio}")
    output_lines.append(f"Total runs: {len(total_results)}")
    output_lines.append("")

    if seed_summaries:
        output_lines.append("Per-seed aggregate (mean +/- std):")
        for entry in seed_summaries:
            seed = entry.get("seed")
            stats = entry.get("stats", {})
            test_stats = stats.get("test", {"mean": 0.0, "std": 0.0})
            output_lines.append(
                f"  Seed {seed}: Test {test_stats['mean']*100:6.2f}% +/- {test_stats['std']*100:6.2f}%"
            )

    stats = _aggregate_split_results(total_results)
    if stats:
        test_stats = stats.get("test", {"mean": 0.0, "std": 0.0})
        output_lines.append("")
        output_lines.append("Overall aggregate (mean +/- std):")
        output_lines.append(
            f"Test {test_stats['mean']*100:6.2f}% +/- {test_stats['std']*100:6.2f}%"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"One-shot summary saved to {filename}")


__all__ = [
    "evaluate_node_level_one_shot_stage",
    "print_node_level_one_shot_summary",
]
