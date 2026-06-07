import datetime
import os
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from torch_geometric.data import Data

from dataset_utils.load_dataset import load_dataset
from utils.utils import set_seed, derive_split_seed
from .node_level import _aggregate_split_results, _format_args
from .base_trainer import BaseAdaptationTrainer


def _to_idx(mask: Optional[torch.Tensor], num_nodes: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.arange(num_nodes, device=device)
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).view(-1)
    return mask


def _load_encoder_bundle(args):
    trainer = BaseAdaptationTrainer(args)
    return trainer._load_encoder_bundle()


def _build_embeddings(args, graph: Data, features: torch.Tensor, encoder_bundle: dict) -> torch.Tensor:
    trainer = BaseAdaptationTrainer(args)
    return trainer._build_embeddings(graph, features, encoder_bundle)




def _run_tabpfn(
    embeddings: torch.Tensor,
    graph: Data,
    args,
) -> Dict[str, float]:
    train_mask = getattr(graph, "train_mask", None)
    val_mask = getattr(graph, "val_mask", None)
    test_mask = getattr(graph, "test_mask", None)

    labels = graph.y
    num_nodes = int(graph.num_nodes)
    train_idx = _to_idx(train_mask, num_nodes, embeddings.device)
    val_idx = _to_idx(val_mask, num_nodes, embeddings.device)
    test_idx = _to_idx(test_mask, num_nodes, embeddings.device)

    X_train = embeddings[train_idx].detach().cpu().numpy()
    y_train = labels[train_idx].detach().cpu().numpy()
    X_test = embeddings[test_idx].detach().cpu().numpy()
    y_test = labels[test_idx].detach().cpu().numpy()

    unique_labels = np.unique(y_train)
    n_classes = int(unique_labels.size)

    try:
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.many_class import ManyClassClassifier
    except Exception as exc:
        raise ImportError("tabpfn_extensions is required for TabPFN evaluation.") from exc

    base_clf = TabPFNClassifier(device=str(args.device))
    if n_classes > 10:
        clf = ManyClassClassifier(
            estimator=base_clf,
            alphabet_size=10,
            n_estimators_redundancy=4,
            random_state=int(getattr(args, "seed", 42)),
        )
    else:
        clf = base_clf

    clf.fit(X_train, y_train)
    classes = getattr(clf, "classes_", unique_labels)
    test_probs = clf.predict_proba(X_test) if len(X_test) > 0 else np.empty((0, len(classes)))
    test_idx_pred = test_probs.argmax(axis=1) if test_probs.size > 0 else np.empty((0,), dtype=int)
    test_preds = classes[test_idx_pred] if test_idx_pred.size > 0 else np.empty((0,), dtype=classes.dtype)

    train_acc = 0.0
    val_acc = 0.0
    test_acc = float((test_preds == y_test).sum()) / len(y_test) if len(y_test) > 0 else 0.0

    return {"train": train_acc, "val": val_acc, "test": test_acc}


def evaluate_node_level_tabpfn_stage(dataset_name, args, results_dir: Optional[str] = None):
    print(f"\n=== TabPFN Evaluating Dataset: {dataset_name} ===")
    device = args.device
    desired_splits = max(1, int(getattr(args, "stage2_splits", 5)))
    shared_random_feature_dim = int(getattr(args, "adapt_shared_random_feature_dim", 0) or 0)
    base_graph = load_dataset(
        dataset_name,
        split_index=0,
        seed=args.seed,
        shared_random_feature_dim=shared_random_feature_dim,
    )
    split_source = getattr(base_graph, "split_source", "random")
    split_count = int(getattr(base_graph, "split_count", 1) or 1)

    if split_source == "public" and split_count > 1:
        num_runs = min(desired_splits, split_count)
        split_indices = list(range(num_runs))
    elif split_source == "public":
        num_runs = desired_splits
        split_indices = [0] * num_runs
    else:
        num_runs = desired_splits
        split_indices = list(range(num_runs))

    print(f"[TabPFN] Using Stage 2 representation: {getattr(args, 'stage2_representation', 'enc')}")
    print(f"[TabPFN] Split source: {split_source} (evaluating {num_runs} split(s))")

    if hasattr(base_graph, "x") and base_graph.x is not None:
        base_graph.x = base_graph.x.to(torch.float32)
    base_graph = base_graph.to(device)

    set_seed(int(args.seed))
    encoder_bundle = _load_encoder_bundle(args)
    base_embeddings = _build_embeddings(args, base_graph, base_graph.x, encoder_bundle)
    base_embeddings = base_embeddings.detach()
    base_embeddings.requires_grad_(False)
    del encoder_bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del base_graph

    split_results: List[Dict[str, float]] = []
    for run_idx, split_index in enumerate(split_indices, start=1):
        run_seed = derive_split_seed(args.seed, run_idx - 1)
        set_seed(run_seed)
        print(f"[TabPFN] Split {run_idx}/{num_runs}: index={split_index}")

        graph: Data = load_dataset(
            dataset_name,
            split_index=split_index,
            seed=args.seed,
            shared_random_feature_dim=shared_random_feature_dim,
        )

        if hasattr(graph, "x") and graph.x is not None:
            graph.x = graph.x.to(torch.float32)

        graph = graph.to(device)

        if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda":
            print(f"Memory usage before tuning: {torch.cuda.memory_allocated(args.device) / 1e6:.2f} MB")

        adapt_result = _run_tabpfn(
            embeddings=base_embeddings,
            graph=graph,
            args=args,
        )

        adapt_result["split_index"] = split_index
        split_results.append(adapt_result)

        del graph

    stats = _aggregate_split_results(split_results)
    best_agg_h = "tabpfn"
    best_agg_train = stats.get("train", {}).get("mean", 0.0)
    best_agg_val = stats.get("val", {}).get("mean", 0.0)
    best_agg_test = stats.get("test", {}).get("mean", 0.0)
    best_agg_train_std = stats.get("train", {}).get("std", 0.0)
    best_agg_val_std = stats.get("val", {}).get("std", 0.0)
    best_agg_test_std = stats.get("test", {}).get("std", 0.0)

    print_node_level_tabpfn_summary(
        dataset_name=dataset_name,
        split_results=split_results,
        results_dir=results_dir,
        args=args,
        split_source=split_source,
        split_count=split_count,
    )

    return {
        "Agg_best_train": best_agg_train,
        "Agg_best_val": best_agg_val,
        "Agg_best_test": best_agg_test,
        "Agg_best_train_std": best_agg_train_std,
        "Agg_best_val_std": best_agg_val_std,
        "Agg_best_test_std": best_agg_test_std,
        "best_agg_h": best_agg_h,
        "num_splits": len(split_results),
        "split_source": split_source,
    }


def print_node_level_tabpfn_summary(
    dataset_name: str,
    split_results: List[Dict[str, float]],
    results_dir: Optional[str],
    args: Optional[Any],
    split_source: Optional[str],
    split_count: Optional[int],
):
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{dataset_name}_tabpfn_summary.txt"

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"TABPFN RESULTS SUMMARY FOR {dataset_name.upper()}")
    output_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args_lines = _format_args(args)
    if args_lines:
        output_lines.extend(args_lines)
    output_lines.append("=" * 80)
    if split_source:
        output_lines.append(f"Split source: {split_source}")
    if split_count is not None:
        output_lines.append(f"Available splits: {split_count}")
    output_lines.append(f"Evaluated splits: {len(split_results)}")
    output_lines.append("")

    if split_results:
        output_lines.append("Per-split evaluation:")
        for idx, result in enumerate(split_results, start=1):
            split_index = result.get("split_index", 0)
            train = result.get("train", 0.0)
            val = result.get("val", 0.0)
            test = result.get("test", 0.0)
            output_lines.append(
                f"  Split {idx} (index={split_index}): "
                f"Train {train:.4f} | Val {val*100:6.2f}% | Test {test*100:6.2f}%"
            )

        stats = _aggregate_split_results(split_results)
        train_stats = stats.get("train", {"mean": 0.0, "std": 0.0})
        val_stats = stats.get("val", {"mean": 0.0, "std": 0.0})
        test_stats = stats.get("test", {"mean": 0.0, "std": 0.0})
        output_lines.append("Aggregate (mean +/- std):")
        output_lines.append(
            f"TABPFN: Train {train_stats['mean']:.4f} +/- {train_stats['std']:.4f} | "
            f"Val {val_stats['mean']*100:6.2f}% +/- {val_stats['std']*100:6.2f}% | "
            f"Test {test_stats['mean']*100:6.2f}% +/- {test_stats['std']*100:6.2f}%"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"TabPFN summary saved to {filename}")


__all__ = [
    "evaluate_node_level_tabpfn_stage",
    "print_node_level_tabpfn_summary",
]
