import datetime
import os
import time
import atexit
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from dataset_utils.load_dataset import load_dataset
from models.setup import setup_model
from utils.utils import set_seed, derive_split_seed
from .base_trainer import BaseAdaptationTrainer


@dataclass(frozen=True)
class PredictorTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 2500
    patience: int = 200


PREDICTOR_TRAIN_CFG = PredictorTrainConfig()

_NODE_LEVEL_TOTAL_EMBED_TIME_SEC: float = 0.0
_NODE_LEVEL_TOTAL_EMBED_CALLS: int = 0


def _record_embed_time(elapsed_sec: float) -> None:
    global _NODE_LEVEL_TOTAL_EMBED_TIME_SEC, _NODE_LEVEL_TOTAL_EMBED_CALLS
    _NODE_LEVEL_TOTAL_EMBED_TIME_SEC += float(elapsed_sec)
    _NODE_LEVEL_TOTAL_EMBED_CALLS += 1


def _print_total_embed_time_summary() -> None:
    if _NODE_LEVEL_TOTAL_EMBED_CALLS <= 0:
        return
    print(
        f"[Node Classification] Total embedding time across datasets: "
        f"{_NODE_LEVEL_TOTAL_EMBED_TIME_SEC:.2f}s "
        f"(calls={_NODE_LEVEL_TOTAL_EMBED_CALLS})"
    )


def save_node_level_embed_time_summary(results_dir: Optional[str]) -> Optional[str]:
    if not results_dir:
        return None
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "node_level_embed_time_summary.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"total_embed_time_sec: {_NODE_LEVEL_TOTAL_EMBED_TIME_SEC:.6f}\n")
        fh.write(f"embed_calls: {_NODE_LEVEL_TOTAL_EMBED_CALLS}\n")
    print(f"[Node Classification] Embedding time summary saved to {out_path}")
    return out_path


atexit.register(_print_total_embed_time_summary)


class PredictorTrainer:
    """Lightweight node-classification predictor trainer with early stopping on val accuracy."""

    def __init__(self, cfg: PredictorTrainConfig = PREDICTOR_TRAIN_CFG) -> None:
        self.cfg = cfg

    def _forward(self, model, graph: Data, features: torch.Tensor):
        # Keep flexible API: some predictors might want (graph, x), others only x.
        try:
            return model(graph, features)
        except TypeError:
            return model(features)

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
        if mask is None:
            return 0.0
        if mask.dtype == torch.bool:
            if not bool(mask.any()):
                return 0.0
            idx = mask
        else:
            if mask.numel() == 0:
                return 0.0
            idx = mask
        preds = logits.argmax(dim=1)
        correct = (preds[idx] == labels[idx]).float().mean()
        return float(correct.item())

    def train(
        self,
        predictor,
        features,
        graph: Data,
        desc: str = "Predictor",
        show_progress: bool = True,
    ) -> dict:
        labels = graph.y
        train_mask = getattr(graph, "train_mask", None)
        val_mask = getattr(graph, "val_mask", None)
        test_mask = getattr(graph, "test_mask", None)

        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        best_val = float("-inf")
        best_test = 0.0
        best_train = 0.0
        best_state = None
        no_improve = 0

        epoch_iter = tqdm(
            range(1, self.cfg.max_epochs + 1),
            desc=desc,
            leave=show_progress,
            disable=not show_progress,
        )
        for epoch in epoch_iter:
            predictor.train()

            logits_full = self._forward(predictor, graph, features)
            if train_mask is None:
                loss = F.cross_entropy(logits_full, labels)
            elif train_mask.dtype == torch.bool:
                loss = F.cross_entropy(logits_full[train_mask], labels[train_mask])
            else:
                loss = F.cross_entropy(logits_full[train_mask], labels[train_mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            predictor.eval()
            with torch.no_grad():
                logits_full = self._forward(predictor, graph, features)
                train_acc = self._accuracy(logits_full, labels, train_mask)
                val_acc = self._accuracy(logits_full, labels, val_mask)
                test_acc = self._accuracy(logits_full, labels, test_mask)

            if val_acc > best_val + 1e-12:
                best_val = val_acc
                best_test = test_acc
                best_train = train_acc
                best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if show_progress:
                epoch_iter.set_postfix(loss=float(loss.item()), val=float(val_acc), best_val=float(best_val))

            if self.cfg.patience is not None and no_improve >= self.cfg.patience:
                break

        if best_state is not None:
            predictor.load_state_dict(best_state)

        return {"train": best_train, "val": best_val, "test": best_test}


def _compute_mean_std(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _aggregate_split_results(split_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key in ("train", "val", "test"):
        values = [float(r[key]) for r in split_results if isinstance(r.get(key), (int, float))]
        agg = _compute_mean_std(values)
        if agg is not None:
            stats[key] = agg
    return stats


class NodeLevelAdaptationTrainer(BaseAdaptationTrainer):
    """Task-specific trainer for Stage 2 node-classification adaptation."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.trainer = PredictorTrainer()

    def per_dataset_adaptation(
        self,
        graph: Data,
        args,
        encoder=None,
        embeddings: Optional[torch.Tensor] = None,
        dataset_name: Optional[str] = None,
        results_dir: Optional[str] = None,
        show_progress: bool = True,
        log_result: bool = True,
    ):
        # PyG: node features in graph.x, labels in graph.y
        features = graph.x
        labels = graph.y
        out_dim = int(labels.max().item()) + 1
        in_dim = features.size(1)

        h = args.enc_num_layers
        if embeddings is None:
            if encoder is None:
                encoder_bundle = self._load_encoder_bundle()
            else:
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()
                encoder_bundle = {"attr": encoder, "primary": encoder}

            t0 = time.perf_counter()
            embeddings = self._build_embeddings(graph, features, encoder_bundle)
            _record_embed_time(time.perf_counter() - t0)
        else:
            embeddings = embeddings.detach()
            embeddings.requires_grad_(False)
            if embeddings.device != args.device:
                embeddings = embeddings.to(args.device)

        predictor_input = embeddings
        embedding_dim = embeddings.size(1)

        predictor = setup_model(
            in_dim=embedding_dim,
            out_dim=out_dim,
            args=args,
            purpose="pred",
        ).to(args.device)

        result = self.trainer.train(
            predictor=predictor,
            features=predictor_input,
            graph=graph,
            desc=f"Adapt {args.arch_type} (h={h})",
            show_progress=show_progress,
        )

        if log_result:
            print(
                f"h = {h:<6} | Train: {result['train']:<10.4f} | "
                f"Val: {result['val']*100:<4.2f}% | Test: {result['test']*100:<4.2f}%"
            )
        del embeddings
        if predictor is not None:
            del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "train": result.get("train", 0.0),
            "val": result.get("val", 0.0),
            "test": result.get("test", 0.0),
            "h": h,
        }

def evaluate_node_level_stage(dataset_name, args, results_dir: Optional[str] = None):
    print(f"\n=== Evaluating Dataset: {dataset_name} ===")
    device = args.device
    tuner = NodeLevelAdaptationTrainer(args)

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

    print(f"[Node Classification] Using Stage 2 representation: {getattr(args, 'stage2_representation', 'enc')}")
    print(f"[Node Classification] Split source: {split_source} (evaluating {num_runs} split(s))")

    if hasattr(base_graph, "x") and base_graph.x is not None:
        base_graph.x = base_graph.x.to(torch.float32)
    base_graph = base_graph.to(device)

    set_seed(int(args.seed))
    encoder_bundle = tuner._load_encoder_bundle()
    t0 = time.perf_counter()
    base_embeddings = tuner._build_embeddings(base_graph, base_graph.x, encoder_bundle)
    _record_embed_time(time.perf_counter() - t0)
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
        print(f"[Node Classification] Split {run_idx}/{num_runs}: index={split_index}")

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

        adapt_result = tuner.per_dataset_adaptation(
            graph=graph,
            args=args,
            dataset_name=dataset_name,
            results_dir=results_dir,
            embeddings=base_embeddings,
        )

        adapt_result["split_index"] = split_index
        split_results.append(adapt_result)

        del graph

    stats = _aggregate_split_results(split_results)
    best_agg_h = split_results[0].get("h") if split_results else None
    best_agg_train = stats.get("train", {}).get("mean", 0.0)
    best_agg_val = stats.get("val", {}).get("mean", 0.0)
    best_agg_test = stats.get("test", {}).get("mean", 0.0)
    best_agg_train_std = stats.get("train", {}).get("std", 0.0)
    best_agg_val_std = stats.get("val", {}).get("std", 0.0)
    best_agg_test_std = stats.get("test", {}).get("std", 0.0)

    model_name = args.arch_type.upper()
    print_node_level_results_summary(
        model_name,
        dataset_name,
        split_results,
        best_agg_h,
        results_dir,
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


def evaluate_node_level_stage_with_embeddings(
    dataset_name,
    args,
    embeddings: torch.Tensor,
    results_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    splits_dir: Optional[str] = None,
):
    print(f"\n=== Evaluating Dataset: {dataset_name} ===")
    device = args.device
    tuner = NodeLevelAdaptationTrainer(args)

    desired_splits = max(1, int(getattr(args, "stage2_splits", 5)))
    shared_random_feature_dim = int(getattr(args, "adapt_shared_random_feature_dim", 0) or 0)
    base_graph = load_dataset(
        dataset_name,
        split_index=0,
        seed=args.seed,
        cache_dir=cache_dir or "../dataset",
        splits_dir=splits_dir or "splits",
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

    print(f"[Node Classification] Using external embeddings for evaluation")
    print(f"[Node Classification] Split source: {split_source} (evaluating {num_runs} split(s))")

    if hasattr(base_graph, "x") and base_graph.x is not None:
        base_graph.x = base_graph.x.to(torch.float32)

    if embeddings is None:
        raise ValueError("embeddings must be provided for external evaluation")
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be 2D [num_nodes, dim]")
    if embeddings.size(0) != base_graph.num_nodes:
        raise ValueError(
            f"embeddings num_nodes ({embeddings.size(0)}) "
            f"!= graph num_nodes ({base_graph.num_nodes})"
        )
    embeddings = embeddings.detach()
    embeddings.requires_grad_(False)
    if embeddings.device != device:
        embeddings = embeddings.to(device)

    del base_graph

    split_results: List[Dict[str, float]] = []
    for run_idx, split_index in enumerate(split_indices, start=1):
        run_seed = derive_split_seed(args.seed, run_idx - 1)
        set_seed(run_seed)
        print(f"[Node Classification] Split {run_idx}/{num_runs}: index={split_index}")

        graph: Data = load_dataset(
            dataset_name,
            split_index=split_index,
            seed=args.seed,
            cache_dir=cache_dir or "../dataset",
            splits_dir=splits_dir or "splits",
            shared_random_feature_dim=shared_random_feature_dim,
        )

        if hasattr(graph, "x") and graph.x is not None:
            graph.x = graph.x.to(torch.float32)

        graph = graph.to(device)

        if torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda":
            print(f"Memory usage before tuning: {torch.cuda.memory_allocated(args.device) / 1e6:.2f} MB")

        adapt_result = tuner.per_dataset_adaptation(
            graph=graph,
            args=args,
            dataset_name=dataset_name,
            results_dir=results_dir,
            embeddings=embeddings,
        )

        adapt_result["split_index"] = split_index
        split_results.append(adapt_result)

        del graph

    stats = _aggregate_split_results(split_results)
    best_agg_h = split_results[0].get("h") if split_results else None
    best_agg_train = stats.get("train", {}).get("mean", 0.0)
    best_agg_val = stats.get("val", {}).get("mean", 0.0)
    best_agg_test = stats.get("test", {}).get("mean", 0.0)
    best_agg_train_std = stats.get("train", {}).get("std", 0.0)
    best_agg_val_std = stats.get("val", {}).get("std", 0.0)
    best_agg_test_std = stats.get("test", {}).get("std", 0.0)

    model_name = getattr(args, "encoder", "MODEL").upper()
    print_node_level_results_summary(
        model_name,
        dataset_name,
        split_results,
        best_agg_h,
        results_dir,
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


def _format_args(args: Optional[Any]) -> List[str]:
    if args is None:
        return []
    if isinstance(args, dict):
        items = args.items()
    else:
        items = vars(args).items() if hasattr(args, "__dict__") else []
    lines = ["ARGUMENTS"]
    for key, value in sorted(items, key=lambda kv: str(kv[0])):
        if isinstance(value, (list, tuple)):
            value_str = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            value_str = str(value)
        lines.append(f"{key}: {value_str}")
    return lines


def print_node_level_results_summary(
    model_name,
    dataset_name,
    split_results: List[Dict[str, float]],
    best_agg_h,
    results_dir: Optional[str] = None,
    args: Optional[Any] = None,
    split_source: Optional[str] = None,
    split_count: Optional[int] = None,
):
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{dataset_name}_detailed.txt"

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"DETAILED RESULTS SUMMARY FOR {dataset_name.upper()}")
    output_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args_lines = _format_args(args)
    if args_lines:
        output_lines.extend(args_lines)
    output_lines.append("=" * 80)

    output_lines.append("\nADAPTATION RESULTS")
    output_lines.append("=" * 60)

    if split_source:
        output_lines.append(f"Split source: {split_source}")
    if split_count is not None:
        output_lines.append(f"Available splits: {split_count}")
    output_lines.append(f"Evaluated splits: {len(split_results)}")

    if split_results and best_agg_h is not None:
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
            f"{model_name} (h={best_agg_h}): "
            f"Train {train_stats['mean']:.4f} +/- {train_stats['std']:.4f} | "
            f"Val {val_stats['mean']*100:6.2f}% +/- {val_stats['std']*100:6.2f}% | "
            f"Test {test_stats['mean']*100:6.2f}% +/- {test_stats['std']*100:6.2f}%"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Detailed results saved to {filename}")


__all__ = [
    "NodeLevelAdaptationTrainer",
    "evaluate_node_level_stage",
    "evaluate_node_level_stage_with_embeddings",
    "print_node_level_results_summary",
    "save_node_level_embed_time_summary",
    "PredictorTrainer",
]
