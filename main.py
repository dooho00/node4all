import argparse
import datetime
import gc
import os
import warnings

import torch

from adaptation.stage2 import (
    evaluate_node_level_stage,
    evaluate_node_level_one_shot_stage,
    evaluate_node_level_tabpfn_stage,
)
from adaptation.node_level import save_node_level_embed_time_summary
from dataset_utils.dataset_registry import get_node_level_dataset_list
from channel_models.setup import normalize_arch_type
from pretrain.stage1 import pretrain
from utils.plot_utils import PLOT_CFG
from utils.summary import print_final_summary, save_stage2_overall_summary
from utils.utils import set_seed
from utils.wandb_utils import WandbConfig, WandbLogger
from visualize_mode import run_visualize_mode


def clear_runtime_caches():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def setup_wandb(args, device):
    exp_name = (
        f"{args.mode}_{args.datasetA}_"
        f"{args.arch_type}_enc_layers_{args.enc_num_layers}_seed_{args.seed}"
    )
    tags = ["Node4All", args.arch_type, args.mode, args.datasetA] + args.wandb_tags
    wandb_config = WandbConfig(
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        experiment_name=exp_name,
        tags=tags,
        notes=args.wandb_notes,
    )

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_ord = None
    if torch.cuda.is_available():
        try:
            cuda_ord = torch.cuda.current_device()
        except Exception:
            cuda_ord = None

    hyperparameters = {
        "mode": args.mode,
        "datasetA": args.datasetA,
        "source_dataset": args.datasetA,
        "seed": args.seed,
        "arch_type": args.arch_type,
        "enc_num_layers": args.enc_num_layers,
        "dec_num_layers": args.dec_num_layers,
        "cgt_hops": args.cgt_hops,
        "cgt_filters": args.cgt_filters,
        "cgt_norms": args.cgt_norms,
        "stage2_splits": args.stage2_splits,
        "predictor_type": args.predictor_type,
        "device": str(device),
        "cuda_visible_devices": cuda_visible,
        "cuda_device_ordinal": int(cuda_ord) if cuda_ord is not None else None,
    }
    if getattr(args, "hp_idx", None) is not None:
        hyperparameters["hp_idx"] = args.hp_idx

    logger = WandbLogger(wandb_config, hyperparameters)
    return logger if logger.init_wandb() else None


def parse_args():
    parser = argparse.ArgumentParser(description="Node4All pretraining and frozen adaptation pipeline")
    add = parser.add_argument
    add("--seed", type=int, default=42, help="Random seed for reproducibility")
    add("--gpu", type=int, default=7, help="GPU device index to use (e.g., 0). Use -1 for CPU only.")
    add("--mode", choices=["pretrain", "adaptation", "both", "visualize"], default="both", help="Which stage to run")
    add("--datasetA", default="node4all", help="Source dataset for Stage 1 pretraining")
    add(
        "--node4all_cfg_json",
        type=str,
        default=None,
        help="JSON string of Node4All config overrides (merged into defaults).",
    )
    add("--enc_checkpoint", type=str, default="checkpoints/cgt_enc.pth", help="Path to save/load the shared encoder checkpoint")

    add("--enc_num_layers", type=int, default=10, help="Number of layers in the encoder")
    add("--embed_dim", type=int, default=8, help="Scalar embedding dimension for encoder backbones")
    add("--dec_num_layers", type=int, default=8, help="Number of layers in the decoder")
    add("--arch_type", type=str, default="cgt", help="Shared encoder/decoder backbone")
    add("--predictor_type", type=str, default="mlp", help="Downstream predictor type for Stage 2 adaptation")

    add("--epochs", type=int, default=500, help="Number of epochs for masked-autoencoder pretraining")
    add("--ssl_lr", type=float, default=1e-3, help="Learning rate for masked-autoencoder pretraining")
    add("--ssl_weight_decay", type=float, default=0.0, help="Weight decay for SSL optimizer")
    add(
        "--ssl_latent_mask_rate",
        type=float,
        default=0.75,
        help="DropNode mask rate applied between encoder and decoder (0 disables).",
    )
    add(
        "--ssl_latent_mask_mode",
        type=str,
        choices=["node"],
        default="node",
        help="DropNode masking mode between encoder and decoder (node only).",
    )
    
    add("--ssl_alpha_l", type=float, default=1.0, help="SCE loss exponent for masked reconstruction")
    add(
        "--enc_feature_budget",
        type=int,
        default=2000_000,
        help="Feature batching budget (num_nodes * feature_chunk) for pretraining memory control",
    )

    add("--hidden_dim", type=int, default=512, help="Hidden dimension used inside SSL encoder/decoder")
    add("--nhead", type=int, default=1, help="Number of attention heads for GAT models")
    add("--feat_drop", type=float, default=0.5, help="Feature dropout used inside SSL encoder")
    add("--attn_drop", type=float, default=0.2, help="Attention dropout used inside SSL encoder")
    add("--negative_slope", type=float, default=0.2, help="LeakyReLU negative slope for SSL attention modules")
    add(
        "--activation",
        type=str,
        default="gelu",
        choices=["relu", "gelu", "prelu", "elu", "none"],
        help="Activation function used in SSL encoder/decoder",
    )
    add("--residual", action="store_true", help="Enable residual connections inside SSL layers")
    add(
        "--norm",
        type=str,
        default="none",
        choices=["none", "layernorm", "batchnorm", "graphnorm"],
        help="Normalization layer applied inside SSL encoder",
    )
    add("--graphsaint_batch_size", type=int, default=1, help="Mini-batch node budget for GraphSAINT random walk sampler")
    add("--graphsaint_walk_length", type=int, default=32, help="Random walk length for GraphSAINT sampler")
    add("--graphsaint_num_steps", type=int, default=1, help="Number of sampling steps per epoch for GraphSAINT")
    add("--graphsaint_sample_coverage", type=float, default=100, help="Optional sample coverage target for GraphSAINT sampler")

    add("--wandb", action="store_true", help="Enable Weights & Biases logging")
    add("--wandb_project", type=str, default="CGT", help="WandB project name")
    add("--wandb_entity", type=str, default=None, help="WandB entity name")
    add("--wandb_tags", type=str, nargs="*", default=[], help="WandB tags for the experiment")
    add("--wandb_notes", type=str, default=None, help="Notes for the experiment")
    add("--hp_idx", type=str, default=None, help="Stable hyperparameter index (logged as config, not a tag)")
    add(
        "--node_level_datasets",
        "--node_cls_datasets",
        dest="node_level_datasets",
        type=str,
        nargs="*",
        default=["all"],
        help="Datasets evaluated for Stage 2 node-classification tasks. Use 'all' for every dataset.",
    )
    add(
        "--node_level_one_shot_datasets",
        dest="node_level_one_shot_datasets",
        type=str,
        nargs="*",
        default=[""],
        help="Datasets evaluated for one-shot node-classification tasks. Use 'all' for every dataset, leave empty to skip.",
    )
    add(
        "--node_level_tabpfn_datasets",
        dest="node_level_tabpfn_datasets",
        type=str,
        nargs="*",
        default=[""],
        help="Datasets evaluated for TabPFN node-classification tasks. Use 'all' for every dataset, leave empty to skip.",
    )
    add(
        "--stage2_representation",
        type=str,
        choices=["enc"],
        default="enc",
        help="Choose which frozen encoder output to use during Stage 2 evaluation and downstream heads",
    )
    add(
        "--stage2_splits",
        type=int,
        default=10,
        help="Number of evaluation splits for Stage 2 (uses public splits when available).",
    )
    add(
        "--results_dir",
        type=str,
        default=None,
        help="Optional base directory for Stage 2 results (defaults to timestamped folder).",
    )
    add(
        "--adapt_shared_random_feature_dim",
        type=int,
        default=0,
        help="Append a shared random feature vector of this size during Stage 2 node-classification adaptation (0 disables).",
    )

    add("--cgt_hops", type=int, default=8, help="Number of CGT multi-hop token depths (includes 0-hop identity).")
    add("--cgt_filters", type=int, default=1, help="Number of CGT filter normalization groups.")
    add("--cgt_norms", type=str, nargs="*", default="sym", help="CGT filter normalization modes: sym, rw, none, or all.")
    add(
        "--cgt_use_self_loops",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Use self-loops during CGT multi-hop tokenization.",
    )

    return parser.parse_args()


def resolve_datasets(selection, available):
    if not selection:
        return []
    selection = [s for s in selection if s]
    if not selection:
        return []
    if "all" in {s.lower() for s in selection}:
        return available
    seen = set()
    resolved = []
    for name in selection:
        if name not in available:
            print(f"[WARN] Dataset '{name}' not found in registry; skipping.")
            continue
        if name not in seen:
            resolved.append(name)
            seen.add(name)
    return resolved


def run_stage(datasets, eval_fn, results_dir, args, wandb_logger, tag_suffix=""):
    if not datasets:
        return {}
    os.makedirs(results_dir, exist_ok=True)
    summaries = {}
    for name in datasets:
        summary = eval_fn(name, args, results_dir)
        summaries[name] = summary
        if wandb_logger and summary:
            wandb_logger.log_stage2_results(f"{name}{tag_suffix}", summary)
    return summaries


def main():
    clear_runtime_caches()
    warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    args = parse_args()
    args.arch_type = normalize_arch_type(args.arch_type)
    is_visualize_mode = args.mode == "visualize"
    PLOT_CFG["tsne"]["enabled"] = is_visualize_mode
    PLOT_CFG["token_noise"]["enabled"] = is_visualize_mode

    if torch.cuda.is_available() and args.gpu is not None and args.gpu >= 0:
        gpu_idx = args.gpu
        if gpu_idx >= torch.cuda.device_count():
            warnings.warn(
                f"Requested GPU index {gpu_idx}, but only {torch.cuda.device_count()} CUDA devices are available. "
                "Falling back to GPU 0."
            )
            gpu_idx = 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    args.device = device
    set_seed(args.seed)
    
    wandb_logger = setup_wandb(args, device) if args.wandb else None

    if args.mode == "visualize":
        run_visualize_mode(args)

    if args.mode in {"pretrain", "both"}:
        print("=" * 60)
        print("STAGE 1: Pretraining Node4All encoder on the source graph distribution")
        print("=" * 60)
        best_val, best_test = pretrain(args)
        if wandb_logger:
            wandb_logger.log_metrics(
                {
                    "stage1/best_val_acc": best_val,
                    "stage1/best_test_acc": best_test,
                    "stage1/dataset": args.datasetA,
                }
            )

    if args.mode in {"adaptation", "both"}:
        print("\n" + "=" * 60)
        print("STAGE 2: Frozen adaptation and evaluation on target datasets")
        print("=" * 60)

        if args.results_dir:
            results_dir = args.results_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"results/{timestamp}_enc_layers_{args.enc_num_layers}"
        os.makedirs(results_dir, exist_ok=True)

        node_datasets = resolve_datasets(args.node_level_datasets, get_node_level_dataset_list())
        one_shot_datasets = resolve_datasets(args.node_level_one_shot_datasets, get_node_level_dataset_list())
        tabpfn_datasets = resolve_datasets(args.node_level_tabpfn_datasets, get_node_level_dataset_list())
        node_dir = os.path.join(results_dir, "node_level") if node_datasets else None
        one_shot_dir = os.path.join(results_dir, "node_level_one_shot") if one_shot_datasets else None
        tabpfn_dir = os.path.join(results_dir, "node_level_tabpfn") if tabpfn_datasets else None

        node_summaries = run_stage(node_datasets, evaluate_node_level_stage, node_dir, args, wandb_logger)
        if node_datasets:
            if node_summaries:
                print_final_summary(node_summaries, node_datasets, node_dir)
            else:
                print("Node classification task: No datasets were successfully processed.")
        else:
            print("Node classification task: No datasets selected.")

        one_shot_summaries = run_stage(one_shot_datasets, evaluate_node_level_one_shot_stage, one_shot_dir, args, wandb_logger, tag_suffix="_one_shot")
        if one_shot_datasets:
            if one_shot_summaries:
                print_final_summary(one_shot_summaries, one_shot_datasets, one_shot_dir)
            else:
                print("One-shot node classification task: No datasets were successfully processed.")
        else:
            print("One-shot node classification task: No datasets selected.")

        tabpfn_summaries = run_stage(tabpfn_datasets, evaluate_node_level_tabpfn_stage, tabpfn_dir, args, wandb_logger, tag_suffix="_tabpfn")
        if tabpfn_datasets:
            if tabpfn_summaries:
                print_final_summary(tabpfn_summaries, tabpfn_datasets, tabpfn_dir)
            else:
                print("TabPFN node classification task: No datasets were successfully processed.")
        else:
            print("TabPFN node classification task: No datasets selected.")

        save_stage2_overall_summary(
            results_dir=results_dir,
            node_results=node_summaries,
        )
        save_node_level_embed_time_summary(results_dir)

    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
