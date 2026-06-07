import datetime
import json
import os

import torch

from adaptation.base_trainer import BaseAdaptationTrainer
from dataset_utils.dataset_registry import get_node_level_dataset_list
from dataset_utils.load_dataset import load_dataset
from utils.plot_utils import get_plot_dir, resolve_plot_base, visualize_graph_embeddings


def _resolve_datasets(selection, available):
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


def _build_node4all_cfg(args):
    cfg = {}
    cfg_json = getattr(args, "node4all_cfg_json", None)
    if not cfg_json:
        return cfg
    try:
        overrides = json.loads(cfg_json)
    except json.JSONDecodeError as exc:
        raise ValueError("node4all_cfg_json must be valid JSON") from exc
    if not isinstance(overrides, dict):
        raise ValueError("node4all_cfg_json must be a JSON object")
    cfg.update(overrides)
    return cfg


def _load_datasetA_graph_for_visualization(args):
    dataset_name = str(getattr(args, "datasetA", "")).strip()
    if dataset_name.lower() == "node4all":
        from dataset_utils.node4all import sample_node4all

        cfg = _build_node4all_cfg(args)
        cfg["seed"] = int(getattr(args, "seed", 42))
        return sample_node4all(cfg=cfg)

    return load_dataset(
        dataset_name,
        split_index=0,
        seed=args.seed,
        shared_random_feature_dim=int(getattr(args, "adapt_shared_random_feature_dim", 0) or 0),
    )


def _resolve_stage2_plot_dir(results_dir):
    plots_root = get_plot_dir()
    if not plots_root:
        return None
    if results_dir:
        base = os.path.basename(str(results_dir).rstrip(os.sep))
        if base in {"node_level"}:
            parent = os.path.basename(os.path.dirname(str(results_dir)))
            if parent:
                return os.path.join(plots_root, parent)
        return os.path.join(plots_root, base)
    return plots_root


def _save_stage2_visuals(*, graph, features, embeddings, dataset_name, results_dir, plot_tag=""):
    if features is None:
        return
    plot_dir = _resolve_stage2_plot_dir(results_dir)
    if not plot_dir:
        return
    safe_name = dataset_name or "dataset"
    base = os.path.join(plot_dir, f"{safe_name}{plot_tag}")
    visualize_graph_embeddings(
        graph=graph,
        features=features,
        embeddings=embeddings,
        out_base=base,
    )


def run_visualize_mode(args):
    print("=" * 60)
    print("VISUALIZE MODE: Generating embedding visualizations only")
    print("=" * 60)

    if not getattr(args, "enc_checkpoint", None):
        raise ValueError("--enc_checkpoint is required in visualize mode.")
    if not os.path.exists(args.enc_checkpoint):
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.enc_checkpoint}")

    plot_base = resolve_plot_base(args.enc_checkpoint)
    if not plot_base:
        raise ValueError("Could not resolve plot output base from enc_checkpoint.")

    trainer = BaseAdaptationTrainer(args)
    encoder_bundle = trainer._load_encoder_bundle()

    pretrain_graph = _load_datasetA_graph_for_visualization(args)
    if hasattr(pretrain_graph, "x") and pretrain_graph.x is not None:
        pretrain_graph.x = pretrain_graph.x.to(torch.float32)
    pretrain_graph = pretrain_graph.to(args.device)

    dataset_name = str(getattr(args, "datasetA", "")).lower()
    if dataset_name == "node4all":
        from dataset_utils.node4all import sample_node4all

        base_seed = int(getattr(args, "seed", 42))
        base_nodes = int(getattr(pretrain_graph, "num_nodes", 0) or 0)
        if base_nodes <= 0:
            size_candidates = [256, 1024, 4096]
        else:
            size_candidates = [
                max(32, int(base_nodes * 0.5)),
                max(32, int(base_nodes)),
                max(32, int(base_nodes * 2.0)),
            ]
        sizes = []
        for size in size_candidates:
            if size not in sizes:
                sizes.append(size)

        base_cfg = _build_node4all_cfg(args)
        for idx, size in enumerate(sizes):
            cfg = dict(base_cfg)
            cfg["num_nodes"] = int(size)
            cfg["seed"] = base_seed + 1000 * idx
            graph = sample_node4all(cfg=cfg).to(args.device)
            if getattr(graph, "x", None) is None:
                continue
            features = graph.x.to(torch.float32)
            embeddings = trainer._build_embeddings(graph, features, encoder_bundle)
            visualize_graph_embeddings(
                graph=graph,
                features=features,
                embeddings=embeddings,
                out_base=plot_base + f"_n{int(size)}",
            )
            print(f"[Visualize] Saved pretrain visuals for node4all size={int(size)}")
    else:
        features = getattr(pretrain_graph, "x", None)
        if features is not None:
            features = features.to(torch.float32)
            embeddings = trainer._build_embeddings(pretrain_graph, features, encoder_bundle)
            visualize_graph_embeddings(
                graph=pretrain_graph,
                features=features,
                embeddings=embeddings,
                out_base=plot_base,
            )
            print(f"[Visualize] Saved pretrain visuals for datasetA={args.datasetA}")
        else:
            print(f"[Visualize] Dataset {args.datasetA} has no features; skipped pretrain-style visualization.")

    if args.results_dir:
        results_dir = args.results_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/{timestamp}_enc_layers_{args.enc_num_layers}"
    os.makedirs(results_dir, exist_ok=True)

    node_datasets = _resolve_datasets(args.node_level_datasets, get_node_level_dataset_list())
    node_dir = os.path.join(results_dir, "node_level") if node_datasets else None

    shared_random_feature_dim = int(getattr(args, "adapt_shared_random_feature_dim", 0) or 0)

    for name in node_datasets:
        graph = load_dataset(
            name,
            split_index=0,
            seed=args.seed,
            shared_random_feature_dim=shared_random_feature_dim,
        )
        if hasattr(graph, "x") and graph.x is not None:
            graph.x = graph.x.to(torch.float32)
        graph = graph.to(args.device)
        features = getattr(graph, "x", None)
        if features is None:
            print(f"[Visualize] Node dataset {name} has no features; skipping.")
            continue
        embeddings = trainer._build_embeddings(graph, features, encoder_bundle)
        _save_stage2_visuals(
            graph=graph,
            features=features,
            embeddings=embeddings,
            dataset_name=name,
            results_dir=node_dir,
        )
        print(f"[Visualize] Saved Stage 2 node visuals for {name}")
