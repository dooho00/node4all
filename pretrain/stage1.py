import json
import os
from typing import Tuple

import torch

from pretrain.attributed_ssl_trainer import AttributedSSLTrainer
from dataset_utils.load_dataset import load_dataset
from channel_models.setup import setup_model

from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler


def _build_node4all_cfg(args) -> dict:
    cfg: dict = {}
    cfg_json = getattr(args, "node4all_cfg_json", None)
    if cfg_json:
        try:
            overrides = json.loads(cfg_json)
        except json.JSONDecodeError as exc:
            raise ValueError("node4all_cfg_json must be valid JSON") from exc
        if not isinstance(overrides, dict):
            raise ValueError("node4all_cfg_json must be a JSON object")
        cfg.update(overrides)
    return cfg


def pretrain(args) -> Tuple[float, float]:
    base_cfg = _build_node4all_cfg(args)
    synthetic_cfg = dict(base_cfg)
    synthetic_cfg["seed"] = int(getattr(args, "seed", 42))

    dataset_name = getattr(args, "datasetA", "")
    if dataset_name == "node4all":
        from dataset_utils.node4all import sample_node4all

        graph = sample_node4all(cfg=synthetic_cfg)
    else:
        graph = load_dataset(dataset_name, split_index=0, synthetic_config=synthetic_cfg)

    return _pretrain_masked_autoencoder(graph, args)


def _pretrain_masked_autoencoder(graph: Data, args) -> Tuple[float, float]:
    dataset_name = args.datasetA
    device = args.device

    print(f"=== Training Node4All masked autoencoder with DropNode on {dataset_name} ===")

    # PyG: features live in graph.x
    features = graph.x.to(device) if graph is not None else None
    encoder = setup_model(
        is_enc=True,
        args=args,
        purpose="enc",
    ).to(device)
    decoder = setup_model(
        is_enc=True,
        args=args,
        purpose="dec",
    ).to(device)

    
    is_node4all_source = getattr(args, "datasetA", "") == "node4all"
    node4all_cfg = _build_node4all_cfg(args)

    def _build_train_loader():
        if is_node4all_source:
            from dataset_utils.node4all import sample_node4all

            def _epoch_loader(epoch: int):
                base_seed = int(getattr(args, "seed", 42))
                epoch_seed = base_seed + 1_000_003 * int(epoch)
                def _iter_batches():
                    cfg = dict(node4all_cfg)
                    cfg["seed"] = int(epoch_seed)
                    yield sample_node4all(cfg=cfg)

                return _iter_batches()

            return _epoch_loader

        else:
            # Prepare PyG data for GraphSAINT sampling
            # GraphSAINT expects CPU Data with edge_index, x, y, and masks
            with torch.no_grad():
                saint_data = Data(
                    x=graph.x.cpu(),
                    y=(graph.y.cpu() if getattr(graph, "y", None) is not None else None),
                    edge_index=graph.edge_index.cpu().long(),
                )
                if hasattr(graph, "train_mask"):
                    saint_data.train_mask = graph.train_mask.cpu()
                if hasattr(graph, "val_mask"):
                    saint_data.val_mask = graph.val_mask.cpu()
                if hasattr(graph, "test_mask"):
                    saint_data.test_mask = graph.test_mask.cpu()

            batch_size = getattr(args, "graphsaint_batch_size", 2048)
            walk_length = getattr(args, "graphsaint_walk_length", 2)
            num_steps = getattr(args, "graphsaint_num_steps", 30)
            sample_coverage = getattr(args, "graphsaint_sample_coverage", 0)
            
            return GraphSAINTRandomWalkSampler(
                saint_data,
                batch_size=batch_size,
                walk_length=walk_length,
                num_steps=num_steps,
                sample_coverage=sample_coverage,
            )

    final_state_dict = {}
    final_result = {}

    # Pass PyG Data `graph` (on CPU/GPU as needed) and `features` to trainers
    enc_trainer = AttributedSSLTrainer(
        args,
        encoder,
        decoder,
    )
    enc_result, enc_state = enc_trainer.train(
        graph,
        features,
        train_loader=_build_train_loader(),
        desc=f"Node4All SSL {args.arch_type}: {dataset_name}",
    )
    final_state_dict["encoder"] = enc_state
    final_result["val"] = enc_result.get("val", 0.0)
    final_result["test"] = enc_result.get("test", 0.0)

    def _safe_save(state, path, label: str):
        if state is None or not path:
            return False
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(state, path)
        print(f"{label} saved to {path}")
        return True

    if "encoder" in final_state_dict:
        _safe_save(
            final_state_dict.get("encoder"),
            getattr(args, "enc_checkpoint", None),
            "Encoder",
        )

    val = final_result.get("val", 0)
    test = final_result.get("test", 0)
    print(f"Masked-autoencoder pretraining complete. Final Valid: {100 * val:.2f}%, Final Test: {100 * test:.2f}%")
    return val, test
