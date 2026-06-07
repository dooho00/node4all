import logging
import os
import os.path as osp
import pickle
import ssl
import sys
import urllib
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import dgl  # used only for dataset backends that still rely on DGL downloads
import numpy as np
import torch
import torch_geometric.datasets as pyg_datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    add_self_loops,
    remove_self_loops,
    coalesce,
)

from sklearn.model_selection import train_test_split

from dataset_utils.dataset_registry import NodeLevelDatasetRegistry
from utils.utils import derive_split_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiny helper to convert DGL → PyG when we must use DGL for download / synthetic
# ---------------------------------------------------------------------------

def _dgl_to_pyg_graph(graph: "dgl.DGLGraph") -> Data:
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0).long()

    x = graph.ndata.get("feat", None)
    y = graph.ndata.get("label", None)

    data = Data(x=x, edge_index=edge_index, y=y)

    for key, value in graph.ndata.items():
        if key in {"feat", "label"}:
            continue
        setattr(data, key, value)

    return data

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):
        if log and "pytest" not in sys.modules:
            logger.info(f"Using existing file {filename}")
        return path

    if log and "pytest" not in sys.modules:
        logger.info(f"Downloading {url}")

    os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path

# ---------------------------------------------------------------------------
# Split helpers (PyG)
# ---------------------------------------------------------------------------

def get_data_split_masks(
    n_nodes: int,
    labels: torch.Tensor,
    num_train_nodes: int,
    test_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_idx = np.arange(len(labels))
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)

    train_idx, test_and_valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    valid_idx, test_idx = train_test_split(
        test_and_valid_idx,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels[test_and_valid_idx],
    )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def _generate_splits_for_pyg_data(
    data: Data,
    data_name: str,
    split_index: int,
    cache_dir: str = "splits",
    seed: int = 42,
) -> Data:
    os.makedirs(cache_dir, exist_ok=True)
    split_id = int(split_index) + 1
    split_path = osp.join(cache_dir, f"{data_name}_seed{seed}_{split_id}.splits")
    derived_seed = derive_split_seed(seed, split_index)

    if osp.exists(split_path):
        logger.info(f"Loading splits for {data_name} from {split_path}")
        train_mask, val_mask, test_mask = pickle.load(open(split_path, "rb"))
    else:
        logger.info(f"Generating a new split for {data_name}")
        labels = data.y.view(-1)
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 1
        desired_train = max(num_classes, 20 * num_classes)
        desired_train = min(desired_train, max(num_classes, data.num_nodes - 2))

        train_mask, val_mask, test_mask = get_data_split_masks(
            n_nodes=data.num_nodes,
            labels=labels,
            num_train_nodes=desired_train,
            test_ratio=0.5,
            seed=derived_seed,
        )
        pickle.dump((train_mask, val_mask, test_mask), open(split_path, "wb"))

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.split_source = "random"
    data.split_count = 1
    data.split_index = int(split_index)
    data.split_seed = int(derived_seed)
    return data

def _preprocess_pyg_graph_for_dataset(
    data,
    add_self_loop: bool = True,
    to_bidirected: bool = True,
):
    edge_index = data.edge_index
    if edge_index is None:
        return data
    num_nodes = data.num_nodes
    edge_index, _ = remove_self_loops(edge_index)
    #print(edge_index.size())
    edge_index = coalesce(edge_index)
    if to_bidirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    data.edge_index = edge_index
    
    # Feature-wise preprocessing
    feat = getattr(data, "x", None)
    # Quantile + identity (z-score) concat
    '''
    if feat is not None and torch.is_floating_point(feat):
        train_mask = getattr(data, "train_mask", None)
        data.x = _quantile_plus_id_features(feat, train_mask)'''

    return data


def _quantile_plus_id_features(
    feat: torch.Tensor,
    train_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if feat.dim() != 2:
        raise ValueError("Feature tensor must be 2D (N, F).")
    num_nodes, num_feats = feat.shape
    if num_feats == 0:
        return feat

    device = feat.device
    dtype = feat.dtype

    if train_mask is not None and train_mask.numel() == num_nodes and bool(train_mask.any().item()):
        fit_feat = feat[train_mask.to(dtype=torch.bool)]
    else:
        fit_feat = feat

    fit_np = fit_feat.detach().cpu().numpy().astype(np.float64, copy=False)
    x_np = feat.detach().cpu().numpy().astype(np.float64, copy=False)

    sorted_cols = []
    for j in range(num_feats):
        col = fit_np[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            sorted_cols.append(np.array([0.0], dtype=np.float64))
        else:
            sorted_cols.append(np.sort(col))

    mean = np.nanmean(fit_np, axis=0)
    std = np.nanstd(fit_np, axis=0)
    std = np.where(std > 0, std, 1.0)

    q_out = np.empty_like(x_np)
    for j in range(num_feats):
        s = sorted_cols[j]
        x_col = x_np[:, j]
        q = np.full(num_nodes, 0.5, dtype=np.float64)
        finite_mask = np.isfinite(x_col)
        if s.size > 1:
            q[finite_mask] = np.searchsorted(s, x_col[finite_mask], side="right") / float(s.size)
        else:
            q[finite_mask] = 0.5
        q_out[:, j] = np.clip(q, 0.0, 1.0)

    z_out = (x_np - mean) / std
    z_out = np.where(np.isfinite(z_out), z_out, 0.0)

    out = np.concatenate([q_out, z_out], axis=1)
    return torch.from_numpy(out).to(device=device, dtype=dtype)
    
# ---------------------------------------------------------------------------
# Main node-classification loader (returns PyG Data)
# ---------------------------------------------------------------------------

def load_dataset(
    data_name: str,
    split_index: int = 0,
    add_self_loop: bool = True,
    to_bidirected: bool = True,
    cache_dir: str = "../dataset",
    seed: int = 42,
    splits_dir: str = "splits",
    synthetic_config: Optional[dict] = None,
    shared_random_feature_dim: int = 0,
    shared_random_feature_seed: Optional[int] = None,
) -> Data:
    """
    Unified node-classification dataset loader.

    Returns:
        PyG Data object with:
          - x: node features (float32)
          - y: node labels (if available)
          - edge_index: graph edges
          - train_mask / val_mask / test_mask: boolean node-classification splits
    """
    registry = NodeLevelDatasetRegistry()

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    data: Optional[Data] = None

    if data_name in registry.data_sources.get("pyg", {}):
        data = _load_pyg_dataset(data_name, registry, split_index, seed, splits_dir, cache_dir)
    elif data_name in registry.data_sources.get("dgl", {}):
        data = _load_dgl_dataset_as_pyg(data_name, registry, split_index, seed, splits_dir, cache_dir)
    elif data_name in registry.data_sources.get("ogb", {}):
        data = _load_ogb_dataset_as_pyg(data_name, registry, cache_dir)
    elif data_name in registry.data_sources.get("heterophilous", {}):
        data = _load_heterophilous_dataset_pyg(data_name, registry, cache_dir, split_index, splits_dir, seed)
    elif synthetic_config is not None:
        data = _load_node4all_synthetic_dataset_as_pyg(data_name, synthetic_config, seed)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    raw_feat = getattr(data, "x", None)
    has_raw_features = raw_feat is not None and raw_feat.numel() > 0
    data = _ensure_node_features(data)
    data = _ensure_feature_dtype(data, dtype=torch.float32)
    data = _preprocess_pyg_graph_for_dataset(data, add_self_loop, to_bidirected)
    data = _maybe_attach_shared_random_features(
        data,
        feature_dim=shared_random_feature_dim,
        seed=seed if shared_random_feature_seed is None else shared_random_feature_seed,
        append=has_raw_features,
    )

    return data


# ---------------------------------------------------------------------------
# Node-classification dataset backends
# ---------------------------------------------------------------------------

def _load_dgl_dataset_as_pyg(
    data_name: str,
    registry: NodeLevelDatasetRegistry,
    split_index: int,
    seed: int,
    splits_dir: str,
    cache_dir: str,
) -> Data:
    dataset_class_name = registry.data_sources["dgl"][data_name]
    dataset_class = getattr(dgl.data, dataset_class_name)

    target_raw_dir = os.path.join(cache_dir, data_name)
    os.makedirs(target_raw_dir, exist_ok=True)
    try:
        dataset = dataset_class(raw_dir=target_raw_dir)
    except TypeError:
        dataset = dataset_class()

    graph = dataset[0]
    ndata = graph.ndata

    split_source = "public"
    split_count = 1
    split_index_used = split_index

    if "train_mask" in ndata and ndata["train_mask"] is not None:
        train_mask = ndata["train_mask"]
        if hasattr(train_mask, "ndim") and train_mask.ndim > 1:
            split_count = int(train_mask.shape[1])
        _handle_existing_splits_dgl(graph, data_name, split_index)
    else:
        split_source = "random"
        graph = _handle_splits_dgl(graph, data_name, split_index, cache_dir=splits_dir, seed=seed)

    data = _dgl_to_pyg_graph(graph)
    if split_count <= 1:
        split_index_used = 0
    data.split_source = split_source
    data.split_count = int(split_count)
    data.split_index = int(split_index_used)
    if split_source == "random":
        data.split_seed = int(derive_split_seed(seed, split_index_used))
    return data


def _load_ogb_dataset_as_pyg(
    data_name: str,
    registry: NodeLevelDatasetRegistry,
    cache_dir: str,
) -> Data:
    from ogb.nodeproppred import PygNodePropPredDataset

    ogb_name = registry.data_sources["ogb"][data_name]
    ogb_root = osp.join(cache_dir, "ogb")
    _cleanup_corrupted_archives(ogb_root)
    dataset = PygNodePropPredDataset(name=ogb_name, root=ogb_root)

    data = dataset[0]
    if hasattr(data, "y") and data.y is not None and data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.view(-1)

    splits = dataset.get_idx_split()
    num_nodes = data.num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[splits["train"]] = True
    val_mask[splits["valid"]] = True
    test_mask[splits["test"]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.split_source = "public"
    data.split_count = 1
    data.split_index = 0

    return data


def _load_node4all_synthetic_dataset_as_pyg(
    data_name: str,
    synthetic_config: Optional[dict],
    seed: int,
) -> Data:
    from dataset_utils.node4all import sample_node4all

    cfg = dict(synthetic_config or {})
    if "seed" not in cfg:
        cfg["seed"] = int(seed)
    data = sample_node4all(cfg=cfg)

    logger.info(
        f"Sampled Node4All synthetic graph '{data_name}' with {int(data.num_nodes)} nodes, "
        f"{int(data.edge_index.size(1))} edges, feat_dim={int(data.x.size(1)) if getattr(data, 'x', None) is not None else 0}"
    )
    return data


def _cleanup_corrupted_archives(directory: str) -> None:
    if not osp.isdir(directory):
        return

    for entry in os.listdir(directory):
        if not entry.lower().endswith(".zip"):
            continue

        path = osp.join(directory, entry)

        try:
            if zipfile.is_zipfile(path):
                continue
        except Exception:
            pass

        logger.warning(f"Removing corrupted archive {path}")
        try:
            os.remove(path)
        except OSError as exc:
            logger.error(f"Failed to remove corrupted archive {path}: {exc}")


def _load_heterophilous_dataset_pyg(
    data_name: str,
    registry: NodeLevelDatasetRegistry,
    cache_dir: str,
    split_index: int,
    splits_dir: str,
    seed: int,
) -> Data:
    dataset_filename = registry.data_sources["heterophilous"][data_name]
    local_path = osp.join(cache_dir, f"{dataset_filename}.npz")

    if osp.exists(local_path):
        dataset = np.load(local_path)
        node_features = torch.tensor(dataset["node_features"])
        edges = torch.tensor(dataset["edges"], dtype=torch.long)
        labels = torch.tensor(dataset["node_labels"])

        edge_index = edges.t().contiguous()
        data = Data(x=node_features, edge_index=edge_index, y=labels)

        if "train_masks" in dataset:
            num_splits = dataset["train_masks"].shape[0]
            split_index = split_index % num_splits

            data.train_mask = torch.tensor(dataset["train_masks"][split_index])
            data.val_mask = torch.tensor(dataset["val_masks"][split_index])
            data.test_mask = torch.tensor(dataset["test_masks"][split_index])
            data.split_source = "public"
            data.split_count = int(num_splits)
            data.split_index = int(split_index)
        else:
            data = _generate_splits_for_pyg_data(data, data_name, split_index, cache_dir=splits_dir, seed=seed)
    else:
        url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{dataset_filename}.npz"
        raw_dir = osp.join(cache_dir, "heterophilous")

        download_path = download_url(url, raw_dir)
        dataset = np.load(download_path)

        node_features = torch.tensor(dataset["node_features"])
        labels = torch.tensor(dataset["node_labels"])
        edges = torch.tensor(dataset["edges"], dtype=torch.long)
        edge_index = edges.t().contiguous()

        train_masks = torch.tensor(dataset["train_masks"]).T
        val_masks = torch.tensor(dataset["val_masks"]).T
        test_masks = torch.tensor(dataset["test_masks"]).T

        data = Data(x=node_features, edge_index=edge_index, y=labels)

        num_splits = train_masks.shape[1]
        split_index = split_index % num_splits

        data.train_mask = train_masks[:, split_index]
        data.val_mask = val_masks[:, split_index]
        data.test_mask = test_masks[:, split_index]
        data.split_source = "public"
        data.split_count = int(num_splits)
        data.split_index = int(split_index)

    return data


def _load_pyg_dataset(
    data_name: str,
    registry: NodeLevelDatasetRegistry,
    split_index: int,
    seed: int,
    splits_dir: str,
    cache_dir: str,
) -> Data:
    dataset_config = registry.data_sources["pyg"][data_name]

    if isinstance(dataset_config, str):
        dataset_class_name = dataset_config
        dataset_params = {}
    else:
        dataset_class_name = dataset_config["class"]
        dataset_params = {k: v for k, v in dataset_config.items() if k != "class"}

    dataset_class = getattr(pyg_datasets, dataset_class_name)

    root_dir = os.path.join(cache_dir, data_name)
    os.makedirs(root_dir, exist_ok=True)

    dataset = dataset_class(root=root_dir, **dataset_params)
    data = dataset[0]

    if hasattr(data, "y") and data.y is not None and data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.view(-1)

    if hasattr(data, "train_mask") and data.train_mask is not None:
        data = _handle_existing_splits_pyg(data, data_name, split_index)
    else:
        data = _generate_splits_for_pyg_data(data, data_name, split_index, cache_dir=splits_dir, seed=seed)

    return data


def _handle_existing_splits_dgl(graph: "dgl.DGLGraph", data_name: str, split_index: int):
    if graph.ndata["train_mask"].ndim > 1:
        num_splits = graph.ndata["train_mask"].shape[1]
        split_index = split_index % num_splits
        graph.ndata["train_mask"] = graph.ndata["train_mask"][:, split_index]
        graph.ndata["val_mask"] = graph.ndata["val_mask"][:, split_index]
        if data_name != "25_wiki_cs":
            graph.ndata["test_mask"] = graph.ndata["test_mask"][:, split_index]


def _handle_splits_dgl(
    graph: "dgl.DGLGraph",
    data_name: str,
    split_index: int,
    cache_dir: str = "splits",
    seed: int = 42,
) -> "dgl.DGLGraph":
    os.makedirs(cache_dir, exist_ok=True)
    split_id = int(split_index) + 1
    split_path = osp.join(cache_dir, f"{data_name}_seed{seed}_{split_id}.splits")
    derived_seed = derive_split_seed(seed, split_index)

    if osp.exists(split_path):
        logger.info(f"Loading splits for {data_name} from {split_path}")
        all_splits = pickle.load(open(split_path, "rb"))
    else:
        logger.info(f"Generating a new split for {data_name}")
        labels = graph.ndata["label"]
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 1
        num_train = 20 * num_classes
        all_splits = get_data_split_masks(graph.num_nodes(), labels, num_train, seed=derived_seed)
        logger.info(f"Generated a split for {data_name}: {all_splits[0].shape}")
        pickle.dump(all_splits, open(split_path, "wb"))

    train_masks, val_masks, test_masks = all_splits

    graph.ndata["train_mask"] = train_masks
    graph.ndata["val_mask"] = val_masks
    graph.ndata["test_mask"] = test_masks

    return graph


def _handle_existing_splits_pyg(data: Data, data_name: str, split_index: int) -> Data:
    num_splits = 1
    if data.train_mask.ndim > 1:
        num_splits = data.train_mask.shape[1]
        split_index = split_index % num_splits
        data.train_mask = data.train_mask[:, split_index]
        data.val_mask = data.val_mask[:, split_index]
        if data_name != "25_wiki_cs":
            data.test_mask = data.test_mask[:, split_index]
    else:
        split_index = 0
    data.split_source = "public"
    data.split_count = int(num_splits)
    data.split_index = int(split_index)
    return data


# ---------------------------------------------------------------------------
# Feature utilities (PyG)
# ---------------------------------------------------------------------------

def _is_one_hot_feature(feat: torch.Tensor, tol: float = 1e-6) -> bool:
    if feat.ndim != 2 or feat.shape[1] <= 1 or feat.numel() == 0:
        return False

    if torch.is_floating_point(feat):
        feat_rounded = feat.round()
        if not torch.allclose(feat, feat_rounded, atol=tol):
            return False
        feat_int = feat_rounded.to(torch.int64)
    else:
        feat_int = feat.to(torch.int64)

    if not torch.all((feat_int == 0) | (feat_int == 1)):
        return False

    row_sums = feat_int.sum(dim=1)
    return bool(torch.all(row_sums == 1))


def _is_index_indicator_feature(feat: torch.Tensor, tol: float = 1e-6) -> bool:
    if feat.ndim == 1:
        values = feat
    elif feat.ndim == 2 and feat.shape[1] == 1:
        values = feat[:, 0]
    else:
        return False

    if values.numel() == 0:
        return False

    if torch.is_floating_point(values):
        values_rounded = values.round()
        if not torch.allclose(values, values_rounded, atol=tol):
            return False
        values_long = values_rounded.to(torch.int64)
    else:
        values_long = values.to(torch.int64)

    n = values_long.numel()
    if n == 0:
        return False

    expected_zero_based = torch.arange(n, device=values_long.device)
    if torch.equal(values_long, expected_zero_based):
        return True
    if torch.equal(values_long, expected_zero_based + 1):
        return True
    return False


def _ensure_node_features(data: Data) -> Data:
    feat = getattr(data, "x", None)
    needs_features = feat is None or feat.numel() == 0

    if needs_features:
        num_nodes = data.num_nodes
        feat = torch.ones((num_nodes, 1), dtype=torch.float32)
        data.x = feat

    return data


def _ensure_feature_dtype(data: Data, dtype: torch.dtype = torch.float32) -> Data:
    feat = getattr(data, "x", None)
    if feat is not None and feat.dtype != dtype:
        if not torch.is_floating_point(feat):
            data.x = feat.to(dtype=dtype)
        else:
            data.x = feat.to(dtype=dtype)
    return data


def _maybe_attach_shared_random_features(
    data: Data,
    *,
    feature_dim: int,
    seed: Optional[int],
    append: bool,
) -> Data:
    dim = int(feature_dim)
    if dim <= 0:
        return data

    num_nodes = int(data.num_nodes or 0)
    if num_nodes <= 0:
        return data

    x = getattr(data, "x", None)
    if x is not None:
        device = x.device
        dtype = x.dtype
    else:
        edge_index = getattr(data, "edge_index", None)
        device = edge_index.device if edge_index is not None else torch.device("cpu")
        dtype = torch.float32

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    shared = torch.randn((1, dim), device=device, generator=generator, dtype=dtype)
    rand_feat = shared.repeat(num_nodes, 1)

    if x is None or not append:
        data.x = rand_feat
        return data

    if x.device != rand_feat.device:
        x = x.to(device)
    if x.dtype != rand_feat.dtype:
        rand_feat = rand_feat.to(x.dtype)
    data.x = torch.cat([x, rand_feat], dim=1)
    return data


def _preprocess_pyg_graph(
    data: Data,
    add_self_loop: bool,
    to_bidirected: bool,
) -> Data:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return data

    num_nodes = data.num_nodes

    edge_index, _ = remove_self_loops(edge_index)

    if to_bidirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    data.edge_index = edge_index

    feat = getattr(data, "x", None)
    if feat is not None:
        is_one_hot = _is_one_hot_feature(feat)
        is_index_indicator = _is_index_indicator_feature(feat)
        '''
        if is_one_hot:
            logger.info("Detected one-hot features; replacing with scalar ones.")
            scalar_feature = torch.ones(
                (data.num_nodes, 1),
                dtype=feat.dtype if torch.is_floating_point(feat) else torch.float32,
                device=feat.device,
            )
            data.x = scalar_feature'''

        # index indicator currently left as is
    return data


# ---------------------------------------------------------------------------
# Graph-level datasets (still PyG based, but unchanged)
# ---------------------------------------------------------------------------

# Graph-level dataset loading and helpers have been removed; graph-level tasks are disabled.
