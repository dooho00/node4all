import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from typing import Optional, Sequence, Tuple
from tqdm import tqdm

from channel_models.channel_base import ChannelGraphBase


def _feature_batch_chunks(
    num_nodes: int,
    num_feats: int,
    feature_budget: int,
) -> Sequence[Tuple[int, int]]:
    """
    Yield (start, end) feature index ranges so that
      num_nodes * (end - start) <= feature_budget.

    If feature_budget <= 0 or very large, returns a single chunk (0, num_feats).
    """
    if feature_budget is None or feature_budget <= 0:
        return [(0, num_feats)]

    max_tokens = max(1, feature_budget)
    max_feats_per_chunk = max(1, max_tokens // max(1, num_nodes))

    if max_feats_per_chunk >= num_feats:
        return [(0, num_feats)]

    chunks = []
    start = 0
    while start < num_feats:
        end = min(num_feats, start + max_feats_per_chunk)
        chunks.append((start, end))
        start = end
    return chunks


def channel_inference_neighbor_sampling(
    model: ChannelGraphBase,
    data: Data,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    num_neighbors: Optional[Sequence[int] | int] = None,
    show_progress: bool = True,
    *,
    mode: str = "auto",
    feature_nf_threshold: int = 1000000,
    feature_budget: Optional[int] = None,
) -> Tensor:
    """
    Inference for any ChannelGraphBase model (CGT, channel-wise GAT, channel-wise GCN).

    There are two independent decisions:

      1) Neighbor batching (decided by N vs batch_size):
           - use neighbor sampling only if batch_size is specified and N > batch_size.
           - if batch_size is None → always use full graph (no neighbor sampling).

      2) Feature batching (decided by N * F):
           - use feature batching over feature dimension if N * F is large.

    Regimes:

      • Full graph, no feature batching
      • Full graph with feature batching
      • Neighbor batching, no feature batching
      • Neighbor batching wrapped inside feature batching
        (outer loop over feature chunks, inner loop over neighbor loader)

    Args:
      model: Channel-wise encoder (ChannelGATEncoder, ChannelGCNEncoder, etc).
      data: PyG Data containing .x and .edge_index.
      batch_size: seed batch size for NeighborLoader.
                  If None → disable neighbor sampling and use full-batch mode.
      num_workers: number of workers for NeighborLoader.
      device: torch.device, taken from model if None.
      num_neighbors: per-layer neighbors for NeighborLoader.
                    If None → [-1] * model.num_layers.
      show_progress: whether to display tqdm bars for both neighbor sampling
                     and feature batching loops.
      mode: "auto", "neighbor", or "full".
            - "full": never use neighbor sampling.
            - "neighbor": try neighbor sampling, but only if
                          batch_size is specified and N > batch_size.
            - "auto": use neighbor sampling if batch_size is specified and
                      N > batch_size, else full batch.
      edge_neighbor_threshold: kept for compatibility, not used.
      feature_nf_threshold: threshold on N * F to prefer feature batching.
      feature_budget: maximum number of scalar tokens per feature chunk
                      (approx N * F_chunk). If None, defaults to feature_nf_threshold.

    Returns:
      out_all: (N, D) tensor of node embeddings or outputs.
    """

    if device is None:
        device = next(model.parameters()).device

    if data.x is None:
        raise ValueError("Data.x is None; ChannelGraphBase models expect node features.")

    x_full = data.x.to(device)
    edge_index_full = data.edge_index.to(device)

    N = data.num_nodes
    F = x_full.size(1)
    E = edge_index_full.size(1)
    nf = N * F

    # If feature_budget not given, tie it to feature_nf_threshold
    if feature_budget is None:
        feature_budget = feature_nf_threshold

    # Resolve an "effective" batch size: None means full-batch only
    effective_batch_size: Optional[int]
    if batch_size is None or batch_size <= 0:
        effective_batch_size = None  # disable neighbor sampling
    else:
        effective_batch_size = batch_size

    # helper to decide per-hop fanouts
    def _resolve_fanouts() -> Sequence[int]:
        if num_neighbors is None:
            return [-1] * model.num_layers
        if isinstance(num_neighbors, int):
            return [num_neighbors] * model.num_layers
        return list(num_neighbors)

    # ------------------------------------------------------------------
    # Full graph paths
    # ------------------------------------------------------------------
    def _full_graph_no_feature_batch() -> Tensor:
        out = model(x_full, edge_index_full, num_nodes=N)
        return out.detach()

    def _full_graph_feature_batched() -> Tensor:
        chunks = _feature_batch_chunks(N, F, feature_budget)
        if len(chunks) == 1:
            # no real need to feature batch
            return _full_graph_no_feature_batch()

        if show_progress:
            chunk_iter = tqdm(
                chunks,
                desc=f"Feature chunks (full graph)",
                total=len(chunks),
            )
        else:
            chunk_iter = chunks

        outs = []
        print(
            f"[channel_inference] Full graph with feature batching over {len(chunks)} chunks "
            f"(N={N}, F={F}, N*F={nf}, feature_budget={feature_budget})."
        )
        for (start, end) in chunk_iter:
            x_chunk = x_full[:, start:end]
            out_chunk = model(x_chunk, edge_index_full, num_nodes=N)
            outs.append(out_chunk)

        out = torch.cat(outs, dim=1)
        return out.detach()

    # ------------------------------------------------------------------
    # Neighbor paths
    # ------------------------------------------------------------------
    def _neighbor_inference_single_feature_chunk(x_chunk: Tensor) -> Tensor:
        """
        Neighbor batching for a given feature chunk.

        This is the inner loop that runs on one feature chunk's features
        but can be used for the full feature set as well.
        """
        fanouts = _resolve_fanouts()
        if len(fanouts) != model.num_layers:
            raise ValueError(
                f"num_neighbors must match number of layers ({model.num_layers}). "
                f"Got {fanouts}."
            )

        if effective_batch_size is None:
            raise RuntimeError(
                "Neighbor inference requested but batch_size is None; "
                "this should not happen under the current decision logic."
            )

        # We construct a shallow Data that uses x_chunk but shares edge_index
        # and num_nodes with the original data.
        tmp_data = Data(
            x=x_chunk,
            edge_index=edge_index_full,
            num_nodes=N,
        )

        loader = NeighborLoader(
            tmp_data,
            num_neighbors=fanouts,
            batch_size=effective_batch_size,
            shuffle=False,
            input_nodes=None,  # all nodes as seeds
            num_workers=num_workers,
        )

        out_all: Optional[Tensor] = None
        iterable = tqdm(
            loader,
            desc="Inference (neighbor sampling)",
            disable=not show_progress,
        )

        for batch in iterable:
            batch = batch.to(device)

            n_id = batch.n_id          # global node indices of subgraph
            seed_size = batch.batch_size

            x_sub = batch.x
            edge_index_sub = batch.edge_index
            num_nodes_sub = batch.num_nodes

            out_sub = model(x_sub, edge_index_sub, num_nodes=num_nodes_sub)

            if out_all is None:
                out_all = out_sub.new_zeros((N,) + out_sub.shape[1:])

            seed_global = n_id[:seed_size]
            out_all[seed_global] = out_sub[:seed_size]

        if out_all is None:
            out_all = torch.empty((N, 0), device=device)
        return out_all.detach()

    def _neighbor_inference_no_feature_batch() -> Tensor:
        print(
            f"[channel_inference] Neighbor batching only "
            f"(N={N}, F={F}, E={E}, N*F={nf}, batch_size={effective_batch_size})."
        )
        return _neighbor_inference_single_feature_chunk(x_full)

    def _neighbor_inference_feature_batched() -> Tensor:
        chunks = _feature_batch_chunks(N, F, feature_budget)
        if len(chunks) == 1:
            return _neighbor_inference_no_feature_batch()

        if show_progress:
            chunk_iter = tqdm(
                chunks,
                desc="Feature chunks (neighbor sampling)",
                total=len(chunks),
            )
        else:
            chunk_iter = chunks

        outs = []
        print(
            f"[channel_inference] Feature batching + neighbor batching "
            f"over {len(chunks)} feature chunks "
            f"(N={N}, F={F}, E={E}, N*F={nf}, feature_budget={feature_budget}, "
            f"batch_size={effective_batch_size})."
        )

        for (start, end) in chunk_iter:
            x_chunk = x_full[:, start:end]
            out_chunk = _neighbor_inference_single_feature_chunk(x_chunk)
            outs.append(out_chunk)

        out = torch.cat(outs, dim=1)
        return out.detach()

    # ------------------------------------------------------------------
    # Decide mode: neighbor vs full, and feature batching yes or no
    # ------------------------------------------------------------------
    if mode not in {"auto", "neighbor", "full"}:
        raise ValueError(f"mode must be one of 'auto', 'neighbor', 'full', got {mode!r}")

    # Decide neighbor usage based on N vs batch_size
    if mode == "full":
        use_neighbor = False
    elif effective_batch_size is None:
        # No batch size → we do not enable neighbor sampling, even in 'auto'.
        if mode == "neighbor":
            print(
                "[channel_inference] mode='neighbor' but batch_size is None; "
                "falling back to full graph."
            )
        use_neighbor = False
    else:
        # effective_batch_size is defined
        if mode == "neighbor":
            use_neighbor = N > effective_batch_size
            if not use_neighbor:
                print(
                    f"[channel_inference] mode='neighbor' but N={N} ≤ batch_size={effective_batch_size}; "
                    "falling back to full graph."
                )
        else:  # "auto"
            use_neighbor = N > effective_batch_size

    # Decide feature batching usage
    use_feature_batch = nf > feature_nf_threshold

    # ------------------------------------------------------------------
    # Execute according to decisions
    # ------------------------------------------------------------------
    if not use_neighbor:
        # full graph path
        if use_feature_batch:
            # print(
            #     f"[channel_inference] FULL → full graph with feature batching "
            #     f"(N={N}, F={F}, N*F={nf} > feature_nf_threshold={feature_nf_threshold})."
            # )
            return _full_graph_feature_batched()
        else:
            # print(
            #     f"[channel_inference] FULL → full graph without feature batching "
            #     f"(N={N}, F={F}, N*F={nf} ≤ feature_nf_threshold={feature_nf_threshold})."
            # )
            return _full_graph_no_feature_batch()
    else:
        # neighbor path
        if use_feature_batch:
            # print(
            #     f"[channel_inference] NEIGHBOR → feature batching + neighbor batching "
            #     f"(N={N}, F={F}, E={E}, N*F={nf} > feature_nf_threshold={feature_nf_threshold}, "
            #     f"batch_size={effective_batch_size})."
            # )
            return _neighbor_inference_feature_batched()
        else:
            # print(
            #     f"[channel_inference] NEIGHBOR → neighbor batching only "
            #     f"(N={N}, F={F}, E={E}, N*F={nf} ≤ feature_nf_threshold={feature_nf_threshold}, "
            #     f"batch_size={effective_batch_size})."
            # )
            return _neighbor_inference_no_feature_batch()
