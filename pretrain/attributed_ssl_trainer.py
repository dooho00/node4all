import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops, subgraph, to_undirected

from .ssl_utils import _feature_batch_chunks, _build_subgraph

class AttributedSSLTrainer:
    """Self-supervised masked autoencoder trainer with DropNode reconstruction."""

    def __init__(
        self,
        args,
        encoder: nn.Module,
        decoder_encoder: Optional[nn.Module],
    ):
        self.args = args
        self.encoder = encoder
        self.decoder_encoder = decoder_encoder
        self.feature_budget = int(getattr(args, "enc_feature_budget", 4_000_000))
        self.latent_mask_rate = float(getattr(args, "ssl_latent_mask_rate", 0.0))
        self.latent_mask_mode = str(getattr(args, "ssl_latent_mask_mode", "element")).lower()
        self.input_mask_value = float(getattr(args, "ssl_input_mask_value", 0.0))
        self.ssl_alpha_l = float(getattr(args, "ssl_alpha_l", 3.0))
        if self.latent_mask_mode not in ("element", "node"):
            raise ValueError("ssl_latent_mask_mode must be one of: element, node")
        self.ssl_loss_fn = "sce" if self.latent_mask_mode == "node" else "mse"

    def _apply_feature_mask(
        self,
        feat_chunk: torch.Tensor,
        mask_override: Optional[torch.Tensor] = None,
        *,
        mask_rate: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_rate = 0.0 if mask_rate is None else float(mask_rate)
        num_nodes, num_feats = feat_chunk.shape

        '''
        mask = torch.rand((num_nodes, num_feats), device=feat_chunk.device) < mask_rate
        masked_input = feat_chunk.clone()
        masked_input[mask] = self.input_mask_value
        '''

        # Input feature masking keeps a small visible context per feature channel.
        mask = torch.zeros((num_nodes, num_feats), dtype=torch.bool, device=feat_chunk.device)
        num_mask_per_dim = num_nodes - 128
        for dim in range(num_feats):
            if num_mask_per_dim <= 0:
                continue
            perm = torch.randperm(num_nodes, device=feat_chunk.device)
            mask_indices = perm[:num_mask_per_dim]
            mask[mask_indices, dim] = True
        
        masked_input = feat_chunk.clone()
        masked_input[mask] = self.input_mask_value

        return masked_input, mask

    def _ensure_undirected_self_loops(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def _sample_node_mask(
        self,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_nodes <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)
        if self.latent_mask_rate <= 0.0:
            return torch.zeros((num_nodes,), dtype=torch.bool, device=device)

        node_mask = torch.rand((num_nodes,), device=device) < self.latent_mask_rate
        # Keep at least one masked node so SCE has a valid supervision target.
        if not bool(node_mask.any().item()):
            node_mask[torch.randint(num_nodes, (1,), device=device)] = True
        # Keep at least one unmasked node when possible to preserve encoder context.
        if num_nodes > 1 and bool(node_mask.all().item()):
            node_mask[torch.randint(num_nodes, (1,), device=device)] = False
        return node_mask

    def _masked_recon_once(
        self,
        graph: Data,
        x_in: torch.Tensor,
        node_norm: Optional[torch.Tensor],
        latent_mask_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = graph.edge_index
        num_nodes = int(graph.num_nodes)
        if self.latent_mask_mode == "node" and (
            self.latent_mask_rate > 0.0 or latent_mask_override is not None
        ):
            node_mask = self._sample_node_mask(
                num_nodes=num_nodes,
                device=x_in.device,
            )
            keep_mask = ~node_mask
            keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
            sub_edge_index, _ = subgraph(
                keep_idx,
                edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            sub_edge_index = self._ensure_undirected_self_loops(sub_edge_index, num_nodes=keep_idx.numel())
            z_keep = self.encoder(x_in[keep_idx], sub_edge_index, num_nodes=keep_idx.numel())
            z = z_keep.new_full((num_nodes, z_keep.size(1)), self.input_mask_value)
            z[keep_idx] = z_keep
            z_mask = node_mask.view(-1, 1).expand_as(z)
            z_masked = z
            edge_index = self._ensure_undirected_self_loops(edge_index, num_nodes=num_nodes)
        else:
            z = self.encoder(x_in, edge_index, num_nodes=num_nodes)

            if self.latent_mask_rate > 0.0 or latent_mask_override is not None:
                z_masked, z_mask = self._apply_feature_mask(
                    z,
                    mask_override=latent_mask_override,
                    mask_rate=self.latent_mask_rate,
                )
            else:
                z_masked, z_mask = z, torch.zeros_like(z, dtype=torch.bool)

        if self.decoder_encoder is None:
            x_hat = z_masked
        else:
            x_hat = self.decoder_encoder(z_masked, edge_index, num_nodes=graph.num_nodes)

        return x_hat, z_mask, z

    def _reconstruction_loss(self, graph: Data, features: torch.Tensor):
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        num_nodes, num_feats = features.shape
        if num_feats == 0:
            zero = features.sum() * 0.0
            return zero, features.new_zeros((graph.num_nodes, 0))

        node_norm = getattr(graph, "node_norm", None)
        total_loss = features.new_zeros(())
        num_chunks = 0
        last_enc_outputs = None

        for start, end in _feature_batch_chunks(
            num_nodes,
            num_feats,
            self.feature_budget,
        ):
            x = features[:, start:end]
            x_hat_view, z_mask, z_view = self._masked_recon_once(
                graph,
                x,
                node_norm,
            )
            last_enc_outputs = z_view

            if self.ssl_loss_fn == "sce":
                x_hat_norm = F.normalize(x_hat_view, p=2, dim=1, eps=1e-12)
                x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)
                cos_sim = (x_hat_norm * x_norm).sum(dim=1)
                loss_elem = (1.0 - cos_sim).clamp_min(0.0).pow(self.ssl_alpha_l)
                if node_norm is not None:
                    loss_elem = loss_elem * node_norm.view(-1)
                recon_mask = z_mask.any(dim=1)
                if recon_mask.any():
                    denom_mask = recon_mask.sum().clamp_min(1)
                    mask_loss = loss_elem[recon_mask].sum() / denom_mask
                else:
                    # Degenerate batch with no dropped nodes: use full-node loss
                    # so the objective remains differentiable.
                    mask_loss = loss_elem.mean()
            else:
                # MSE loss
                loss_elem = (x_hat_view - x).pow(2)

                # Relative error loss 68.6
                #diff = (x_hat_view - x).abs()
                #denom = x_hat_view.abs() + x.abs()
                #loss_elem = (diff / denom + 1e-9)

                # log-space MSE loss 69.01
                #log_x_hat = torch.log1p(x_hat_view.abs()) * x_hat_view.sign()
                #log_x = torch.log1p(x.abs()) * x.sign()
                #loss_elem = (log_x_hat - log_x).pow(2)

                # Normalized MSE loss 68.75
                #norm_x = x.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
                #norm_x_hat = x_hat_view.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
                #scaled_x = x / norm_x
                #scaled_x_hat = x_hat_view / norm_x_hat
                #loss_elem = (scaled_x_hat - scaled_x).pow(2)


                if node_norm is not None:
                    loss_elem = loss_elem * node_norm.view(-1, 1)

                #recon_mask = z_mask
                recon_mask = torch.ones_like(loss_elem, dtype=torch.bool)
                denom_mask = recon_mask.sum().clamp_min(1)
                mask_loss = loss_elem[recon_mask].sum() / denom_mask

            total_loss = total_loss + mask_loss
            num_chunks += 1

        total_loss = total_loss / max(num_chunks, 1)
        return total_loss, last_enc_outputs

    def train(
        self,
        graph: Data,
        features: torch.Tensor,
        desc: str = "Encoder SSL",
        train_loader=None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, torch.Tensor]]]:
        params = list(self.encoder.parameters())
        if self.decoder_encoder is not None:
            params += list(self.decoder_encoder.parameters())

        optimizer = torch.optim.Adam(
            params,
            lr=self.args.ssl_lr,
            weight_decay=self.args.ssl_weight_decay,
        )
        best_loss = float("inf")
        metrics: Dict[str, float] = {}
        last_loss: Optional[float] = None

        epoch_iter = tqdm(range(self.args.epochs), desc=desc, leave=False)
        for epoch in epoch_iter:
            self.encoder.train()
            if self.decoder_encoder is not None:
                self.decoder_encoder.train()

            if train_loader is None:
                loss, _ = self._reconstruction_loss(graph, features)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                torch.nn.utils.clip_grad_norm_(features, 1.0)
                optimizer.step()
            else:
                epoch_loss = 0.0
                total_weight = 0.0
                total_dims = 0
                total_sparsify = 0
                total_calls = 0
                epoch_loader = train_loader(epoch) if callable(train_loader) else train_loader
                batch_iter = tqdm(epoch_loader, desc=f"{desc} | epoch {epoch + 1}", leave=False)
                for batch in batch_iter:
                    batch = batch.to(self.args.device, non_blocking=True)
                    stats = getattr(batch, "node4all_auto_stats", None)
                    if isinstance(stats, dict):
                        total_dims += int(stats.get("total_dims", 0))
                        total_sparsify += int(stats.get("sparsify_dims", 0))
                        total_calls += int(stats.get("num_calls", 0))
                    subgraph: Data = _build_subgraph(batch, self.args).to(self.args.device, non_blocking=True)
                    loss, _ = self._reconstruction_loss(subgraph, subgraph.x)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    torch.nn.utils.clip_grad_norm_(subgraph.x, 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_weight += 1.0
                    batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})

                if total_weight > 0:
                    loss = torch.tensor(epoch_loss / total_weight, device=self.args.device)
                if total_dims > 0:
                    p_sparse = total_sparsify / max(total_dims, 1)
                    msg = (
                        f"[node4all auto] epoch {epoch + 1}: "
                        f"sparsify_prob={p_sparse:.3f}"
                    )
                    if total_calls > 0:
                        msg += f", calls={total_calls}"
                    print(msg)

            current_loss = float(loss.item())
            last_loss = current_loss
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss

            epoch_iter.set_postfix({"loss": f"{current_loss:.4f}", "best": f"{best_loss:.4f}"})

        last_state = {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}

        final_loss = last_loss if last_loss is not None else 0.0
        return {"val": -final_loss, "test": metrics.get("test", 0.0)}, last_state

__all__ = ["AttributedSSLTrainer"]
