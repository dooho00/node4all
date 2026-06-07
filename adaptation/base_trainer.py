"""Shared utilities for adaptation trainers."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from channel_models.setup import setup_model
from channel_models.batch_inference import channel_inference_neighbor_sampling

class BaseAdaptationTrainer:
    """Provides shared helpers for adaptation trainers."""

    def __init__(self, args) -> None:
        self.args = args

    def _torch_load_checkpoint(self, path: str):
        load_kwargs = {"map_location": self.args.device}
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)
        except Exception as exc:
            warnings.warn(
                f"Failed to load {path} with weights_only=True ({exc}); retrying with weights_only=False.",
                RuntimeWarning,
            )
            return torch.load(path, **load_kwargs)

    def _try_load_encoder_state(self, path: Optional[str]):
        if not path:
            return None
        if not os.path.exists(path):
            return None
        checkpoint = self._torch_load_checkpoint(path)
        print(f"Loaded encoder checkpoint from {path}")
        return checkpoint

    def _load_encoder_bundle(
        self,
    ) -> Dict[str, torch.nn.Module]:
        """
        Load pretrained encoder from checkpoint.
        """
        bundle: Dict[str, torch.nn.Module] = {}

        enc_state = self._try_load_encoder_state(getattr(self.args, "enc_checkpoint", None))

        def _init_encoder(state_dict) -> torch.nn.Module:
            def _safe_load(module: torch.nn.Module, sd: dict) -> None:
                try:
                    module.load_state_dict(sd)
                except RuntimeError as exc:
                    missing, unexpected = module.load_state_dict(sd, strict=False)
                    warnings.warn(
                        f"Non-strict load_state_dict fallback due to: {exc}. "
                        f"missing={len(missing)}, unexpected={len(unexpected)}",
                        RuntimeWarning,
                    )

            encoder = setup_model(
                is_enc=True,
                args=self.args,
                purpose="enc",
            ).to(self.args.device)
            _safe_load(encoder, state_dict)
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            return encoder

        if enc_state is not None:
            bundle["enc"] = _init_encoder(enc_state)
        if not bundle and enc_state is None:
            raise RuntimeError("No pretrained encoder states were found in the provided checkpoints.")
        return bundle

    def _build_embeddings(
        self,
        graph,
        features: torch.Tensor,
        encoder_bundle: Dict[str, torch.nn.Module],
    ) -> torch.Tensor:
        """
        Build node representations using the encoder.
        """
        encoder = encoder_bundle.get("enc")
        if encoder:
            encoder.eval()
            encoder.requires_grad_(False)

        device = self.args.device

        def _encode(encoder, graph):
            with torch.no_grad():
                encoder.eval()
                encoder.requires_grad_(False)
                embeddings = channel_inference_neighbor_sampling(
                    model=encoder,
                    data=graph,
                )
            return embeddings

        return _encode(encoder, graph)

__all__ = ["BaseAdaptationTrainer"]
