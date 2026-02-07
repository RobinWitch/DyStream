"""
Streaming inference engine for DyStream.
Aligned with app.py raw-audio inference path.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StreamingInferenceEngine:
    """
    Streaming inference engine for real-time motion generation.
    Uses one_clip_only_inference to generate one frame per chunk.
    """

    def __init__(
        self,
        dystream_model,
        device: torch.device,
        config: Dict[str, Any],
        noise_scheduler=None,
        ema=None
    ):
        self.model = dystream_model
        self.device = device
        self.config = config

        # Generation parameters
        self.denoising_steps = config.get('denoising_steps', 5)
        self.inpainting_length = getattr(dystream_model, 'inpainting_length', config.get('inpainting_length', 94))
        self.motion_dim = 512  # Motion latent dimension

        # Audio parameters
        self.audio_sr = config.get('audio_sample_rate', 16000)
        self.pose_fps = config.get('target_fps', 25)

        # KV-cache for GPT autoregressive loop
        self.model._use_kv_cache = config.get('use_kv_cache', True)
        # torch.compile for GPT block loop (requires use_kv_cache=True)
        self.model._use_torch_compile_gpt_loop = config.get('use_torch_compile_gpt_loop', False)

        # CFG parameters
        self.cfg_audio = config.get('cfg_audio', 0.5)
        self.cfg_audio_other = config.get('cfg_audio_other', 0.5)
        self.cfg_anchor = config.get('cfg_anchor', 0.0)
        self.cfg_all = config.get('cfg_all', 1.0)

        # Noise scheduler / EMA (from model_loader)
        self.noise_scheduler = noise_scheduler
        self.ema = ema

        # Anchor motion latent (1, 1, 512)
        self.motion_anchor: Optional[torch.Tensor] = None
        # Rolling past motion (1, inpainting_length, 512)
        self.past_motion: Optional[torch.Tensor] = None

    def initialize_context(self, motion_anchor):
        """
        Initialize anchor motion (single frame).

        Args:
            motion_anchor: Motion latent as numpy (512,)/(1,512)/(T,512) or torch.Tensor
        """
        if motion_anchor is None:
            raise ValueError("motion_anchor is None")

        if isinstance(motion_anchor, torch.Tensor):
            t = motion_anchor.detach().float()
        else:
            t = torch.as_tensor(np.asarray(motion_anchor), dtype=torch.float32)

        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[0] > 1:
            t = t[:1]

        anchor_tensor = t.unsqueeze(0).to(self.device)  # (1, 1, 512)
        self.motion_anchor = anchor_tensor
        self.past_motion = anchor_tensor.repeat(1, self.inpainting_length, 1)

        logger.info(f"Initialized motion anchor with shape: {self.motion_anchor.shape}")

    def generate_next_frame(
        self,
        audio_self_features: torch.Tensor,
        audio_other_features: torch.Tensor,
        audio_self_raw: np.ndarray,
        audio_other_raw: np.ndarray,
        mode: str = 'speaker'
    ) -> torch.Tensor:
        """
        Generate next motion frame given precomputed audio features.

        Args:
            audio_self_features: Audio features for self (1, T, 768)
            audio_other_features: Audio features for other (1, T, 768)
            audio_self_raw: Raw audio samples for self (N,)
            audio_other_raw: Raw audio samples for other (N,)
            mode: 'speaker' or 'listener'

        Returns:
            Next motion latent (1, 1, 512)
        """
        if self.motion_anchor is None or self.past_motion is None:
            raise RuntimeError("Motion anchor not initialized. Call initialize_context() first.")

        # Convert raw audio to tensors (for compatibility with one_clip_only_inference)
        audio_self_tensor = torch.as_tensor(audio_self_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio_other_tensor = torch.as_tensor(audio_other_raw, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Override CFG parameters (same as app.py)
        self.model.cfg_audio = self.cfg_audio
        self.model.cfg_audio_other = self.cfg_audio_other
        self.model.cfg_anchor = self.cfg_anchor
        self.model.cfg_all = self.cfg_all

        # Run inference (same as app.py)
        if self.ema is not None:
            self.ema.to(self.device)
            ctx = self.ema.average_parameters(self.model.parameters())
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx, torch.no_grad():
            motion_latent_pred = self.model.one_clip_only_inference(
                per_compute_audio_feature=audio_self_features,
                per_compute_audio_other_feature=audio_other_features,
                audio_self=audio_self_tensor,
                past_audio_self=None,
                audio_other=audio_other_tensor,
                past_audio_other=None,
                past_motion=self.past_motion,
                gen_frames=1,
                anchor_latent=self.motion_anchor,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=int(self.denoising_steps),
            )

            if motion_latent_pred is None or motion_latent_pred.shape[1] == 0:
                return self.motion_anchor.clone()

            # Update past motion (rolling)
            self.past_motion = torch.cat([self.past_motion, motion_latent_pred], dim=1)
            self.past_motion = self.past_motion[:, -self.inpainting_length:]

            return motion_latent_pred[:, -1:, :]

    def update_config(self, new_config: Dict[str, Any]):
        """Update generation parameters in real-time."""
        if 'denoising_steps' in new_config:
            self.denoising_steps = new_config['denoising_steps']
            logger.info(f"Updated denoising steps to: {self.denoising_steps}")

        if 'cfg_audio' in new_config:
            self.cfg_audio = new_config['cfg_audio']
        if 'cfg_audio_other' in new_config:
            self.cfg_audio_other = new_config['cfg_audio_other']
        if 'cfg_anchor' in new_config:
            self.cfg_anchor = new_config['cfg_anchor']
        if 'cfg_all' in new_config:
            self.cfg_all = new_config['cfg_all']

    def reset(self):
        """Reset the inference engine state."""
        self.motion_anchor = None
        self.past_motion = None
        logger.info("Inference engine reset")
