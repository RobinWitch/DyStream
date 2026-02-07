"""
Visualization utilities for rendering motion latents to video frames.
Adapted from tools/visualization_0416/ for streaming use.
"""

import torch
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CombinedRenderPipeline(torch.nn.Module):
    """Wraps flow_estimator + face_generator so they can be torch.compiled as one graph."""

    def __init__(self, flow_estimator, face_generator):
        super().__init__()
        self.flow_estimator = flow_estimator
        self.face_generator = face_generator

    def forward(self, anchor_motion, motion_latent, face_features):
        optical_flow = self.flow_estimator(anchor_motion, motion_latent)
        rgb_frame = self.face_generator(optical_flow, face_features)
        return rgb_frame


class FrameRenderer:
    """Renders motion latents to video frames."""

    def __init__(
        self,
        face_encoder,
        face_generator,
        flow_estimator,
        device: torch.device,
        jpeg_quality: int = 80,
        combined_pipeline=None,
    ):
        self.face_encoder = face_encoder
        self.face_generator = face_generator
        self.flow_estimator = flow_estimator
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.combined_pipeline = combined_pipeline

        # Cached values for efficiency
        self.cached_face_features = None
        self.cached_reference_image = None
        self.anchor_motion = None  # Set by initialize_reference()

    def render_frame(
        self,
        motion_latent: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> bytes:
        """
        Render a single motion latent to JPEG image.

        Args:
            motion_latent: Motion latent tensor (1, 1, 512) or (1, 512)
            reference_image: Optional reference image tensor

        Returns:
            JPEG bytes
        """
        with torch.no_grad():
            # Ensure correct shape
            if motion_latent.dim() == 2:
                motion_latent = motion_latent.unsqueeze(1)  # (1, 1, 512)

            # Generate frame
            # Note: This is a simplified placeholder - actual implementation
            # depends on the visualization model's forward signature
            frame = self._generate_frame(motion_latent, reference_image)

            # Convert to PIL Image
            frame_pil = self._tensor_to_pil(frame)

            # Encode to JPEG
            jpeg_bytes = self._encode_jpeg(frame_pil)

            return jpeg_bytes

    def render_frame_raw(
        self,
        motion_latent: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Render a single motion latent to raw RGB bytes.

        Returns:
            (width, height, rgb_bytes)
        """
        with torch.no_grad():
            if motion_latent.dim() == 2:
                motion_latent = motion_latent.unsqueeze(1)  # (1, 1, 512)

            frame = self._generate_frame(motion_latent, reference_image)
            rgb = self._tensor_to_uint8(frame)
            h, w, _ = rgb.shape
            return w, h, rgb.tobytes()

    def render_frame_numpy(
        self,
        motion_latent: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Render a single motion latent to numpy uint8 array (H, W, 3).
        Avoids the tobytes()/frombuffer() round-trip of render_frame_raw.
        """
        with torch.no_grad():
            if motion_latent.dim() == 2:
                motion_latent = motion_latent.unsqueeze(1)
            frame = self._generate_frame(motion_latent, reference_image)
            return self._tensor_to_uint8(frame)

    def render_frame_gpu(
        self,
        motion_latent: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Render and return both GPU tensor and numpy uint8.

        Returns:
            (gpu_tensor (1,3,H,W) in [-1,1], numpy_uint8 (H,W,3))
        """
        with torch.no_grad():
            if motion_latent.dim() == 2:
                motion_latent = motion_latent.unsqueeze(1)
            frame = self._generate_frame(motion_latent, reference_image)
            return frame, self._tensor_to_uint8(frame)

    def initialize_reference(
        self,
        reference_image: torch.Tensor,
        anchor_motion: torch.Tensor
    ):
        """
        Initialize reference image and anchor motion for frame generation.

        Args:
            reference_image: Reference image tensor (1, 3, 512, 512)
            anchor_motion: Anchor motion latent (1, T, 512) or (T, 512)
        """
        self.cached_reference_image = reference_image

        # Ensure anchor_motion has correct shape and device
        if anchor_motion.dim() == 2:
            anchor_motion = anchor_motion.unsqueeze(0)  # (1, T, 512)

        # Use first frame as anchor, ensure on correct device
        self.anchor_motion = anchor_motion[:, 0:1, :].to(self.device)  # (1, 1, 512)

        # Pre-encode reference image features
        with torch.no_grad():
            self.cached_face_features = self.face_encoder(reference_image)

        logger.info(f"Initialized reference: anchor_motion={self.anchor_motion.shape}, "
                   f"face_features={self.cached_face_features.shape}")

    def _generate_frame(
        self,
        motion_latent: torch.Tensor,
        reference_image: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate RGB frame from motion latent using visualization pipeline.
        Reference: app.py:380-407 (latents_to_video_frames)

        Args:
            motion_latent: Motion latent (1, 1, 512)
            reference_image: Reference image (optional, will use cached if None)

        Returns:
            RGB frame tensor (1, 3, 512, 512)
        """
        # 1. Update reference image if provided (skip if already cached)
        if reference_image is not None and self.cached_reference_image is None:
            logger.info("Updating reference image and face features")
            self.cached_reference_image = reference_image
            with torch.no_grad():
                self.cached_face_features = self.face_encoder(reference_image)

        # 2. Check that anchor motion is set
        if self.anchor_motion is None:
            logger.error("Anchor motion not set. Call initialize_reference() first.")
            # Return a black frame as fallback
            return torch.zeros(1, 3, 512, 512, device=self.device)

        # 3. Check that face features are cached
        if self.cached_face_features is None:
            logger.error("Face features not cached")
            return torch.zeros(1, 3, 512, 512, device=self.device)

        # 4. Ensure motion_latent has correct shape and device
        if motion_latent.dim() == 2:
            motion_latent = motion_latent.unsqueeze(1)  # (1, 1, 512)
        motion_latent = motion_latent.to(self.device)

        # 5-6. Render: flow estimation + face generation
        with torch.no_grad():
            if self.combined_pipeline is not None:
                rgb_frame = self.combined_pipeline(
                    self.anchor_motion, motion_latent, self.cached_face_features
                )
            else:
                optical_flow = self.flow_estimator(
                    self.anchor_motion, motion_latent
                )
                rgb_frame = self.face_generator(
                    optical_flow, self.cached_face_features
                )

        return rgb_frame

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image.

        Args:
            tensor: Image tensor (1, 3, H, W) in range [-1, 1]

        Returns:
            PIL Image
        """
        np_image = self._tensor_to_uint8(tensor)
        return Image.fromarray(np_image)

    def _tensor_to_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to uint8 RGB array (H, W, 3).
        Performs [-1,1]→[0,255] on GPU then transfers the smaller uint8 tensor.
        """
        tensor = tensor.squeeze(0)
        # GPU-side: [-1,1] → [0,255] as uint8 (transfers 0.75MB vs 3MB float32)
        tensor = ((tensor + 1.0) * 127.5).clamp(0, 255).byte()
        return tensor.permute(1, 2, 0).cpu().numpy()

    def _encode_jpeg(self, image: Image.Image) -> bytes:
        """
        Encode PIL Image to JPEG bytes.

        Args:
            image: PIL Image

        Returns:
            JPEG bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
        jpeg_bytes = buffer.getvalue()
        return jpeg_bytes

    def render_multiple_frames(
        self,
        motion_latents: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> list:
        """
        Render multiple motion latents to JPEG images.

        Args:
            motion_latents: Motion latents (1, T, 512)
            reference_image: Optional reference image

        Returns:
            List of JPEG bytes
        """
        frames = []

        for i in range(motion_latents.shape[1]):
            motion_latent = motion_latents[:, i:i+1, :]  # (1, 1, 512)
            frame_bytes = self.render_frame(motion_latent, reference_image)
            frames.append(frame_bytes)

        return frames

    def set_jpeg_quality(self, quality: int):
        """Set JPEG encoding quality (0-100)."""
        self.jpeg_quality = max(0, min(100, quality))
        logger.info(f"JPEG quality set to {self.jpeg_quality}")


def load_reference_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess reference image.

    Args:
        image_path: Path to image file
        device: Target device

    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Resize to 512x512
    image = image.resize((512, 512), Image.LANCZOS)

    # Convert to tensor
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize to [-1, 1]
    image_np = image_np * 2 - 1

    # Convert to torch tensor (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)

    return image_tensor


def process_uploaded_image(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    """
    Process uploaded image bytes.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG)
        device: Target device

    Returns:
        Preprocessed image tensor
    """
    # Load from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Resize to 512x512
    image = image.resize((512, 512), Image.LANCZOS)

    # Convert to tensor
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize to [-1, 1]
    image_np = image_np * 2 - 1

    # Convert to torch tensor (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)

    return image_tensor
