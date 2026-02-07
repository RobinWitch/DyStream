"""
Incremental audio processor for streaming inference.
Buffers raw audio chunks with a sliding window for app.py-aligned inference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IncrementalAudioProcessor:
    """
    Processes audio chunks incrementally with a sliding window.
    Maintains a raw-audio buffer for short-window inference.
    """

    def __init__(
        self,
        dystream_model,
        device: torch.device,
        sample_rate: int = 16000,
        window_frames: int = 96,   # matches cbh_window_length
        chunk_size: int = 640,     # 40ms at 16kHz
        target_fps: int = 25,
        lookahead_ms_speaker: int = 60,
        lookahead_ms_listener: int = 0,
        update_every_n_chunks: int = 1,
        wav2vec_trt: dict = None,
    ):
        self.dystream_model = dystream_model
        # TRT encoder overrides (avoids modifying dystream model parameters)
        self._wav2vec_trt = wav2vec_trt or {}
        self.device = device
        self.sample_rate = sample_rate
        self.window_frames = window_frames
        self.chunk_size = chunk_size
        self.target_fps = target_fps
        self.update_every_n_chunks = max(1, int(update_every_n_chunks))

        self.frame_samples = int(self.sample_rate / self.target_fps)
        self.frame_ms = 1000.0 / self.target_fps
        # Use floor to avoid exceeding latency target
        self.lookahead_frames_speaker = int(lookahead_ms_speaker // self.frame_ms)
        self.lookahead_frames_listener = int(lookahead_ms_listener // self.frame_ms)

        max_lookahead_frames = max(self.lookahead_frames_speaker, self.lookahead_frames_listener)
        self.window_samples = self.window_frames * self.frame_samples
        self.max_lookahead_samples = max_lookahead_frames * self.frame_samples

        # Audio ring buffer (pre-allocated numpy array)
        self._buf_capacity = self.window_samples + self.max_lookahead_samples
        self._buf = np.zeros(self._buf_capacity, dtype=np.float32)
        self._buf_len = 0  # how many valid samples in buffer

        # Track how much audio we've processed
        self.total_samples_processed = 0
        self._chunk_counter = {"speaker": 0, "listener": 0}
        self._last_features = {"speaker": None, "listener": None}

        # Cache for zero-audio Wav2Vec2 features (identical every call)
        self._zero_features_cache = {}

    def add_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        mode: str = 'speaker'
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]:
        """
        Add an audio chunk and return audio features if enough audio accumulated.

        Args:
            audio_chunk: Audio samples (640 samples for 40ms)
            mode: 'speaker' or 'listener' - determines audio routing

        Returns:
            Tuple of (audio_self_features, audio_other_features, audio_self_raw, audio_other_raw) or None
        """
        if audio_chunk is None:
            return None

        # Ensure numpy array
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        else:
            audio_chunk = audio_chunk.astype(np.float32, copy=False)

        # Flatten to 1D if needed
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.reshape(-1)

        # Add chunk to ring buffer
        n = len(audio_chunk)
        if self._buf_len + n > self._buf_capacity:
            # Shift left to make room (drop oldest samples)
            shift = self._buf_len + n - self._buf_capacity
            self._buf[:self._buf_len - shift] = self._buf[shift:self._buf_len]
            self._buf_len -= shift
        self._buf[self._buf_len:self._buf_len + n] = audio_chunk
        self._buf_len += n
        self.total_samples_processed += n

        # Check if we have enough audio for processing
        min_ready_samples = self.chunk_size
        if mode == 'speaker':
            min_ready_samples += self.lookahead_frames_speaker * self.frame_samples
        else:
            min_ready_samples += self.lookahead_frames_listener * self.frame_samples
        if self._buf_len < min_ready_samples:
            return None

        lookahead_frames = self.lookahead_frames_speaker if mode == 'speaker' else self.lookahead_frames_listener
        lookahead_samples = lookahead_frames * self.frame_samples
        total_needed = self.window_samples + lookahead_samples

        # Extract raw audio window (pad left with zeros if insufficient history)
        if self._buf_len < total_needed:
            audio_window = np.zeros(total_needed, dtype=np.float32)
            audio_window[total_needed - self._buf_len:] = self._buf[:self._buf_len]
        else:
            audio_window = self._buf[self._buf_len - total_needed:self._buf_len].copy()

        if lookahead_samples > 0:
            audio_window_no_lookahead = audio_window[:-lookahead_samples]
        else:
            audio_window_no_lookahead = audio_window

        # Route audio based on mode
        if mode == 'speaker':
            # AI acts as speaker: user audio -> audio_self
            audio_self_window = audio_window_no_lookahead
            audio_other_window = np.zeros_like(audio_window_no_lookahead)
            audio_self_for_features = audio_window
            audio_other_for_features = np.zeros_like(audio_window)
        else:  # listener
            # AI acts as listener: user audio -> audio_other
            audio_self_window = np.zeros_like(audio_window_no_lookahead)
            audio_other_window = audio_window_no_lookahead
            audio_self_for_features = np.zeros_like(audio_window)
            audio_other_for_features = audio_window

        # Extract features (using model's audio encoder)
        self._chunk_counter[mode] += 1
        cached = self._last_features.get(mode)
        need_update = (self._chunk_counter[mode] % self.update_every_n_chunks == 0) or (cached is None)
        if need_update:
            # In speaker mode, audio_other is all-zero → use cache
            # In listener mode, audio_self is all-zero → use cache
            enc_self = self._get_encoder('audio_encoder_face')
            enc_other = self._get_encoder('audio_encoder_face_other')
            if mode == 'speaker':
                features_self = self._extract_features(audio_self_for_features, enc_self)
                features_self = self._trim_or_pad_features(features_self, self.window_frames)
                features_other = self._get_zero_features(enc_other, "other")
            else:
                features_self = self._get_zero_features(enc_self, "self")
                features_other = self._extract_features(audio_other_for_features, enc_other)
                features_other = self._trim_or_pad_features(features_other, self.window_frames)

            self._last_features[mode] = (features_self, features_other)
        else:
            features_self, features_other = cached

        return features_self, features_other, audio_self_window, audio_other_window

    def _get_encoder(self, attr_name: str):
        """Get encoder, preferring TRT override if available."""
        if attr_name in self._wav2vec_trt:
            return self._wav2vec_trt[attr_name]
        return getattr(self.dystream_model, attr_name)

    def _get_zero_features(self, encoder, cache_key: str) -> torch.Tensor:
        """Return cached Wav2Vec2 features for an all-zero audio window."""
        if cache_key not in self._zero_features_cache:
            zero_audio = np.zeros(self.window_samples + self.max_lookahead_samples, dtype=np.float32)
            features = self._extract_features(zero_audio, encoder)
            features = self._trim_or_pad_features(features, self.window_frames)
            self._zero_features_cache[cache_key] = features
        return self._zero_features_cache[cache_key]

    def _get_audio_processor(self):
        if hasattr(self.dystream_model, 'audio_processor'):
            return self.dystream_model.audio_processor
        from transformers import Wav2Vec2Processor
        if not hasattr(self, '_audio_processor'):
            self._audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        return self._audio_processor

    def _extract_features(self, audio_window: np.ndarray, encoder) -> torch.Tensor:
        """
        Extract Wav2Vec2 features from raw audio window.
        Returns tensor of shape (1, T, 768) on device.
        """
        with torch.no_grad():
            processor = self._get_audio_processor()
            inputs = processor(
                [audio_window],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            padded_input = torch.concat(
                [inputs.input_values, torch.zeros([1, 80], device=self.device)],
                dim=-1
            )

            features = encoder(padded_input)["high_level"]

            # Downsample to target_fps
            features = F.interpolate(
                features.transpose(1, 2),
                scale_factor=(self.target_fps / 50),
                mode="linear",
                align_corners=True
            ).transpose(1, 2)

            return features

    def _trim_or_pad_features(self, features: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = features.shape[1]
        if current_len == target_len:
            return features
        if current_len > target_len:
            return features[:, -target_len:]
        # pad with zeros
        pad = torch.zeros(
            features.shape[0], target_len - current_len, features.shape[2],
            device=features.device, dtype=features.dtype
        )
        return torch.cat([features, pad], dim=1)

    def reset(self):
        """Reset the audio processor state."""
        self._buf[:] = 0
        self._buf_len = 0
        self.total_samples_processed = 0
        self._chunk_counter = {"speaker": 0, "listener": 0}
        self._last_features = {"speaker": None, "listener": None}

    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        return (self._buf_len / self.sample_rate) * 1000

    def is_buffer_ready(self) -> bool:
        """Check if buffer has enough audio for processing."""
        return self._buf_len >= self.window_samples
