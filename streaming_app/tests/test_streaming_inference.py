"""
Regression tests for StreamingInferenceEngine.
Requires GPU + models loaded.
"""

import numpy as np
import pytest
import torch

from streaming_app.models.audio_processor import IncrementalAudioProcessor
from streaming_app.models.streaming_inference import StreamingInferenceEngine


@pytest.fixture
def engine(dystream_model, device, config, noise_scheduler, ema):
    return StreamingInferenceEngine(
        dystream_model=dystream_model,
        device=device,
        config=config,
        noise_scheduler=noise_scheduler,
        ema=ema,
    )


@pytest.fixture
def audio_processor(dystream_model, device, config):
    return IncrementalAudioProcessor(
        dystream_model=dystream_model,
        device=device,
        sample_rate=config.get("audio_sample_rate", 16000),
        window_frames=config.get("audio_window_frames", 96),
        chunk_size=config.get("audio_chunk_size", 640),
        target_fps=config.get("target_fps", 25),
        lookahead_ms_speaker=config.get("audio_lookahead_ms_speaker", 60),
        lookahead_ms_listener=config.get("audio_lookahead_ms_listener", 0),
        update_every_n_chunks=config.get("audio_feature_update_every_n_chunks", 1),
    )


def _get_features(audio_processor, config, mode="speaker"):
    """Feed enough zero audio to get one set of features."""
    sr = config.get("audio_sample_rate", 16000)
    chunk_size = config.get("audio_chunk_size", 640)
    window_frames = config.get("audio_window_frames", 96)
    total = (window_frames + 5) * (sr // 25)
    audio = np.zeros(total, dtype=np.float32)
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = audio_processor.add_audio_chunk(chunk, mode=mode)
        if result is not None:
            return result
    return None


class TestInitializeContext:
    def test_initialize_context_shapes(self, engine, device):
        anchor = np.random.randn(512).astype(np.float32)
        engine.initialize_context(anchor)

        assert engine.motion_anchor is not None
        assert engine.motion_anchor.shape == (1, 1, 512)
        assert engine.motion_anchor.device.type == device.type

        assert engine.past_motion is not None
        assert engine.past_motion.shape == (1, engine.inpainting_length, 512)

    def test_initialize_context_from_tensor(self, engine, device):
        anchor = torch.randn(1, 512, device=device)
        engine.initialize_context(anchor)

        assert engine.motion_anchor.shape == (1, 1, 512)

    def test_initialize_context_multi_frame(self, engine, device):
        """Multi-frame anchor → uses first frame only."""
        anchor = np.random.randn(10, 512).astype(np.float32)
        engine.initialize_context(anchor)
        assert engine.motion_anchor.shape == (1, 1, 512)


class TestGenerateNextFrame:
    def test_generate_next_frame_deterministic(self, engine, audio_processor, config, device):
        """Same inputs → same motion output (with manual seeding)."""
        anchor = np.zeros(512, dtype=np.float32)
        r = _get_features(audio_processor, config, "speaker")
        assert r is not None
        feat_self, feat_other, raw_self, raw_other = r

        results = []
        for _ in range(2):
            engine.initialize_context(anchor)
            torch.manual_seed(0)
            motion = engine.generate_next_frame(
                feat_self, feat_other, raw_self, raw_other, "speaker"
            )
            results.append(motion.clone())

        torch.testing.assert_close(results[0], results[1], atol=1e-4, rtol=1e-4)

    def test_generate_next_frame_shape(self, engine, audio_processor, config, device):
        anchor = np.zeros(512, dtype=np.float32)
        engine.initialize_context(anchor)

        r = _get_features(audio_processor, config, "speaker")
        assert r is not None
        feat_self, feat_other, raw_self, raw_other = r

        motion = engine.generate_next_frame(
            feat_self, feat_other, raw_self, raw_other, "speaker"
        )
        assert motion.shape == (1, 1, 512)
        assert motion.device.type == device.type
