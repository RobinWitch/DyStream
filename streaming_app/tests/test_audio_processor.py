"""
Regression tests for IncrementalAudioProcessor.
Requires GPU + models loaded.
"""

import numpy as np
import pytest
import torch

from streaming_app.models.audio_processor import IncrementalAudioProcessor


@pytest.fixture
def make_processor(dystream_model, device, config):
    """Factory to create audio processors with default config."""
    def _make(**overrides):
        kwargs = dict(
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
        kwargs.update(overrides)
        return IncrementalAudioProcessor(**kwargs)
    return _make


def _feed_until_output(processor, audio, chunk_size, mode="speaker"):
    """Feed audio chunks until processor returns features."""
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = processor.add_audio_chunk(chunk, mode=mode)
        if result is not None:
            return result
    return None


class TestAudioFeaturesDeterministic:
    """Same audio input → same features from two separate processors."""

    def test_audio_features_deterministic(self, make_processor, config):
        chunk_size = config.get("audio_chunk_size", 640)
        sr = config.get("audio_sample_rate", 16000)
        window_frames = config.get("audio_window_frames", 96)
        # Generate enough audio for one output
        np.random.seed(42)
        total = (window_frames + 5) * (sr // 25)
        audio = np.random.randn(total).astype(np.float32) * 0.1

        p1 = make_processor()
        p2 = make_processor()

        r1 = _feed_until_output(p1, audio, chunk_size, "speaker")
        r2 = _feed_until_output(p2, audio, chunk_size, "speaker")

        assert r1 is not None and r2 is not None, "Processor should produce output"
        feat_self_1, feat_other_1, _, _ = r1
        feat_self_2, feat_other_2, _, _ = r2

        torch.testing.assert_close(feat_self_1, feat_self_2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(feat_other_1, feat_other_2, atol=1e-5, rtol=1e-5)


class TestZeroAudioFeaturesConstant:
    """Zero audio → Wav2Vec2 output is the same every time (precondition for caching)."""

    def test_zero_audio_features_constant(self, make_processor, config, device):
        chunk_size = config.get("audio_chunk_size", 640)
        sr = config.get("audio_sample_rate", 16000)
        window_frames = config.get("audio_window_frames", 96)

        p = make_processor()
        total = (window_frames + 10) * (sr // 25)

        # Feed different non-zero audio but in speaker mode, audio_other is zero
        results = []
        for trial in range(2):
            p.reset()
            np.random.seed(trial + 100)
            audio = np.random.randn(total).astype(np.float32) * 0.1
            r = _feed_until_output(p, audio, chunk_size, "speaker")
            assert r is not None
            results.append(r[1])  # features_other (should be from zero audio)

        # Both features_other should be identical (same zero input)
        torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)


class TestBufferDuration:
    def test_buffer_duration_accurate(self, make_processor, config):
        sr = config.get("audio_sample_rate", 16000)
        chunk_size = config.get("audio_chunk_size", 640)
        chunk_ms = chunk_size / sr * 1000

        p = make_processor()
        chunk = np.zeros(chunk_size, dtype=np.float32)

        p.add_audio_chunk(chunk, mode="speaker")
        dur = p.get_buffer_duration_ms()
        assert abs(dur - chunk_ms) < 0.1, f"Expected ~{chunk_ms}ms, got {dur}ms"

        p.add_audio_chunk(chunk, mode="speaker")
        dur = p.get_buffer_duration_ms()
        assert abs(dur - 2 * chunk_ms) < 0.1


class TestFeatureShapes:
    def test_feature_shapes(self, make_processor, config):
        chunk_size = config.get("audio_chunk_size", 640)
        sr = config.get("audio_sample_rate", 16000)
        window_frames = config.get("audio_window_frames", 96)

        p = make_processor()
        total = (window_frames + 5) * (sr // 25)
        audio = np.zeros(total, dtype=np.float32)

        r = _feed_until_output(p, audio, chunk_size, "speaker")
        assert r is not None

        feat_self, feat_other, raw_self, raw_other = r
        assert feat_self.shape == (1, window_frames, 768)
        assert feat_other.shape == (1, window_frames, 768)
        assert raw_self.ndim == 1
        assert raw_other.ndim == 1
