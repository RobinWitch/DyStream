"""
Regression tests for FrameRenderer / visualization pipeline.
Requires GPU + models loaded.
"""

import numpy as np
import pytest
import torch

from streaming_app.utils.visualization import FrameRenderer


@pytest.fixture(scope="module")
def renderer(vis_models, device):
    return FrameRenderer(
        face_encoder=vis_models["face_encoder"],
        face_generator=vis_models["face_generator"],
        flow_estimator=vis_models["flow_estimator"],
        device=device,
    )


@pytest.fixture(scope="module")
def initialized_renderer(renderer, device):
    """Renderer with reference + anchor set (random, just for shape testing)."""
    torch.manual_seed(0)
    ref_image = torch.randn(1, 3, 512, 512, device=device)
    anchor = torch.randn(1, 1, 512, device=device)
    renderer.initialize_reference(ref_image, anchor)
    return renderer


class TestTensorToUint8:
    def test_tensor_to_uint8_range(self, renderer, device):
        tensor = torch.randn(1, 3, 64, 64, device=device)
        result = renderer._tensor_to_uint8(tensor)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_tensor_to_uint8_values_neg1(self, renderer, device):
        """Input all -1.0 → should map to 0."""
        tensor = torch.full((1, 3, 4, 4), -1.0, device=device)
        result = renderer._tensor_to_uint8(tensor)
        assert result.min() == 0
        assert result.max() == 0

    def test_tensor_to_uint8_values_pos1(self, renderer, device):
        """Input all +1.0 → should map to 255."""
        tensor = torch.full((1, 3, 4, 4), 1.0, device=device)
        result = renderer._tensor_to_uint8(tensor)
        assert result.min() == 255
        assert result.max() == 255

    def test_tensor_to_uint8_deterministic(self, renderer, device):
        torch.manual_seed(123)
        tensor = torch.randn(1, 3, 32, 32, device=device)
        r1 = renderer._tensor_to_uint8(tensor.clone())
        r2 = renderer._tensor_to_uint8(tensor.clone())
        np.testing.assert_array_equal(r1, r2)


class TestRenderFrameRaw:
    def test_render_frame_raw_shape(self, initialized_renderer, device):
        motion = torch.randn(1, 1, 512, device=device)
        w, h, rgb_bytes = initialized_renderer.render_frame_raw(motion)
        assert w == 512
        assert h == 512
        frame = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
        assert frame.shape == (512, 512, 3)
