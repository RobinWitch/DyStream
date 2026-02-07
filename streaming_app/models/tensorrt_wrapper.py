"""
TensorRT acceleration for DyStream streaming pipeline.

Two backends:
1. ONNX → native TRT engine (Wav2Vec2: standard ops, fixed input shape)
2. torch_tensorrt via torch.compile (face_generator, diffusion_head: 5D grid_sample etc.)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── ONNX → native TRT engine ───────────────────────────────────────────────


class TRTEngine:
    """Native TRT engine loaded from an .engine file."""

    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt

        self.device = device
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self._trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=device)

        # Discover I/O
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._output_shapes: List[Tuple[int, ...]] = []
        self._output_dtypes: List[torch.dtype] = []

        _dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
        }

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = _dtype_map.get(
                    self.engine.get_tensor_dtype(name), torch.float32
                )
                self._output_shapes.append(shape)
                self._output_dtypes.append(dtype)

        # Pre-allocate output buffers (reused across calls)
        self._outputs = [
            torch.empty(s, dtype=d, device=device)
            for s, d in zip(self._output_shapes, self._output_dtypes)
        ]

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        for name, inp in zip(self._input_names, inputs):
            self.context.set_tensor_address(name, inp.contiguous().data_ptr())

        for name, buf in zip(self._output_names, self._outputs):
            self.context.set_tensor_address(name, buf.data_ptr())

        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return self._outputs[0] if len(self._outputs) == 1 else tuple(self._outputs)


def export_onnx(
    model: nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    onnx_path: str,
    input_names: List[str],
    output_names: List[str],
    opset: int = 17,
):
    """Export a PyTorch module to ONNX using the legacy JIT-based exporter."""
    import onnx as _onnx
    import os

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Force legacy exporter (JIT-based, not onnxscript) for TRT compatibility
    old_val = os.environ.get("ONNX_EXPORT_LEGACY", None)
    os.environ["ONNX_EXPORT_LEGACY"] = "1"
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        if old_val is None:
            os.environ.pop("ONNX_EXPORT_LEGACY", None)
        else:
            os.environ["ONNX_EXPORT_LEGACY"] = old_val

    # Re-save with all weights embedded (in case of external data files)
    onnx_model = _onnx.load(onnx_path, load_external_data=True)
    _onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

    # Clean up stale external data files
    data_file = Path(onnx_path + ".data")
    if data_file.exists():
        data_file.unlink()

    logger.info(f"ONNX exported → {onnx_path}")


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: float = 4.0,
):
    """Build a native TRT engine from an ONNX file (cached to disk)."""
    import tensorrt as trt

    trt_log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_log)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_log)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    logger.info("Building TRT engine (may take minutes)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    logger.info(f"TRT engine saved → {engine_path}")


# ─── Wav2Vec2 wrappers ─────────────────────────────────────────────────────


class _Wav2VecForONNX(nn.Module):
    """Thin wrapper: returns tensor (not dict) so ONNX export succeeds."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)["high_level"]


class Wav2Vec2TRTWrapper(nn.Module):
    """
    Drop-in replacement for WrapedWav2Vec.
    Pads input to a fixed size, runs TRT engine, returns dict.
    Inherits nn.Module so it can be assigned via setattr on nn.Module parents.
    """

    def __init__(self, trt_engine: TRTEngine, fixed_len: int):
        super().__init__()
        self._engine = trt_engine
        self._fixed_len = fixed_len

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cur = x.shape[-1]
        if cur < self._fixed_len:
            x = F.pad(x, (0, self._fixed_len - cur))
        elif cur > self._fixed_len:
            x = x[..., : self._fixed_len]
        features = self._engine(x)
        # If padded, the extra frames at the end are harmless; the caller
        # trims with _trim_or_pad_features anyway.
        return {"high_level": features}


def build_wav2vec_trt(
    encoder: nn.Module,
    device: torch.device,
    cache_dir: str,
    name: str = "wav2vec",
    input_len: int = 62160,
    fp16: bool = True,
) -> Wav2Vec2TRTWrapper:
    """Export Wav2Vec2 encoder → ONNX → TRT → Wav2Vec2TRTWrapper."""
    onnx_path = str(Path(cache_dir) / f"{name}.onnx")
    engine_path = str(Path(cache_dir) / f"{name}.engine")

    if not Path(engine_path).exists():
        wrapper = _Wav2VecForONNX(encoder)
        dummy = torch.randn(1, input_len, device=device)
        if not Path(onnx_path).exists():
            export_onnx(
                wrapper,
                (dummy,),
                onnx_path,
                input_names=["audio"],
                output_names=["features"],
            )
        build_trt_engine(onnx_path, engine_path, fp16=fp16)

    engine = TRTEngine(engine_path, device)

    # Verify correctness
    dummy = torch.randn(1, input_len, device=device)
    with torch.no_grad():
        ref_out = encoder(dummy)["high_level"]
        trt_out = engine(dummy)
    diff = (ref_out.float() - trt_out.float()).abs().max().item()
    logger.info(f"Wav2Vec2 TRT ({name}) max diff = {diff:.6f}")
    if diff > 0.05:
        logger.warning(f"Wav2Vec2 TRT diff too large ({diff:.4f}), falling back")
        raise RuntimeError(f"TRT accuracy check failed: diff={diff:.4f}")

    return Wav2Vec2TRTWrapper(engine, fixed_len=input_len)


# ─── torch_tensorrt (for face_generator, diffusion_head) ────────────────────


# ─── FaceMesh TRT ─────────────────────────────────────────────────────────


class FaceMeshTRT:
    """
    TRT-accelerated face landmark detector.
    Input:  GPU tensor (1, 3, H, W) in [-1, 1]
    Output: landmarks (478, 3), confidence float
    """

    def __init__(self, engine_path: str, device: torch.device):
        self._engine = TRTEngine(engine_path, device)
        self._device = device

    @torch.no_grad()
    def __call__(self, frame_tensor: torch.Tensor):
        """
        Args:
            frame_tensor: (1, 3, H, W) float32 in [-1, 1] on GPU
        Returns:
            landmarks: (478, 3) float32 — normalized x, y, z
            confidence: float — face detection score
        """
        # [-1, 1] → [0, 1]
        x = (frame_tensor + 1.0) * 0.5
        # Resize to 256×256
        if x.shape[-2] != 256 or x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        # Run TRT
        landmarks_raw, confidence, extra = self._engine(x)
        # (1, 1, 1, 1434) → (478, 3)
        landmarks = landmarks_raw.view(478, 3)
        return landmarks, confidence.item()


def load_facemesh_trt(
    engine_path: str,
    device: torch.device,
    onnx_path: Optional[str] = None,
) -> FaceMeshTRT:
    """Load or build FaceMesh TRT engine."""
    if not Path(engine_path).exists():
        if onnx_path and Path(onnx_path).exists():
            logger.info(f"Building FaceMesh TRT engine from {onnx_path}")
            build_trt_engine(onnx_path, engine_path, fp16=True)
        else:
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}. "
                f"Build it first or provide onnx_path."
            )
    return FaceMeshTRT(engine_path, device)


class BlendShapeTRT:
    """
    TRT-accelerated blendshape estimator.
    Input:  landmarks (478, 3) from FaceMeshTRT
    Output: 52 ARKit blendshape coefficients
    """

    # 146 landmark indices used by the blendshape model (MediaPipe V2 subset)
    # Source: kLandmarksSubsetIdxs in mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc
    LANDMARK_INDICES = [
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53,
        54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91,
        93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148,
        149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168,
        172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263,
        267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297,
        300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336,
        338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
        384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415,
        454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ]

    def __init__(self, engine_path: str, device: torch.device):
        self._engine = TRTEngine(engine_path, device)
        self._device = device
        self._indices = torch.tensor(self.LANDMARK_INDICES, dtype=torch.long, device=device)

    @torch.no_grad()
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmarks: (478, 3) from FaceMeshTRT
        Returns:
            blendshapes: (52,) float32 coefficients
        """
        # Select 146 landmarks, take only (x, y)
        selected = landmarks[self._indices, :2]  # (146, 2)
        inp = selected.unsqueeze(0)  # (1, 146, 2)
        return self._engine(inp)


def load_blendshape_trt(
    engine_path: str,
    device: torch.device,
    onnx_path: Optional[str] = None,
) -> BlendShapeTRT:
    """Load or build BlendShape TRT engine."""
    if not Path(engine_path).exists():
        if onnx_path and Path(onnx_path).exists():
            logger.info(f"Building BlendShape TRT engine from {onnx_path}")
            build_trt_engine(onnx_path, engine_path, fp16=True)
        else:
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}. "
                f"Build it first or provide onnx_path."
            )
    return BlendShapeTRT(engine_path, device)


def compile_with_torch_tensorrt(
    model: nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    fp16: bool = True,
    cache_path: Optional[str] = None,
) -> nn.Module:
    """
    Compile a PyTorch module using torch_tensorrt as torch.compile backend.
    Handles 5D grid_sample, custom ops, etc. that ONNX cannot export.
    """
    import torch_tensorrt  # noqa: F401

    precisions = {torch.float32}
    if fp16:
        precisions.add(torch.float16)

    compiled = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": precisions,
            "truncate_double": True,
            "min_block_size": 1,
            "use_python_runtime": True,
        },
    )

    # Trigger TRT engine build (first forward)
    logger.info("Triggering TRT engine build (first forward)…")
    with torch.no_grad():
        _ = compiled(*sample_inputs)
    logger.info("torch_tensorrt compile done")

    return compiled
