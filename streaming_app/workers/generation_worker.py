"""
Background worker for asynchronous motion generation and frame rendering.
Separates heavy computation from WebSocket audio reception.

Supports two modes:
1. Async mode (WebSocket): 3 asyncio tasks with to_thread for GPU work.
2. GPU thread mode (WebRTC): single dedicated thread, reads from
   StreamingState.audio_queue, writes to StreamingState.frame_slot.
"""

import asyncio
import base64
import io
import queue
import threading
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
import time

from PIL import Image

logger = logging.getLogger(__name__)


class GenerationWorker:
    """
    Background worker for motion generation and frame rendering.
    Runs in separate asyncio task to prevent blocking audio reception.
    """

    def __init__(
        self,
        inference_engine,
        frame_renderer,
        audio_processor,
        session_id: str,
        facemesh=None,
        blendshape=None,
        jpeg_quality: int = 80,
        state=None,
    ):
        self.inference_engine = inference_engine
        self.frame_renderer = frame_renderer
        self.audio_processor = audio_processor
        self.session_id = session_id
        self.facemesh = facemesh
        self.blendshape = blendshape
        self.jpeg_quality = jpeg_quality
        self.device = getattr(inference_engine, "device", None)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda and self.device is None:
            self.device = torch.device("cuda")

        # WebRTC shared state (None for WebSocket mode)
        self.state = state

        # Queues for async processing (WebSocket mode only)
        self.audio_queue = asyncio.Queue(maxsize=10)    # Audio chunks to process
        self.feature_queue = asyncio.Queue(maxsize=10)  # Extracted audio features
        self.motion_queue = asyncio.Queue(maxsize=10)   # Motion latents
        self.frame_queue = asyncio.Queue(maxsize=5)     # (jpeg_base64, timestamp) tuples

        # Control flags
        self.running = False
        self.task = None
        self._tasks = []
        self._gpu_thread: Optional[threading.Thread] = None

        # Metrics
        self.frames_generated = 0
        self.total_audio_time = 0.0
        self.total_motion_time = 0.0
        self.total_frame_time = 0.0
        self.total_mesh_time = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Async mode (WebSocket) — existing implementation
    # ──────────────────────────────────────────────────────────────────

    async def start(self):
        """Start the worker task (async/WebSocket mode)."""
        if self.running:
            logger.warning(f"Worker for session {self.session_id} already running")
            return

        self.running = True
        # Pipeline tasks
        self._tasks = [
            asyncio.create_task(self._audio_loop()),
            asyncio.create_task(self._motion_loop()),
            asyncio.create_task(self._frame_loop()),
        ]
        logger.info(f"Worker started for session {self.session_id}")

    async def stop(self):
        """Stop the worker task (async/WebSocket mode)."""
        if not self.running:
            return

        self.running = False

        # Wait for task to complete
        if self._tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"Worker task timeout for session {self.session_id}")
                for t in self._tasks:
                    t.cancel()

        self._log_final_stats()

    async def queue_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        mode: str = 'speaker'
    ):
        """Queue an audio chunk for processing."""
        try:
            await asyncio.wait_for(
                self.audio_queue.put((audio_chunk, mode)),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            logger.warning(f"Audio queue full for session {self.session_id}, dropping chunk")

    async def get_next_frame(self, timeout: float = 0.5) -> Optional[Tuple[str, float]]:
        """
        Get the next generated frame.

        Returns:
            (jpeg_base64, timestamp) tuple or None if timeout
        """
        try:
            result = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return None

    async def _audio_loop(self):
        """Audio chunk -> features."""
        logger.info(f"Audio loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    audio_chunk, mode = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()
                result = await asyncio.to_thread(self.audio_processor.add_audio_chunk, audio_chunk, mode)
                audio_time = time.time() - t0
                self.total_audio_time += audio_time

                if result is None:
                    continue

                try:
                    await asyncio.wait_for(self.feature_queue.put((result, mode)), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Feature queue full for session {self.session_id}, dropping features")
            except Exception as e:
                logger.error(f"Error in audio loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Audio loop stopped for session {self.session_id}")

    async def _motion_loop(self):
        """Features -> motion latent."""
        logger.info(f"Motion loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    (audio_self_features, audio_other_features, audio_self_raw, audio_other_raw), mode = await asyncio.wait_for(
                        self.feature_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()
                motion_latent = await asyncio.to_thread(
                    self.inference_engine.generate_next_frame,
                    audio_self_features,
                    audio_other_features,
                    audio_self_raw,
                    audio_other_raw,
                    mode
                )
                motion_time = time.time() - t0
                self.total_motion_time += motion_time

                try:
                    await asyncio.wait_for(self.motion_queue.put(motion_latent), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Motion queue full for session {self.session_id}, dropping motion")
            except Exception as e:
                logger.error(f"Error in motion loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Motion loop stopped for session {self.session_id}")

    async def _frame_loop(self):
        """Motion latent -> JPEG base64."""
        logger.info(f"Frame loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    motion_latent = await asyncio.wait_for(self.motion_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()

                def _run_render():
                    # Render frame (GPU tensor + numpy)
                    frame_gpu, frame_np = self.frame_renderer.render_frame_gpu(motion_latent)

                    # FaceMesh + BlendShape (metrics only)
                    mesh_time = 0.0
                    if self.facemesh is not None:
                        t_mesh = time.time()
                        landmarks, confidence = self.facemesh(frame_gpu)
                        if self.blendshape is not None:
                            blendshapes = self.blendshape(landmarks)
                        mesh_time = time.time() - t_mesh

                    # numpy -> JPEG -> base64
                    img = Image.fromarray(frame_np)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=self.jpeg_quality)
                    jpeg_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
                    return jpeg_b64, mesh_time

                jpeg_b64, mesh_time = await asyncio.to_thread(_run_render)
                frame_time = time.time() - t0
                self.total_frame_time += frame_time
                self.total_mesh_time += mesh_time

                timestamp = time.time()

                try:
                    await asyncio.wait_for(self.frame_queue.put((jpeg_b64, timestamp)), timeout=0.1)
                    self.frames_generated += 1

                    if self.frames_generated % 25 == 0:
                        n = max(1, self.frames_generated)
                        logger.info(
                            f"Session {self.session_id} - Frame {self.frames_generated}: "
                            f"audio={self.total_audio_time/n*1000:.1f}ms, "
                            f"motion={self.total_motion_time/n*1000:.1f}ms, "
                            f"render={self.total_frame_time/n*1000:.1f}ms, "
                            f"mesh={self.total_mesh_time/n*1000:.1f}ms"
                        )
                except asyncio.TimeoutError:
                    logger.warning(f"Frame queue full for session {self.session_id}, dropping frame")
            except Exception as e:
                logger.error(f"Error in frame loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Frame loop stopped for session {self.session_id}")

    # ──────────────────────────────────────────────────────────────────
    # GPU thread mode (WebRTC) — all GPU ops on a single thread
    # ──────────────────────────────────────────────────────────────────

    def start_gpu_thread(self):
        """Start the dedicated GPU inference thread (WebRTC mode)."""
        if self.state is None:
            raise RuntimeError("GPU thread mode requires StreamingState (state=...)")
        self.running = True
        self._gpu_thread = threading.Thread(
            target=self._gpu_inference_loop, daemon=True, name=f"gpu-{self.session_id[:8]}"
        )
        self._gpu_thread.start()
        logger.info(f"GPU thread started for session {self.session_id}")

    def stop_gpu_thread(self):
        """Stop the GPU inference thread."""
        self.running = False
        if self.state:
            self.state.running = False
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=3.0)
            self._gpu_thread = None
        self._log_final_stats()

    def _gpu_inference_loop(self):
        """Dedicated GPU thread: sequential audio→motion→render→frame_slot pipeline."""
        logger.info(f"GPU inference loop started for session {self.session_id}")
        state = self.state

        while self.running and state.running:
            # 1. Get audio chunk (blocking with timeout)
            try:
                chunk, mode = state.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 2. Audio feature extraction (~2.5ms)
                t0 = time.time()
                result = self.audio_processor.add_audio_chunk(chunk, mode)
                self.total_audio_time += time.time() - t0

                if result is None:
                    continue

                features_self, features_other, audio_self, audio_other = result

                # 3. Motion inference (~20ms)
                t0 = time.time()
                motion = self.inference_engine.generate_next_frame(
                    features_self, features_other, audio_self, audio_other, mode
                )
                self.total_motion_time += time.time() - t0

                # 4. Render frame (~14ms)
                t0 = time.time()
                frame_gpu, frame_np = self.frame_renderer.render_frame_gpu(motion)
                self.total_frame_time += time.time() - t0

                # 5. FaceMesh + BlendShape (metrics, ~1.5ms)
                t0 = time.time()
                if self.facemesh is not None:
                    landmarks, _ = self.facemesh(frame_gpu)
                    if self.blendshape is not None:
                        self.blendshape(landmarks)
                self.total_mesh_time += time.time() - t0

                # 6. Pass frame to asyncio (VideoTrack)
                state.frame_slot.put(frame_np)
                self.frames_generated += 1

                if self.frames_generated % 25 == 0:
                    n = self.frames_generated
                    logger.info(
                        f"Session {self.session_id} - Frame {n}: "
                        f"audio={self.total_audio_time/n*1000:.1f}ms, "
                        f"motion={self.total_motion_time/n*1000:.1f}ms, "
                        f"render={self.total_frame_time/n*1000:.1f}ms, "
                        f"mesh={self.total_mesh_time/n*1000:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Error in GPU loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"GPU inference loop stopped for session {self.session_id}")

    # ──────────────────────────────────────────────────────────────────
    # Shared utilities
    # ──────────────────────────────────────────────────────────────────

    def _log_final_stats(self):
        n = max(1, self.frames_generated)
        logger.info(
            f"Worker stopped for session {self.session_id}. "
            f"Generated {self.frames_generated} frames. "
            f"Avg times - audio: {self.total_audio_time/n:.3f}s, "
            f"motion: {self.total_motion_time/n:.3f}s, "
            f"frame: {self.total_frame_time/n:.3f}s, "
            f"mesh: {self.total_mesh_time/n:.3f}s"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics."""
        if self.frames_generated == 0:
            return {
                'frames_generated': 0,
                'avg_audio_time_ms': 0,
                'avg_motion_time_ms': 0,
                'avg_frame_time_ms': 0,
                'avg_mesh_time_ms': 0,
                'avg_total_time_ms': 0
            }

        n = self.frames_generated
        return {
            'frames_generated': n,
            'avg_audio_time_ms': (self.total_audio_time / n) * 1000,
            'avg_motion_time_ms': (self.total_motion_time / n) * 1000,
            'avg_frame_time_ms': (self.total_frame_time / n) * 1000,
            'avg_mesh_time_ms': (self.total_mesh_time / n) * 1000,
            'avg_total_time_ms': ((self.total_audio_time + self.total_motion_time + self.total_frame_time) / n) * 1000
        }
