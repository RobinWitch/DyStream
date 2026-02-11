"""
Custom WebRTC tracks for DyStream.

DyStreamVideoTrack: pulls frames from FrameSlot and encodes as VP8.
AudioReceiver: receives Opus audio from WebRTC, resamples 48kHz→16kHz,
               and feeds chunks into the GPU thread via audio_queue.
"""

import asyncio
import logging
import queue

import numpy as np
from aiortc import MediaStreamTrack, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from streaming_app.webrtc.state import StreamingState

logger = logging.getLogger(__name__)


class DyStreamVideoTrack(VideoStreamTrack):
    """Video track that reads generated frames from FrameSlot.

    Extends VideoStreamTrack which provides next_timestamp() for proper
    PTS/time_base management needed by the VP8 encoder.
    """

    def __init__(self, state: StreamingState):
        super().__init__()
        self.state = state

    async def recv(self):
        frame_np = await self.state.frame_slot.get()  # (H, W, 3) uint8 RGB
        frame = VideoFrame.from_ndarray(frame_np, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


class AudioReceiver:
    """Receives WebRTC audio track, resamples 48kHz→16kHz, feeds audio_queue.

    WebRTC browsers typically send Opus at 48kHz. The DyStream pipeline
    expects 16kHz mono float32 chunks of 640 samples (40ms).
    """

    def __init__(
        self,
        track: MediaStreamTrack,
        state: StreamingState,
        sample_rate_in: int = 48000,
        sample_rate_out: int = 16000,
        chunk_size: int = 640,
    ):
        self.track = track
        self.state = state
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out
        self.chunk_size = chunk_size
        self._buf = np.zeros(0, dtype=np.float32)

    async def run(self):
        from scipy.signal import resample_poly

        up = self.sample_rate_out
        down = self.sample_rate_in
        # Simplify ratio
        from math import gcd
        g = gcd(up, down)
        up, down = up // g, down // g  # 1, 3

        logger.info("AudioReceiver started (resample %d→%d, chunk=%d)",
                     self.sample_rate_in, self.sample_rate_out, self.chunk_size)

        while True:
            try:
                frame = await self.track.recv()
            except MediaStreamError:
                logger.info("AudioReceiver: track ended")
                break

            # frame.to_ndarray() returns int16 shape (channels, samples)
            audio_int16 = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
            audio_16k = resample_poly(audio_int16, up, down)
            self._buf = np.concatenate([self._buf, audio_16k])

            while len(self._buf) >= self.chunk_size:
                chunk = self._buf[: self.chunk_size].copy()
                self._buf = self._buf[self.chunk_size :]
                try:
                    self.state.audio_queue.put_nowait((chunk, self.state.mode))
                except queue.Full:
                    pass  # drop oldest if GPU can't keep up

        logger.info("AudioReceiver stopped")
