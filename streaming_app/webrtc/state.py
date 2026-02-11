"""
Shared state for WebRTC streaming sessions.
FrameSlot provides thread-safe frame passing from GPU thread to asyncio event loop.
StreamingState holds per-session control state.
"""

import asyncio
import queue
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


class FrameSlot:
    """GPU thread → asyncio event loop thread-safe frame handoff.

    GPU thread calls put() to store the latest frame and wake the asyncio side.
    asyncio coroutines call get() to await the next frame.
    """

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event = asyncio.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def put(self, frame: np.ndarray):
        """Called from GPU thread."""
        with self._lock:
            self._frame = frame
        if self._loop:
            self._loop.call_soon_threadsafe(self._event.set)

    async def get(self) -> np.ndarray:
        """Called from asyncio event loop."""
        await self._event.wait()
        self._event.clear()
        with self._lock:
            return self._frame


@dataclass
class StreamingState:
    """Per-session shared state between asyncio IO and GPU inference thread."""

    session_id: str
    mode: str = "speaker"
    is_streaming: bool = False
    running: bool = True
    audio_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10))
    frame_slot: FrameSlot = field(default_factory=FrameSlot)
    config: dict = field(default_factory=dict)
