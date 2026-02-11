"""
WebRTC session management for DyStream.

Handles SDP offer/answer exchange, pipeline construction,
and lifecycle management for WebRTC connections.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from aiortc import RTCPeerConnection, RTCSessionDescription

from streaming_app.models.audio_processor import IncrementalAudioProcessor
from streaming_app.models.streaming_inference import StreamingInferenceEngine
from streaming_app.utils.visualization import FrameRenderer, load_reference_image
from streaming_app.webrtc.state import StreamingState
from streaming_app.webrtc.tracks import AudioReceiver, DyStreamVideoTrack
from streaming_app.workers.generation_worker import GenerationWorker

logger = logging.getLogger(__name__)

# Active peer connections (for cleanup)
_peer_connections: Dict[str, RTCPeerConnection] = {}


def _build_pipeline(
    session,
    state: StreamingState,
    app_state: Dict[str, Any],
) -> GenerationWorker:
    """Build the full inference pipeline for a WebRTC session.

    Reuses the same component construction logic as the WebSocket path in main.py.
    """
    config = app_state["config"]
    models = app_state["models"]
    device = app_state["device"]

    vis_models = models.get("visualization")
    wav2vec_trt = models.get("wav2vec_trt")

    # Audio processor
    audio_processor = IncrementalAudioProcessor(
        dystream_model=models["dystream"],
        device=device,
        sample_rate=config.get("audio_sample_rate", 16000),
        window_frames=config.get("audio_window_frames", 96),
        chunk_size=config.get("audio_chunk_size", 640),
        target_fps=config.get("target_fps", 25),
        lookahead_ms_speaker=config.get("audio_lookahead_ms_speaker", 60),
        lookahead_ms_listener=config.get("audio_lookahead_ms_listener", 0),
        update_every_n_chunks=config.get("audio_feature_update_every_n_chunks", 1),
        wav2vec_trt=wav2vec_trt,
    )

    # Inference engine
    inference_engine = StreamingInferenceEngine(
        dystream_model=models["dystream"],
        device=device,
        config=config,
        noise_scheduler=models.get("noise_scheduler"),
        ema=models.get("dystream_ema"),
    )
    inference_engine.initialize_context(session.motion_anchor)

    # Frame renderer
    frame_renderer = None
    if vis_models:
        frame_renderer = FrameRenderer(
            face_encoder=vis_models["face_encoder"],
            face_generator=vis_models["face_generator"],
            flow_estimator=vis_models["flow_estimator"],
            device=device,
            jpeg_quality=config.get("frame_encoding_quality", 80),
            combined_pipeline=vis_models.get("combined_pipeline"),
        )
        # Initialize reference image
        anchor_tensor = torch.from_numpy(session.motion_anchor).float().to(device)
        from streaming_app.websocket.session_manager import SessionManager

        ref_image_path = (
            Path(session.reference_image_path)
            if session.reference_image_path
            else Path(SessionManager.DEFAULT_AVATARS[0]["path"])
        )
        if ref_image_path.exists():
            ref_img_tensor = load_reference_image(str(ref_image_path), device)
            frame_renderer.initialize_reference(ref_img_tensor, anchor_tensor)
            logger.info("Frame renderer initialized for WebRTC session %s", state.session_id)
        else:
            logger.warning("Reference image not found: %s", ref_image_path)

    # Generation worker (GPU-thread mode for WebRTC)
    worker = GenerationWorker(
        inference_engine=inference_engine,
        frame_renderer=frame_renderer,
        audio_processor=audio_processor,
        session_id=state.session_id,
        facemesh=app_state.get("facemesh"),
        blendshape=app_state.get("blendshape"),
        state=state,
    )

    return worker


async def handle_offer(request_body: dict, app_state: Dict[str, Any]) -> dict:
    """Handle POST /api/webrtc/offer — SDP exchange + pipeline setup.

    Args:
        request_body: {"sdp": str, "type": "offer"}
        app_state: shared server state (config, models, device, session_manager, ...)

    Returns:
        {"sdp": str, "type": "answer", "session_id": str}
    """
    offer = RTCSessionDescription(sdp=request_body["sdp"], type=request_body["type"])
    pc = RTCPeerConnection()

    # Create session via existing SessionManager
    session_manager = app_state["session_manager"]
    session = session_manager.create_session()

    # Create shared state
    state = StreamingState(session_id=session.session_id)
    state.is_streaming = True

    # Build inference pipeline
    worker = _build_pipeline(session, state, app_state)

    # Add server→client video track
    video_track = DyStreamVideoTrack(state)
    pc.addTrack(video_track)

    # Track this peer connection
    _peer_connections[session.session_id] = pc

    # Handle incoming audio track from client
    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            logger.info("WebRTC audio track received for session %s", session.session_id)
            receiver = AudioReceiver(track, state)
            asyncio.ensure_future(receiver.run())

    # Handle DataChannel for control messages
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("DataChannel '%s' opened for session %s", channel.label, session.session_id)

        @channel.on("message")
        def on_message(msg):
            try:
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "start":
                    state.is_streaming = True
                elif msg_type == "stop":
                    state.is_streaming = False
                elif msg_type == "mode_switch":
                    state.mode = data.get("mode", "speaker")
                    session_manager.switch_mode(session.session_id, state.mode)
                elif msg_type == "config_update":
                    new_config = data.get("config", {})
                    session_manager.update_session_config(session.session_id, new_config)
                    worker.inference_engine.update_config(new_config)

                # Acknowledge
                channel.send(json.dumps({"type": "status", "status": f"{msg_type}_ok"}))
            except Exception as e:
                logger.error("DataChannel message error: %s", e)
                channel.send(json.dumps({"type": "error", "message": str(e)}))

    # SDP exchange
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Start GPU inference thread
    state.frame_slot.set_loop(asyncio.get_event_loop())
    worker.start_gpu_thread()

    # Cleanup on disconnect
    @pc.on("connectionstatechange")
    async def on_state_change():
        logger.info("WebRTC connection state: %s (session %s)", pc.connectionState, session.session_id)
        if pc.connectionState in ("failed", "closed"):
            worker.stop_gpu_thread()
            session_manager.delete_session(session.session_id)
            _peer_connections.pop(session.session_id, None)
            logger.info("WebRTC session %s cleaned up", session.session_id)

    logger.info("WebRTC session %s established", session.session_id)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session.session_id,
    }
