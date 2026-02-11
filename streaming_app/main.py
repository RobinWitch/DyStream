"""
FastAPI WebSocket server for DyStream streaming inference.
"""

import asyncio
import base64
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import streaming modules
from streaming_app.models.model_loader import model_loader
from streaming_app.models.audio_processor import IncrementalAudioProcessor
from streaming_app.models.streaming_inference import StreamingInferenceEngine
from streaming_app.websocket.connection_manager import ConnectionManager
from streaming_app.websocket.session_manager import SessionManager
from streaming_app.utils.visualization import FrameRenderer
from streaming_app.workers.generation_worker import GenerationWorker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DyStream Streaming Server", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config: Dict[str, Any] = {}
models: Dict[str, Any] = {}
connection_manager = ConnectionManager()
session_manager = None
device = None

# Optimized model components (loaded once, shared across sessions)
facemesh_trt = None
blendshape_trt = None

# Per-session resources
audio_processors: Dict[str, IncrementalAudioProcessor] = {}
inference_engines: Dict[str, StreamingInferenceEngine] = {}
frame_renderers: Dict[str, FrameRenderer] = {}
generation_workers: Dict[str, GenerationWorker] = {}


@app.on_event("startup")
async def startup_event():
    """Load models and initialize on startup."""
    global config, models, session_manager, device, facemesh_trt, blendshape_trt

    logger.info("Starting DyStream streaming server...")

    # Load config
    config_path = Path(__file__).parent / "config" / "streaming_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['streaming']

    logger.info(f"Loaded config: {config}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    logger.info("Loading models... This may take a few minutes.")
    try:
        models = model_loader.load_all_models(config)
        logger.info("Models loaded successfully!")

        # Load FaceMesh + BlendShape TRT (optional, best-effort)
        try:
            from streaming_app.models.tensorrt_wrapper import load_facemesh_trt, load_blendshape_trt
            facemesh_trt = load_facemesh_trt(
                'checkpoints/facemesh/face_landmarks_detector_fp16.engine',
                device,
                onnx_path='checkpoints/facemesh/face_landmarks_detector_1x3x256x256.onnx',
            )
            blendshape_trt = load_blendshape_trt(
                'checkpoints/facemesh/face_blendshapes_fp16.engine',
                device,
                onnx_path='checkpoints/facemesh/face_blendshapes.onnx',
            )
            logger.info("FaceMesh + BlendShape TRT loaded")
        except Exception as e:
            logger.warning(f"FaceMesh/BlendShape TRT not available: {e}")

        # Initialize session manager
        session_manager = SessionManager(config, models, device)

        logger.info("Server startup complete!")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/")
async def get_index():
    """Serve the frontend HTML."""
    index_path = Path(__file__).parent.parent / "streaming_frontend" / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>Frontend not yet available</h1>", status_code=404)

    with open(index_path, 'r') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@app.get("/api/status")
async def get_status():
    """Get server status."""
    from streaming_app.models.model_loader import ModelLoader
    return JSONResponse({
        'status': 'running',
        'device': str(device),
        'models_loaded': ModelLoader._models_loaded,
        'active_sessions': connection_manager.get_connection_count(),
        'max_sessions': config.get('max_concurrent_sessions', 2)
    })


@app.get("/api/avatars")
async def get_avatars():
    """Get list of default avatars."""
    return JSONResponse({
        'avatars': SessionManager.get_default_avatars()
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming."""
    session_id = None

    try:
        # Initialize session
        session = session_manager.create_session()
        session_id = session.session_id

        # Connect WebSocket
        await connection_manager.connect(websocket, session_id)

        # Send initialization message
        await connection_manager.send_text(session_id, {
            'type': 'init',
            'session_id': session_id,
            'config': session.to_dict()
        })

        # Get optimized components
        vis_models = models.get('visualization')
        wav2vec_trt = models.get('wav2vec_trt')

        # Initialize audio processor (with wav2vec TRT if available)
        audio_processors[session_id] = IncrementalAudioProcessor(
            dystream_model=models['dystream'],
            device=device,
            sample_rate=config.get('audio_sample_rate', 16000),
            window_frames=config.get('audio_window_frames', 96),
            chunk_size=config.get('audio_chunk_size', 640),
            target_fps=config.get('target_fps', 25),
            lookahead_ms_speaker=config.get('audio_lookahead_ms_speaker', 60),
            lookahead_ms_listener=config.get('audio_lookahead_ms_listener', 0),
            update_every_n_chunks=config.get('audio_feature_update_every_n_chunks', 1),
            wav2vec_trt=wav2vec_trt,
        )

        # Initialize inference engine
        inference_engines[session_id] = StreamingInferenceEngine(
            dystream_model=models['dystream'],
            device=device,
            config=config,
            noise_scheduler=models.get('noise_scheduler'),
            ema=models.get('dystream_ema')
        )

        # Initialize motion context with session's motion anchor
        inference_engines[session_id].initialize_context(session.motion_anchor)

        # Initialize frame renderer for this session
        if vis_models:
            frame_renderers[session_id] = FrameRenderer(
                face_encoder=vis_models['face_encoder'],
                face_generator=vis_models['face_generator'],
                flow_estimator=vis_models['flow_estimator'],
                device=device,
                jpeg_quality=config.get('frame_encoding_quality', 80),
                combined_pipeline=vis_models.get('combined_pipeline'),
            )

            # Initialize reference image and anchor motion
            anchor_tensor = torch.from_numpy(session.motion_anchor).float().to(device)

            from streaming_app.utils.visualization import load_reference_image
            ref_image_path = Path(session.reference_image_path) if session.reference_image_path else \
                            Path(SessionManager.DEFAULT_AVATARS[0]['path'])

            if ref_image_path.exists():
                ref_img_tensor = load_reference_image(str(ref_image_path), device)
                frame_renderers[session_id].initialize_reference(ref_img_tensor, anchor_tensor)
                logger.info(f"Frame renderer initialized for session {session_id}")
            else:
                logger.warning(f"Reference image not found: {ref_image_path}")
        else:
            logger.warning("Visualization models not available")

        # Initialize and start generation worker
        if vis_models:
            generation_workers[session_id] = GenerationWorker(
                inference_engine=inference_engines[session_id],
                frame_renderer=frame_renderers[session_id],
                audio_processor=audio_processors[session_id],
                session_id=session_id,
                facemesh=facemesh_trt,
                blendshape=blendshape_trt,
                jpeg_quality=config.get('frame_encoding_quality', 80),
            )
            await generation_workers[session_id].start()
            logger.info(f"Generation worker started for session {session_id}")

        logger.info(f"Session {session_id} initialized and ready")

        # Start frame sender task
        frame_sender_task = asyncio.create_task(frame_sender_loop(session_id))

        # Message handling loop (JSON-only)
        while True:
            message_type, data = await connection_manager.receive_message(websocket)

            if message_type is None:
                break

            if message_type == 'text':
                await handle_text_message(session_id, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        if session_id:
            await connection_manager.send_error(session_id, str(e))

    finally:
        # Cleanup
        if session_id:
            # Stop generation worker
            if session_id in generation_workers:
                await generation_workers[session_id].stop()
                del generation_workers[session_id]

            connection_manager.disconnect(session_id)
            session_manager.delete_session(session_id)

            # Clean up processors, engines, and renderers
            if session_id in audio_processors:
                del audio_processors[session_id]
            if session_id in inference_engines:
                del inference_engines[session_id]
            if session_id in frame_renderers:
                del frame_renderers[session_id]

            logger.info(f"Session {session_id} cleaned up")


async def handle_text_message(session_id: str, message_text: str):
    """Handle JSON text messages (all message types)."""
    try:
        message = json.loads(message_text)
        msg_type = message.get('type')

        if msg_type == 'audio':
            await handle_audio_chunk(session_id, message.get('data', ''))

        elif msg_type == 'reference_image':
            await handle_reference_image(session_id, message.get('data', ''))

        elif msg_type == 'start':
            session = session_manager.get_session(session_id)
            session.is_streaming = True
            await connection_manager.send_status(session_id, 'streaming_started')
            logger.info(f"Session {session_id} started streaming")

        elif msg_type == 'stop':
            session = session_manager.get_session(session_id)
            session.is_streaming = False
            await connection_manager.send_status(session_id, 'streaming_stopped')
            logger.info(f"Session {session_id} stopped streaming")

        elif msg_type == 'mode_switch':
            new_mode = message.get('mode')
            session_manager.switch_mode(session_id, new_mode)
            await connection_manager.send_status(session_id, 'mode_switched', {'mode': new_mode})

        elif msg_type == 'config_update':
            new_config = message.get('config', {})
            session_manager.update_session_config(session_id, new_config)
            inference_engines[session_id].update_config(new_config)
            await connection_manager.send_status(session_id, 'config_updated', {'config': new_config})

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await connection_manager.send_error(session_id, str(e))


async def handle_audio_chunk(session_id: str, audio_base64: str):
    """Decode base64 audio and queue for processing."""
    session = session_manager.get_session(session_id)

    if not session.is_streaming:
        return

    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

        if session_id in generation_workers:
            await generation_workers[session_id].queue_audio_chunk(
                audio_chunk,
                mode=session.mode
            )

    except Exception as e:
        logger.error(f"Error handling audio chunk: {e}", exc_info=True)
        await connection_manager.send_error(session_id, str(e))


async def frame_sender_loop(session_id: str):
    """Continuously send generated frames to the client as JSON."""
    logger.info(f"Frame sender loop started for session {session_id}")

    try:
        while session_id in generation_workers:
            result = await generation_workers[session_id].get_next_frame(timeout=0.1)

            if result is not None:
                jpeg_b64, timestamp = result
                await connection_manager.send_frame(session_id, jpeg_b64, timestamp)

                session = session_manager.get_session(session_id)
                if session:
                    session.frames_generated += 1

            await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error in frame sender loop for session {session_id}: {e}", exc_info=True)

    logger.info(f"Frame sender loop stopped for session {session_id}")


async def handle_reference_image(session_id: str, image_base64: str):
    """Process uploaded reference image from base64."""
    try:
        image_data = base64.b64decode(image_base64)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_data)
            image_path = f.name

        session_manager.update_reference_image(session_id, image_path)

        await connection_manager.send_status(session_id, 'reference_image_updated_next_session')
        logger.info(f"Session {session_id} reference image stored for next session")

    except Exception as e:
        logger.error(f"Error processing reference image: {e}")
        await connection_manager.send_error(session_id, str(e))


# ── WebRTC endpoint ──────────────────────────────────────────────

@app.post("/api/webrtc/offer")
async def webrtc_offer(request: Request):
    """WebRTC SDP offer → answer exchange. Sets up full inference pipeline."""
    from streaming_app.webrtc.session import handle_offer

    body = await request.json()
    result = await handle_offer(body, app_state={
        "config": config,
        "models": models,
        "device": device,
        "session_manager": session_manager,
        "facemesh": facemesh_trt,
        "blendshape": blendshape_trt,
    })
    return JSONResponse(result)


# Mount static files (if any)
static_path = Path(__file__).parent.parent / "streaming_frontend"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
