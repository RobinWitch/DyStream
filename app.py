"""
DyStream Gradio Demo
====================
Streaming Dyadic Talking Heads Generation via FlowMatching-based Autoregressive Model with optional dyadic conversation support.
"""

import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import cv2
import imageio
import gradio as gr
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from torch_ema import ExponentialMovingAverage

# ────────────────────────────────────────────────────────────────────────────
# Path setup
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(PROJECT_ROOT, "tools", "visualization_0416")
VIS_MODEL_DIR = os.path.join(VIS_DIR, "utils", "model_0506")

# 1) Add project root first and grab what we need
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project-level helpers immediately (before VIS paths pollute sys.path)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("project_utils", os.path.join(PROJECT_ROOT, "utils.py"))
_project_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_project_utils)

instantiate_motion_gen = _project_utils.instantiate_motion_gen
Config = _project_utils.Config

# 2) Add VIS_DIR to sys.path (for utils.face_detector etc.)
#    Do NOT add VIS_MODEL_DIR directly – it has its own `model/` package that
#    would shadow the project-root `model/` package.  Instead, we merge the two
#    `model` package namespaces below.
if VIS_DIR not in sys.path:
    sys.path.append(VIS_DIR)

# 3) Merge the two `model` namespaces so that both
#    `model.motion_generation.*` (project root) and
#    `model.head_animation.*`   (vis tools) are importable.
import model as _model_pkg  # loads PROJECT_ROOT/model
_vis_model_dir_model = os.path.join(VIS_MODEL_DIR, "model")
if _vis_model_dir_model not in _model_pkg.__path__:
    _model_pkg.__path__.append(_vis_model_dir_model)

# ────────────────────────────────────────────────────────────────────────────
# Global model holders (lazy-loaded)
# ────────────────────────────────────────────────────────────────────────────
_dystream_model = None
_dystream_cfg = None
_dystream_ema = None
_noise_scheduler = None
_vis_ctx = None  # visualization context: transform, face_encoder, flow_estimator, face_generator
_motion_encoder = None  # for extracting motion latent from image
_face_detector = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────────
# Model Loading
# ────────────────────────────────────────────────────────────────────────────

def load_dystream_model():
    """Load the DyStream motion generation model."""
    global _dystream_model, _dystream_cfg, _dystream_ema, _noise_scheduler

    if _dystream_model is not None:
        return

    print("[DyStream] Loading motion generation model...")
    config_path = os.path.join(PROJECT_ROOT, "configs", "motion_gen", "sample.yaml")

    # Build config with overrides (same as run.sh)
    override_args = {
        "exp_name": "gradio_demo",
        "model.module_name": "model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder",
        "resume_ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "last.ckpt"),
    }
    _dystream_cfg = Config(config_path, override_args)

    # Instantiate model
    _dystream_model = instantiate_motion_gen(
        module_name=_dystream_cfg.model.module_name,
        class_name=_dystream_cfg.model.class_name,
        cfg=_dystream_cfg.model,
        hfstyle=False,
    )

    # Load checkpoint
    ckpt_path = _dystream_cfg.resume_ckpt
    if os.path.exists(ckpt_path):
        print(f"[DyStream] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # The checkpoint stores state_dict at top level (from LightningModule)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip "model." prefix if from Lightning checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                new_state_dict[k] = v
        _dystream_model.load_state_dict(new_state_dict, strict=False)
        print("[DyStream] Checkpoint loaded successfully.")

        # Load EMA if available
        _dystream_ema = ExponentialMovingAverage(
            _dystream_model.parameters(), decay=_dystream_cfg.model.ema_decay
        )
        if "ema_state" in checkpoint:
            _dystream_ema.load_state_dict(checkpoint["ema_state"])
            print("[DyStream] EMA state loaded.")
    else:
        print(f"[DyStream] WARNING: Checkpoint not found at {ckpt_path}")

    _dystream_model.eval().to(DEVICE)

    # Noise scheduler
    _noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **OmegaConf.to_container(_dystream_cfg.noise_scheduler_kwargs, resolve=True)
    )
    print("[DyStream] Motion generation model ready.")


def load_visualization_model():
    """Load the visualization model for converting motion latents to video."""
    global _vis_ctx

    if _vis_ctx is not None:
        return

    print("[Visualization] Loading visualization model...")
    config_path = os.path.join(VIS_DIR, "configs", "head_animator_best_0506.yaml")
    config = OmegaConf.load(config_path)

    # Fix relative checkpoint path
    vis_ckpt = config.resume_ckpt
    if not os.path.isabs(vis_ckpt):
        vis_ckpt = os.path.normpath(os.path.join(VIS_DIR, vis_ckpt))

    # Load vis tools' own instantiate via importlib (to avoid conflicts with
    # the project-root utils.py and the VIS_DIR utils/ package).
    _vis_utils_spec = _ilu.spec_from_file_location(
        "vis_tools_utils", os.path.join(VIS_MODEL_DIR, "utils.py")
    )
    _vis_utils_mod = _ilu.module_from_spec(_vis_utils_spec)
    _vis_utils_spec.loader.exec_module(_vis_utils_mod)
    vis_instantiate = _vis_utils_mod.instantiate

    module_cls = vis_instantiate(config.model, instantiate_module=False)
    model = module_cls(config=config)

    checkpoint = torch.load(vis_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval().to(DEVICE)

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    _vis_ctx = {
        "transform": transform,
        "flow_estimator": model.flow_estimator,
        "face_generator": model.face_generator,
        "face_encoder": model.face_encoder,
        "motion_encoder": model.motion_encoder,
    }
    print("[Visualization] Visualization model ready.")


def load_face_detector():
    """Load MediaPipe face detector for image preprocessing."""
    global _face_detector

    if _face_detector is not None:
        return

    print("[FaceDetector] Loading face detector...")
    # Load FaceDetector via importlib to avoid 'utils' name conflict
    _fd_spec = _ilu.spec_from_file_location(
        "vis_face_detector",
        os.path.join(VIS_DIR, "utils", "face_detector.py"),
    )
    _fd_mod = _ilu.module_from_spec(_fd_spec)
    _fd_spec.loader.exec_module(_fd_mod)
    FaceDetector = _fd_mod.FaceDetector

    model_path = os.path.join(VIS_DIR, "utils", "face_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        print("[FaceDetector] Downloading face landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    _face_detector = FaceDetector(
        mediapipe_model_asset_path=model_path,
        face_detection_confidence=0.5,
        num_faces=1,
    )
    print("[FaceDetector] Face detector ready.")


# ────────────────────────────────────────────────────────────────────────────
# Image Processing
# ────────────────────────────────────────────────────────────────────────────

def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    sw *= scale
    sh *= scale
    scaled = [cx - sw, cy - sh, cx + sw, cy + sh]
    scaled[0] = np.clip(scaled[0], 0, w)
    scaled[2] = np.clip(scaled[2], 0, w)
    scaled[1] = np.clip(scaled[1], 0, h)
    scaled[3] = np.clip(scaled[3], 0, h)
    return scaled


def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise Exception("Invalid mask")
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask = np.zeros((hd, wd, 3), dtype=np.uint8)
    mask[y0:y1, x0:x1, :] = 255
    if return_pil:
        return Image.fromarray(mask)
    return mask


def generate_crop_bounding_box(h, w, center, size=512):
    half_size = size // 2
    y1 = max(center[0] - half_size, 0)
    x1 = max(center[1] - half_size, 0)
    y2 = min(center[0] + half_size, h)
    x2 = min(center[1] + half_size, w)
    return [x1, y1, x2, y2]


def crop_from_bbox(image, center, bbox, size=512):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    half_size = size // 2
    cropped = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    cropped[(y1 - (center[0] - half_size)):(y2 - (center[0] - half_size)),
            (x1 - (center[1] - half_size)):(x2 - (center[1] - half_size))] = image[y1:y2, x1:x2]
    return cropped


def process_image(image_pil, crop=True, union_bbox_scale=1.6):
    """
    Process uploaded image: face detection, crop, resize, mask, extract motion latent.
    Returns: (resized_image_pil, masked_image_pil, motion_latent_tensor)
    """
    load_face_detector()
    load_visualization_model()

    cfg_path = os.path.join(VIS_DIR, "configs", "audio_head_animator.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    from torchvision import transforms
    pixel_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.5], [0.5]),
    ])
    resize_transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)

    img = image_pil.convert("RGB")
    img_np = np.array(img)
    state = torch.get_rng_state()

    det_res = _face_detector.get_face_xy_rotation_and_keypoints(
        img_np, cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
    )

    if not det_res or len(det_res[6]) == 0:
        raise gr.Error("No face detected. Please upload an image with a clear face.")

    person_id = 0
    mouth_bbox = np.array(det_res[6][person_id])
    eye_bbox = det_res[7][person_id]
    face_contour = np.array(det_res[8][person_id])
    left_eye_bbox = eye_bbox["left_eye"]
    right_eye_bbox = eye_bbox["right_eye"]

    if crop:
        face_bbox = det_res[5][person_id]
        x1, y1 = face_bbox[0]
        x2, y2 = face_bbox[1]
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        width = x2 - x1
        height = y2 - y1
        max_size = int(max(width, height) * union_bbox_scale)
        hd, wd = img.size[1], img.size[0]
        crop_bbox = generate_crop_bounding_box(hd, wd, center, max_size)
        img_array = np.array(img)
        cropped_img = crop_from_bbox(img_array, center, crop_bbox, size=max_size)
        img = Image.fromarray(cropped_img)

        det_res = _face_detector.get_face_xy_rotation_and_keypoints(
            cropped_img, cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
        )
        if not det_res or len(det_res[6]) == 0:
            raise gr.Error("No face detected after cropping. Please try a different image.")
        mouth_bbox = np.array(det_res[6][person_id])
        eye_bbox = det_res[7][person_id]
        face_contour = np.array(det_res[8][person_id])
        left_eye_bbox = eye_bbox["left_eye"]
        right_eye_bbox = eye_bbox["right_eye"]

    def augmentation(images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            transformed = [transforms.functional.to_tensor(img_item) for img_item in images]
            return transform(torch.stack(transformed, dim=0))
        return transform(transforms.functional.to_tensor(images))

    pixel_values_ref = augmentation([img], pixel_transform, state)
    pixel_values_ref = (pixel_values_ref + 1) / 2
    new_hd, new_wd = img.size[1], img.size[0]

    mouth_mask = resize_transform(get_mask(mouth_bbox, new_hd, new_wd, scale=1.0))
    left_eye_mask = resize_transform(get_mask(left_eye_bbox, new_hd, new_wd, scale=1.0))
    right_eye_mask = resize_transform(get_mask(right_eye_bbox, new_hd, new_wd, scale=1.0))
    face_contour_resized = resize_transform(Image.fromarray(face_contour))

    eye_mask = np.bitwise_or(np.array(left_eye_mask), np.array(right_eye_mask))
    combined_mask = np.bitwise_or(eye_mask, np.array(mouth_mask))

    combined_mask_tensor = torch.from_numpy(combined_mask / 255.0).permute(2, 0, 1).unsqueeze(0)
    face_contour_tensor = torch.from_numpy(np.array(face_contour_resized) / 255.0).permute(2, 0, 1).unsqueeze(0)

    masked_ref = pixel_values_ref * combined_mask_tensor + face_contour_tensor * (1 - combined_mask_tensor)
    masked_ref = masked_ref.clamp(0, 1)

    # Convert to PIL
    resized_np = (pixel_values_ref.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    masked_np = (masked_ref.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    resized_pil = Image.fromarray(resized_np)
    masked_pil = Image.fromarray(masked_np)

    # Extract motion latent using motion encoder
    # NOTE: main.py passes the RESIZED (clean) image to img_to_latent.py, NOT the masked one.
    #       See main.py _get_latent line 438-441: ori_resize_abs is _resize.png
    vis_transform = _vis_ctx["transform"]
    resized_img_tensor = vis_transform(resized_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        motion_latent = _vis_ctx["motion_encoder"](resized_img_tensor)[0]  # [1, 512]

    return resized_pil, masked_pil, motion_latent.cpu()


# ────────────────────────────────────────────────────────────────────────────
# Motion Latent to Video
# ────────────────────────────────────────────────────────────────────────────

def latents_to_video_frames(motion_latents, ref_image_pil):
    """Convert motion latents to video frames using the visualization model."""
    load_visualization_model()

    transform = _vis_ctx["transform"]
    face_encoder = _vis_ctx["face_encoder"]
    flow_estimator = _vis_ctx["flow_estimator"]
    face_generator = _vis_ctx["face_generator"]

    ref_img_tensor = transform(ref_image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    # motion_latents: [1, T, 512] or [T, 512]
    if motion_latents.dim() == 3:
        motion_latents = motion_latents.squeeze(0)
    motion_latents = motion_latents.to(DEVICE).float()
    num_frames = motion_latents.shape[0]

    with torch.no_grad():
        face_feat = face_encoder(ref_img_tensor)
        recon_list = []
        for i in range(num_frames):
            tgt = flow_estimator(motion_latents[0:1], motion_latents[i:i + 1])
            recon_list.append(face_generator(tgt, face_feat))

    recon = torch.cat(recon_list, dim=0)
    video_np = recon.permute(0, 2, 3, 1).cpu().numpy()
    video_np = np.clip((video_np + 1) / 2 * 255, 0, 255).astype("uint8")
    return video_np


def save_video_with_audio(video_frames, audio_path, output_path, fps=25):
    """Save video frames as mp4, optionally mux with audio."""
    temp_mp4 = output_path.replace(".mp4", "_temp.mp4")
    with imageio.get_writer(temp_mp4, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    if audio_path and os.path.exists(audio_path):
        try:
            import moviepy.editor as mpe
            clip = mpe.VideoFileClip(temp_mp4)
            audio = mpe.AudioFileClip(audio_path)
            # Trim audio to match video length
            video_duration = len(video_frames) / fps
            if audio.duration > video_duration:
                audio = audio.subclip(0, video_duration)
            clip = clip.set_audio(audio)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            clip.close()
            audio.close()
            os.remove(temp_mp4)
        except Exception as e:
            print(f"[Warning] Failed to mux audio: {e}, saving video without audio.")
            if os.path.exists(temp_mp4):
                shutil.move(temp_mp4, output_path)
    else:
        shutil.move(temp_mp4, output_path)

    return output_path


# ────────────────────────────────────────────────────────────────────────────
# DyStream Inference
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    image_input,
    speaker_audio_path,
    listener_audio_path,
    denoising_steps,
    cfg_audio,
    cfg_audio_other,
    cfg_anchor,
    cfg_all,
    progress=gr.Progress(track_tqdm=True),
    precomputed_npz_path=None,
    precomputed_ref_img_path=None,
    video_audio_path=None,
):
    """
    Full inference pipeline:
    1. Process image -> motion latent  (or use precomputed)
    2. Run DyStream inference
    3. Convert motion latents to video
    """
    if image_input is None and precomputed_npz_path is None:
        raise gr.Error("Please upload a reference face image.")
    if speaker_audio_path is None:
        raise gr.Error("Please upload speaker audio.")

    # ── Step 1: Load all models ──────────────────────────────────────────
    progress(0.0, desc="Loading models...")
    load_dystream_model()
    load_visualization_model()

    # ── Step 2: Get motion latent & reference image ──────────────────────
    progress(0.1, desc="Processing image...")

    if precomputed_npz_path is not None and os.path.exists(precomputed_npz_path):
        # ── Use pre-computed files (same as run.sh) ──
        data = np.load(precomputed_npz_path, allow_pickle=True)
        try:
            motion_latent_np = data["motion_latent"]
        except KeyError:
            motion_latent_np = data["random_data"]
        motion_latent_cpu = torch.from_numpy(motion_latent_np)  # [N, 512] or [1, 512]

        ref_img_path = precomputed_ref_img_path or str(data.get("ref_img_path", ""))
        if os.path.exists(ref_img_path):
            resized_pil = Image.open(ref_img_path).convert("RGB")
        else:
            raise gr.Error(f"Reference image not found: {ref_img_path}")
        masked_pil = resized_pil  # for display only
    else:
        # ── On-the-fly processing for custom uploads ──
        if isinstance(image_input, np.ndarray):
            image_pil = Image.fromarray(image_input)
        else:
            image_pil = image_input
        resized_pil, masked_pil, motion_latent_cpu = process_image(image_pil)

    # ── Step 3: Prepare audio ────────────────────────────────────────────
    progress(0.2, desc="Processing audio...")
    audio_sr = int(OmegaConf.select(_dystream_cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(OmegaConf.select(_dystream_cfg.config, "model.pose_fps", default=25))

    audio_self, _ = librosa.load(speaker_audio_path, sr=audio_sr)
    additional_motion_seq = _dystream_model.inpainting_length
    audio_self = np.concatenate([
        np.zeros(additional_motion_seq * int(audio_sr / pose_fps)),
        audio_self,
    ], axis=0)
    audio_tensor = torch.from_numpy(audio_self).float().unsqueeze(0).to(DEVICE)

    # Listener audio (optional)
    if listener_audio_path is not None and os.path.exists(listener_audio_path):
        audio_other, _ = librosa.load(listener_audio_path, sr=audio_sr)
        audio_other = np.concatenate([
            np.zeros(additional_motion_seq * int(audio_sr / pose_fps)),
            audio_other,
        ], axis=0)
        audio_other_tensor = torch.from_numpy(audio_other).float().unsqueeze(0).to(DEVICE)
    else:
        audio_other_tensor = torch.zeros_like(audio_tensor).to(DEVICE)

    # ── Step 4: Prepare motion latent input ──────────────────────────────
    #    Matches main.py _inference_one_file exactly:
    #      motion_latent = torch.from_numpy(...).unsqueeze(0)   # [1, N, 512]
    #      motion_latent_in = motion_latent[:,0:1,:].repeat(1,t,1)
    #      anchor_motion = motion_latent[:,0:1,:]
    progress(0.3, desc="Preparing inference input...")
    motion_latent = motion_latent_cpu.to(DEVICE)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.unsqueeze(0)  # [1, 512]
    if motion_latent.dim() == 2:
        motion_latent = motion_latent.unsqueeze(0)  # [1, N, 512]
    # Take first frame only (same as main.py: motion_latent[:,0:1,:])
    t = audio_tensor.shape[1] // int(audio_sr / pose_fps)
    motion_latent_in = motion_latent[:, 0:1, :].repeat(1, t, 1)  # [1, T, 512]

    # ── Step 5: Override CFG parameters ──────────────────────────────────
    _dystream_model.cfg_audio = cfg_audio
    _dystream_model.cfg_audio_other = cfg_audio_other
    _dystream_model.cfg_anchor = cfg_anchor
    _dystream_model.cfg_all = cfg_all

    # ── Step 6: Run DyStream inference ───────────────────────────────────
    progress(0.4, desc="Generating motion sequence...")
    denoising_steps = int(denoising_steps)

    if _dystream_ema is not None:
        _dystream_ema.to(DEVICE)
        ctx = _dystream_ema.average_parameters(_dystream_model.parameters())
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        motion_latent_pred = _dystream_model.inference(
            audio_tensor,
            audio_other=audio_other_tensor,
            init_motion=motion_latent_in,
            cond_motion=motion_latent_in,
            anchor_motion=motion_latent[:, 0:1, :],
            noise_scheduler=_noise_scheduler,
            num_inference_steps=denoising_steps,
        )
        # Remove the inpainting prefix
        motion_latent_pred = motion_latent_pred[:, additional_motion_seq:]

    progress(0.7, desc="Rendering video frames...")

    # ── Step 7: Convert motion latents to video ──────────────────────────
    video_frames = latents_to_video_frames(motion_latent_pred, resized_pil)

    progress(0.9, desc="Compositing video...")

    # ── Step 8: Save video ───────────────────────────────────────────────
    # Determine which audio to mux with the final video:
    #   - If video_audio_path is provided (e.g. full mixed audio), use it
    #   - Else if listener audio exists, mix speaker + listener into one track
    #   - Else use speaker audio only
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.mp4")

    final_audio = video_audio_path
    if final_audio is None or not os.path.exists(final_audio):
        if listener_audio_path is not None and os.path.exists(listener_audio_path):
            # Mix speaker + listener audio for the video soundtrack
            try:
                import soundfile as sf
                sp, sr1 = librosa.load(speaker_audio_path, sr=None)
                ls, sr2 = librosa.load(listener_audio_path, sr=sr1)
                min_len = min(len(sp), len(ls))
                mixed = sp[:min_len] + ls[:min_len]
                mixed_path = os.path.join(output_dir, "mixed_audio.wav")
                sf.write(mixed_path, mixed, sr1)
                final_audio = mixed_path
            except Exception as e:
                print(f"[Warning] Failed to mix audio: {e}, using speaker audio only.")
                final_audio = speaker_audio_path
        else:
            final_audio = speaker_audio_path

    save_video_with_audio(video_frames, final_audio, output_path, fps=pose_fps)

    progress(1.0, desc="Done!")

    num_frames = motion_latent_pred.shape[1]
    duration = num_frames / pose_fps
    info_text = (
        f"Frames: {num_frames}\n"
        f"Duration: {duration:.2f}s\n"
        f"FPS: {pose_fps}\n"
        f"Denoising Steps: {denoising_steps}\n"
        f"CFG (audio): {cfg_audio}, CFG (listener): {cfg_audio_other}\n"
        f"CFG (anchor): {cfg_anchor}, CFG (all): {cfg_all}"
    )

    return output_path, resized_pil, masked_pil, info_text


# ────────────────────────────────────────────────────────────────────────────
# Demo with pre-loaded samples
# ────────────────────────────────────────────────────────────────────────────

def update_sample_preview(sample_choice):
    """Return preview assets (face image, speaker audio, listener audio) for the selected sample."""
    if sample_choice == "Sample 1: Dyadic Conversation":
        image_path = os.path.join(PROJECT_ROOT, "img_files", "3.png")
        speaker_audio = os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_1.wav")
        listener_audio = os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_0.wav")
    else:
        image_path = os.path.join(PROJECT_ROOT, "img_files", "11.png")
        speaker_audio = os.path.join(PROJECT_ROOT, "wav_files", "11.wav")
        listener_audio = None

    return image_path, speaker_audio, listener_audio


def run_sample_demo(sample_choice, denoising_steps, cfg_audio, cfg_audio_other, cfg_anchor, cfg_all, progress=gr.Progress(track_tqdm=True)):
    """Run inference on pre-loaded sample data using the full pipeline from raw image."""
    if sample_choice == "Sample 1: Dyadic Conversation":
        image_path = os.path.join(PROJECT_ROOT, "img_files", "3.png")
        speaker_audio = os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_1.wav")
        listener_audio = os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_0.wav")
        # Full combined audio for the final video (same as main.py uses audio_path)
        full_audio = os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_full.wav")
    else:
        image_path = os.path.join(PROJECT_ROOT, "img_files", "11.png")
        speaker_audio = os.path.join(PROJECT_ROOT, "wav_files", "11.wav")
        listener_audio = None
        full_audio = None

    image_pil = Image.open(image_path).convert("RGB")
    return run_inference(
        image_pil, speaker_audio, listener_audio,
        denoising_steps, cfg_audio, cfg_audio_other, cfg_anchor, cfg_all,
        progress=progress,
        video_audio_path=full_audio,
    )


# ────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 1.5em;
    font-size: 1.1em;
}
.param-group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
}
"""

def build_ui():
    with gr.Blocks(
        title="DyStream - Streaming Dyadic Talking Heads Generation via FlowMatching-based Autoregressive Model",
    ) as demo:
        # ── Header ──
        gr.Markdown(
            "# DyStream: Streaming Dyadic Talking Heads Generation via FlowMatching-based Autoregressive Model",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "Upload a reference face image and audio to generate a talking head video. "
            "Supports both single-speaker and dyadic conversation modes.",
            elem_classes=["subtitle"],
        )

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════
            # Tab 1 - Sample Demos
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Sample Demos"):
                gr.Markdown("### Try with pre-loaded samples")
                with gr.Row():
                    with gr.Column(scale=1):
                        sample_choice = gr.Radio(
                            choices=[
                                "Sample 1: Dyadic Conversation",
                                "Sample 2: Speaker Only",
                            ],
                            value="Sample 1: Dyadic Conversation",
                            label="Select Sample",
                        )

                        gr.Markdown("### Inference Parameters")
                        with gr.Group(elem_classes=["param-group"]):
                            sample_denoising_steps = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Denoising Steps",
                            )
                            with gr.Row():
                                sample_cfg_audio = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.5, step=0.1,
                                    label="CFG Audio",
                                )
                                sample_cfg_audio_other = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.5, step=0.1,
                                    label="CFG Listener",
                                )
                            with gr.Row():
                                sample_cfg_anchor = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.0, step=0.1,
                                    label="CFG Anchor",
                                )
                                sample_cfg_all = gr.Slider(
                                    minimum=0, maximum=3.0, value=1.0, step=0.1,
                                    label="CFG All",
                                )

                        sample_btn = gr.Button(
                            "Run Sample",
                            variant="primary",
                            size="lg",
                        )

                        # Dynamic preview of selected sample inputs
                        gr.Markdown("### Selected Sample Inputs")
                        sample_preview_image = gr.Image(
                            value=os.path.join(PROJECT_ROOT, "img_files", "3.png"),
                            label="Reference Face Image",
                            height=200,
                            interactive=False,
                        )
                        sample_preview_speaker_audio = gr.Audio(
                            value=os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_1.wav"),
                            label="Speaker Audio",
                            interactive=False,
                        )
                        sample_preview_listener_audio = gr.Audio(
                            value=os.path.join(PROJECT_ROOT, "wav_files", "_sgIH81kj78-Scene-005+audio_v3_0.wav"),
                            label="Listener Audio",
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        sample_output_video = gr.Video(
                            label="Generated Talking Head Video",
                            height=400,
                        )
                        with gr.Row():
                            sample_output_resized = gr.Image(
                                label="Preprocessed Image",
                                height=200,
                            )
                            sample_output_masked = gr.Image(
                                label="Masked Image",
                                height=200,
                            )
                        sample_output_info = gr.Textbox(
                            label="Generation Info",
                            lines=6,
                            interactive=False,
                        )

                sample_btn.click(
                    fn=run_sample_demo,
                    inputs=[
                        sample_choice, sample_denoising_steps,
                        sample_cfg_audio, sample_cfg_audio_other, sample_cfg_anchor, sample_cfg_all,
                    ],
                    outputs=[sample_output_video, sample_output_resized, sample_output_masked, sample_output_info],
                )

                # Update preview when sample selection changes
                sample_choice.change(
                    fn=update_sample_preview,
                    inputs=[sample_choice],
                    outputs=[sample_preview_image, sample_preview_speaker_audio, sample_preview_listener_audio],
                )

            # ══════════════════════════════════════════════════════════════
            # Tab 2 - Custom Input
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Custom Input"):
                with gr.Row():
                    # ── Left: inputs ──
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        image_input = gr.Image(
                            label="Reference Face Image",
                            type="pil",
                            height=300,
                        )
                        speaker_audio = gr.Audio(
                            label="Speaker Audio (required)",
                            type="filepath",
                        )
                        listener_audio = gr.Audio(
                            label="Listener Audio (optional, for dyadic mode)",
                            type="filepath",
                        )

                        gr.Markdown("### Inference Parameters")
                        with gr.Group(elem_classes=["param-group"]):
                            denoising_steps = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Denoising Steps",
                            )
                            with gr.Row():
                                cfg_audio = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.5, step=0.1,
                                    label="CFG Audio",
                                )
                                cfg_audio_other = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.5, step=0.1,
                                    label="CFG Listener",
                                )
                            with gr.Row():
                                cfg_anchor = gr.Slider(
                                    minimum=0, maximum=3.0, value=0.0, step=0.1,
                                    label="CFG Anchor",
                                )
                                cfg_all = gr.Slider(
                                    minimum=0, maximum=3.0, value=1.0, step=0.1,
                                    label="CFG All",
                                )

                        generate_btn = gr.Button(
                            "Generate Video",
                            variant="primary",
                            size="lg",
                        )

                    # ── Right: outputs ──
                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        output_video = gr.Video(
                            label="Generated Talking Head Video",
                            height=400,
                        )
                        with gr.Row():
                            output_resized = gr.Image(
                                label="Preprocessed Image",
                                height=200,
                            )
                            output_masked = gr.Image(
                                label="Masked Image",
                                height=200,
                            )
                        output_info = gr.Textbox(
                            label="Generation Info",
                            lines=6,
                            interactive=False,
                        )

                generate_btn.click(
                    fn=run_inference,
                    inputs=[
                        image_input, speaker_audio, listener_audio,
                        denoising_steps, cfg_audio, cfg_audio_other, cfg_anchor, cfg_all,
                    ],
                    outputs=[output_video, output_resized, output_masked, output_info],
                )

            # ══════════════════════════════════════════════════════════════
            # Tab 3 - About
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("About"):
                gr.Markdown("""
## DyStream: Streaming Dyadic Talking Heads Generation via FlowMatching-based Autoregressive Model

### Introduction
DyStream is a flow matching-based autoregressive model for generating talking head videos from dyadic audio in realtime.

For more details, please refer to the [website](https://robinwitch.github.io/DyStream-Page) and [paper](https://arxiv.org/pdf/2512.24408).

### Supported Modes
- **Speaker Only**: Generates talking head motion using only the speaker's audio.
- **Dyadic Conversation**: Uses both speaker and listener audio to generate more natural conversational motion.

### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| Denoising Steps | Flow matching sampling steps | 5 |
| CFG Audio | Guidance strength for speaker audio | 0.5 |
| CFG Listener | Guidance strength for listener audio | 0.5 |
| CFG Anchor | Guidance strength for anchor motion | 0.0 |
| CFG All | Global guidance strength | 1.0 |
                """)

    return demo


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure localhost is not routed through proxy
    import os as _os
    for _var in ("no_proxy", "NO_PROXY"):
        _cur = _os.environ.get(_var, "")
        if "localhost" not in _cur:
            _os.environ[_var] = f"localhost,127.0.0.1,{_cur}" if _cur else "localhost,127.0.0.1"

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
