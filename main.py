# Standard library imports
import os
import json
from datetime import datetime
import time
import random

# Third-party library imports
import numpy as np
import librosa
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from lightning import LightningModule
from lightning import Trainer, seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from pytorch_fid.fid_score import calculate_frechet_distance

# Local imports
from utils import save_config_and_codes, instantiate_motion_gen, load_metrics, load_config

# short utility functions
def custom_collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# inference callback
class InferenceCallback(Callback):
    def __init__(self, save_dir, inference_step, steps_interval):
        self.save_dir = save_dir
        self.inference_step = inference_step
        self.steps_interval = steps_interval
        
    def on_train_start(self, trainer, pl_module):
        pl_module.logger.experiment.define_metric("videos", step_metric="inference_steps")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero and ((trainer.global_step % self.steps_interval == 0) or (pl_module.cfg.test_first and trainer.global_step == 1)):
            test_save_path = os.path.join(self.save_dir, f"test_{trainer.global_step}")
            os.makedirs(test_save_path, exist_ok=True)
            infer_dict = self.inference_step(
                pl_module.cfg.config.data.test_meta_paths,  # correct test path from config
                test_save_path,
                steps=trainer.global_step,
                noise_scheduler=pl_module.val_noise_scheduler
            )
            for key, value in infer_dict["metrics"].items():
                pl_module.log(f"{key}", value, on_step=True, on_epoch=False)
            
            for key, save_video_dir_dya in infer_dict["saved_videos"].items():
                video_to_log = []
                if not os.path.exists(save_video_dir_dya): continue
                # sort the files by name
                for save_file in sorted(os.listdir(save_video_dir_dya)):
                    if save_file.endswith(".mp4"):  
                        wandb_video = wandb.Video(os.path.join(save_video_dir_dya, save_file), caption=f"{trainer.global_step:06d}-{save_file}")
                        video_to_log.append(wandb_video)
                    if len(video_to_log) > 50: break
                pl_module.logger.experiment.log({f"videos_{key}": video_to_log, "inference_steps": trainer.global_step})
                
        if trainer.is_global_zero and pl_module.cfg.is_test:
            trainer.should_stop = True 

# motion gen lightning module
class MotionGenLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate_motion_gen(module_name=cfg.model.module_name, class_name=cfg.model.class_name, cfg=cfg.model, hfstyle=False)
        
        # Move model to device first
        self.model = self.model.to(self.device)
        for name, param in self.model.named_parameters():
            if "freeze" in name:
                param.requires_grad = False
                rank_zero_info(f"Freezing {name}")
        
        # Initialize EMA with model parameters that are already on the correct device
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=cfg.model.ema_decay)
        
        # noise schedulers
        self.val_noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler_kwargs)
        self.train_noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler_kwargs)
        
        # logging
        self.last_batch_end_time, self.batch_ready_time = None, None
        # validation outputs for FID calculation
        self.validation_step_outputs = []
        self.threshold = cfg.threshold
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.solver.learning_rate,
            betas=(self.cfg.solver.adam_beta1, self.cfg.solver.adam_beta2),
            weight_decay=self.cfg.solver.adam_weight_decay,
            eps=self.cfg.solver.adam_epsilon
        )
        lr_scheduler = get_scheduler(
            self.cfg.solver.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.solver.lr_warmup_steps * self.cfg.solver.gradient_accumulation_steps,
            num_training_steps=self.cfg.solver.max_train_steps * self.cfg.solver.gradient_accumulation_steps
        )
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": "step",  
            "frequency": 1
        }
    }

    @staticmethod
    def denoising_loss_fn(cfg, model_pred, target, noise_scheduler, timesteps, frame_weight=3.0):

        bs, t, d = model_pred.shape
        
        per_frame_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")  # [bs, t, d]
        per_frame_loss = per_frame_loss.mean(dim=-1)  # [bs, t] 
        

        weights = torch.ones_like(per_frame_loss)  # [bs, t]
        if t > 5:
            weights[:, -5:] = frame_weight
        else:
            weights[:, :] = frame_weight 
        
        weighted_loss = (per_frame_loss * weights).sum() / weights.sum()
        
        return weighted_loss
    
    @staticmethod
    def parse_name(fname):
        base = fname[:-4]
        parts = base.split('_')
        return '_'.join(parts[:2]), '_'.join(parts[2:])
    

    def _step(self, batch, is_training=True):
        audio = batch["audio"].to(self.device)
        motion_latent = batch["motion_latent"].to(self.device)
        style_latent = batch["style_latent"].to(self.device)
        audio_other = batch["audio_other"].to(self.device)
        style_latent_other = batch["style_latent_other"].to(self.device)
        
        # am_ratio = int(self.cfg.model.audio_fps/self.cfg.model.pose_fps)
        # motion_latent = motion_latent[:, self.cfg.model.prev_audio_frames:]
        # audio, prev_audio = audio[:, self.cfg.model.prev_audio_frames*am_ratio:], audio[:, self.cfg.model.seed_frames * am_ratio : (self.cfg.model.seed_frames + self.cfg.model.prev_audio_frames) * am_ratio]
        # audio_other, prev_audio_other = audio_other[:, self.cfg.model.prev_audio_frames*am_ratio:], audio_other[:, self.cfg.model.seed_frames * am_ratio : (self.cfg.model.seed_frames + self.cfg.model.prev_audio_frames) * am_ratio]
        # style_latent = style_latent[:, self.cfg.model.prev_audio_frames:]
        # style_latent_other = style_latent_other[:, self.cfg.model.prev_audio_frames:]

        # cond_motion, motion_latent = motion_latent[:, :self.cfg.model.seed_frames], motion_latent[:, self.cfg.model.seed_frames:]
        # audio = audio[:, int(self.cfg.model.seed_frames*self.cfg.model.audio_fps/self.cfg.model.pose_fps):]
        # audio_other = audio_other[:, int(self.cfg.model.seed_frames*self.cfg.model.audio_fps/self.cfg.model.pose_fps):]
        # style_latent = style_latent[:, self.cfg.model.seed_frames:]
        # style_latent_other = style_latent_other[:, self.cfg.model.seed_frames:]

        bs, t, _ = motion_latent.shape   
        prev_audio = torch.zeros([bs,80],device=audio.device)
        prev_audio_other = torch.zeros([bs,80],device=audio.device)
        noise = torch.randn_like(motion_latent)


        noise_scheduler = self.train_noise_scheduler if is_training else self.val_noise_scheduler
        indices = torch.randint(0, len(noise_scheduler.timesteps), (bs,))
        # please note that the timesteps are in reverse order inside the scheduler, 
        # the above indices is uniformly distributed, so there is no need to reverse the timesteps
        # if you design new indices sampling strategy, you need to reverse the timesteps
        timesteps = noise_scheduler.timesteps[indices].to(self.device)
        noisy_latents = noise_scheduler.scale_noise(sample=motion_latent, timestep=timesteps, noise=noise)
        
        threshold = self.threshold
        mask = torch.rand(bs, t,1, device=self.device) < threshold
        mask_noise_latent = noisy_latents * mask.float() + motion_latent * (1 - mask.float())    
    
        linear_weights = torch.linspace(0, 1, t, device=self.device).view(1, t, 1)  # shape: [1, t, 1]

        mask_noise_latent = mask_noise_latent * (1 - linear_weights) + motion_latent * linear_weights
        
        k = min(10, t)
        start_idx = t - k
        random_indices = torch.randint(start_idx, t, (bs,), device=motion_latent.device)

        anchor_motion = motion_latent[torch.arange(bs), random_indices].unsqueeze(1)
        
        
        motion_pred = self.model(
            face_latent_gt=mask_noise_latent,
            noise_face_latent=noisy_latents,
            time_step=timesteps,
            audio=audio,
            audio_other = audio_other,
            anchor_latent = anchor_motion,
            prev_audio=prev_audio,
            prev_audio_other=prev_audio_other,
        )


        target = motion_latent

        loss = self.denoising_loss_fn(self.cfg, motion_pred, target[:,1:], noise_scheduler, timesteps)
        return {
            'loss': loss,
            'real_motion': motion_latent,
            'pred_motion': motion_pred
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.last_batch_end_time = time.time()
        self.ema.to(self.device)
        self.ema.update()
        if self.global_step % 100 == 0:
            self.log('ema_decay', self.ema.decay, sync_dist=True)
            # Calculate average difference using vectorized operations
            with torch.no_grad():
                model_params = torch.cat([p.flatten() for p in self.model.parameters() if p.requires_grad])
                ema_params = torch.cat([
                    self.ema.shadow_params[i].flatten()
                    for i, (name, p) in enumerate(self.model.named_parameters()) if p.requires_grad
                ])
                avg_diff = torch.abs(model_params - ema_params).mean().item()
                self.log('ema_diff/avg', avg_diff, sync_dist=True)

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.time()

    def training_step(self, batch, batch_idx):
        net_start_time = time.time()
        result = self._step(batch, is_training=True)
        net_end_time = time.time()
        data_time = self.batch_ready_time - self.last_batch_end_time if self.last_batch_end_time is not None else 0.0
        net_time = net_end_time - net_start_time
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, prog_bar=True)
        self.log("data_time", data_time, on_step=True, prog_bar=True)
        self.log("net_time", net_time, on_step=True, prog_bar=True)
        self.log("train_loss", result['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return result['loss']

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters(self.model.parameters()):
            with torch.no_grad():
                result = self._step(batch, is_training=False)
        
        self.validation_step_outputs.append({
            'real_motion': result['real_motion'].cpu(),
            'pred_motion': result['pred_motion'].cpu()
        })
        
        self.log("val_loss", result['loss'], on_step=False, on_epoch=True, sync_dist=True)
        return result['loss']
    
    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return
            
        all_real = torch.cat([x['real_motion'] for x in self.validation_step_outputs], dim=0)
        all_pred = torch.cat([x['pred_motion'] for x in self.validation_step_outputs], dim=0)
        
        if self.trainer.world_size > 1:
            all_real = self.all_gather(all_real)
            all_pred = self.all_gather(all_pred)
            if all_real.dim() > 3:
                all_real = all_real.view(-1, all_real.size(-2), all_real.size(-1))
                all_pred = all_pred.view(-1, all_pred.size(-2), all_pred.size(-1))
        
        bs, n, d = all_real.shape
        real_features = all_real.view(-1, d).cpu().numpy()
        pred_features = all_pred.view(-1, d).cpu().numpy()
        
        if self.trainer.is_global_zero:
            try:
                mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
                mu2, sigma2 = np.mean(pred_features, axis=0), np.cov(pred_features, rowvar=False)
                fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                self.log("val_fid", fid_score, on_epoch=True, sync_dist=False)
            except Exception as e:
                rank_zero_info(f"FID calculation failed: {e}")
        
        self.validation_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        checkpoint['ema_state'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'ema_state' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state'])

        if self.cfg.solver.lr_reset:
            rank_zero_info("Resetting learning rate on resume.")
            new_lr = self.cfg.solver.learning_rate
            if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
                for param_group in checkpoint['optimizer_states'][0].get('param_groups', []):
                    param_group['lr'] = new_lr
                    rank_zero_info(f"  - New LR set to: {param_group['lr']}")
            if 'lr_schedulers' in checkpoint and checkpoint['lr_schedulers']:
                checkpoint['lr_schedulers'][0]['base_lrs'] = [new_lr]
                rank_zero_info(f"  - New LR Scheduler set to: {new_lr}")

    def _get_test_list(self, data_meta_path, dataset_type, save_path, select_range=None, is_test=False):
        test_list = []
        test_list.extend(json.load(open(data_meta_path, "r")))
        test_list = [item for item in test_list if item.get("mode") == "test_wild" and item.get("dataset_type") == dataset_type]
        test_list = sorted(test_list, key=lambda x: x["video_id"])
        
        if select_range is not None and not is_test:
            seen_ids = set()
            test_list = [item for item in test_list if not (self.parse_name(item["video_id"])[0] in seen_ids or seen_ids.add(self.parse_name(item["video_id"])[0]))]
            start, end = select_range
            test_list = test_list[start:end]

        current_save_path = os.path.join(save_path, dataset_type)
        os.makedirs(current_save_path, exist_ok=True)
        save_video_dir = os.path.join(current_save_path, 'single_reconstruct')
        test_loss = 0
        total_length = 0
        return test_list, current_save_path, save_video_dir, test_loss, total_length
    
    def _inference_one_file(self, test_file, test_loss, total_length, ref_img_path, save_path, calculate_loss=False):
        audio, _ = librosa.load(test_file["audio_self_path"], sr=self.cfg.model.audio_sr)
        additional_motion_seq = self.model.inpainting_length
        audio = np.concatenate([np.zeros((additional_motion_seq * int(self.cfg.model.audio_sr / self.cfg.model.pose_fps))),audio], axis=0)
        audio = torch.from_numpy(audio).to(self.device).unsqueeze(0)
        
        audio_other_path = test_file["audio_other_path"]
        if audio_other_path is not None:
            audio_other, _ = librosa.load(audio_other_path, sr=self.cfg.model.audio_sr)
            additional_motion_seq = self.model.inpainting_length
            audio_other = np.concatenate([np.zeros((additional_motion_seq * int(self.cfg.model.audio_sr / self.cfg.model.pose_fps))),audio_other], axis=0)
            audio_other = torch.from_numpy(audio_other).to(self.device).unsqueeze(0)
        else:
            audio_other = torch.zeros_like(audio).to(self.device)

        audio = audio.float()
        audio_other = audio_other.float()
        # motion seed
        try:
            motion_latent = np.load(test_file["motion_self_path"], allow_pickle=True)["motion_latent"]
        except:
            motion_latent = np.load(test_file["motion_self_path"], allow_pickle=True)["random_data"]
        motion_latent = torch.from_numpy(motion_latent).to(self.device).unsqueeze(0)
        t = audio.shape[1] // int(self.cfg.model.audio_sr / self.cfg.model.pose_fps)
        motion_latent_in = motion_latent[:,0:1,:].repeat(1,t,1)
        style_motion = None
        # print("audio", audio.shape, "motion_latent_in", motion_latent_in.shape)
        with torch.no_grad():
            motion_latent_pred = self.model.inference(
                audio, cond_motion=motion_latent_in, 
                audio_other=audio_other, 
                init_motion=motion_latent_in,
                anchor_motion=motion_latent[:,0:1,:],
                noise_scheduler = self.val_noise_scheduler,
                num_inference_steps = self.cfg.validation.denoising_steps,
                )
            motion_latent_pred = motion_latent_pred[:, additional_motion_seq:]

        if calculate_loss:
            minimum_length = min(t, motion_latent.shape[1],motion_latent_pred.shape[1])
            current_loss = torch.abs(motion_latent[:,0:minimum_length,:] - motion_latent_pred[:,0:minimum_length,:]).mean()
            test_loss += current_loss * t
        total_length += t   
        # save the latent
        np.savez(
            os.path.join(save_path, f"{test_file['video_id']}_cfg_fusion_{self.cfg.model.cfg_fusion}_cfg_prev_motion_{self.cfg.model.cfg_prev_motion}_cfg_anchor_{self.cfg.model.cfg_anchor}_output.npz"),
            motion_latent=motion_latent_pred.cpu().numpy(),
            audio_path=test_file["audio_path"],
            ref_img_path=ref_img_path,
            video_id=test_file["video_id"])
        return motion_latent_pred, test_loss, total_length

    def _render_and_evaluate(self, save_path, save_video_dir, metrics, dataset_type):
        
        current_path = os.getcwd()
        cmd_vis = (
            f"cd {self.cfg.tools_path}/tools/visualization_0416/ && "
            f"python latent_to_video.py "
            f"--save_fps {self.cfg.model.pose_fps} --npz_dir {save_path} "
            f"--save_dir {save_video_dir} --version '0506' "
            f"&& cd {current_path}"
        )
        rank_zero_info(f"Running command: {cmd_vis}")
        os.system(cmd_vis)

        if self.cfg.model.eval_metrics:
            cmd_eval = f"cd {self.cfg.tools_path}/tools/evaluation_video && python eval_all_in_one.py --video_pred_path {save_video_dir} --metrics lipsync var --verbose"
            rank_zero_info(f"Running command: {cmd_eval}")
            os.system(cmd_eval)
        
        text_path = os.path.join(save_video_dir, "metrics.txt")
        try: 
            metrics_saved = load_metrics(text_path)
            new_metrics = {}
            for key, value in metrics_saved.items():
                new_metrics[dataset_type+"_"+key] = value
            metrics.update(new_metrics)
        except:
            rank_zero_info(f"{dataset_type} metrics not saved")
        rank_zero_info(metrics)

    def _get_latent(self, test_file):
        current_path = os.getcwd()
        if not os.path.exists(test_file["motion_self_path"]):
            ori_img_path = test_file["resampled_video_path"].replace(".mp4", ".png")
            masked_img_path = ori_img_path.replace(".png", "_masked.png")
            video_path = test_file["resampled_video_path"]
            latent_path = test_file["motion_self_path"]
            
            if not os.path.exists(ori_img_path) and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(ori_img_path)
                    rank_zero_info(f"Saved first frame to {ori_img_path}")
                else:
                    print(f"Failed to extract frame from video: {video_path}")
                    return None
            ori_img_abs = os.path.abspath(ori_img_path)
            print(masked_img_path, ori_img_path)
            if not os.path.exists(masked_img_path) and os.path.exists(ori_img_path):
                cmd_mask = f"cd {self.cfg.tools_path}/tools/visualization_0416 && python img_to_mask.py --image_path {ori_img_abs} --save_path {ori_img_abs} --crop True --union_bbox_scale 1.6 && cd {current_path}"
                rank_zero_info(f"Running command: {cmd_mask}")
                os.system(cmd_mask)

            ori_img_path = ori_img_path.replace(".png", "_resize.png")
            ori_resize_abs = os.path.abspath(ori_img_path)
            latent_path_abs = os.path.abspath(latent_path)
            cmd_latent = f"cd {self.cfg.tools_path}/tools/visualization_0416 && python img_to_latent.py --mask_image_path {ori_resize_abs} --save_npz_path {latent_path_abs} --version {self.cfg.model.version} && cd {current_path}"
            rank_zero_info(f"Running command: {cmd_latent}")
            os.system(cmd_latent)
        else:
            if ".png" in test_file["resampled_video_path"] and "resize" not in test_file["resampled_video_path"]:
                png_path = test_file["resampled_video_path"].replace(".png", "_resize.png")
            else:
                png_path = test_file["resampled_video_path"].replace(".mp4", "_resize.png") 
            # print(png_path)
            if os.path.exists(png_path):
                ori_img_path = png_path
            else:
                ori_img_path = test_file["motion_self_path"].replace(f"motion_latent_{self.cfg.model.version}", "original_video").replace(".npz", ".mp4")
        return ori_img_path

    def inference_step(self, test_path, save_path, **kwargs):
        with self.ema.average_parameters(self.model.parameters()):
            # init 
            steps = self.global_step
            noise_scheduler = self.val_noise_scheduler
            actual_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            actual_model.eval()
            metrics = {}  
            save_videos = {}

            if self.cfg.model.infer_32:
                pose_length_backup = actual_model.cfg.pose_length
                actual_model.cfg.pose_length = 32
            if test_path is None:
                test_path = []
            # short video
            all_test_list = []
            for data_meta_path in test_path:
                test_list, short_vid_save_path, short_vid_save_video_dir, test_loss, total_length = self._get_test_list(data_meta_path, "speaker_only", save_path, select_range=(0, 4), is_test=True)
                all_test_list.extend(test_list)
            # for data_meta_path in test_path:
            #     test_list, _, _, _, _ = self._get_test_list(data_meta_path, "dyadic", save_path, select_range=(0, 4))
            #     all_test_list.extend(test_list)
            # for data_meta_path in test_path:
            #     test_list, _, _, _, _ = self._get_test_list(data_meta_path, "bad_cases", save_path, select_range=(0, 4))
            #     all_test_list.extend(test_list)
            test_list = all_test_list
            rank_zero_info(f"Test Short Video List: {test_list}")
            if len(test_list) > 0:
                for test_file in tqdm(test_list, desc="Testing Short Video"):
                    ref_img_path = self._get_latent(test_file)
                    if ref_img_path is None: continue
                    motion_latent_pred, test_loss, total_length = self._inference_one_file(test_file, test_loss, total_length, ref_img_path, short_vid_save_path, calculate_loss=False)
                # metrics["speaker_only_latent_l1"] = test_loss.cpu().numpy()/total_length
                self._render_and_evaluate(short_vid_save_path, short_vid_save_video_dir, metrics, "speaker_only")
                save_videos["speaker_only"] = short_vid_save_video_dir
            
            # short video testset 2
            all_test_list = []
            for data_meta_path in test_path:
                test_list, interal_save_path, interal_save_video_dir, test_loss, total_length = self._get_test_list(data_meta_path, "internal", save_path, select_range=(0, 4), is_test=True)
                all_test_list.extend(test_list)
            test_list = all_test_list
            rank_zero_info(f"Test Short Video Dataset 2 List: {test_list}")
            if len(test_list) > 0:
                for test_file in tqdm(test_list, desc="Testing Short Video Dataset 2"):
                    ref_img_path = self._get_latent(test_file)
                    if ref_img_path is None: continue
                    motion_latent_pred, test_loss, total_length = self._inference_one_file(test_file, test_loss, total_length, ref_img_path, interal_save_path, calculate_loss=False)
                self._render_and_evaluate(interal_save_path, interal_save_video_dir, metrics, "internal")
                save_videos["internal"] = interal_save_video_dir

            # long video
            all_test_list = []
            for data_meta_path in test_path:
                test_list, infp_save_path, infp_save_video_dir, test_loss, total_length = self._get_test_list(data_meta_path, "dyadic", save_path, select_range=(0, 4), is_test=True)
                all_test_list.extend(test_list)
            test_list = all_test_list
            rank_zero_info(f"Test Long Video List: {test_list}")
            if len(test_list) > 0:
                for test_file in tqdm(test_list, desc="Testing Long Video"):
                    # get latent 
                    ref_img_path = self._get_latent(test_file)
                    if ref_img_path is None: continue
                    motion_latent_pred, test_loss, total_length = self._inference_one_file(test_file, test_loss, total_length, ref_img_path, infp_save_path, calculate_loss=False)
                self._render_and_evaluate(infp_save_path, infp_save_video_dir, metrics, "dyadic")
                save_videos["dyadic"] = infp_save_video_dir

            if self.cfg.model.infer_32:
                actual_model.cfg.pose_length = pose_length_backup
            return {
                "saved_videos": save_videos,
                "metrics": metrics
            }
                        

def main():
    # init
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(cfg.save_dir, exist_ok=True)    
    rank_zero_info(f"Save dir: {cfg.save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}")
    
    logger = None
    if not cfg.debug and cfg.get('logger', None) is not None:
        os.environ["WANDB_API_KEY"] = cfg.logger.wandb.wandb_key
        logger = WandbLogger(
            project=cfg.logger.wandb.project,
            name=cfg.exp_name,
            #entity=cfg.logger.wandb.entity,
            config=OmegaConf.to_container(cfg.config, resolve=True),
            dir=cfg.save_dir
        )
    
    # dataloader
    train_dataset = instantiate_motion_gen(
        module_name=cfg.data.module_name,
        class_name=cfg.data.class_name,
        cfg=cfg.config,
        split="train"
    )
    test_dataset = instantiate_motion_gen(
        module_name=cfg.data.module_name,
        class_name=cfg.data.class_name,
        cfg=cfg.config,
        split="test_wild"
    )
    rank_zero_info(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        persistent_workers=True,
        prefetch_factor=8,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
    )
    
    # model
    model = MotionGenLightningModule(cfg)
    # when and how to save the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{step}",   
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,                
        save_last=True,               
        save_on_train_epoch_end=False,  
    )
    # when and how to do inference
    inference_callback = InferenceCallback(cfg.save_dir, model.inference_step, cfg.validation.test_steps)
    # when to do validation
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_gpus = len(cuda_visible_devices.split(","))
    max_validation_steps = len(train_dataloader) // num_gpus
    validation_steps = min(cfg.validation.validation_steps, max_validation_steps)
    rank_zero_info(f"val every {validation_steps} steps (requested: {cfg.validation.validation_steps}, max: {max_validation_steps}), dataloader len {len(train_dataloader)}, gpus {num_gpus}")
    # trainer
    if not cfg.test:
        callbacks = [checkpoint_callback, inference_callback]
    else:
        callbacks = [inference_callback]
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        val_check_interval=validation_steps,  
        default_root_dir=cfg.save_dir,
    )
    # backup config and codes
    if trainer.is_global_zero: save_config_and_codes(cfg, cfg.save_dir)
    
    if cfg.is_test:
        rank_zero_info("Running in inference-only mode (is_test=True)")
        if cfg.resume_ckpt:
            checkpoint = torch.load(cfg.resume_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            if 'ema_state' in checkpoint:
                model.ema.load_state_dict(checkpoint['ema_state'])
            rank_zero_info(f"Loaded checkpoint from {cfg.resume_ckpt}")
        
        model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            result = model.inference_step(
                cfg.config.data.test_meta_paths,
                cfg.save_dir,
                steps=0,
                noise_scheduler=model.val_noise_scheduler
            )
        rank_zero_info(f"Inference completed. Results: {result['metrics']}")
        rank_zero_info(f"Videos saved to: {result['saved_videos']}")
    else:
        trainer.validate(model, dataloaders=test_dataloader)
        trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader, ckpt_path=cfg.resume_ckpt)
    
    if not cfg.debug: wandb.finish()

# infer 
if __name__ == "__main__":
    main()