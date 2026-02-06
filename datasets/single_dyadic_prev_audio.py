import random
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import json 
import librosa

class DyadicTalking(Dataset):
    def __init__(
        self,
        cfg=None,
        split='train',
    ):
        super().__init__()
        self.cfg = cfg

        vid_meta = []
        if split == "train":
            data_meta_path = cfg.data.meta_paths
        else:
            data_meta_path = cfg.data.val_meta_paths
        for data_meta_path in data_meta_path:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]

        self.data_list = self.vid_meta
        self.fps = cfg.model.pose_fps
        self.audio_sr = cfg.model.audio_sr
        self.training_mode = cfg.data.training_mode
        self.mode = split
        
        self.frame_length = cfg.model.cbh_window_length
        # self.mean = 0
        # self.std = 1
        # self.generate_cache = False
        # self.face_id = 0

    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        # return motion * torch.from_numpy(std).to(motion.device) + torch.from_numpy(mean).to(motion.device)
        return motion * torch.tensor(std).to(motion.device) + torch.tensor(mean).to(motion.device)
    
    def __len__(self):
        return len(self.data_list)
    
    def get_item(self, item):
        data_item = self.data_list[item]
        # meta information
        # video_metadata = np.load(data_item["metadata_path"], allow_pickle=True)["arr_0"].tolist()
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        edx = sdx+self.frame_length
        dataset_type = data_item["dataset_type"]
        face_id = random.choice([0, 1]) if dataset_type == "dyadic" else 0
        if self.training_mode is not None and dataset_type == "dyadic" and self.mode == "train":
            self_cond = data_item["self_syncnet_conf"] 
            other_cond = data_item["other_syncnet_conf"]
            if self.training_mode == "speaker_only":
                if self_cond > other_cond:
                    face_id = 0
                else:
                    face_id = 1
            elif self.training_mode == "listener_only":
                if self_cond > other_cond:
                    face_id = 1
                else:
                    face_id = 0
        # audio 
        # here audio self means audio_face_id_0, audio_other means audio_face_id_1.
        audio_self_path, audio_other_path = data_item["audio_self_path"], data_item["audio_other_path"]
        if face_id == 1:
            audio_self_path, audio_other_path = audio_other_path, audio_self_path
        
        audio, _ = librosa.load(audio_self_path, sr=self.audio_sr)
        audio_old = audio.copy()
        sdx_audio = sdx * int((1 / self.cfg.model.pose_fps) * self.audio_sr)
        edx_audio = edx * int((1 / self.cfg.model.pose_fps) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
        audio_tensor = torch.from_numpy(audio).float()
        diff_audio = edx_audio - sdx_audio - audio_tensor.shape[0]
        if diff_audio != 0:
            print("padding audio", diff_audio, audio_tensor.shape[0], sdx_audio, edx_audio, audio_old.shape, sdx, edx, data_item["video_id"], data_item["frames"], data_item["audio_self_path"])
            if diff_audio > 0:
                audio_tensor = torch.cat([audio_tensor, audio_tensor[-1:].repeat(diff_audio)], dim=0)
            else:
                audio_tensor = audio_tensor[:self.cfg.model.pose_length]
                
        if audio_other_path is not None:
            audio_other, _ = librosa.load(audio_other_path, sr=self.audio_sr)
            audio_other = audio_other[sdx_audio:edx_audio]
            audio_other_tensor = torch.from_numpy(audio_other).float()
            diff_audio_other = edx_audio - sdx_audio - audio_other_tensor.shape[0]
            if diff_audio_other != 0:
                # print("padding audio_other", diff_audio_other)
                if diff_audio_other > 0:
                    audio_other_tensor = torch.cat([audio_other_tensor, audio_other_tensor[-1:].repeat(diff_audio_other)], dim=0)
                else:
                    audio_other_tensor = audio_other_tensor[:self.cfg.model.pose_length]
        else:
            audio_other_tensor = torch.zeros_like(audio_tensor)
        # print(audio_tensor.shape, audio_other_tensor.shape)

        # motion latent 
        motion_self_path, motion_other_path = data_item["motion_self_path"], data_item["motion_other_path"]
        if face_id == 1:
            motion_self_path, motion_other_path = motion_other_path, motion_self_path

        motion_dict = np.load(motion_self_path, allow_pickle=True)
        motion = motion_dict["random_data"][sdx:edx]
        # print(motion_dict["random_data"].shape, sdx, edx)
        # motion = self.normalize(motion, self.mean, self.std)        
        if np.random.rand() > self.cfg.data.random_mix:
            length = motion_dict["random_data"].shape[0] - self.cfg.model.pose_length - self.cfg.model.prev_audio_frames
            ref_sdx = np.random.randint(0, length)
            ref_edx = ref_sdx + self.cfg.model.pose_length + self.cfg.model.prev_audio_frames
            ref_motion = motion_dict["random_data"][ref_sdx:ref_edx]
        else:
            ref_motion = motion
        motion_tensor = torch.from_numpy(motion).float()
        ref_motion_tensor = torch.from_numpy(ref_motion).float()
        # if motion_tensor.shape[0] != (self.cfg.model.pose_length + self.cfg.model.prev_audio_frames):
        #     print("padding motion", motion_tensor.shape[0])
        #     # motion_tensor = torch.cat([motion_tensor, motion_tensor[-1:].repeat(self.cfg.model.pose_length - motion_tensor.shape[0], 1)], dim=0)
        
        if motion_other_path is not None:
            motion_dict_other = np.load(motion_other_path, allow_pickle=True)
            motion_other = motion_dict_other["random_data"][sdx:edx]
            if np.random.rand() > self.cfg.data.random_mix:
                length = motion_dict_other["random_data"].shape[0] - self.cfg.model.pose_length - self.cfg.model.prev_audio_frames
                ref_sdx = np.random.randint(0, length)
                ref_edx = ref_sdx + self.cfg.model.pose_length + self.cfg.model.prev_audio_frames
                ref_motion_other = motion_dict_other["random_data"][ref_sdx:ref_edx]
            else:
                ref_motion_other = motion_other
            motion_other_tensor = torch.from_numpy(motion_other).float()
            ref_motion_other_tensor = torch.from_numpy(ref_motion_other).float()
        else:
            motion_other_tensor, ref_motion_other_tensor = torch.zeros_like(motion_tensor), torch.zeros_like(ref_motion_tensor)
        
        # print(motion_tensor.shape, ref_motion_tensor.shape, motion_other_tensor.shape, ref_motion_other_tensor.shape)

        return dict(
            motion_latent=motion_tensor,
            audio=audio_tensor, 
            style_latent=ref_motion_tensor,
            motion_latent_other=motion_other_tensor,
            audio_other=audio_other_tensor,
            style_latent_other=ref_motion_other_tensor,
            video_id=data_item["video_id"],
        )

    def __getitem__(self, item):
        return self.get_item(item)
    
# CUDA_VISIABLE_DEVICES=3 torchrun --nproc_per_node 1 --nnodes 1 --master_port 29503 train_emage_audio.py --config ./configs/dyanic_base_2.yaml --evaluation --wandb     
if __name__ == "__main__": 
    import argparse
    from tqdm import tqdm
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="/home/weili/haiyang/PantoMatrix/configs/dyanic_base.yaml")
    parser.add_argument("--face_id", type=int, default=0)
    args = parser.parse_args()
    # train_test_spilt
    # generate_cache = True
    # face_id = args.face_id
    cfg = OmegaConf.load(args.cfg)
    dataset = DyadicTalking(cfg, split="train")
    print(len(dataset))
    # dataset.generate_cache = generate_cache
    # dataset.face_id = face_id
    dataloader= torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    fail_count = 0
    for idx, data in tqdm(enumerate(dataloader)):
        print(data.keys())
        print(data["motion_latent"].shape)
        print(data["audio"].shape)
        print(data["style_latent"].shape)
        print(data["motion_latent_other"].shape)
        print(data["audio_other"].shape)
        print(data["style_latent_other"].shape)
        print(data["face_video"].shape)
        print(data["hybrid_face_video"].shape)
        print(data["ref_face_img"].shape)
