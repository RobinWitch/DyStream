# DyStream: Streaming Dyadic Talking Heads Generation via Flow Matching-based Autoregressive Model

[Paper](https://arxiv.org/pdf/2512.24408) | [Webpage](https://robinwitch.github.io/DyStream-Page)

We are gradually releasing the code for this project.

## TODO List
- [x] Offline video generation
- [ ] Online video generation
- [ ] Training code

## Setup

### Environment
Create a Python environment using conda:
```bash
conda create -n dystream_py11 python=3.11
conda activate dystream_py11
pip install -r requirements.txt
```

### Download Checkpoints
Download the required checkpoints and tools:
```bash
git clone https://huggingface.co/robinwitch/DyStream
cd DyStream
mv tools ../
mv checkpoints ../
cd ..
rm -rf DyStream
```

## Quick Start

Run the demo with a single command:
```bash
bash run.sh
```

## Running with Your Own Data

### Configuration
Configuration files can be referenced and changed in `data_json/sample_files.json`. We provide examples for two scenarios:
1. Speaker audio only
2. Speaker and listener audio tracks

### Scenario 1: Speaker Audio Only

Example configuration:
```json
{
    "origin_video_path": null,
    "resampled_video_path": "img_files/11.png",
    "audio_path": "wav_files/11.wav",
    "audio_self_path": "wav_files/11.wav",
    "audio_other_path": null,
    "motion_self_path": "img_files/11.npz",
    "motion_other_path": null,
    "mode": "test_wild",
    "dataset_type": "dyadic",
    "video_id": "single_speaker_11_11"
}
```

**To use your own image and audio:**
- Modify the following fields: `resampled_video_path`, `audio_path`, `audio_self_path`, `motion_self_path`, and `video_id`
- **Required files**: `resampled_video_path` and `audio_self_path` must exist
- `audio_path` should be identical to `audio_self_path` in this scenario
- `motion_self_path` can be set by changing the file extension of `resampled_video_path` to `.npz`. This file will be automatically generated during runtime if it doesn't exist
- `video_id` can be any identifier for organizing your outputs

### Scenario 2: Speaker and Listener Audio

Example configuration:
```json
{
    "origin_video_path": null,
    "resampled_video_path": "img_files/3.png",
    "audio_path": "wav_files/_sgIH81kj78-Scene-005+audio_full.wav",
    "audio_self_path": "wav_files/_sgIH81kj78-Scene-005+audio_v3_1.wav",
    "audio_other_path": "wav_files/_sgIH81kj78-Scene-005+audio_v3_0.wav",
    "motion_self_path": "img_files/3.npz",
    "motion_other_path": null,
    "mode": "test_wild",
    "dataset_type": "dyadic",
    "video_id": "_sgIH81kj78-Scene-005+audio_v3_2"
}
```

**To use your own image and audio:**
- Modify the following fields: `resampled_video_path`, `audio_path`, `audio_self_path`, `audio_other_path`, `motion_self_path`, and `video_id`
- **Required files**: `resampled_video_path`, `audio_self_path`, and `audio_other_path` must exist
- `audio_self_path`: speaker audio track
- `audio_other_path`: listener audio track
- `audio_path`: combined audio containing both speaker and listener tracks. This is only used for final video rendering and audio merging, not for inference
- `motion_self_path` can be set by changing the file extension of `resampled_video_path` to `.npz`. This file will be automatically generated during runtime if it doesn't exist
- `video_id` can be any identifier for organizing your outputs



## Citation
If you find this work useful, please consider citing:
```bibtex
@article{chen2025dystream,
  title={DyStream: Streaming Dyadic Talking Heads Generation via Flow Matching-based Autoregressive Model},
  author={Bohong Chen and Haiyang Liu},
  journal={ArXiv},
  year={2025},
  volume={abs/2512.24408},
}
```