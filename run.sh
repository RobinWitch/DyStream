#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"
echo "Current working directory: $(pwd)"

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config configs/motion_gen/sample.yaml \
    --override \
    exp_name=test \
    model.module_name=model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder \
    resume_ckpt=checkpoints/last.ckpt

echo ""
echo "Finished! Videos saved to outputs/ "

