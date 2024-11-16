#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --output=sbatch_out/finetune.%A.%a.out
#SBATCH --error=sbatch_err/finetune.%A.%a.err
#SBATCH --job-name=labram_finetune

. /etc/profile
module load anaconda/3
conda activate labram

export CUDA_LAUNCH_BLOCKING=1

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
        --output_dir ./checkpoints/finetune_alex_physion_v2/ \
        --log_dir ./log/finetune_alex_physion_v2/ \
        --model labram_base_patch200_200 \
        --finetune ./checkpoints/labram_base_alex_physion_150/checkpoint-149.pth \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 15 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0