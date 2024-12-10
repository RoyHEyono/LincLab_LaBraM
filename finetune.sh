#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32GB
#SBATCH --time=90:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --output=sbatch_out/wandb_finetune.%A.%a.out
#SBATCH --error=sbatch_err/wandb_finetune.%A.%a.err
#SBATCH --job-name=labram_wandb

. /etc/profile
module load anaconda/3
conda activate labram

export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=""
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

OMP_NUM_THREADS=1 torchrun --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
        --output_dir ./checkpoints/finetune_paper_physion_onlyoutlier/ \
        --log_dir ./log/finetune_paper_physion_onlyoutlier/ \
        --model labram_base_patch200_200 \
        --finetune ./checkpoints/paper_labram-base.pth \
        --weight_decay 0.5 \
        --batch_size 32 \
        --lr 0.0001 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 50 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 45 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0