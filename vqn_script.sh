#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/vqn.%A.%a.out
#SBATCH --error=sbatch_err/vqn.%A.%a.err
#SBATCH --job-name=vqn


. /etc/profile
module load anaconda/3
conda activate labram


OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 run_vqnsp_training.py \
    --output_dir ./checkpoints/vqnsp/ \
    --log_dir ./log/vqnsp/ \
    --model vqnsp_encoder_base_decoder_3x200x12 \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 64 \
    --quantize_kmeans_init \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4  \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 20 