#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=16GB
#SBATCH --time=60:00:00
#SBATCH --cpus-per-gpu=2
#SBATCH --output=sbatch_out/tokenizer.%A.%a.out
#SBATCH --error=sbatch_err/tokenizer.%A.%a.err
#SBATCH --job-name=vqn


. /etc/profile
module load anaconda/3
# conda create -n labram python=3.11
conda activate labram
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install tensorboardX
# pip install -r requirements.txt
export CUDA_LAUNCH_BLOCKING=1
# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 run_vqnsp_training.py \
#     --output_dir ./checkpoints/vqnsp_alex_physion_200/ \
#     --log_dir ./log/vqnsp_alex_physion_200/ \
#     --model vqnsp_encoder_base_decoder_3x200x12 \
#     --codebook_n_emd 8192 \
#     --codebook_emd_dim 64 \
#     --quantize_kmeans_init \
#     --batch_size 128 \
#     --opt adamw \
#     --opt_betas 0.9 0.99 \
#     --weight_decay 1e-4  \
#     --warmup_epochs 10 \
#     --epochs 200 \
#     --save_ckpt_freq 100 

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 run_labram_pretraining.py \
        --output_dir ./checkpoints/labram_base_alex_physion_150/ \
        --log_dir ./log/labram_base_alex_physion_150/ \
        --model labram_base_patch200_1600_8k_vocab \
        --input_size 1600 \
        --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
        --tokenizer_weight ./checkpoints/vqnsp_alex_physion_200/checkpoint-199.pth \
        --batch_size 64 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 150 \
        --save_ckpt_freq 50 \
        --codebook_dim 64 \
        --gradient_accumulation_steps 1
