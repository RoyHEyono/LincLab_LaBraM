#!/usr/bin/env bash
#SBATCH --array=0-728%100  # 729 random configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --output=sbatch_out/arrayjob/wandb_finetune.%A.%a.out
#SBATCH --error=sbatch_err/arrayjob/wandb_finetune.%A.%a.err
#SBATCH --job-name=labram_arrayjob

# Load environment
. /etc/profile
module load anaconda/3
conda activate labram

export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=""
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# Load parameters from file
random_configs_file='labram_hparams.json'
random_index=$((SLURM_ARRAY_TASK_ID + 100))
random_params=$(python -c "import json; import sys; f=open('$random_configs_file'); configs=json.load(f); f.close(); print(json.dumps(configs[$random_index]))")
lr=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr'])")
weight_decay=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['weight_decay'])")
drop=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['drop'])")
layer_decay=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['layer_decay'])")
batch_size=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['batch_size'])")
drop_path=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['drop_path'])")

master_port=$((29400 + SLURM_ARRAY_TASK_ID))
OMP_NUM_THREADS=1 torchrun --rdzv_endpoint=localhost:$master_port --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
        --output_dir ./checkpoints/finetune_paper_physion_onlyoutlier/ \
        --log_dir ./log/finetune_paper_physion_onlyoutlier/ \
        --model labram_base_patch200_200 \
        --finetune ./checkpoints/paper_labram-base.pth \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --lr $lr \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 50 \
        --layer_decay $layer_decay \
        --drop_path $drop_path \
        --drop $drop \
        --dist_eval \
        --save_ckpt_freq 45 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0
