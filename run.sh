#!/bin/bash

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=0
# # 340M model training scripts
export DELTA_UPDATE_RULE="delta"
NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/delta_net_340M/batch32.seqlen2048.warmup1024.update1.steps15000.lr3e-4 \
  --model.config configs/delta_net_340M.json \
  --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 2 \
  --training.steps 15000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /home/ubuntu/poria-cvpr-2026/leijingdi/Data/SlimPajama-627B-Merged \
  --training.dataset_split train \
  --training.num_workers 2 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.streaming \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2000 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1


export DELTA_UPDATE_RULE="efla"
NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/delta_net_340M_EFLA_scale_fixed_q_norm/batch32.seqlen2048.warmup1024.update1.steps15000.lr3e-4 \
  --model.config configs/delta_net_340M.json \
  --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.2 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 2 \
  --training.steps 15000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /home/ubuntu/poria-cvpr-2026/leijingdi/Data/SlimPajama-627B-Merged \
  --training.dataset_split train \
  --training.num_workers 2 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.streaming \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2000 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1









#### 1.3B model training scripts

# export DELTA_UPDATE_RULE="delta"
# NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
#   --job.config_file flame/models/fla.toml \
#   --job.dump_folder exp/delta_net_1B/batch32.seqlen2048.warmup1024.update1.steps24000.lr3e-4 \
#   --model.config configs/delta_net_1B.json \
#   --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
#   --optimizer.name AdamW \
#   --optimizer.eps 1e-15 \
#   --optimizer.lr 3e-4 \
#   --lr_scheduler.warmup_steps 1024 \
#   --lr_scheduler.lr_min 0.1 \
#   --lr_scheduler.decay_type cosine \
#   --training.batch_size 16 \
#   --training.seq_len 2048 \
#   --training.gradient_accumulation_steps 16 \
#   --training.steps 4000 \
#   --training.max_norm 1.0 \
#   --training.skip_nan_inf \
#   --training.dataset /home/ubuntu/poria-cvpr-2026/leijingdi/Data/SlimPajama-627B-Merged \
#   --training.dataset_split train \
#   --training.num_workers 2 \
#   --training.prefetch_factor 2 \
#   --training.seed 42 \
#   --training.compile \
#   --training.streaming \
#   --training.tensor_parallel_degree 1 \
#   --training.disable_loss_parallel \
#   --checkpoint.interval 500 \
#   --checkpoint.load_step -1 \
#   --metrics.log_freq 1

# export DELTA_UPDATE_RULE="efla"
# NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
#   --job.config_file flame/models/fla.toml \
#   --job.dump_folder exp/delta_net_1B_EFLA_scale_fixed/batch32.seqlen2048.warmup1024.update1.steps24000.lr3e-4 \
#   --model.config configs/delta_net_1B.json \
#   --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
#   --optimizer.name AdamW \
#   --optimizer.eps 1e-15 \
#   --optimizer.lr 3e-4 \
#   --lr_scheduler.warmup_steps 1024 \
#   --lr_scheduler.lr_min 0.2 \
#   --lr_scheduler.decay_type cosine \
#   --training.batch_size 16 \
#   --training.seq_len 2048 \
#   --training.gradient_accumulation_steps 16 \
#   --training.steps 4000 \
#   --training.max_norm 1.0 \
#   --training.skip_nan_inf \
#   --training.dataset /home/ubuntu/poria-cvpr-2026/leijingdi/Data/SlimPajama-627B-Merged \
#   --training.dataset_split train \
#   --training.num_workers 2 \
#   --training.prefetch_factor 2 \
#   --training.seed 42 \
#   --training.compile \
#   --training.streaming \
#   --training.tensor_parallel_degree 1 \
#   --training.disable_loss_parallel \
#   --checkpoint.interval 500 \
#   --checkpoint.load_step -1 \
#   --metrics.log_freq 1


