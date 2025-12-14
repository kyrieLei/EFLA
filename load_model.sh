#!/bin/bash

python -m flame.utils.convert_dcp_to_hf \
  --path /home/ubuntu/poria-cvpr-2026/leijingdi/flame/exp/delta_net_340M_EFLA_scale_fixed/batch32.seqlen2048.warmup1024.update1.steps14500.lr3e-4 \
  --step 8000 \
  --config configs/delta_net_340M.json \
  --tokenizer mistralai/Mistral-7B-v0.1