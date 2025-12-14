#!/bin/bash

cd 3rdparty/flash-linear-attention

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation
# MODEL='/home/ubuntu/poria-cvpr-2026/leijingdi/flame/exp/delta_net_340M/batch32.seqlen2048.warmup1024.update1.steps7000.lr3e-4'
# accelerate launch -m evals.harness --model hf  \
#     --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True  \
#     --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
#     --batch_size 128  \
#     --num_fewshot 0  \
#     --device cuda  \
#     --show_config  \
#     --trust_remote_code

export DELTA_UPDATE_RULE="efla"
MODEL='/home/ubuntu/poria-cvpr-2026/leijingdi/flame/exp/delta_net_340M_EFLA_norm/batch32.seqlen2048.warmup1024.update1.steps4096.lr3e-4'
OUTPUT='/home/ubuntu/poria-cvpr-2026/leijingdi/outputs'
accelerate launch -m evals.harness \
    --output_path $OUTPUT \
    --tasks niah_single_2 \
    --model_args pretrained=$MODEL,dtype=bfloat16,max_length=8192,trust_remote_code=True \
    --metadata='{"max_seq_lengths":[1024,2048,4096,8192]}' \
    --batch_size 16 \
    --show_config  \
    --trust_remote_code

# export DELTA_UPDATE_RULE="delta"
# MODEL='/home/ubuntu/poria-cvpr-2026/leijingdi/flame/exp/delta_net_340M/batch32.seqlen2048.warmup1024.update1.steps4096.lr3e-4'
# OUTPUT='/home/ubuntu/poria-cvpr-2026/leijingdi/outputs'
# accelerate launch -m evals.harness \
#     --output_path $OUTPUT \
#     --tasks niah_single_2 \
#     --model_args pretrained=$MODEL,dtype=bfloat16,max_length=8192,trust_remote_code=True \
#     --metadata='{"max_seq_lengths":[1024,2048,4096,8192]}' \
#     --batch_size 16 \
#     --show_config  \
#     --trust_remote_code
# --tasks longbench,wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
