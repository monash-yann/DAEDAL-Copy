#!/bin/bash
set -e


BASE_OUTPUT_PATH="./results/baseline"
MODEL_PATH="/homebck/home/xiangzhong_guest/LLADA/llada_sampling_system/models/LLaDA-8B-Instruct"
#MODEL_PATH="/data/AnYilin/models/LLaDA-8B-Instruct/"


TASKS=("gsm8k")
#TASKS=("humaneval" "mbpp")
LENGTHS=(256)
#LENGTHS=(32 64 128 256 512 1024 2048)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<Baseline>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            --model LLaDA \
            --tasks "${task}" \
            --batch_size 8 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "block_length=256,gen_length=${length},steps=${length},cfg_scale=0.0,remasking="low_confidence" " \
            --num_fewshot 0 \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code
          #            -m dllm_eval \ =>ignored
        
        python metrics/${task}.py \
            --model_path "${MODEL_PATH}" \
            --res_path "${OUTPUT_PATH}"
    done
done
