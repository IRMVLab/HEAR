#!/bin/bash
LLM=qwen2_vl   #qwen2_vl  paligemma
LLM_MODEL_SIZE=2B #3B
# LLM_MODEL_SIZE=2_8B
ACTION_HEAD=dit_diffusion_policy  #act #unet_diffusion_policy dit_diffusion_policy

echo '7.5h'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${POLICY_ROOT}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${POLICY_ROOT}/outputs}"

PRETRAIN="${PRETRAIN:-${MODEL_ROOT}/${ACTION_HEAD}_pretrain/checkpoint-60000}"
DIT_PRETRAIN="${DIT_PRETRAIN:-${MODEL_ROOT}/dit_pretrain.ckpt}"
TASK_NAME="${TASK_NAME:-folding_two_shirts_by_drag}"

if [ "${LLM}" == "paligemma" ]; then
  echo "Using PaliGemma"
  mnop="${MODEL_NAME_OR_PATH:-${MODEL_ROOT}/vla-paligemma-3b-pt-224}"
else
  mnop="${MODEL_NAME_OR_PATH:-${MODEL_ROOT}/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct}"
fi

mnop="$PRETRAIN"

OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/${LLM}_${LLM_MODEL_SIZE}/${TASK_NAME}_Stage3}"
if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi

mkdir -p $OUTPUT/src
cp -r "${POLICY_ROOT}/aloha_scripts" "$OUTPUT/src/"
cp -r "${POLICY_ROOT}/scripts" "$OUTPUT/"
cp -r "${POLICY_ROOT}/data_utils" "$OUTPUT/src/"
cp -r "${POLICY_ROOT}/qwen2_vla" "$OUTPUT/src/"
cp -r "${POLICY_ROOT}/policy_heads" "$OUTPUT/src/"

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 "${POLICY_ROOT}/train_vla.py" \
  --deepspeed "${POLICY_ROOT}/scripts/zero2.json" \
  --use_reasoning True \
  --lora_enable False \
  --action_dim 14 \
  --state_dim 14 \
  --flash_attn True \
  --chunk_size 50 \
  --lora_module "vit llm" \
  --load_pretrain False \
  --history_images_length 1 \
  --model_pretrain $PRETRAIN \
  --load_pretrain_dit False \
  --pretrain_dit_path $DIT_PRETRAIN \
  --ground_truth_reasoning False \
  --using_all_reasoning_hidden False \
  --using_film True \
  --using_ema False \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "DiT_H" \
  --with_llm_head True \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --lora_r 64 \
  --lora_alpha 256 \
  --episode_first False \
  --task_name $TASK_NAME \
  --model_name_or_path $mnop \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 20000 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log | tee $OUTPUT/log.log

for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${mnop}/preprocessor_config.json $dir
        cp ${mnop}/chat_template.json $dir
    fi
done

mv ./60030.log $OUTPUT
echo $OUTPUT
