#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=old_train
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=32G

# ============================================================================
# Old-code training on marvin.
#
#   cd analyse-old-code/old-code
#   sbatch train.sh [DATASET] [MODEL] [KEEP_CHECKPOINTS]
#
# - venv: old-code/old_env (setup_env.sh); base weights + HF cache on Lustre.
# - Checkpoints go to Lustre ($CKPT_ROOT, see marvin_common.sh); only the last
#   KEEP_CHECKPOINTS (default 2) checkpoint-* dirs are kept (--save_total_limit).
# - Logs go to old-code/logs/. On success, <run_dir>/training_complete.flag.
# - train.py cannot resume, so an unfinished run dir is wiped and restarted.
#   A finished one is left alone unless FORCE=1.
# ============================================================================
set -euo pipefail

find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        [ -d "$dir/.git" ] && { printf '%s\n' "$dir"; return 0; }
        dir="$(dirname "$dir")"
    done
    return 1
}
START_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OLD_ROOT="$(find_repo_root "$START_DIR")" || { echo "ERROR: no repo root (.git) above $START_DIR" >&2; exit 1; }
cd "$OLD_ROOT"
source "$OLD_ROOT/marvin_common.sh"

mkdir -p "$LOG_DIR"
JOB_NAME="${SLURM_JOB_NAME:-old_train}"
rotate_logs "$JOB_NAME" 3 "$LOG_DIR"
exec > "$LOG_DIR/${JOB_NAME}_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/${JOB_NAME}_${SLURM_JOB_ID:-manual}.err"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_DISABLED="true"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Configuration (run hyper-parameters: see marvin_common.sh)
# ---------------------------------------------------------------------------
DATASET="${1:-stratgeqa_agent}"
MODEL="${2:-meta-llama/Llama-2-7b-chat-hf}"
#MODEL=meta-llama/Llama-3.1-8B-Instruct
#MODEL=Qwen/Qwen2.5-7B-Instruct
KEEP="${3:-$KEEP_CHECKPOINTS}"
MODE=supervised

activate_old_env
export TQDM_DISABLE=1
load_hf_token
ensure_base_model "$MODEL"
BASE_MODEL_LOCAL="$(local_model_dir "$MODEL")"

RUN_DIR="$(run_dir "$DATASET" "$MODEL")"
case "$RUN_DIR" in "$CKPT_ROOT"/*) ;; *) echo "ERROR: unexpected RUN_DIR $RUN_DIR" >&2; exit 1 ;; esac
if [ -f "$RUN_DIR/training_complete.flag" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Already finished: $RUN_DIR (FORCE=1 to retrain)"
    exit 0
fi
rm -rf "$RUN_DIR"          # no resume support in train.py -> clean restart
mkdir -p "$RUN_DIR"

echo "======================================================================"
echo " OLD-CODE TRAIN (marvin)"
echo "  Dataset      : $DATASET"
echo "  Model        : $MODEL"
echo "  Base weights : $BASE_MODEL_LOCAL"
echo "  Venv         : $VIRTUAL_ENV"
echo "  HF_HOME      : $HF_HOME"
echo "  Checkpoints  : $RUN_DIR (keep last $KEEP)"
echo "  Node         : $(hostname)"
echo "======================================================================"

python -u train.py \
    --model_name_or_path "$BASE_MODEL_LOCAL" \
    --hf_hub_token "$HF_TOKEN" \
    --add_soft_prompts $ADD_SOFT_PROMPT \
    --num_general_prefix_tokens $N_PREFIX \
    --num_special_prefix_tokens $N_SPECIAL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset "$DATASET" \
    --fp16 True \
    --output_dir "$RUN_DIR" \
    --model_max_length 850 \
    --num_train_epochs 12 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit "$KEEP" \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --optim "adamw_torch" \
    --gradient_accumulation_steps 16 \
    --embedding_model_name "$MODEL" \
    --extract_step_type_tokens $STEP_TYPE \
    --num_plan_types 5 \
    --num_test 1200 \
    --lora_module mlp \
    --int8_training True
    # --gradient_checkpointing \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --sharded_ddp "zero_dp_2 offload" \
    # --fsdp "full_shard offload" \

touch "$RUN_DIR/training_complete.flag"
echo "Training complete: $RUN_DIR"
ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null || true
