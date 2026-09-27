#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=old_eval
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=32G

# ============================================================================
# Old-code evaluation on marvin.
#
#   cd analyse-old-code/old-code
#   sbatch eval.sh [DATASET] [MODEL] [CHECKPOINT]
#
#   CHECKPOINT  a checkpoint-XXX dir. Default: newest checkpoint-* of the
#               Lustre run dir train.sh wrote for DATASET/MODEL.
#
# Results (in this folder):
#   save_data/<DATASET>/<MODEL>/<checkpoint-XXX>_output.json
#   save_data/summary.tsv   (one line per eval: dataset, model, checkpoint, accuracy)
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
JOB_NAME="${SLURM_JOB_NAME:-old_eval}"
JOB_ID="${SLURM_JOB_ID:-manual$$}"
LOG_OUT="$LOG_DIR/${JOB_NAME}_${JOB_ID}.out"
rotate_logs "$JOB_NAME" 3 "$LOG_DIR"
exec > "$LOG_OUT" 2> "$LOG_DIR/${JOB_NAME}_${JOB_ID}.err"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_DISABLED="true"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Configuration (run hyper-parameters: see marvin_common.sh)
# ---------------------------------------------------------------------------
DATASET="${1:-truthfulqa_agent}"
MODEL="${2:-meta-llama/Llama-2-7b-chat-hf}"
# MODEL=meta-llama/Llama-3.1-8B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct

RUN_DIR="$(run_dir "$DATASET" "$MODEL")"
CHECKPOINT="${3:-$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)}"
[ -n "$CHECKPOINT" ] && [ -d "$CHECKPOINT" ] || {
    echo "ERROR: no checkpoint found (arg 3 or $RUN_DIR/checkpoint-*)" >&2
    exit 1
}
CHECKPOINT="$(cd "$CHECKPOINT" && pwd)"
RESULT="$(result_file "$DATASET" "$MODEL" "$(basename "$CHECKPOINT")")"
TMP_OUT="$RESULTS_DIR/.tmp_$JOB_ID"

activate_old_env
export TQDM_DISABLE=1
load_hf_token
ensure_base_model "$MODEL"
BASE_MODEL_LOCAL="$(local_model_dir "$MODEL")"

echo "======================================================================"
echo " OLD-CODE EVAL (marvin)"
echo "  Dataset      : $DATASET"
echo "  Model        : $MODEL"
echo "  Base weights : $BASE_MODEL_LOCAL"
echo "  Checkpoint   : $CHECKPOINT"
echo "  Venv         : $VIRTUAL_ENV"
echo "  Result       : $RESULT"
echo "  Node         : $(hostname)"
echo "======================================================================"

rm -rf "$TMP_OUT"
python -u eval.py \
    --base_model_name_or_path "$BASE_MODEL_LOCAL" \
    --hf_hub_token "$HF_TOKEN" \
    --model_name_or_path "$CHECKPOINT" \
    --output_dir "$TMP_OUT" \
    --add_soft_prompts $ADD_SOFT_PROMPT \
    --parameter_efficient_mode $EFFICIENT \
    --dataset "$DATASET" \
    --batch_size 1 \
    --max_length 850 \
    --seed 300 \
    --extract_step_type_tokens $STEP_TYPE \
    --embedding_model_name "$MODEL" \
    --num_plan_types 5 \
    --num_test 1200 \
    --load_in_8bit True
    # --prompt_template alpaca \
    # --use_calculator True \

# eval.py names its file after --base_model_name_or_path (the Lustre path), so
# it lands deep inside TMP_OUT; move it to the per-checkpoint result name.
OUT_JSON="$(find "$TMP_OUT" -type f -name '*_output.json' | head -n1)"
[ -n "$OUT_JSON" ] || { echo "ERROR: eval.py wrote no *_output.json under $TMP_OUT" >&2; exit 1; }
mkdir -p "$(dirname "$RESULT")"
mv -f "$OUT_JSON" "$RESULT"
rm -rf "$TMP_OUT"

ACC="$(sed -n 's/^Accuracy:[[:space:]]*//p' "$LOG_OUT" | tail -n1)"
[ -f "$RESULTS_DIR/summary.tsv" ] || printf 'date\tdataset\tmodel\tcheckpoint\taccuracy\tjob\n' > "$RESULTS_DIR/summary.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$DATASET" "$MODEL" "$(basename "$CHECKPOINT")" "${ACC:-?}" "$JOB_ID" >> "$RESULTS_DIR/summary.tsv"
echo "Results: $RESULT  (accuracy ${ACC:-?})"
