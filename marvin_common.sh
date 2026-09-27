#!/bin/bash
# ============================================================================
# marvin_common.sh — shared settings for setup_env.sh, train.sh, eval.sh and
# heart_beat.sh (sourced, never run directly). Callers must set OLD_ROOT
# (this folder's repo root) before sourcing.
#
# WHERE THINGS LIVE
#   old-code/old_env/            venv (built by setup_env.sh from requirements.txt)
#   old-code/logs/               Slurm job logs + heart_beat.log
#   old-code/save_data/          eval results + summary.tsv
#   old-code/journal.txt         heart_beat job journal
#   $LUSTRE_OLD/checkpoints/     training checkpoints (only the last
#                                KEEP_CHECKPOINTS per run, via save_total_limit)
#   $MODEL_STORE (Lustre)        base weights, shared with Disentangling-Reasoning
#   $HF_HOME (Lustre)            HF cache, shared with Disentangling-Reasoning
#
# Reuses Disentangling-Reasoning's script/model_resources.sh (AMD module tree,
# Lustre model store, log rotation) and script/download_model.sh.
# ============================================================================

: "${OLD_ROOT:?OLD_ROOT must be set before sourcing marvin_common.sh}"

# Disentangling-Reasoning checkout (default: sibling of analyse-old-code/).
DR_ROOT="${DR_ROOT:-$(dirname "$(dirname "$OLD_ROOT")")/Disentangling-Reasoning}"
if [ ! -f "$DR_ROOT/script/model_resources.sh" ]; then
    echo "ERROR: $DR_ROOT/script/model_resources.sh not found -- set DR_ROOT=..." >&2
    return 1 2>/dev/null || exit 1
fi
SERVER=marvin
source "$DR_ROOT/script/model_resources.sh"

PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"

# --- paths ------------------------------------------------------------------
ENV_DIR="${ENV_DIR:-$OLD_ROOT/old_env}"
ENV_READY_FLAG="$ENV_DIR/.env_ready"
LOG_DIR="$OLD_ROOT/logs"
RESULTS_DIR="$OLD_ROOT/save_data"
LUSTRE_ROOT="$(dirname "$(model_store_root)")"      # /lustre/scratch/data/s76kalav_hpc-memory
LUSTRE_OLD="${LUSTRE_OLD:-$LUSTRE_ROOT/old-code}"
CKPT_ROOT="${CKPT_ROOT:-$LUSTRE_OLD/checkpoints}"

# How many checkpoint-* dirs to keep per run on Lustre (--save_total_limit).
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"

# --- run hyper-parameters (single source of truth for train + eval paths) ---
ADD_SOFT_PROMPT=True
N_PREFIX=3
N_SPECIAL=2
EFFICIENT=lora+prompt-tuning
STEP_TYPE=memory
LR=2e-4

# run_dir DATASET MODEL -> Lustre dir train.sh writes checkpoints into
run_dir() {
    echo "$CKPT_ROOT/$2/$1/step_type=$STEP_TYPE-$N_PREFIX-$N_SPECIAL-efficient=$EFFICIENT-lr=$LR-soft-prompt=$ADD_SOFT_PROMPT"
}

# result_file DATASET MODEL CHECKPOINT_NAME -> eval output json (in this folder)
result_file() {
    echo "$RESULTS_DIR/$1/$2/$3_output.json"
}

# Loads the AMD Python module and activates old_env (+ AVX-512 workaround,
# see model_resources.sh). Fails if setup_env.sh hasn't completed.
activate_old_env() {
    ensure_modulepath_for_host
    module load "$PYTHON_MODULE" || return 1
    [ -f "$ENV_READY_FLAG" ] || {
        echo "ERROR: $ENV_DIR not ready -- run: sbatch setup_env.sh" >&2
        return 1
    }
    source "$ENV_DIR/bin/activate" || return 1
    export MKL_ENABLE_INSTRUCTIONS=AVX2
    export DNNL_MAX_CPU_ISA=AVX2
}

# HF_TOKEN from the environment, else from old-code/.env or Disentangling-Reasoning/.env.
load_hf_token() {
    local f
    if [ -z "${HF_TOKEN:-}" ]; then
        for f in "$OLD_ROOT/.env" "$DR_ROOT/.env"; do
            [ -f "$f" ] || continue
            HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=//p' "$f" | tail -n1 | tr -d "\"'")"
            [ -n "$HF_TOKEN" ] && break
        done
    fi
    [ -n "${HF_TOKEN:-}" ] || {
        echo "ERROR: HF_TOKEN not set (export it or put HF_TOKEN=... in $OLD_ROOT/.env)" >&2
        return 1
    }
    export HF_TOKEN
}

# Makes sure $1 is in the Lustre model store; echoes nothing, fails on error.
ensure_base_model() {
    export_hf_home_for_server
    SLURM_SUBMIT_DIR="$DR_ROOT" bash "$DR_ROOT/script/download_model.sh" "$1" "$SERVER"
}
