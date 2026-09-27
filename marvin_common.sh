#!/bin/bash
# ============================================================================
# marvin_common.sh — shared settings for setup_env.sh, train.sh, eval.sh and
# heart_beat.sh (sourced, never run directly). Callers must set OLD_ROOT
# (this folder's repo root) before sourcing.
#
# WHERE THINGS LIVE
#   old-code/old_env/            venv (built by setup_env.sh from requirements.txt)
#   old-code/logs/               Slurm job logs + heart_beat.log
#   old-code/save_data/          eval results + summary.tsv (committed + pushed)
#   old-code/logs/startup_errors.log  errors before a job's own log exists
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

# Every job's Slurm stdout/stderr is /dev/null until it redirects into logs/,
# so anything failing before that is logged here instead.
startup_error() {
    mkdir -p "$OLD_ROOT/logs"
    echo "[$(date +'%F %T')] ${SLURM_JOB_NAME:-manual} job=${SLURM_JOB_ID:-} host=$(hostname): $*" \
        | tee -a "$OLD_ROOT/logs/startup_errors.log" >&2
}

# Disentangling-Reasoning checkout (venv-independent helpers + Lustre model
# store). DR_ROOT=... wins; otherwise the first of: next to old-code, next to
# its parent, ~/Disentangling-Reasoning, or any checkout up to 3 levels below
# $HOME (its folder name differs between machines).
find_dr_root() {
    local c
    for c in "${DR_ROOT:-}" \
             "$(dirname "$OLD_ROOT")/Disentangling-Reasoning" \
             "$(dirname "$(dirname "$OLD_ROOT")")/Disentangling-Reasoning" \
             "$HOME/Disentangling-Reasoning"; do
        [ -n "$c" ] && [ -f "$c/script/model_resources.sh" ] && [ -f "$c/script/download_model.sh" ] \
            && { (cd "$c" && pwd); return 0; }
    done
    c="$(find "$HOME" -maxdepth 4 -path '*/script/download_model.sh' -not -path "$OLD_ROOT/*" 2>/dev/null | head -n1)"
    [ -n "$c" ] && { dirname "$(dirname "$c")"; return 0; }
    return 1
}
if ! DR_ROOT="$(find_dr_root)"; then
    startup_error "Disentangling-Reasoning checkout not found (need script/model_resources.sh + script/download_model.sh) -- set DR_ROOT=/path/to/it"
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

# Commits save_data/ (only that path) and pushes it to the branch's upstream.
# Called by eval.sh when a result is written and by heart_beat.sh every pass
# (retries anything a job couldn't push). Best-effort: never fails the caller.
# COMMIT_RESULTS=0 disables it. A mkdir lock serializes parallel evals.
commit_results() {
    [ "${COMMIT_RESULTS:-1}" = "1" ] || return 0
    local lock="$OLD_ROOT/.git/results_commit.lock" waited=0 rc=0
    until mkdir "$lock" 2>/dev/null; do
        # stale lock (killed job) after 10 min
        if [ $(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || date +%s) )) -gt 600 ]; then
            rm -rf "$lock"; continue
        fi
        [ "$waited" -ge 300 ] && { echo "commit_results: lock busy, skipping (next heart_beat pass retries)"; return 0; }
        sleep 5; waited=$((waited + 5))
    done
    (
        cd "$OLD_ROOT" || exit 1
        git add -A -- save_data || exit 1
        if ! git diff --cached --quiet -- save_data; then
            n="$(git diff --cached --name-only -- save_data | wc -l)"
            git -c user.name="$(git config user.name || echo "${USER:-heart_beat}")" \
                -c user.email="$(git config user.email || echo "${USER:-heart_beat}@$(hostname -f 2>/dev/null || hostname)")" \
                commit -q -m "results: $n file(s) from ${SLURM_JOB_NAME:-manual} ${SLURM_JOB_ID:-} ($(date +'%F %T'))" -- save_data || exit 1
            echo "commit_results: committed $n file(s)"
        fi
        # push whatever is ahead of upstream (also earlier commits whose push failed)
        [ -n "$(git rev-list '@{u}..HEAD' 2>/dev/null)" ] || exit 0
        git push -q 2>&1 || { git pull -q --rebase --autostash 2>&1 && git push -q 2>&1; } || exit 1
        echo "commit_results: pushed to $(git rev-parse --abbrev-ref '@{u}')"
    ) || rc=$?
    rm -rf "$lock"
    [ "$rc" -eq 0 ] || echo "commit_results: git commit/push failed (rc=$rc) -- will retry next heart_beat pass"
    return 0
}

# Makes sure $1 is in the Lustre model store; echoes nothing, fails on error.
ensure_base_model() {
    export_hf_home_for_server
    SLURM_SUBMIT_DIR="$DR_ROOT" bash "$DR_ROOT/script/download_model.sh" "$1" "$SERVER"
}
