#!/bin/bash
#SBATCH --partition=sgpu_devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=old_setup_env
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=32G

# ============================================================================
# Build old-code's own venv (old-code/old_env) on a marvin GPU node from
# requirements.txt (transformers 4.31 etc. -- the old code imports internals
# that newer transformers removed, so it can't share new_env_a100).
#
#   cd analyse-old-code/old-code
#   sbatch setup_env.sh              # no-op if old_env is already ready
#   FORCE=1 sbatch setup_env.sh      # rebuild
#
# Built on a GPU node (AMD EPYC) with the AMD module tree, same as
# Disentangling-Reasoning/setup-env/build_env_a100_marvin.slurm. pip cache and
# temp files go to Lustre. heart_beat.sh submits this automatically.
# old_env/.env_ready is written only after all verification passed.
# ============================================================================
set -uo pipefail

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
cd "$OLD_ROOT" || exit 1
source "$OLD_ROOT/marvin_common.sh" || exit 1

mkdir -p "$LOG_DIR"
JOB_NAME="${SLURM_JOB_NAME:-old_setup_env}"
rotate_logs "$JOB_NAME" 3 "$LOG_DIR"
exec > "$LOG_DIR/${JOB_NAME}_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/${JOB_NAME}_${SLURM_JOB_ID:-manual}.err"

ts() { date +'%F %T'; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

echo "[$(ts)] OLD_ROOT=$OLD_ROOT  ENV_DIR=$ENV_DIR"
echo "[$(ts)] Node: $(hostname)  CPU: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"

if [ -f "$ENV_READY_FLAG" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(ts)] $ENV_DIR already ready -- nothing to do (FORCE=1 to rebuild)."
    exit 0
fi

# ---------------------------------------------------------------------------
# AMD-tree Python (Intel-tree modules SIGILL on marvin's AMD GPU nodes)
# ---------------------------------------------------------------------------
module purge
[ -d "$AMD_MODULEPATH_MARVIN" ] || die "AMD module path not found: $AMD_MODULEPATH_MARVIN (run this on a marvin GPU node)"
export MODULEPATH="$AMD_MODULEPATH_MARVIN:/opt/software/modulefiles"
module load "$PYTHON_MODULE" || die "module load $PYTHON_MODULE failed"
python3 -c "import ctypes" || die "ctypes broken in $PYTHON_MODULE"
echo "[$(ts)] Python: $(python3 --version) ($(which python3))"

export MKL_ENABLE_INSTRUCTIONS=AVX2
export DNNL_MAX_CPU_ISA=AVX2
export PYTHONUNBUFFERED=1

# pip cache / build temp on Lustre, not $HOME
export PIP_CACHE_DIR="$LUSTRE_OLD/pip_cache"
export TMPDIR="$LUSTRE_OLD/tmp/setup_${SLURM_JOB_ID:-manual}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# ---------------------------------------------------------------------------
# Fresh venv
# ---------------------------------------------------------------------------
case "$ENV_DIR" in "$OLD_ROOT"/*) ;; *) die "refusing to delete ENV_DIR outside $OLD_ROOT: $ENV_DIR" ;; esac
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR" || die "venv creation failed"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools || die "pip upgrade failed"

# requirements.txt as-is, except torchaudio==0.13.1+cu116: that build doesn't
# exist on PyPI and doesn't match torch 2.4.1; the code doesn't use it.
REQ="$TMPDIR/requirements.txt"
grep -v -E '^torchaudio==' "$OLD_ROOT/requirements.txt" > "$REQ"
echo "[$(ts)] Installing $(grep -c . "$REQ") pinned packages ..."
python -m pip install -r "$REQ" || die "pip install -r requirements.txt failed"

# ---------------------------------------------------------------------------
# Verification: CUDA + the old code's full import chain
# ---------------------------------------------------------------------------
echo "[$(ts)] Verifying ..."
PYTHONPATH="$OLD_ROOT" python - <<'PY' || die "verification failed"
import torch, transformers, peft, accelerate, bitsandbytes, tokenizers
print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"tokenizers   {tokenizers.__version__}")
print(f"peft         {peft.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
assert torch.cuda.is_available(), "CUDA not available"
torch.ones(1, device="cuda").add_(1)
# same imports train.py / eval.py do
from model.load_model import MyAutoModelForCausalLM
from model.peft_model import MyPeftModelForCausalLM
from model.my_trainer import MyTrainer
from load_data.preprocess import StrategyQAData_Ours, CommonsenseQAData_Ours, TruthfulQAData_Ours
print("old-code imports OK")
PY

# Informational: can this transformers version read each model's config?
# (Llama-3.1 / Qwen2.5 need newer transformers than 4.31.) Only checks models
# already in the Lustre store; never fails the build.
for m in meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
    d="$(local_model_dir "$m")"
    if local_model_ready "$m"; then
        if python -c "import transformers,sys; transformers.AutoConfig.from_pretrained(sys.argv[1])" "$d" >/dev/null 2>&1; then
            echo "[$(ts)] config check: $m  OK"
        else
            echo "[$(ts)] config check: $m  NOT LOADABLE with transformers $(python -c 'import transformers; print(transformers.__version__)')"
        fi
    else
        echo "[$(ts)] config check: $m  (not in model store yet, skipped)"
    fi
done

touch "$ENV_READY_FLAG"
echo "[$(ts)] Done: $ENV_DIR ($(du -sh "$ENV_DIR" | cut -f1))"
