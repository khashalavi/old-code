#!/bin/bash
#SBATCH --partition=sgpu_medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=old_heart_beat
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem=2G
# NOTE: CPU-only orchestrator (no --gres). Keep TIME_LIMIT_SECONDS in sync
# with --time above.
# ============================================================================
# heart_beat.sh — hourly driver for the old code on marvin: env setup ->
# training for every DATASET x MODEL -> eval of every kept checkpoint.
#
#   cd analyse-old-code/old-code
#   sbatch heart_beat.sh             # runs ~24h, then re-submits itself
#   ./heart_beat.sh --status         # print state, submit nothing
#   ./heart_beat.sh --dry-run --once # show what would be submitted
#   ./heart_beat.sh --once           # one pass, then exit
#
# ONE PASS
#   1. setup  : old_env/.env_ready missing -> submit setup_env.sh. Training
#               jobs submitted while it runs get --dependency=afterok:<setup>.
#   2. train  : per DATASET x MODEL, done = <lustre run dir>/training_complete.flag.
#               Otherwise, if its job is not RUNNING/PENDING -> (re)submit train.sh.
#   3. eval   : per finished run, per kept checkpoint-* (only KEEP_CHECKPOINTS
#               exist on Lustre), done = save_data/<ds>/<model>/<ckpt>_output.json.
#               Otherwise (re)submit eval.sh for that checkpoint.
#   Each job key is (re)submitted at most MAX_ATTEMPTS times, then GAVE_UP
#   (raise MAX_ATTEMPTS or delete the key's lines in journal.txt to retry).
#
# JOURNAL (old-code/journal.txt, append-only)
#   <timestamp> | <EVENT> | <key> | job=<id>
#   key:   setup::old_env | train::<dataset>::<model> | eval::<dataset>::<model>::<checkpoint>
#   EVENT: SUBMITTED RUNNING FINISHED DIED GAVE_UP SUBMIT_FAILED <slurm-state>
# Log: old-code/logs/heart_beat.log
# ============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# Selection matrix + knobs
# ---------------------------------------------------------------------------
DATASETS=(stratgeqa_agent commonsenseqa_agent truthfulqa_agent)
MODELS=(
    meta-llama/Llama-2-7b-chat-hf
    # Need newer transformers than requirements.txt's 4.31 -- setup_env.sh's
    # "config check" lines tell you whether they load in old_env:
    # meta-llama/Llama-3.1-8B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
)
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"   # checkpoint-* kept per run on Lustre
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
CHECK_INTERVAL="${CHECK_INTERVAL:-3600}"
TIME_LIMIT_SECONDS=86400
REINIT_MARGIN=900
# Extra sbatch args per job type (override partition/time here, e.g.
# TRAIN_SBATCH="--partition=sgpu_medium --time=24:00:00" if 8h is too short).
TRAIN_SBATCH="${TRAIN_SBATCH:-}"
EVAL_SBATCH="${EVAL_SBATCH:-}"
SETUP_SBATCH="${SETUP_SBATCH:-}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
ONCE=0; DRY_RUN=0; STATUS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --once)    ONCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --status)  STATUS_ONLY=1; ONCE=1 ;;
        -h|--help) sed -n '12,38p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths, log, journal
# ---------------------------------------------------------------------------
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
cd "$OLD_ROOT" || exit 1      # jobs submitted from here get SLURM_SUBMIT_DIR=$OLD_ROOT
source "$OLD_ROOT/marvin_common.sh" || exit 1

mkdir -p "$LOG_DIR"
HB_LOG="$LOG_DIR/heart_beat.log"
JOURNAL="$OLD_ROOT/journal.txt"
[ -f "$JOURNAL" ] || printf '# old-code heart_beat journal — <ts> | <EVENT> | <key> | job=<id>\n' > "$JOURNAL"
SLURM_USER="${USER:-$(id -un)}"

log() {
    local msg="[$(date +'%F %T')] $*"
    echo "$msg"
    echo "$msg" >> "$HB_LOG"
}
journal_append() {   # EVENT KEY [JOBID]
    [ "$DRY_RUN" -eq 1 ] || [ "$STATUS_ONLY" -eq 1 ] && return 0
    printf '%s | %s | %s | job=%s\n' "$(date +'%F %T')" "$1" "$2" "${3:--}" >> "$JOURNAL"
}
journal_lines()      { grep -F " | $1 | job=" "$JOURNAL" 2>/dev/null; }
journal_last_event() { journal_lines "$1" | tail -n1 | awk -F' \\| ' '{print $2}'; }
journal_last_jobid() { journal_lines "$1" | grep -oE 'job=[0-9]+' | tail -n1 | cut -d= -f2; }
journal_count()      { journal_lines "$1" | awk -F' \\| ' -v e="$2" '$2==e' | wc -l; }

snapshot_queue() {
    # Distinguish "squeue failed" from "queue empty": on failure we must not
    # treat every job as dead and mass-resubmit.
    if QUEUE="$(squeue -u "$SLURM_USER" -h -o '%i|%T' 2>/dev/null)"; then QUEUE_OK=1; else QUEUE_OK=0; QUEUE=""; fi
}
queue_state_of() { awk -F'|' -v id="$1" '$1==id{print $2; exit}' <<< "$QUEUE"; }

# ---------------------------------------------------------------------------
# track KEY DONE LABEL CMD...
#   Drives one job key through the state machine. Sets TRACK to one of
#   done | active | submitted | gave_up | failed | needed | would,
#   and TRACK_JOBID to the live job id (active/submitted).
# ---------------------------------------------------------------------------
declare -A COUNT
track() {
    local key="$1" done="$2" label="$3"; shift 3
    local last jobid state newid attempts
    TRACK_JOBID=""
    last="$(journal_last_event "$key")"

    if [ "$done" -eq 1 ]; then
        [ "$last" = "FINISHED" ] || { journal_append FINISHED "$key" "$(journal_last_jobid "$key")"; log "FINISHED     $label"; }
        TRACK=done; COUNT[$TRACK]=$(( ${COUNT[$TRACK]:-0} + 1 )); return
    fi

    case "$last" in
        SUBMITTED|RUNNING)
            jobid="$(journal_last_jobid "$key")"
            state="$(queue_state_of "$jobid")"
            case "$state" in
                RUNNING)
                    [ "$last" = "RUNNING" ] || { journal_append RUNNING "$key" "$jobid"; log "RUNNING      $label (job $jobid)"; }
                    TRACK=active; TRACK_JOBID="$jobid"; COUNT[$TRACK]=$(( ${COUNT[$TRACK]:-0} + 1 )); return ;;
                PENDING|CONFIGURING|COMPLETING|RESIZING|REQUEUED|SUSPENDED)
                    TRACK=active; TRACK_JOBID="$jobid"; COUNT[$TRACK]=$(( ${COUNT[$TRACK]:-0} + 1 )); return ;;
                "")
                    journal_append DIED "$key" "$jobid"; log "DIED         $label (job $jobid gone, not done)" ;;
                *)
                    journal_append "$state" "$key" "$jobid"; log "ENDED        $label (job $jobid state=$state)" ;;
            esac ;;
    esac

    attempts="$(journal_count "$key" SUBMITTED)"
    if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
        [ "$last" = "GAVE_UP" ] || { journal_append GAVE_UP "$key"; log "GAVE_UP      $label ($attempts attempts -- see logs/)"; }
        TRACK=gave_up; COUNT[$TRACK]=$(( ${COUNT[$TRACK]:-0} + 1 )); return
    fi

    if [ "$STATUS_ONLY" -eq 1 ]; then
        log "NEEDS RUN    $label"; TRACK=needed
    elif [ "$DRY_RUN" -eq 1 ]; then
        log "WOULD SUBMIT $label -> $*"; TRACK=would
    else
        newid="$("$@" 2>>"$HB_LOG")"; newid="${newid%%;*}"
        if [[ "$newid" =~ ^[0-9]+$ ]]; then
            journal_append SUBMITTED "$key" "$newid"
            log "SUBMITTED    $label -> job $newid (attempt $((attempts + 1))/$MAX_ATTEMPTS)"
            TRACK=submitted; TRACK_JOBID="$newid"
        else
            journal_append SUBMIT_FAILED "$key"
            log "ERROR: submit failed for $label (output: '$newid')"
            TRACK=failed
        fi
    fi
    COUNT[$TRACK]=$(( ${COUNT[$TRACK]:-0} + 1 ))
}

# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------
run_pass() {
    log "===== pass start (dry_run=$DRY_RUN status=$STATUS_ONLY keep=$KEEP_CHECKPOINTS) ====="
    COUNT=()
    snapshot_queue
    if [ "$QUEUE_OK" -eq 0 ]; then
        log "WARNING: squeue unavailable; skipping this pass to avoid duplicate submissions."
        return 0
    fi

    # 1) environment
    local dep=() done
    done=0; [ -f "$ENV_READY_FLAG" ] && done=1
    track "setup::old_env" "$done" "setup old_env" \
        sbatch --parsable --job-name=old_setup_env $SETUP_SBATCH "$OLD_ROOT/setup_env.sh"
    case "$TRACK" in
        done) ;;
        active|submitted) dep=(--dependency="afterok:$TRACK_JOBID" --kill-on-invalid-dep=yes) ;;
        needed|would) ;;   # status / dry-run: keep going to show the plan
        *) log "old_env not available (setup $TRACK) -- no training/eval this pass."; log "===== pass end ====="; return 0 ;;
    esac

    local ds model tag rd ck name res
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            tag="$(model_tag "$model")"
            rd="$(run_dir "$ds" "$model")"

            # 2) training
            done=0; [ -f "$rd/training_complete.flag" ] && done=1
            track "train::$ds::$model" "$done" "train $ds / $model" \
                sbatch --parsable --job-name="ot_${ds:0:3}_${tag}" $TRAIN_SBATCH ${dep[@]+"${dep[@]}"} \
                    "$OLD_ROOT/train.sh" "$ds" "$model" "$KEEP_CHECKPOINTS"
            [ "$TRACK" = "done" ] || continue

            # 3) eval every kept checkpoint
            for ck in "$rd"/checkpoint-*; do
                [ -d "$ck" ] || continue
                name="$(basename "$ck")"
                res="$(result_file "$ds" "$model" "$name")"
                done=0; [ -f "$res" ] && done=1
                track "eval::$ds::$model::$name" "$done" "eval  $ds / $model / $name" \
                    sbatch --parsable --job-name="oe_${ds:0:3}_${tag}" $EVAL_SBATCH \
                        "$OLD_ROOT/eval.sh" "$ds" "$model" "$ck"
            done
        done
    done

    local k summary=""
    for k in "${!COUNT[@]}"; do summary+="$k=${COUNT[$k]} "; done
    log "summary: ${summary:-nothing tracked}"
    log "===== pass end ====="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [ "$ONCE" -eq 1 ]; then
    run_pass
    exit 0
fi

REINIT_AT=$(( TIME_LIMIT_SECONDS - REINIT_MARGIN ))
while true; do
    run_pass
    if [ $(( SECONDS + CHECK_INTERVAL )) -ge "$REINIT_AT" ]; then
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
            newid="$(sbatch --parsable "$OLD_ROOT/heart_beat.sh" 2>>"$HB_LOG")"
            log "Approaching wall clock -> re-submitted heart_beat as job ${newid:-<failed>}. Exiting."
            exit 0
        fi
        log "Standalone run: re-exec'ing a fresh heart_beat process."
        exec "$0" "$@"
    fi
    sleep "$CHECK_INTERVAL"
done
