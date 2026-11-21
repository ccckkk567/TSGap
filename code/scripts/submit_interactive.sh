#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {all|bc-only|mains-only} [--bc-segments N] [--main-segments M] [--bc-resume] [--bc-time HH:MM:SS] [--main-time HH:MM:SS]" >&2
    exit 2
}
mode=${1:-}
case "$mode" in all|bc-only|mains-only) shift ;; *) usage ;; esac
bc_segments=1 main_segments=2 bc_resume=0
bc_time=08:00:00 main_time=06:00:00
while (( $# )); do
    case "$1" in
        --bc-resume) bc_resume=1; shift; continue ;;
        --bc-segments) bc_segments=${2:?Missing BC segment count} ;;
        --main-segments) main_segments=${2:?Missing Main segment count} ;;
        --bc-time) bc_time=${2:?Missing BC walltime} ;;
        --main-time) main_time=${2:?Missing Main walltime} ;;
        *) usage ;;
    esac
    shift 2
done
for count in "$bc_segments" "$main_segments"; do
    [[ $count =~ ^[1-9][0-9]{0,8}$ ]] || usage
done
for limit in "$bc_time" "$main_time"; do
    [[ $limit =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ && $limit != 00:00:00 && ! $limit > 18:00:00 ]] ||
        { echo "testqueue requires a positive HH:MM:SS up to 18:00:00" >&2; exit 2; }
done
[[ $mode != mains-only || $bc_resume == 0 ]] || usage
if [[ $mode != mains-only && -z ${TSGAP_REPO:-} && -n ${TSGAP_MAIN_REPO:-} ]]; then
    echo "TSGAP_MAIN_REPO is only supported with mains-only" >&2
    exit 2
fi

ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${TSGAP_REPO:-${TSGAP_MAIN_REPO:-$ROOT/code/TSGap}}"
[[ $REPO == /* ]] || { echo "Repository must be an absolute path" >&2; exit 2; }
CODE="$REPO/code"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
case "$(hostname -s)" in orion-login01|orion-login02) ;; *) echo "Use an Orion login node" >&2; exit 1 ;; esac
for cmd in git sbatch squeue flock; do
    command -v "$cmd" >/dev/null || { echo "Missing $cmd" >&2; exit 1; }
done
for name in run_bc.sbatch run_chosen_sft.sbatch run_teacher_sft.sbatch run_guided_sft.sbatch submit_interactive.sh; do
    cmp -s "$SCRIPT_DIR/$name" "$CODE/scripts/$name" ||
        { echo "Missing or stale $name. Copy the five entries together from $CODE/scripts." >&2; exit 1; }
done
export TSGAP_INTERACTIVE_CONFIG="${TSGAP_INTERACTIVE_CONFIG:-$CODE/examples/teaching/interactive_config.yaml}"
[[ $TSGAP_INTERACTIVE_CONFIG == /* && -f $TSGAP_INTERACTIVE_CONFIG ]] || { echo "Expected an absolute config path" >&2; exit 1; }
module load python/3.11.9
cd "$CODE"
bootstrap="$ROOT/envs/rollout/bin/python"
[[ -x $bootstrap ]] || bootstrap="$ROOT/envs/alfworld/bin/python"
settings=$("$bootstrap" -c 'import sys; from scripts.teaching_experiment import orion_settings; print(*orion_settings(sys.argv[1]))' "$TSGAP_INTERACTIVE_CONFIG")
read -r RUN prefix backend <<< "$settings"
[[ $RUN =~ ^[a-zA-Z0-9_-]+$ && $prefix =~ ^[a-zA-Z0-9_-]+$ ]] || usage
VENV="$ROOT/envs/alfworld"
if [[ $backend == direct ]]; then
    VENV="$ROOT/envs/rollout"
    if [[ $mode == all || ( $mode == bc-only && $bc_resume == 0 ) ]]; then
        echo "Run init-bc, then use bc-only --bc-resume and mains-only separately." >&2
        exit 2
    fi
fi
mkdir -p "$ROOT/logs"
record="$ROOT/logs/$RUN-submissions.tsv"
exec 9>>"$record"
flock -n 9 || { echo "Another submission is using $record" >&2; exit 1; }
diagnose='Inspect squeue -u "$USER" -o "%.18i %.32j %.10T %.48E %R" and the submission record.'
queued=$(squeue -h -u "$USER" -o '%i %j') || { echo "$diagnose" >&2; exit 1; }
if ! awk -v mode="$mode" -v prefix="$prefix" 'FILENAME == ARGV[1] {stages[$3]=$1; next}
    NF == 2 && mode == "mains-only" &&
        ($1 in stages ? stages[$1] == "bc" : $2 ~ ("^" prefix "-bc(-[0-9]+)?$")) {next}
    NF && (NF != 2 || $1 in stages || index($2, prefix "-") == 1 || (prefix == "tsgap-32b" && $2 == "tsgap-experiment")) {print; conflict=1}
    END {exit conflict}' "$record" - <<< "$queued"; then
    echo "Possible active writer for $RUN. $diagnose See $record" >&2
    exit 1
fi

TSGAP_EXPECTED_COMMIT=$(git -C "$REPO" rev-parse HEAD)
export TSGAP_EXPECTED_COMMIT
printf 'commit=%s config=%s run=%s mode=%s bc_segments=%s main_segments=%s\n' \
    "$TSGAP_EXPECTED_COMMIT" "$TSGAP_INTERACTIVE_CONFIG" "$RUN" "$mode" "$bc_segments" "$main_segments"
git -C "$REPO" status --short --branch
preflight=fresh
if [[ $mode == mains-only ]]; then preflight=main; elif (( bc_resume )); then preflight=resume; fi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" -c \
    'import sys; from scripts.teaching_experiment import check_orion_run; check_orion_run(*sys.argv[1:])' \
    "$TSGAP_INTERACTIVE_CONFIG" "$preflight"

accepted=()
on_exit() {
    if (( $1 != 0 )); then
        echo "Submission stopped. Confirmed IDs: ${accepted[*]:-none}. $diagnose See $record" >&2
        echo "Check which segments are missing before submitting anything else." >&2
    fi
}
trap 'on_exit $?' EXIT
printf 'stage\tsegment\tjob_id\tdependency\tscript\tcommit\n'
submit() {
    local stage=$1 segment=$2 dependency=$3 limit=$4 script=$5
    shift 5
    local args=(--parsable --export=ALL --time="$limit" --job-name="$prefix-$stage-$segment")
    if [[ -n $dependency ]]; then args+=(--dependency="$dependency"); fi
    local reply
    reply=$(sbatch "${args[@]}" "$script" "$@")
    job_id=${reply%%;*}
    [[ $job_id =~ ^[1-9][0-9]*$ ]] || { echo "Unexpected sbatch reply: $reply" >&2; exit 1; }
    accepted+=("$job_id")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$stage" "$segment" "$job_id" "${dependency:--}" \
        "$script" "$TSGAP_EXPECTED_COMMIT" | tee -a "$record"
}

bc_tail=
if [[ $mode != mains-only ]]; then
    for ((i=1; i<=bc_segments; i++)); do
        args=()
        if (( bc_resume || i > 1 )); then args+=(--resume); fi
        submit bc "$i" "${bc_tail:+afterany:$bc_tail}" "$bc_time" "$SCRIPT_DIR/run_bc.sbatch" "${args[@]}"
        bc_tail=$job_id
    done
fi
if [[ $mode != bc-only ]]; then
    for group in chosen teacher guided; do
        previous=
        for ((i=1; i<=main_segments; i++)); do
            dependency=${bc_tail:+afterok:$bc_tail}
            if [[ -n $previous ]]; then dependency="${dependency:+$dependency,}afterany:$previous"; fi
            submit "$group" "$i" "$dependency" "$main_time" "$SCRIPT_DIR/run_${group}_sft.sbatch"
            previous=$job_id
        done
    done
fi
