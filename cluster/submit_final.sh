#!/bin/bash
# Usage, from the repo whose code the jobs must run (sbatch records it as SLURM_SUBMIT_DIR):
#   env -u ALLOW_DIRTY -u DRYRUN bash cluster/submit_final.sh      the final graph (clean code only)
#   DRYRUN=1 bash cluster/submit_final.sh                          the same graph, tiny, in a modified throwaway copy
# Submits every job of the final graph at once, with dependencies: the safe base's extended study, then per base
# GP fit -> 3 confirm runs -> selection -> chosen-beta final runs, the beta = 0 final runs, and per base the test.
# Prints each job id and name, writes them with their sbatch arguments to cluster/final_jobs.txt and prints the
# CPU-h requested. DRYRUN=1: small resources, 2 extension jobs, n-eval 5, test on 3 selection seeds, ALLOW_DIRTY=1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

A=ppo_5m700k_goal2
B=ppo_5m_goal2_safe
P=pass1
LIST=cluster/final_jobs.txt
MAX_SUBMIT=30  # QOS MaxSubmitJobsPU: pending + running jobs per user

# Same pathspec as the dirty check of the runs
DIRTY=$(git status --porcelain --untracked-files=all -- src scripts configs 'cluster/*.sh')
if [ -n "${DRYRUN:-}" ]; then
    if [ -z "$DIRTY" ]; then
        echo "DRYRUN is set but the code is clean: the dry run belongs in a modified copy" >&2
        exit 1
    fi
    MODE="dry run"
    export ALLOW_DIRTY=1
    FT_RES=(--cpus-per-task=2 --mem=12G --time=0:30:00)
    SEL_RES=(--cpus-per-task=2 --mem=4G --time=0:20:00)
    TEST_RES=(--cpus-per-task=2 --mem=8G --time=0:30:00)
    N_EXT=2
    NEVAL=(--n-eval 5)
    TEST=(selection 3)
else
    if [ -n "${ALLOW_DIRTY:-}" ] || [ -n "$DIRTY" ]; then
        echo "the final graph needs clean code and ALLOW_DIRTY unset; uncommitted: ${DIRTY:-none}" >&2
        exit 1
    fi
    MODE="final graph"
    # Same as the scripts' headers, passed explicitly so the job list shows them
    FT_RES=(--cpus-per-task=8 --mem=36G --time=4:00:00)
    SEL_RES=(--cpus-per-task=2 --mem=8G --time=0:20:00)
    TEST_RES=(--cpus-per-task=8 --mem=16G --time=6:00:00)
    N_EXT=8
    NEVAL=()
    TEST=(test)
fi
if [ -e "$LIST" ]; then
    echo "$LIST exists: the graph was already submitted from here" >&2
    exit 1
fi
# Refuse up front rather than stop halfway with a partial graph
N_JOBS=$((N_EXT + 20))  # 4 ref + per base: fit, 3 confirm, select, 2 ft, test
N_QUEUED=$(squeue -u "$USER" -h | wc -l)
if [ $((N_QUEUED + N_JOBS)) -gt "$MAX_SUBMIT" ]; then
    echo "$N_QUEUED jobs queued + $N_JOBS to submit > $MAX_SUBMIT allowed" >&2
    exit 1
fi
echo "mode: $MODE, $N_JOBS jobs, $N_QUEUED already queued"

TOTAL_SECS=0
# sub <var> <sbatch args...>: submits, stores the job id in <var>, adds the requested CPU-seconds
sub() {
    local var="$1"
    shift
    local out id cpus=0 secs=0 name="" a h m s
    out=$(sbatch --parsable "$@")
    id="${out%%;*}"
    for a in "$@"; do
        case "$a" in
            --cpus-per-task=*) cpus="${a#*=}" ;;
            --time=*) IFS=: read -r h m s <<< "${a#*=}"; secs=$((10#$h * 3600 + 10#$m * 60 + 10#$s)) ;;
            --job-name=*) name="${a#*=}" ;;
        esac
    done
    TOTAL_SECS=$((TOTAL_SECS + cpus * secs))
    printf -v "$var" '%s' "$id"
    echo "$id $name" | tee -a "$LIST"
    echo "    sbatch $*" >> "$LIST"
}

# Safe base's extended study: one trial per job; jobs after the budget or a stop exit at once
EXT_ID=
for k in $(seq 1 "$N_EXT"); do
    sub EXT_ID --job-name="ext_B_$k" "${FT_RES[@]}" ${EXT_ID:+--dependency=afterany:$EXT_ID} \
        cluster/sweep_trial.sh "$B" "$P" --beta-range 0.1 300 --study-tag ext \
        --warm-start "runs/sweeps/${B}_${P}/journal.log" "${NEVAL[@]}"
done

# chosen <letter> <base> <journal> <fit dependency or "">: fit -> 3 confirm -> select -> 2 ft jobs (FT_<letter>1/2)
chosen() {
    local L="$1" BASE="$2" JOURNAL="$3" DEP="$4" SEL="runs/sweeps/${2}_${P}_select" FIT SELECT C0 C1 C2
    sub FIT --job-name="fit_$L" "${SEL_RES[@]}" ${DEP:+--dependency=$DEP} \
        cluster/beta_select.sh fit "$BASE" --journal "$JOURNAL" --out-dir "$SEL"
    for r in 0 1 2; do
        sub "C$r" --job-name="conf_${L}_r$r" "${FT_RES[@]}" --dependency="afterok:$FIT" \
            cluster/sweep_ref.sh "$BASE" "$P" --beta-file "$SEL/top3.json" --rank "$r" --pool confirm --seeds 0,1,2 \
            "${NEVAL[@]}"
    done
    sub SELECT --job-name="sel_$L" "${SEL_RES[@]}" --dependency="afterok:$C0:$C1:$C2" \
        cluster/beta_select.sh select "$BASE" "$P" --top3 "$SEL/top3.json" --out-dir "$SEL"
    sub "FT_${L}1" --job-name="f_ft_${L}_s0-2" "${FT_RES[@]}" --dependency="afterok:$SELECT" \
        cluster/sweep_ref.sh "$BASE" "$P" --beta-file "$SEL/chosen_beta.json" --pool final --seeds 0,1,2 "${NEVAL[@]}"
    sub "FT_${L}2" --job-name="f_ft_${L}_s3-5" "${FT_RES[@]}" --dependency="afterok:$SELECT" \
        cluster/sweep_ref.sh "$BASE" "$P" --beta-file "$SEL/chosen_beta.json" --pool final --seeds 3,4,5 "${NEVAL[@]}"
}
chosen A "$A" "runs/sweeps/${A}_${P}/journal.log" ""
chosen B "$B" "runs/sweeps/${B}_${P}_ext/journal.log" "afterany:$EXT_ID"

# beta = 0 final runs last: only the tests need them
sub REF_A1 --job-name=f_ref_A_s0-2 "${FT_RES[@]}" cluster/sweep_ref.sh "$A" "$P" --pool final --seeds 0,1,2 "${NEVAL[@]}"
sub REF_A2 --job-name=f_ref_A_s3-5 "${FT_RES[@]}" cluster/sweep_ref.sh "$A" "$P" --pool final --seeds 3,4,5 "${NEVAL[@]}"
sub REF_B1 --job-name=f_ref_B_s0-2 "${FT_RES[@]}" cluster/sweep_ref.sh "$B" "$P" --pool final --seeds 0,1,2 "${NEVAL[@]}"
sub REF_B2 --job-name=f_ref_B_s3-5 "${FT_RES[@]}" cluster/sweep_ref.sh "$B" "$P" --pool final --seeds 3,4,5 "${NEVAL[@]}"

sub TEST_A --job-name=test_A "${TEST_RES[@]}" --dependency="afterok:$FT_A1:$FT_A2:$REF_A1:$REF_A2" \
    cluster/final_test.sh "$A" "$P" "runs/sweeps/${A}_${P}_select/chosen_beta.json" "${TEST[@]}"
sub TEST_B --job-name=test_B "${TEST_RES[@]}" --dependency="afterok:$FT_B1:$FT_B2:$REF_B1:$REF_B2" \
    cluster/final_test.sh "$B" "$P" "runs/sweeps/${B}_${P}_select/chosen_beta.json" "${TEST[@]}"

printf 'CPU-h requested (jobs x CPUs x time limit), total: %.2f\n' "$(echo "$TOTAL_SECS / 3600" | bc -l)"
