#!/usr/bin/env bash
# Remove what failed array tasks left behind, so they can be resubmitted with restart on.
#
#   ./clean_failed.sh JOBID INDEX...          # list what would be removed
#   ./clean_failed.sh --yes JOBID INDEX...    # remove it
#   ./clean_failed.sh 47602600 2 3 5 6 12 14 15
#
# Per index it removes the stage 1 to 3 step folders of output/case_N (the step0_*
# downloads are kept, they are valid and restart skips them) and the sandbox folders
# in array/ that the task's own SLURM logs name. Sandboxes of running tasks are not
# named in those logs, so they are left alone. The SLURM logs themselves are kept.
set -euo pipefail

BASE=/gpfs/projects/irb93/ruben/ab_wf
OUT=$BASE/output
ARRAY=$BASE/biobb_wf_antibody/array

DELETE=0
if [[ ${1:-} == --yes ]]; then DELETE=1; shift; fi
if (( $# < 2 )); then sed -n '2,12p' "$0"; exit 1; fi
JOBID=$1; shift

remove() {
    if (( DELETE )); then rm -rf -- "$1"; echo "removed $1"; else echo "would remove $1"; fi
}

for i in "$@"; do
    state=$(sacct -n -X -P -j "${JOBID}_${i}" -o State 2>/dev/null | head -1 || true)
    if [[ $state == RUNNING || $state == PENDING ]]; then
        echo "skip case_$i: task ${JOBID}_${i} is $state"; continue
    fi
    case_dir=$OUT/case_$i
    [[ -d $case_dir ]] || { echo "skip case_$i: no $case_dir"; continue; }
    for step in "$case_dir"/step[123]_*/; do
        [[ -d $step ]] && remove "${step%/}"
    done
    logs=("$OUT"/array_logs/ab_"${JOBID}_${i}".{out,err})
    for sandbox in $(grep -ohs 'sandbox_[0-9a-f-]\{36\}' "${logs[@]}" | sort -u); do
        [[ -d $ARRAY/$sandbox ]] && remove "$ARRAY/$sandbox"
    done
done
(( DELETE )) || echo "dry run, add --yes to remove"
