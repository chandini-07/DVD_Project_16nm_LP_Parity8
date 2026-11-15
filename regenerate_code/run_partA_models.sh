#!/bin/bash
# Part A (transistor ML models): regenerates every Part A model deliverable (~1 hour).
# Prerequisite: Stage 1 must have generated:
#   data/nmos_on.csv
#   data/nmos_off.csv
#   data/pmos_on.csv
#   data/pmos_off.csv
set -e
mkdir -p "$(dirname "$0")"/data "$(dirname "$0")"/results "$(dirname "$0")"/plots "$(dirname "$0")"/netlists "$(dirname "$0")"/models
cd "$(dirname "$0")/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CASES="nmos_on nmos_off pmos_on pmos_off"

wait_jobs() {
    local stage="$1"
    shift
    local failed=0
    for entry in "$@"; do
        local case_name="${entry%%:*}"
        local pid="${entry#*:}"
        local status=0
        wait "$pid" || status=$?
        if [ "$status" -ne 0 ]; then
            echo "Error: ${stage} failed for case '${case_name}' (PID ${pid}, exit status ${status})" >&2
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "Error: One or more parallel background jobs failed in ${stage}." >&2
        return 1
    fi
}

# Stage 1: Data look-over
python3 -W ignore explore.py

# Stage 2: Hyperparameter tuning on validation set
python3 -W ignore tune_case.py nmos_on

# Stage 3: Candidate model comparison (validation set)
jobs=()
for c in $CASES; do
    python3 -W ignore compare_case.py $c &
    jobs+=("$c:$!")
done
wait_jobs "Stage 3 (compare_case)" "${jobs[@]}"

python3 -W ignore - <<'PY'
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/compare_*.csv"))]).to_csv("../results/model_comparison_validation.csv", index=False)
PY

# Stage 4: Initialize candidate final models with uniform-trained MLPs
for c in $CASES; do cp ../models/${c}__mlp.joblib ../models/final_${c}.joblib; done

# Stage 5: Baseline uniform test evaluation (produces results/test_chosen_baseline.csv)
python3 -W ignore test_chosen.py --baseline

# Stage 6: Baseline rail evaluation and rail dataset generation (produces data/rail_test_<case>.csv and results/rail_check_baseline.csv)
python3 -W ignore rail_check.py --baseline

# Stage 7: Rail-augmented retraining (reads data/rail_test_<case>.csv -> produces augmented models/final_<case>.joblib)
jobs=()
for c in $CASES; do
    python3 -W ignore retrain_case.py $c &
    jobs+=("$c:$!")
done
wait_jobs "Stage 7 (retrain_case)" "${jobs[@]}"

python3 -W ignore - <<'PY'
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/retrain_*.csv"))]).to_csv("../results/retrain_before_after.csv", index=False)
PY

# Stage 8: Final test set evaluation (produces results/test_chosen.csv and plots/pred_vs_spice_<case>.png - Deliverable 5)
python3 -W ignore test_chosen.py

# Stage 9: Final rail check evaluation (reuses verified data/rail_test_<case>.csv -> produces results/rail_check.csv)
python3 -W ignore rail_check.py

# Stage 10: Final prediction verification
python3 -W ignore test_predict.py

echo "earlier parts done."
