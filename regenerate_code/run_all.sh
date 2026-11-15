#!/usr/bin/env bash
# ==============================================================================
# DVD Project (16nm LP CMOS Parity8) - Master Pipeline Orchestration
# Regenerates the complete ML-driven PVT variation-aware leakage estimation project
# from physical models through transistor ML models, gate models, and parity checker.
# ==============================================================================
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_STAGE="Initialization"
MASTER_LOG=""
FORCE_FULL=false
VERIFY_ONLY=false
AUTO_CONFIRM=false
PYTHON_BIN="${PYTHON_BIN:-}"

# ------------------------------------------------------------------------------
# Usage / Help
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pipeline Orchestration Options:
  -h, --help        Show this help message and exit
  --verify-only     Non-destructive verification of existing deliverables across
                    Part A, Part B, and Part C without re-running SPICE or retraining.
                    (True read-only: does not modify files, bytecode, or logs)
  --force-full      Force full end-to-end pipeline regeneration (~1.5 hours)
                    overwriting existing datasets, models, plots, and results
  -y, --yes         Automatic yes to prompts (for non-interactive execution)

Examples:
  $(basename "$0") --verify-only    # Safely verify existing project artifacts (read-only)
  $(basename "$0") --force-full     # Full pipeline regeneration (prompts for confirmation)
  $(basename "$0") --force-full -y  # Full pipeline regeneration without prompting
EOF
}

# ------------------------------------------------------------------------------
# Error Handling & Traps
# ------------------------------------------------------------------------------
on_error() {
    local exit_code="$1"
    local line_no="$2"
    local last_cmd="$3"
    echo "" >&2
    echo "======================================================================" >&2
    echo "PIPELINE FAILED" >&2
    echo "Failed Stage : ${CURRENT_STAGE}" >&2
    echo "Command      : ${last_cmd}" >&2
    echo "Script Line  : ${line_no}" >&2
    echo "Exit Code    : ${exit_code}" >&2
    echo "Timestamp    : $(date '+%Y-%m-%d %H:%M:%S')" >&2
    if [ -n "${MASTER_LOG:-}" ] && [ -f "${MASTER_LOG}" ]; then
        echo "Log File     : ${MASTER_LOG}" >&2
        {
            echo ""
            echo "======================================================================"
            echo "PIPELINE FAILED"
            echo "Failed Stage : ${CURRENT_STAGE}"
            echo "Command      : ${last_cmd}"
            echo "Script Line  : ${line_no}"
            echo "Exit Code    : ${exit_code}"
            echo "Timestamp    : $(date '+%Y-%m-%d %H:%M:%S')"
            echo "======================================================================"
        } >> "${MASTER_LOG}"
    fi
    echo "======================================================================" >&2
    exit "${exit_code}"
}
trap 'on_error $? $LINENO "$BASH_COMMAND"' ERR

# ------------------------------------------------------------------------------
# Logging Helpers
# ------------------------------------------------------------------------------
setup_logging() {
    local results_dir="${SCRIPT_DIR}/results"
    mkdir -p "${results_dir}"
    MASTER_LOG="${results_dir}/run_all.log"

    # Safe log rotation: archive non-empty existing master log
    if [ -s "${MASTER_LOG}" ]; then
        local ts
        ts="$(date -r "${MASTER_LOG}" '+%Y%m%d_%H%M%S' 2>/dev/null || date '+%Y%m%d_%H%M%S')"
        local archive="${results_dir}/run_all_${ts}.log"
        mv "${MASTER_LOG}" "${archive}"
    fi
    touch "${MASTER_LOG}"
}

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    if [ -n "${MASTER_LOG:-}" ] && [ -f "${MASTER_LOG}" ]; then
        echo "$msg" >> "${MASTER_LOG}"
    fi
}

# ------------------------------------------------------------------------------
# Preflight Environment Verification (Separated for Modes)
# ------------------------------------------------------------------------------
check_python_preflight() {
    log "Checking Python environment..."

    local py_candidate=""
    if [ -n "${PYTHON_BIN:-}" ]; then
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "Error: Specified PYTHON_BIN '${PYTHON_BIN}' is not executable." >&2
            return 1
        fi
        if ! "$PYTHON_BIN" -B -c "import sklearn, pandas, numpy, joblib, scipy, matplotlib" >/dev/null 2>&1; then
            echo "Error: Python environment at '${PYTHON_BIN}' is missing required packages (sklearn, pandas, numpy, joblib, scipy, matplotlib)." >&2
            return 1
        fi
        py_candidate="$PYTHON_BIN"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ] && "${VIRTUAL_ENV}/bin/python3" -B -c "import sklearn, pandas, numpy, joblib, scipy, matplotlib" >/dev/null 2>&1; then
        py_candidate="${VIRTUAL_ENV}/bin/python3"
    elif command -v python3 >/dev/null 2>&1 && python3 -B -c "import sklearn, pandas, numpy, joblib, scipy, matplotlib" >/dev/null 2>&1; then
        py_candidate="$(command -v python3)"
    else
        # Check standard project virtualenv paths
        for venv_path in \
            "${SCRIPT_DIR}/../../dvd_person2/.venv/bin/python3" \
            "${HOME}/DVD_ML_PROJ/dvd_person2/.venv/bin/python3" \
            "/home/j-chandini-gayathri/DVD_ML_PROJ/dvd_person2/.venv/bin/python3"; do
            if [ -x "$venv_path" ] && "$venv_path" -B -c "import sklearn, pandas, numpy, joblib, scipy, matplotlib" >/dev/null 2>&1; then
                py_candidate="$venv_path"
                break
            fi
        done
    fi

    if [ -z "$py_candidate" ]; then
        echo "Error: No suitable Python environment found with required packages." >&2
        echo "Required packages: scikit-learn, pandas, numpy, joblib, scipy, matplotlib." >&2
        echo "Please activate a virtual environment (e.g., source ~/DVD_ML_PROJ/dvd_person2/.venv/bin/activate)" >&2
        echo "or specify PYTHON_BIN=/path/to/python3." >&2
        return 1
    fi

    export PYTHON_BIN="$py_candidate"
    export PATH="$(dirname "$py_candidate"):${PATH}"

    # Verify Python packages without bytecode creation
    "$PYTHON_BIN" -B -c "import sklearn, pandas, numpy, joblib, scipy, matplotlib" 2>/dev/null || {
        echo "Error: Python environment at '${PYTHON_BIN}' is missing required packages (sklearn, pandas, numpy, joblib, scipy, matplotlib)." >&2
        return 1
    }

    log "Python Interpreter: ${PYTHON_BIN} ($("$PYTHON_BIN" --version))"
}

check_ngspice_preflight() {
    log "Checking SPICE engine (ngspice)..."
    if ! command -v ngspice >/dev/null 2>&1; then
        echo "Error: 'ngspice' executable not found in PATH." >&2
        echo "Please install ngspice or add its directory to PATH." >&2
        return 1
    fi
    local ng_ver
    ng_ver="$(ngspice --version 2>&1 | head -n 1)"
    log "SPICE Engine: $(command -v ngspice) (${ng_ver})"
}

# ------------------------------------------------------------------------------
# Deliverable Validation Helpers (Deep Validation, Read-Only, Bytecode-Free)
# ------------------------------------------------------------------------------
verify_file() {
    local path="$1"
    local desc="$2"
    if [ ! -f "$path" ]; then
        echo "Error: Missing required deliverable: ${desc}" >&2
        echo "Expected file: ${path}" >&2
        return 1
    fi
    if [ ! -s "$path" ]; then
        echo "Error: Deliverable file is empty (0 bytes): ${desc}" >&2
        echo "File: ${path}" >&2
        return 1
    fi
}

validate_csv_deliverable() {
    local file_path="$1"
    local desc="$2"
    local required_cols="$3"
    local min_rows="$4"
    local positive_cols="${5:-}"
    local finite_cols="${6:-}"
    local nonneg_cols="${7:-}"

    "$PYTHON_BIN" -B -c '
import sys, os, pandas as pd, numpy as np

path = sys.argv[1]
desc = sys.argv[2]
req_cols = [c.strip() for c in sys.argv[3].split(",") if c.strip()]
min_rows = int(sys.argv[4])
pos_cols = [c.strip() for c in sys.argv[5].split(",") if c.strip()]
fin_cols = [c.strip() for c in sys.argv[6].split(",") if c.strip()]
nonneg_cols = [c.strip() for c in sys.argv[7].split(",") if c.strip()]

if not os.path.isfile(path):
    sys.stderr.write(f"Error: Missing deliverable file for {desc}: {path}\n")
    sys.exit(1)

if os.path.getsize(path) == 0:
    sys.stderr.write(f"Error: Deliverable file is empty (0 bytes) for {desc}: {path}\n")
    sys.exit(1)

try:
    df = pd.read_csv(path)
except Exception as e:
    sys.stderr.write(f"Error: Failed to parse CSV for {desc} ({path}): {e}\n")
    sys.exit(1)

missing = [c for c in req_cols if c not in df.columns]
if missing:
    sys.stderr.write(f"Error: {desc} ({path}) missing required columns: {missing}\n")
    sys.exit(1)

if len(df) < min_rows:
    sys.stderr.write(f"Error: {desc} ({path}) has {len(df)} rows, expected at least {min_rows}\n")
    sys.exit(1)

for c in set(fin_cols + pos_cols + nonneg_cols):
    if c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any() or np.isinf(vals).any():
            nan_cnt = int(vals.isna().sum() + np.isinf(vals).sum())
            sys.stderr.write(f"Error: {desc} ({path}) column \"{c}\" contains {nan_cnt} NaN or Inf values\n")
            sys.exit(1)

for c in pos_cols:
    if c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if (vals <= 0).any():
            bad_cnt = int((vals <= 0).sum())
            sys.stderr.write(f"Error: {desc} ({path}) column \"{c}\" contains {bad_cnt} non-positive values (must be > 0)\n")
            sys.exit(1)

for c in nonneg_cols:
    if c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if (vals < 0).any():
            bad_cnt = int((vals < 0).sum())
            sys.stderr.write(f"Error: {desc} ({path}) column \"{c}\" contains {bad_cnt} negative values (must be >= 0)\n")
            sys.exit(1)
' "$file_path" "$desc" "$required_cols" "$min_rows" "$positive_cols" "$finite_cols" "$nonneg_cols"
}

verify_all_deliverables() {
    local root="$1"
    local layout="$2" # "canonical" or "local"
    local missing=0

    log "Validating project deliverables (layout: ${layout}, root: ${root})..."

    if [ "$layout" = "canonical" ]; then
        # Canonical directory layout
        validate_csv_deliverable "${root}/data/shared_corners/corners_200.csv" \
            "Shared PVT Corners (canonical)" "corner,vdd,temp" 200 "vdd" "temp" || missing=$((missing + 1))
        
        for c in nmos_on nmos_off pmos_on pmos_off; do
            validate_csv_deliverable "${root}/data/PartA_transistor/${c}.csv" \
                "Part A dataset ${c} (canonical)" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib" 10000 "vdd,w,toxe,ndep,u0,xj" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib" || missing=$((missing + 1))
            verify_file "${root}/models/PartA_transistor/final_${c}.joblib" "Final model ${c} (canonical)" || missing=$((missing + 1))
            verify_file "${root}/netlists/PartA_transistor/fet_${c}.cir" "Part A netlist fet_${c} (canonical)" || missing=$((missing + 1))
        done

        validate_csv_deliverable "${root}/results/PartA_transistor/test_chosen.csv" \
            "Part A Test Evaluation Results (canonical)" "case,model,r2,median_err,p95_err" 4 "" "r2,median_err,p95_err" "median_err,p95_err" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/PartA_transistor/rail_check.csv" \
            "Part A Rail Evaluation Results (canonical)" "case,rail_median,rail_p95,uniform_median,uniform_p95" 4 "" "rail_median,rail_p95,uniform_median,uniform_p95,rail_id_median,rail_ig_median,rail_is_median,rail_ib_median" "rail_median,rail_p95,uniform_median,uniform_p95" || missing=$((missing + 1))

        validate_csv_deliverable "${root}/data/PartB_gates/gate_spice.csv" \
            "Part B Gate SPICE Reference (canonical)" "gate,ab,corner,vdd,temp,spice_leak" 3600 "vdd,spice_leak" "temp" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/PartB_gates/gate_ml.csv" \
            "Part B Gate ML Predictions (canonical)" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,rel_err" 3600 "vdd,spice_leak,ml_leak" "corner,vdd,temp,spice_leak,ml_leak,rel_err" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/PartB_gates/gate_nodes.csv" \
            "Part B Gate Node Voltages (canonical)" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV" 3600 "vdd,spice_leak,ml_leak,ml_leak_mlV" "corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/PartB_gates/gate_results.csv" \
            "Part B Gate Accuracy Metrics (canonical)" "gate,ab,spice_median,ml_median" 18 "spice_median,ml_median" "" || missing=$((missing + 1))
        verify_file "${root}/plots/PartB_gates/gate_parity.png" "Part B gate parity plot (canonical)" || missing=$((missing + 1))

        validate_csv_deliverable "${root}/data/PartC_parity_checker/circuit_spice.csv" \
            "Part C Parity SPICE Reference (canonical)" "vec,corner,vdd,temp,spice_leak,parity" 51200 "vdd,spice_leak" "temp" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/PartC_parity_checker/circuit_ml.csv" \
            "Part C Parity ML Predictions (canonical)" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice" 51200 "vdd,spice_leak,ml_leak,ml_leak_gatemodel,sum_gate_spice" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/PartC_parity_checker/circuit_results.csv" \
            "Part C Parity Accuracy Metrics (canonical)" "set,n,spice_median,ml_median" 12 "spice_median,ml_median" "" || missing=$((missing + 1))
        verify_file "${root}/plots/PartC_parity_checker/circuit_parity.png" "Part C circuit parity plot (canonical)" || missing=$((missing + 1))
        verify_file "${root}/netlists/PartC_parity_checker/parity8_example.cir" "Part C parity netlist (canonical)" || missing=$((missing + 1))

    else
        # Local regenerate_code flat layout
        validate_csv_deliverable "${root}/data/corners_200.csv" \
            "Shared PVT Corners (local)" "corner,vdd,temp" 200 "vdd" "temp" || missing=$((missing + 1))

        for c in nmos_on nmos_off pmos_on pmos_off; do
            validate_csv_deliverable "${root}/data/${c}.csv" \
                "Part A dataset ${c} (local)" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib" 10000 "vdd,w,toxe,ndep,u0,xj" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib" || missing=$((missing + 1))
            verify_file "${root}/models/final_${c}.joblib" "Final model ${c} (local)" || missing=$((missing + 1))
            verify_file "${root}/netlists/${c}.cir" "Part A netlist ${c} (local)" || missing=$((missing + 1))
        done

        validate_csv_deliverable "${root}/results/test_chosen.csv" \
            "Part A Test Evaluation Results (local)" "case,model,r2,median_err,p95_err" 4 "" "r2,median_err,p95_err" "median_err,p95_err" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/rail_check.csv" \
            "Part A Rail Evaluation Results (local)" "case,rail_median,rail_p95,uniform_median,uniform_p95" 4 "" "rail_median,rail_p95,uniform_median,uniform_p95,rail_id_median,rail_ig_median,rail_is_median,rail_ib_median" "rail_median,rail_p95,uniform_median,uniform_p95" || missing=$((missing + 1))
        verify_file "${root}/results/retrain_before_after.csv" "Part A retrain results (local)" || missing=$((missing + 1))
        verify_file "${root}/results/model_comparison_validation.csv" "Part A model comparison (local)" || missing=$((missing + 1))

        validate_csv_deliverable "${root}/data/gate_spice.csv" \
            "Part B Gate SPICE Reference (local)" "gate,ab,corner,vdd,temp,spice_leak" 3600 "vdd,spice_leak" "temp" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/gate_ml.csv" \
            "Part B Gate ML Predictions (local)" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,rel_err" 3600 "vdd,spice_leak,ml_leak" "corner,vdd,temp,spice_leak,ml_leak,rel_err" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/gate_nodes.csv" \
            "Part B Gate Node Voltages (local)" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV" 3600 "vdd,spice_leak,ml_leak,ml_leak_mlV" "corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/gate_results.csv" \
            "Part B Gate Accuracy Metrics (local)" "gate,ab,spice_median,ml_median" 18 "spice_median,ml_median" "" || missing=$((missing + 1))
        verify_file "${root}/plots/gate_parity.png" "Part B gate parity plot (local)" || missing=$((missing + 1))

        validate_csv_deliverable "${root}/data/circuit_spice.csv" \
            "Part C Parity SPICE Reference (local)" "vec,corner,vdd,temp,spice_leak,parity" 51200 "vdd,spice_leak" "temp" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/data/circuit_ml.csv" \
            "Part C Parity ML Predictions (local)" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice" 51200 "vdd,spice_leak,ml_leak,ml_leak_gatemodel,sum_gate_spice" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice" || missing=$((missing + 1))
        validate_csv_deliverable "${root}/results/circuit_results.csv" \
            "Part C Parity Accuracy Metrics (local)" "set,n,spice_median,ml_median" 12 "spice_median,ml_median" "" || missing=$((missing + 1))
        verify_file "${root}/plots/circuit_parity.png" "Part C circuit parity plot (local)" || missing=$((missing + 1))
        verify_file "${root}/netlists/parity8_example.cir" "Part C parity netlist (local)" || missing=$((missing + 1))
    fi

    if [ "$missing" -ne 0 ]; then
        return 1
    fi
    return 0
}

# Per-stage post-execution validation
verify_partA_data_outputs() {
    log "Verifying Part A data deliverables..."
    validate_csv_deliverable "${SCRIPT_DIR}/data/corners_200.csv" \
        "200 shared corners dataset" "corner,vdd,temp" 200 "vdd" "temp"
    for c in nmos_on nmos_off pmos_on pmos_off; do
        validate_csv_deliverable "${SCRIPT_DIR}/data/${c}.csv" \
            "Part A dataset ${c}" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib" 10 "vdd,w,toxe,ndep,u0,xj" "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib"
        verify_file "${SCRIPT_DIR}/netlists/${c}.cir" "Part A netlist for ${c}"
    done
    for p in vgs vds temp; do
        verify_file "${SCRIPT_DIR}/plots/trend_${p}.png" "Part A trend plot for ${p}"
    done
    log "Part A data deliverables verified."
}

verify_partA_models_outputs() {
    log "Verifying Part A model deliverables..."
    for c in nmos_on nmos_off pmos_on pmos_off; do
        verify_file "${SCRIPT_DIR}/models/final_${c}.joblib" "Final retrained ML model for ${c}"
    done
    validate_csv_deliverable "${SCRIPT_DIR}/results/test_chosen.csv" \
        "Final test evaluation results" "case,model,r2,median_err,p95_err" 4 "" "r2,median_err,p95_err" "median_err,p95_err"
    validate_csv_deliverable "${SCRIPT_DIR}/results/rail_check.csv" \
        "Part A rail check evaluation results" "case,rail_median,rail_p95,uniform_median,uniform_p95" 4 "" "rail_median,rail_p95,uniform_median,uniform_p95,rail_id_median,rail_ig_median,rail_is_median,rail_ib_median" "rail_median,rail_p95,uniform_median,uniform_p95"
    verify_file "${SCRIPT_DIR}/results/retrain_before_after.csv" "Retraining comparison results"
    verify_file "${SCRIPT_DIR}/results/model_comparison_validation.csv" "Candidate model validation comparison"
    log "Part A model deliverables verified."
}

verify_partB_gates_outputs() {
    log "Verifying Part B gate deliverables..."
    validate_csv_deliverable "${SCRIPT_DIR}/data/gate_spice.csv" \
        "Part B gate SPICE reference" "gate,ab,corner,vdd,temp,spice_leak" 3600 "vdd,spice_leak" "temp"
    validate_csv_deliverable "${SCRIPT_DIR}/data/gate_ml.csv" \
        "Part B gate ML predictions" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,rel_err" 3600 "vdd,spice_leak,ml_leak" "corner,vdd,temp,spice_leak,ml_leak,rel_err"
    validate_csv_deliverable "${SCRIPT_DIR}/data/gate_nodes.csv" \
        "Part B gate node voltages" "gate,ab,corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV" 3600 "vdd,spice_leak,ml_leak,ml_leak_mlV" "corner,vdd,temp,spice_leak,ml_leak,ml_out,ml_leak_mlV"
    validate_csv_deliverable "${SCRIPT_DIR}/results/gate_results.csv" \
        "Part B gate accuracy metrics" "gate,ab,spice_median,ml_median" 18 "spice_median,ml_median" ""
    verify_file "${SCRIPT_DIR}/plots/gate_parity.png" "Part B gate parity plot"
    log "Part B gate deliverables verified."
}

verify_partC_parity_outputs() {
    log "Verifying Part C parity checker deliverables..."
    validate_csv_deliverable "${SCRIPT_DIR}/data/circuit_spice.csv" \
        "Part C parity checker SPICE reference" "vec,corner,vdd,temp,spice_leak,parity" 51200 "vdd,spice_leak" "temp"
    validate_csv_deliverable "${SCRIPT_DIR}/data/circuit_ml.csv" \
        "Part C parity checker ML predictions" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice" 51200 "vdd,spice_leak,ml_leak,ml_leak_gatemodel,sum_gate_spice" "vec,corner,vdd,temp,spice_leak,spice_vpar,parity,ml_leak,ml_leak_gatemodel,sum_gate_spice"
    validate_csv_deliverable "${SCRIPT_DIR}/results/circuit_results.csv" \
        "Part C parity accuracy metrics" "set,n,spice_median,ml_median" 12 "spice_median,ml_median" ""
    verify_file "${SCRIPT_DIR}/plots/circuit_parity.png" "Part C parity scatter plot"
    verify_file "${SCRIPT_DIR}/netlists/parity8_example.cir" "Part C 8-bit parity netlist"
    log "Part C parity checker deliverables verified."
}

# ------------------------------------------------------------------------------
# Non-Destructive Verification Mode (--verify-only)
# ------------------------------------------------------------------------------
run_verification_mode() {
    CURRENT_STAGE="Verification Mode (--verify-only)"
    log "======================================================================"
    log "RUNNING IN NON-DESTRUCTIVE VERIFICATION MODE"
    log "No SPICE simulations, training, or file writes will be performed."
    log "======================================================================"

    # Only run Python preflight (do NOT require ngspice in --verify-only)
    check_python_preflight

    # Locate deliverables: check local regenerate_code first if populated, then canonical project tree
    local target_root=""
    local layout=""

    if [ -n "${DVD_PROJECT_ROOT:-}" ]; then
        if [ -f "${DVD_PROJECT_ROOT}/data/shared_corners/corners_200.csv" ]; then
            target_root="${DVD_PROJECT_ROOT}"
            layout="canonical"
            log "Found project deliverables at DVD_PROJECT_ROOT: ${target_root}"
        elif [ -f "${DVD_PROJECT_ROOT}/data/corners_200.csv" ]; then
            target_root="${DVD_PROJECT_ROOT}"
            layout="local"
            log "Found local project deliverables at DVD_PROJECT_ROOT: ${target_root}"
        else
            echo "Error: Could not locate deliverables in specified DVD_PROJECT_ROOT: ${DVD_PROJECT_ROOT}" >&2
            return 1
        fi
    elif [ -f "${SCRIPT_DIR}/data/corners_200.csv" ]; then
        target_root="${SCRIPT_DIR}"
        layout="local"
        log "Found local regenerated deliverables at ${target_root}"
    else
        for candidate in \
            "${SCRIPT_DIR}/.." \
            "${HOME}/DVD_ML_PROJ/DVD_Project_16nm_LP_Parity8" \
            "/home/j-chandini-gayathri/DVD_ML_PROJ/DVD_Project_16nm_LP_Parity8"; do
            if [ -n "$candidate" ] && [ -f "${candidate}/data/shared_corners/corners_200.csv" ]; then
                target_root="$(cd "$candidate" && pwd)"
                layout="canonical"
                log "Found canonical project deliverables at ${target_root}"
                break
            fi
        done
    fi

    if [ -z "$target_root" ]; then
        echo "Error: Could not locate deliverables in local directory (${SCRIPT_DIR}) or canonical project paths." >&2
        return 1
    fi

    # Perform thorough deliverable validation (read-only)
    verify_all_deliverables "$target_root" "$layout"

    # Fast non-destructive sanity tests (zero SPICE)
    log "Running fast non-destructive sanity tests..."

    # Check ML inference across all 4 transistor models
    log "Checking ML model load and inference capabilities across all 4 Part A models..."
    local models_dir=""
    if [ "$layout" = "local" ] && [ -d "${target_root}/models" ]; then
        models_dir="${target_root}/models"
    elif [ "$layout" = "canonical" ] && [ -d "${target_root}/models/PartA_transistor" ]; then
        models_dir="${target_root}/models/PartA_transistor"
    fi

    if [ -n "$models_dir" ]; then
        "$PYTHON_BIN" -B -c '
import joblib, sys, os
models_dir = sys.argv[1]
for script_cand in [
    os.path.join(models_dir, "..", "scripts"),
    os.path.join(models_dir, "..", "regenerate_code", "scripts"),
    os.path.join(os.path.dirname(os.path.dirname(models_dir)), "regenerate_code", "scripts"),
    os.path.join(os.path.dirname(os.path.dirname(models_dir)), "scripts", "PartA_transistor", "ml_models"),
    models_dir
]:
    if os.path.isdir(script_cand) and script_cand not in sys.path:
        sys.path.insert(0, os.path.abspath(script_cand))
cases = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
for c in cases:
    p = os.path.join(models_dir, f"final_{c}.joblib")
    if not os.path.isfile(p):
        sys.stderr.write(f"Error: Missing model file: {p}\n")
        sys.exit(1)
    try:
        obj = joblib.load(p)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load model {c} from {p}: {e}\n")
        sys.exit(1)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not hasattr(model, "predict"):
        sys.stderr.write(f"Error: Model object for {c} in {p} missing predict() method\n")
        sys.exit(1)
    print(f"ML model load check passed for {c} ({type(model).__name__})")
' "$models_dir"
    fi

    log "Checking 8-bit parity checker Boolean circuit logic..."
    "$PYTHON_BIN" -B -c '
import sys, os
sys.path.insert(0, os.path.join("'"${SCRIPT_DIR}"'", "scripts"))
import circuit
bad = [v for v in range(256) if circuit.logic_levels(v)[circuit.OUT] != bin(v).count("1") % 2]
assert len(bad) == 0, f"Boolean parity mismatch for vectors: {bad}"
print("Boolean parity checker logic: 256/256 vectors verified")
'

    log "======================================================================"
    log "ALL PROJECT DELIVERABLES AND SANITY CHECKS PASSED (100% VERIFIED)"
    log "======================================================================"
    return 0
}

# ------------------------------------------------------------------------------
# Stage Execution Wrapper
# ------------------------------------------------------------------------------
run_stage() {
    local stage_name="$1"
    local script_name="$2"
    CURRENT_STAGE="${stage_name}"

    log "======================================================================"
    log "STARTING STAGE: ${stage_name}"
    log "Script        : ${script_name}"
    log "======================================================================"
    local t_start
    t_start=$(date +%s)

    # Execute child script using bash
    # Capture stdout and stderr into MASTER_LOG while preserving live terminal output
    if [ -n "${MASTER_LOG:-}" ] && [ -f "${MASTER_LOG}" ]; then
        bash "${SCRIPT_DIR}/${script_name}" 2>&1 | tee -a "${MASTER_LOG}"
    else
        bash "${SCRIPT_DIR}/${script_name}"
    fi

    local t_end
    t_end=$(date +%s)
    local elapsed=$((t_end - t_start))
    log "COMPLETED STAGE: ${stage_name} in ${elapsed}s ($((elapsed / 60))m $((elapsed % 60))s)"
    log "======================================================================"
}

# ------------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------------
main() {
    # Parse CLI Arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            --verify-only|--check)
                VERIFY_ONLY=true
                shift
                ;;
            --force-full)
                FORCE_FULL=true
                shift
                ;;
            -y|--yes)
                AUTO_CONFIRM=true
                shift
                ;;
            *)
                echo "Error: Unknown option '$1'. Use --help for usage." >&2
                exit 2
                ;;
        esac
    done

    # Handle verification mode (TRUE READ-ONLY: zero file writes, no log creation)
    if [ "$VERIFY_ONLY" = true ]; then
        run_verification_mode
        exit 0
    fi

    # Check for existing data before full regeneration
    local existing_data=false
    if [ -f "${SCRIPT_DIR}/data/corners_200.csv" ] || \
       [ -f "${SCRIPT_DIR}/data/nmos_on.csv" ] || \
       [ -f "${SCRIPT_DIR}/data/gate_spice.csv" ] || \
       [ -f "${SCRIPT_DIR}/data/circuit_spice.csv" ] || \
       [ -f "${SCRIPT_DIR}/models/final_nmos_on.joblib" ] || \
       [ -f "${SCRIPT_DIR}/results/test_chosen.csv" ]; then
        existing_data=true
    fi

    if [ "$existing_data" = true ] && [ "$FORCE_FULL" = false ]; then
        echo "" >&2
        echo "======================================================================" >&2
        echo "SAFETY REFUSAL: Existing generated data detected in ${SCRIPT_DIR}." >&2
        echo "" >&2
        echo "Full regeneration will overwrite datasets, retrain all ML models," >&2
        echo "and run ~95,000 SPICE simulations (about 1.5 hours of computation)." >&2
        echo "" >&2
        echo "To safely verify existing deliverables without re-simulating, run:" >&2
        echo "  $0 --verify-only" >&2
        echo "" >&2
        echo "To force full regeneration and overwrite existing results, run:" >&2
        echo "  $0 --force-full" >&2
        echo "======================================================================" >&2
        exit 1
    fi

    # Interactive confirmation for --force-full if not auto-confirmed
    if [ "$FORCE_FULL" = true ] && [ "$AUTO_CONFIRM" = false ]; then
        echo "WARNING: Full regeneration will completely overwrite datasets and models (~1.5 hours)."
        read -r -p "Are you sure you want to proceed? [y/N]: " reply
        if [[ ! "$reply" =~ ^[Yy]$ ]]; then
            echo "Full regeneration aborted by user."
            exit 1
        fi
    fi

    # ONLY NOW initialize logging (ensures aborted runs or verify-only never touch filesystem)
    setup_logging

    # Begin Full Execution
    local overall_start
    overall_start=$(date +%s)
    log "======================================================================"
    log "STARTING FULL PIPELINE REGENERATION"
    log "Working Directory : ${SCRIPT_DIR}"
    log "Timestamp         : $(date '+%Y-%m-%d %H:%M:%S')"
    log "======================================================================"

    check_python_preflight
    check_ngspice_preflight

    # Stage 1: Part A Data Generation
    run_stage "Part A (Data Generation & Physical Validation)" "run_partA_data.sh"
    verify_partA_data_outputs

    # Stage 2: Part A Model Training & Evaluation
    run_stage "Part A (Transistor ML Training & Retraining)" "run_partA_models.sh"
    verify_partA_models_outputs

    # Stage 3: Part B Gate-Level SPICE & ML
    run_stage "Part B (Gate-Level SPICE & ML Leakage)" "run_partB_gates.sh"
    verify_partB_gates_outputs

    # Stage 4: Part C 8-bit Parity Checker
    run_stage "Part C (8-bit Parity Checker SPICE & ML)" "run_partC_parity_checker.sh"
    verify_partC_parity_outputs

    local overall_end
    overall_end=$(date +%s)
    local total_elapsed=$((overall_end - overall_start))
    CURRENT_STAGE="Completed"

    log "======================================================================"
    log "ALL PIPELINE STAGES COMPLETED SUCCESSFULLY"
    log "Total Duration : ${total_elapsed}s ($((total_elapsed / 60))m $((total_elapsed % 60))s)"
    log "Master Log     : ${MASTER_LOG}"
    log "======================================================================"
}

main "$@"
