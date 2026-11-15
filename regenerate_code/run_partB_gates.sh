#!/bin/bash
# Part B (gate level): regenerates every gate deliverable.
# Prerequisite: Stage 1 and Stage 2 must have generated:
#   data/corners_200.csv
#   models/final_nmos_on.joblib
#   models/final_nmos_off.joblib
#   models/final_pmos_on.joblib
#   models/final_pmos_off.joblib
# Order: SPICE reference -> ML leakage -> parity plot -> ML node voltages -> example netlists. All seeds fixed upstream (corners.py).
set -e
mkdir -p "$(dirname "$0")"/data "$(dirname "$0")"/results "$(dirname "$0")"/plots "$(dirname "$0")"/netlists "$(dirname "$0")"/models
cd "$(dirname "$0")/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ngspice --version | head -n 2
python3 -W ignore gate_spice.py          # data/gate_spice.csv   (3600 SPICE runs, about 1 min)
python3 -W ignore gate_ml.py             # data/gate_ml.csv, results/gate_results.csv  (deliverable 6)
python3 -W ignore gate_plot.py           # plots/gate_parity.png (deliverable 7)
python3 -W ignore gate_nodes.py          # data/gate_nodes.csv, results/gate_node_results.csv
python3 -W ignore make_gate_netlists.py  # netlists/gate_*.cir   (deliverable 2, gates)
echo "earlier parts done."
