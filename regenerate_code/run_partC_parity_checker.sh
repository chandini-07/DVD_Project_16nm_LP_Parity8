#!/bin/bash
# Part C (8-bit parity checker): regenerates every circuit deliverable.
# Prerequisite: Stage 1 and Stage 3 must have generated:
#   data/corners_200.csv
#   data/gate_nodes.csv
set -e
mkdir -p "$(dirname "$0")"/data "$(dirname "$0")"/results "$(dirname "$0")"/plots "$(dirname "$0")"/netlists "$(dirname "$0")"/models
cd "$(dirname "$0")/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -W ignore circuit.py              # smoke test: 28 NAND2, Boolean parity check, 4 SPICE runs
python3 -W ignore circuit_spice.py        # data/circuit_spice.csv  (51,200 SPICE runs, about 8 min)
python3 -W ignore circuit_ml.py           # data/circuit_ml.csv, results/circuit_results.csv   (deliverable 8)
python3 -W ignore circuit_plot.py         # plots/circuit_parity.png                           (deliverable 8)
python3 -W ignore make_circuit_netlist.py # netlists/parity8_example.cir                       (deliverable 2, circuit)
echo "earlier parts done."
