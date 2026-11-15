#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$0")"/data "$(dirname "$0")"/results "$(dirname "$0")"/plots "$(dirname "$0")"/netlists "$(dirname "$0")"/models
cd "$(dirname "$0")"
rm -f results/generation_log.txt
python3 scripts/make_netlists.py   # 4 example netlists
python3 scripts/corners.py         # 200 shared corners
python3 scripts/generate.py        # 4 CSVs x 10,000 rows
python3 scripts/kirchhoff.py       # Kirchhoff check
python3 scripts/trends.py          # 3 trend plots
