"""Step 9: how accurate are the models at the voltages gates really use (Vd, Vs exactly 0 or VDD)?"""
import os
import numpy as np, pandas as pd
from sampler import draw_case
from runner import run_batch, ROOT
from predict import predict_currents
from mlcommon import CASES, TARGETS, FLOOR, RAW

N = 300
rows = []
uni = pd.read_csv(ROOT + "/results/test_chosen.csv").set_index("case")
for ci, case in enumerate(CASES):
    if not os.path.exists("%s/models/final_%s.joblib" % (ROOT, case)):
        print("skipping", case); continue
    rng = np.random.default_rng([777, ci])
    df = draw_case(case, N, rng)
    r = rng.integers(0, 2, (N, 2))                       # each of Vd, Vs is exactly 0 or exactly VDD
    df["vd"], df["vs"] = r[:, 0] * df["vdd"], r[:, 1] * df["vdd"]
    I = run_batch(case, df)                              # SPICE truth
    out = df.copy(); out[TARGETS] = I
    out.to_csv("%s/data/rail_test_%s.csv" % (ROOT, case), index=False, float_format="%.10e")
    P = predict_currents(case, *[df[k].to_numpy() for k in RAW])
    rel = np.abs(P - I) / np.maximum(np.abs(I), FLOOR)
    row = dict(case=case, rail_median=np.median(rel), rail_p95=np.percentile(rel, 95),
               uniform_median=uni.loc[case, "median_err"], uniform_p95=uni.loc[case, "p95_err"])
    for j, k in enumerate(TARGETS):
        row["rail_%s_median" % k] = np.median(rel[:, j])
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(ROOT + "/results/rail_check.csv", index=False)
print(res.to_string(index=False, float_format=lambda x: "%.4g" % x))
