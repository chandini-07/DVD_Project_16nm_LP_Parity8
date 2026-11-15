"""Rail-mix VALIDATION set (3000/case, seed [555,ci]) used for model development decisions. Not used for training."""
import os, time, numpy as np, pandas as pd
from sampler import CASES, draw_case
from runner import run_batch, ROOT
from mlcommon import TARGETS
N = 3000
for ci, case in enumerate(CASES):
    out = "%s/data/railval_%s.csv" % (ROOT, case)
    if os.path.exists(out): continue
    rng = np.random.default_rng([555, ci]); df = draw_case(case, N, rng)
    for k in ("vd", "vs"):
        u = rng.random(N); v = df[k].to_numpy().copy()
        v[u < 0.4] = 0.0; m = (u >= 0.4) & (u < 0.8); v[m] = df["vdd"].to_numpy()[m]; df[k] = v
    I = run_batch(case, df); ok = np.isfinite(I).all(axis=1)
    df = df[ok].copy(); df[TARGETS] = I[ok]; df.to_csv(out, index=False, float_format="%.10e"); print(case, len(df), flush=True)
