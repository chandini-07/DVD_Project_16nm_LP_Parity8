"""Extra TRAINING data (never used for validation/test): 24k rail-mix + 16k uniform samples per case.
Seeds [31337, case_index] are disjoint from the 10k split (42), aug (2024), rail_test (777), gate corners."""
import sys, os, time, numpy as np, pandas as pd
from sampler import CASES, draw_case
from runner import run_batch, ROOT
from mlcommon import TARGETS
N_RAIL, N_UNI = 24000, 16000
for ci, case in enumerate(CASES):
    out = "%s/data/big_%s.csv" % (ROOT, case)
    if os.path.exists(out): continue
    t0 = time.time(); rng = np.random.default_rng([31337, ci])
    df = draw_case(case, N_RAIL + N_UNI, rng)
    for k in ("vd", "vs"):
        u = rng.random(N_RAIL); v = df[k].to_numpy().copy()
        a = v[:N_RAIL].copy(); a[u < 0.4] = 0.0
        m = (u >= 0.4) & (u < 0.8); a[m] = df["vdd"].to_numpy()[:N_RAIL][m]
        v[:N_RAIL] = a; df[k] = v
    I = run_batch(case, df); ok = np.isfinite(I).all(axis=1)
    df = df[ok].copy(); df[TARGETS] = I[ok]
    df.to_csv(out, index=False, float_format="%.10e")
    print("%s: %d rows (%d failed) %.0fs" % (case, len(df), int((~ok).sum()), time.time() - t0), flush=True)
