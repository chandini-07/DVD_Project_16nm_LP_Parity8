import os, sys, time
import numpy as np, pandas as pd
from sampler import CASES, draw_case
from runner import run_batch, ROOT

COLS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
        "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    outdir = os.path.join(ROOT, "data") if N == 10000 else os.path.join(ROOT, "data", "test")
    os.makedirs(outdir, exist_ok=True)
    log = []
    for ci, case in enumerate(CASES):
        t0 = time.time()
        rng = np.random.default_rng([42, ci])        # fixed seed, different for each case
        parts, have, n_run, n_fail = [], 0, 0, 0
        while have < N:                              # keep drawing until we have N valid samples
            need = N - have
            df = draw_case(case, int(need * 1.02) + 5, rng)
            I = run_batch(case, df)
            ok = np.isfinite(I).all(axis=1)
            n_run += len(df); n_fail += int((~ok).sum())
            d = df[ok].copy()
            d[["id", "ig", "is", "ib"]] = I[ok]
            parts.append(d); have += len(d)
            print("%s: %d / %d valid" % (case, have, N), flush=True)
        out = pd.concat(parts).iloc[:N][COLS].reset_index(drop=True)
        out.to_csv(os.path.join(outdir, case + ".csv"), index=False, float_format="%.10e")
        log.append("%s: rows=%d simulated=%d failed=%d time=%.0fs" % (case, len(out), n_run, n_fail, time.time() - t0))
        print(log[-1], flush=True)
    open(os.path.join(ROOT, "results", "generation_log.txt"), "a").write("\n".join(log) + "\n")

if __name__ == "__main__":
    main()
