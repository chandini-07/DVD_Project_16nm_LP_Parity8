"""Part C SPICE reference: 8-bit parity checker, 256 input vectors x 200 corners -> data/circuit_spice.csv
Usage: python3 circuit_spice.py            (all cores)
       python3 circuit_spice.py --limit 20 (quick test: first 20 jobs, serial)"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from circuit import run_circuit, logic_levels, OUT, ROOT

CS = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv")).to_dict("records")

def work(job):
    vec, k = job
    c = CS[k]
    r = run_circuit(vec, c)
    if r is None:
        return (vec, k, c["vdd"], c["temp"], np.nan, np.nan)
    return (vec, k, c["vdd"], c["temp"], r["leak"], r["vpar"])

if __name__ == "__main__":
    jobs = [(v, k) for k in range(len(CS)) for v in range(256)]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    if limit:
        jobs = jobs[:limit]
    t0 = time.time(); rows = []
    if limit:
        for j in jobs:
            rows.append(work(j))
    else:
        with Pool(os.cpu_count()) as p:
            for i, r in enumerate(p.imap(work, jobs, chunksize=64), 1):
                rows.append(r)
                if i % 5000 == 0:
                    print("%d / %d  (%.0f s)" % (i, len(jobs), time.time() - t0), flush=True)
    df = pd.DataFrame(rows, columns=["vec", "corner", "vdd", "temp", "spice_leak", "spice_vpar"])
    df["parity"] = [bin(v).count("1") % 2 for v in df.vec]
    out = os.path.join(ROOT, "data", "circuit_spice_test.csv" if limit else "circuit_spice.csv")
    df.to_csv(out, index=False)
    right = ((df.spice_vpar > df.vdd / 2).astype(int) == df.parity)
    print("%d rows in %.0f s; failures: %d; non-positive leak: %d; wrong parity output: %d"
          % (len(df), time.time() - t0, int(df.spice_leak.isna().sum()), int((df.spice_leak <= 0).sum()), int((~right).sum())))
    print("leak min/median/max: %.3e / %.3e / %.3e A" % (df.spice_leak.min(), df.spice_leak.median(), df.spice_leak.max()))
    print("saved", out)
