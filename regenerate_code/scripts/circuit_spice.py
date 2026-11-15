"""Part C SPICE reference: 8-bit parity checker, 256 input vectors x 200 corners -> data/circuit_spice.csv
Usage: python3 circuit_spice.py            (all cores)
       python3 circuit_spice.py --limit 20 (quick test: first 20 jobs, serial)"""
import os, sys, time, tempfile
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

def parse_limit(argv):
    if "--limit" not in argv:
        return None
    idx = argv.index("--limit")
    if idx + 1 >= len(argv):
        print("Error: --limit requires an integer argument", file=sys.stderr)
        sys.exit(2)
    val_str = argv[idx + 1]
    try:
        val = int(val_str)
    except ValueError:
        print(f"Error: --limit requires an integer argument, got '{val_str}'", file=sys.stderr)
        sys.exit(2)
    if val <= 0:
        print(f"Error: --limit must be a positive integer, got {val}", file=sys.stderr)
        sys.exit(2)
    return val

if __name__ == "__main__":
    jobs = [(v, k) for k in range(len(CS)) for v in range(256)]
    limit = parse_limit(sys.argv)
    if limit is not None:
        jobs = jobs[:limit]
    t0 = time.time(); rows = []
    if limit is not None:
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
    
    right = ((df.spice_vpar > df.vdd / 2).astype(int) == df.parity)
    n_fail = int(df.spice_leak.isna().sum())
    n_nonpos = int((df.spice_leak <= 0).sum())
    n_wrong = int((~right).sum())
    print("%d rows in %.0f s; failures: %d; non-positive leak: %d; wrong parity output: %d"
          % (len(df), time.time() - t0, n_fail, n_nonpos, n_wrong))
          
    if n_fail > 0 or n_nonpos > 0 or n_wrong > 0 or len(df) != len(jobs):
        log_path = os.path.join(ROOT, "results", "circuit_spice_failures.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Circuit SPICE batch failed: failures={n_fail}, nonpos={n_nonpos}, wrong_parity={n_wrong}, total_rows={len(df)} expected={len(jobs)}\n")
            for idx, row in df[df.spice_leak.isna() | (df.spice_leak <= 0) | (~right)].iterrows():
                f.write(f"vec={int(row.vec)} corner={int(row.corner)}: leak={row.spice_leak} vpar={row.spice_vpar} parity={row.parity}\n")
        print(f"Error: circuit_spice failed. Log written to {log_path}", file=sys.stderr)
        sys.exit(1)
        
    out = os.path.join(ROOT, "data", "circuit_spice_test.csv" if limit is not None else "circuit_spice.csv")
    target_dir = os.path.dirname(out)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="circuit_spice_", suffix=".tmp", dir=target_dir)
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    print("leak min/median/max: %.3e / %.3e / %.3e A" % (df.spice_leak.min(), df.spice_leak.median(), df.spice_leak.max()))
    print("saved", out)
