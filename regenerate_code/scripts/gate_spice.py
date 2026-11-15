import os, sys, tempfile, numpy as np, pandas as pd
from gates import run_gate, CASES18, ROOT

NODES = ("leak", "vout", "x1", "y1", "n1")

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
    cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
    limit = parse_limit(sys.argv)
    
    # Generate list of jobs
    jobs = [(gate, ab, c) for gate, ab in CASES18 for c in cs.to_dict("records")]
    if limit is not None:
        jobs = jobs[:limit]
    
    expected_rows = len(jobs)
    rows = []
    failures = []
    
    for gate, ab, c in jobs:
        r = run_gate(gate, ab, c)
        if r is None:
            failures.append(f"Gate {gate} inputs={ab} corner={c['corner']}: simulation or validation failed")
            r = {}
        row = {"gate": gate, "ab": ab, "corner": int(c["corner"]), "vdd": c["vdd"], "temp": c["temp"]}
        row.update({"spice_" + n: r.get(n, np.nan) for n in NODES})
        rows.append(row)
        
    df = pd.DataFrame(rows)
    n_fail = int(df["spice_leak"].isna().sum())
    n_nonpos = int((df["spice_leak"] <= 0).sum())
    print(f"{len(df)} rows; failures: {n_fail}; non-positive leak: {n_nonpos}")
    
    if failures or n_fail > 0 or n_nonpos > 0 or len(df) != expected_rows:
        log_path = os.path.join(ROOT, "results", "gate_spice_failures.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Gate SPICE batch failed: failures={n_fail}, nonpos={n_nonpos}, rows={len(df)}, expected={expected_rows}\n")
            f.write("\n".join(failures) + "\n")
        print(f"Error: gate_spice failed. Log written to {log_path}", file=sys.stderr)
        sys.exit(1)
        
    out_csv = os.path.join(ROOT, "data", "gate_spice_test.csv" if limit is not None else "gate_spice.csv")
    target_dir = os.path.dirname(out_csv)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="gate_spice_", suffix=".tmp", dir=target_dir)
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_csv)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    print(df.groupby(["gate", "ab"])["spice_leak"].agg(["min", "median", "max"]).to_string())
    print("saved", out_csv)
