import os, numpy as np, pandas as pd
from gates import run_gate, CASES18, ROOT

NODES = ("leak", "vout", "x1", "y1", "n1")

if __name__ == "__main__":
    cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
    rows = []
    for gate, ab in CASES18:
        for c in cs.to_dict("records"):
            r = run_gate(gate, ab, c) or {}
            row = {"gate": gate, "ab": ab, "corner": int(c["corner"]), "vdd": c["vdd"], "temp": c["temp"]}
            row.update({"spice_" + n: r.get(n, np.nan) for n in NODES})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "data", "gate_spice.csv"), index=False)
    print(len(df), "rows; failures:", int(df["spice_leak"].isna().sum()), "; non-positive leak:", int((df["spice_leak"] <= 0).sum()))
    print(df.groupby(["gate", "ab"])["spice_leak"].agg(["min", "median", "max"]).to_string())
