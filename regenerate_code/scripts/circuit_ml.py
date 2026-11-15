"""Part C ML prediction: sum of the Part B NAND2 gate-model leakages over the 28 gates of the parity checker.
Reads data/gate_nodes.csv (Part B) and data/circuit_spice.csv. Writes data/circuit_ml.csv, results/circuit_results.csv."""
import os, sys
import numpy as np, pandas as pd
from circuit import GL, logic_levels, ROOT

spice_file = "circuit_spice_test.csv" if "--test" in sys.argv else "circuit_spice.csv"
cs = pd.read_csv(os.path.join(ROOT, "data", spice_file))
gn = pd.read_csv(os.path.join(ROOT, "data", "gate_nodes.csv"))
nd = gn[gn.gate == "NAND2"]
def table(col):                      # array [ab, corner] of NAND2 leakage
    T = np.zeros((4, 200))
    for ab in range(4):
        d = nd[nd.ab == ab].sort_values("corner")
        T[ab] = d[col].to_numpy()
    return T
T_ml, T_mlsp, T_sp = table("ml_leak_mlV"), table("ml_leak"), table("spice_leak")

# state of every NAND2 (ab = 2*A + B, same convention as Part B) for every input vector
STATE = np.zeros((256, len(GL)), dtype=int)
for v in range(256):
    L = logic_levels(v)
    for j, (idx, a, b, o) in enumerate(GL):
        STATE[v, j] = 2 * L[a] + L[b]

vec, k = cs.vec.to_numpy(), cs.corner.to_numpy()
def total(T):
    return np.array([T[STATE[v], c].sum() for v, c in zip(vec, k)])
cs["ml_leak"] = total(T_ml)            # fully ML gate models (ML-solved node voltages)
cs["ml_leak_gatemodel"] = total(T_mlsp)  # Part B gate model fed with SPICE node voltages of the standalone gate
cs["sum_gate_spice"] = total(T_sp)     # sum of the standalone gate SPICE leakages (no ML): isolates the composition error
cs.to_csv(os.path.join(ROOT, "data", "circuit_ml.csv"), index=False)

def stats(d, pred, name):
    e = (d[pred] - d.spice_leak) / d.spice_leak
    return {"set": name, "n": len(d), "spice_median": d.spice_leak.median(), "ml_median": d[pred].median(),
            "bias_%": 100 * e.median(), "median_err_%": 100 * e.abs().median(), "p95_err_%": 100 * e.abs().quantile(.95),
            "mean_acc_%": 100 * (1 - e.abs().mean())}
rows = [stats(cs, "ml_leak", "ALL")]
rows += [stats(cs[cs.parity == p], "ml_leak", "parity %d" % p) for p in (0, 1)]
cs["hw"] = [bin(v).count("1") for v in cs.vec]
rows += [stats(cs[cs.hw == h], "ml_leak", "ones=%d" % h) for h in range(9)]
t = pd.DataFrame(rows)
t.to_csv(os.path.join(ROOT, "results", "circuit_results.csv" if "--test" not in sys.argv else "circuit_results_test.csv"), index=False)
pd.set_option("display.width", 200); print(t.to_string(index=False, float_format=lambda x: "%.3g" % x))
print()
for pred, name in (("ml_leak", "fully-ML composition"), ("ml_leak_gatemodel", "gate model w/ SPICE node voltages"), ("sum_gate_spice", "sum of standalone-gate SPICE (no ML)")):
    e = ((cs[pred] - cs.spice_leak) / cs.spice_leak).abs()
    print("%-40s mean acc %.2f%%  median err %.2f%%  p95 %.2f%%  max %.1f%%" % (name, 100 * (1 - e.mean()), 100 * e.median(), 100 * e.quantile(.95), 100 * e.max()))
