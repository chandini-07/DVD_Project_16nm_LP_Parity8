"""Deliverable 2 (gates): one example SPICE netlist per gate and input combination, at corner 0."""
import os, pandas as pd
from gates import gate_netlist, CASES18, ROOT

cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
c = cs.iloc[0].to_dict()
os.makedirs(os.path.join(ROOT, "netlists"), exist_ok=True)
for g, ab in CASES18:
    text, _ = gate_netlist(g, ab, c)
    path = os.path.join(ROOT, "netlists", "gate_%s_%d.cir" % (g.lower(), ab))
    open(path, "w").write(text)
print("wrote", len(CASES18), "gate netlists (corner 0) to netlists/")
