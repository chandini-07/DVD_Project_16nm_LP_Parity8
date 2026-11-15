"""Deliverable 2 (test circuit): example netlist of the 8-bit parity checker (corner 0, input vector 181 = 10110101)."""
import os, pandas as pd
from circuit import circuit_netlist, ROOT

c = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv")).iloc[0].to_dict()
vec = 181
os.makedirs(os.path.join(ROOT, "netlists"), exist_ok=True)
path = os.path.join(ROOT, "netlists", "parity8_example.cir")
open(path, "w").write(circuit_netlist(vec, c))
print("wrote", path)
