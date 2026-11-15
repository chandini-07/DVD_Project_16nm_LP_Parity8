import os
import numpy as np, pandas as pd
from sampler import NOM
from runner import ROOT

N_CORNERS = 200
rng = np.random.default_rng([42, 999])        # fixed seed, separate from the datasets
d = {"corner": np.arange(N_CORNERS),
     "vdd":  rng.uniform(0.9, 1.1, N_CORNERS),
     "temp": rng.uniform(-40, 125, N_CORNERS)}
for dev, prefix in (("nmos", "n_"), ("pmos", "p_")):          # NMOS draw, then PMOS draw
    for p, nom in NOM[dev].items():
        d[prefix + p] = nom * (1 + rng.uniform(-0.1, 0.1, N_CORNERS))
df = pd.DataFrame(d)
df.to_csv(os.path.join(ROOT, "data", "corners_200.csv"), index=False, float_format="%.10e")
print(df.agg(["min", "max"]).T.to_string())
print("rows:", len(df))
