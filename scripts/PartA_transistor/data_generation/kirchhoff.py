import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
rows = []
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for c in CASES:
    df = pd.read_csv(os.path.join(ROOT, "data", c + ".csv"))
    assert len(df) == 10000 and not df.isna().any().any(), c        # size + no NaN
    I = df[["id", "ig", "is", "ib"]].to_numpy()
    res = np.abs(I.sum(axis=1))             # |Id + Ig + Is + Ib|   (should be ~0)
    big = np.abs(I).max(axis=1)             # largest of the 4 currents in that row
    rel = res / big
    rows.append(dict(case=c,
                     median_abs=np.median(res), max_abs=res.max(),
                     median_rel=np.median(rel), p99_rel=np.percentile(rel, 99), max_rel=rel.max(),
                     frac_rel_below_1e3=(rel < 1e-3).mean()))
    ax[0].hist(np.log10(rel + 1e-30), bins=60, histtype="step", label=c)
    ax[1].hist(np.log10(res + 1e-30), bins=60, histtype="step", label=c)
ax[0].set_xlabel("log10( |sum| / max|I| )   (relative residual)")
ax[1].set_xlabel("log10( |sum| )   (absolute residual, A)")
for a in ax:
    a.set_ylabel("samples"); a.legend(); a.grid(alpha=.3)
fig.suptitle("Kirchhoff check: Id + Ig + Is + Ib should be 0")
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "plots", "kirchhoff_check.png"), dpi=200)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results", "kirchhoff_check.csv"), index=False)
print(out.to_string(index=False, float_format=lambda x: "%.3e" % x))
