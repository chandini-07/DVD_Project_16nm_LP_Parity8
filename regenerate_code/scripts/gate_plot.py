import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gates import ROOT

res = pd.read_csv(os.path.join(ROOT, "data", "gate_ml.csv"))
colors = {"INV": "#1f77b4", "NAND2": "#ff7f0e", "NOR2": "#2ca02c", "AND2": "#d62728", "OR2": "#9467bd"}
fig, ax = plt.subplots(figsize=(6.2, 6), dpi=150)
lo, hi = res.spice_leak.min() * 0.7, res.spice_leak.max() * 1.4
x = np.array([lo, hi])
ax.fill_between(x, x * 0.9, x * 1.1, color="0.85", label="±10 %")
ax.plot(x, x, "k-", lw=1, label="ML = SPICE")
for g, d in res.groupby("gate"):
    ax.scatter(d.spice_leak, d.ml_leak, s=6, alpha=0.5, color=colors[g], label="%s (%d)" % (g, len(d)))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
e = (res.ml_leak - res.spice_leak).abs() / res.spice_leak
ax.set_title("Gate leakage, 16nm LP, 18 cases x 200 corners\nmean accuracy %.1f%%, median error %.1f%%, p95 %.1f%%" % (100 * (1 - e.mean()), 100 * e.median(), 100 * e.quantile(.95)), fontsize=10)
ax.legend(fontsize=8, loc="upper left"); ax.set_aspect("equal"); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = os.path.join(ROOT, "plots", "gate_parity.png"); fig.savefig(out); print("saved", out)
