import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from circuit import ROOT

cs = pd.read_csv(os.path.join(ROOT, "data", "circuit_ml.csv"))
fig, ax = plt.subplots(figsize=(6.2, 6), dpi=150)
lo, hi = cs.spice_leak.min() * 0.8, cs.spice_leak.max() * 1.25
x = np.array([lo, hi])
ax.fill_between(x, x * 0.9, x * 1.1, color="0.88", label="±10 %")
ax.fill_between(x, x * 0.95, x * 1.05, color="0.78", label="±5 %")
ax.plot(x, x, "k-", lw=1, label="ML = SPICE")
for p, col, name in ((0, "#1f77b4", "even parity (output 0)"), (1, "#d62728", "odd parity (output 1)")):
    d = cs[cs.parity == p]
    ax.scatter(d.spice_leak, d.ml_leak, s=2, alpha=0.25, color=col, label="%s, n=%d" % (name, len(d)), rasterized=True)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
e = (cs.ml_leak - cs.spice_leak).abs() / cs.spice_leak
ax.set_title("8-bit parity checker, 16nm LP: 256 vectors x 200 corners\nmean accuracy %.1f%%, median error %.1f%%, p95 %.1f%%" % (100 * (1 - e.mean()), 100 * e.median(), 100 * e.quantile(.95)), fontsize=10)
ax.legend(fontsize=8, loc="upper left", markerscale=4); ax.set_aspect("equal"); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = os.path.join(ROOT, "plots", "circuit_parity.png"); fig.savefig(out); print("saved", out)
