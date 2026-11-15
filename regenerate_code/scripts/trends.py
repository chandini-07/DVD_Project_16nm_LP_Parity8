import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sampler import CASES, NOM, LMIN
from runner import run_one, ROOT

def base(case, **kw):
    dev, vg_f, vb_f = CASES[case]
    on, vdd = case.endswith("_on"), 1.0
    if dev == "nmos": vd, vs = (0.0, 0.0) if on else (vdd, 0.0)
    else:             vd, vs = (vdd, vdd) if on else (0.0, vdd)
    s = dict(vdd=vdd, temp=27.0, w=4 * LMIN, vd=vd, vs=vs, vg=vg_f * vdd, vb=vb_f * vdd, **NOM[dev])
    s.update(kw)
    return s

def cur(case, k, **kw):                       # k: 0=id, 1=ig
    return abs(run_one((case, base(case, **kw)))[k])

def plot(fname, title, xlabel, ylabel, curves):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, x, y, color in curves:
        ax.plot(x, y, "-o", ms=3, label=label, color=color)
        print("%-30s first=%.3e last=%.3e ratio=%.1f" % (label, y[0], y[-1], y[-1] / y[0]))
    ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "plots", fname), dpi=200); plt.close(fig)

os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
col = {"nmos": "C0", "pmos": "C3"}
fac = np.linspace(0.9, 1.1, 21)

# 1) OFF current vs temperature
T = np.arange(-40, 126, 5.0)
plot("trend1_off_current_vs_temperature.png", "OFF current vs temperature", "Temperature (C)", "|Id| (A)",
     [(c + " |Id|", T, [cur(c, 0, temp=t) for t in T], col[c[:4]]) for c in ("nmos_off", "pmos_off")])

# 2) OFF current vs vth0
curves = []
for c in ("nmos_off", "pmos_off"):
    nom = NOM[CASES[c][0]]["vth0"]
    curves.append((c + " |Id|", abs(nom) * fac, [cur(c, 0, vth0=nom * f) for f in fac], col[c[:4]]))
plot("trend2_off_current_vs_vth0.png", "OFF current vs |vth0|", "|vth0| (V)", "|Id| (A)", curves)

# 3) gate current vs toxe (ON state)
curves = []
for c in ("nmos_on", "pmos_on"):
    nom = NOM[CASES[c][0]]["toxe"]
    curves.append((c + " |Ig|", nom * fac * 1e9, [cur(c, 1, toxe=nom * f) for f in fac], col[c[:4]]))
plot("trend3_gate_current_vs_toxe.png", "Gate current vs toxe", "toxe (nm)", "|Ig| (A)", curves)
