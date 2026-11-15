import os, shutil, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlcommon import load_case, inv_symlog, metrics, CASES, TARGETS, ROOT

CHOSEN = {c: "mlp" for c in CASES}        # best on validation (lowest 95th-percentile error, R2 as tiebreaker)
os.makedirs(ROOT + "/plots", exist_ok=True)
rows = []
for case in CASES:
    path = "%s/models/final_%s.joblib" % (ROOT, case)
    if not os.path.exists(path):
        print("skipping", case, "(model file not found)"); continue
    d = load_case(case)
    model = joblib.load(path)["model"]
    Tp = model.predict(d["Xte"])               # TEST set: used only here, once
    Ip = inv_symlog(Tp)
    m = metrics(d["Ite"], Ip, d["Tte"], Tp)
    row = dict(case=case, model=CHOSEN[case], r2=m["r2"], median_err=m["median"], p95_err=m["p95"])
    for k in TARGETS:
        row[k + "_median"], row[k + "_p95"] = m[k + "_median"], m[k + "_p95"]
    rows.append(row)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for j, k in enumerate(TARGETS):
        a = ax[j]
        a.scatter(d["Ite"][:, j], Ip[:, j], s=4, alpha=.4)
        lim = np.abs(np.r_[d["Ite"][:, j], Ip[:, j]]).max() * 2
        a.plot([-lim, lim], [-lim, lim], "r-", lw=.8)
        a.set_xscale("symlog", linthresh=1e-15); a.set_yscale("symlog", linthresh=1e-15)
        a.set_title("%s   median err %.1f%%" % (k, 100 * m[k + "_median"]), fontsize=10)
        a.set_xlabel("SPICE (A)"); a.set_ylabel("ML prediction (A)"); a.grid(alpha=.3)
    fig.suptitle("%s: predicted vs SPICE on the test set (%s)" % (case, CHOSEN[case]))
    fig.tight_layout(); fig.savefig("%s/plots/pred_vs_spice_%s.png" % (ROOT, case), dpi=200); plt.close(fig)
out = pd.DataFrame(rows)
out.to_csv(ROOT + "/results/test_chosen.csv", index=False)
print(out[["case", "model", "r2", "median_err", "p95_err"]].to_string(index=False, float_format=lambda x: "%.4g" % x))
print()
print(out[["case"] + [k + "_median" for k in TARGETS] + [k + "_p95" for k in TARGETS]].to_string(index=False, float_format=lambda x: "%.3g" % x))
