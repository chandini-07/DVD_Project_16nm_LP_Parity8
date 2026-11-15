"""Step 10: add rail-focused training samples and retrain the neural network for one case.
Usage: python3 scripts/retrain_case.py <case> [n_aug]"""
import sys, os, joblib, shutil, time
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sampler import draw_case
from runner import run_batch, ROOT
from mlcommon import load_case, features, symlog, inv_symlog, metrics, CASES, TARGETS, RAW

case = sys.argv[1]
N_AUG = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
ci = CASES.index(case)

# 1) rail-focused samples: Vd and Vs are each exactly 0 (40 %), exactly VDD (40 %) or uniform (20 %)
aug_path = "%s/data/aug_%s.csv" % (ROOT, case)
if not os.path.exists(aug_path):
    t0 = time.time()
    rng = np.random.default_rng([2024, ci])
    df = draw_case(case, N_AUG, rng)
    for k in ("vd", "vs"):
        u = rng.random(N_AUG); v = df[k].to_numpy().copy()
        v[u < 0.4] = 0.0
        m = (u >= 0.4) & (u < 0.8); v[m] = df["vdd"].to_numpy()[m]
        df[k] = v
    I = run_batch(case, df)
    ok = np.isfinite(I).all(axis=1)
    df = df[ok].copy(); df[TARGETS] = I[ok]
    df.to_csv(aug_path, index=False, float_format="%.10e")
    print("%s: simulated %d rail-focused samples (%d failed) in %.0fs" % (case, N_AUG, int((~ok).sum()), time.time() - t0), flush=True)
aug = pd.read_csv(aug_path)

# 2) train on original training rows + augmented rows (validation / test sets are unchanged)
d = load_case(case)
Xa = d["scaler"].transform(features(aug[RAW]).to_numpy()); Ta = symlog(aug[TARGETS].to_numpy())
X, T = np.r_[d["Xtr"], Xa], np.r_[d["Ttr"], Ta]
t0 = time.time()
new = TransformedTargetRegressor(
    regressor=MLPRegressor(hidden_layer_sizes=(128, 128, 64), solver="lbfgs", max_iter=3000, alpha=1e-6, random_state=42),
    transformer=StandardScaler()).fit(X, T)
print("%s: trained on %d rows in %.0fs" % (case, len(X), time.time() - t0), flush=True)

# 3) compare old (uniform-only) and new model on the uniform test set and on the rail test set
old = joblib.load("%s/models/%s__mlp.joblib" % (ROOT, case))["model"]
rt = pd.read_csv("%s/data/rail_test_%s.csv" % (ROOT, case))
Xr = d["scaler"].transform(features(rt[RAW]).to_numpy()); Ir = rt[TARGETS].to_numpy()
rows = []
for mname, model in (("uniform-trained", old), ("rail-augmented", new)):
    for tname, Xe, Ie in (("uniform test", d["Xte"], d["Ite"]), ("rail test", Xr, Ir)):
        Tp = model.predict(Xe)
        m = metrics(Ie, inv_symlog(Tp), symlog(Ie), Tp)
        row = dict(case=case, model=mname, test=tname, r2=m["r2"], median_err=m["median"], p95_err=m["p95"])
        for k in TARGETS:
            row[k + "_median"] = m[k + "_median"]
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv("%s/results/retrain_%s.csv" % (ROOT, case), index=False)
print(res.to_string(index=False, float_format=lambda x: "%.4g" % x))

# 4) the rail-augmented model becomes the final model used by predict_currents()
joblib.dump(dict(model=new, scaler=d["scaler"]), "%s/models/aug_%s__mlp.joblib" % (ROOT, case))
shutil.copy("%s/models/aug_%s__mlp.joblib" % (ROOT, case), "%s/models/final_%s.joblib" % (ROOT, case))
print("saved models/final_%s.joblib (rail-augmented)" % case)
