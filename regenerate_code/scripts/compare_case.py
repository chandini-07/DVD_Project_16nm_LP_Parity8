import sys, time, joblib
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from mlcommon import load_case, inv_symlog, metrics, ROOT, SEED

def final_models():
    """the tuned setting of each model family (tuned on the nmos_on validation set, Step 5)"""
    return {
        "poly_ridge": make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=1, n_jobs=1, random_state=SEED),
        "grad_boost": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=2000, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-3, random_state=SEED)),
        "mlp": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(128, 128, 64), solver="lbfgs", max_iter=3000,
                                   alpha=1e-6, random_state=SEED),
            transformer=StandardScaler()),
    }

if __name__ == "__main__":
    case = sys.argv[1]
    d = load_case(case)
    rows = []
    for name, model in final_models().items():
        t0 = time.time()
        model.fit(d["Xtr"], d["Ttr"])
        Tp = model.predict(d["Xva"])
        m = metrics(d["Iva"], inv_symlog(Tp), d["Tva"], Tp)
        row = dict(case=case, model=name, r2=m["r2"], median_err=m["median"], p95_err=m["p95"], seconds=time.time() - t0)
        for k in ("id", "ig", "is", "ib"):
            row[k + "_median"], row[k + "_p95"] = m[k + "_median"], m[k + "_p95"]
        rows.append(row)
        if name != "random_forest":     # the forest is huge on disk; it is quick to refit if it ever wins
            joblib.dump(dict(model=model, scaler=d["scaler"]), "%s/models/%s__%s.joblib" % (ROOT, case, name))
        print("done", case, name, "%.0fs" % row["seconds"], flush=True)
    pd.DataFrame(rows).to_csv("%s/results/compare_%s.csv" % (ROOT, case), index=False)
    print("finished", case)
