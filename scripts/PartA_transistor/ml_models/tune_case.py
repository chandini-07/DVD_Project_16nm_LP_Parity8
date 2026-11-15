import sys, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from mlcommon import load_case, inv_symlog, metrics, ROOT, SEED

def hgb(**kw):
    return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=SEED, **kw))
def mlp(**kw):
    return TransformedTargetRegressor(regressor=MLPRegressor(random_state=SEED, **kw), transformer=StandardScaler())

CONFIGS = {
    "grad_boost": {
        "gb_default":  lambda: hgb(max_iter=500),
        "gb_slow_15":  lambda: hgb(max_iter=2000, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-3),
        "gb_deep_63":  lambda: hgb(max_iter=1500, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1e-3),
    },
    "random_forest": {
        "rf_leaf1":    lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=SEED),
        "rf_half":     lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=SEED),
    },
    "mlp": {
        "mlp_adam_128x128x64": lambda: mlp(hidden_layer_sizes=(128, 128, 64), early_stopping=True, max_iter=500),
        "mlp_lbfgs_64x64":     lambda: mlp(hidden_layer_sizes=(64, 64), solver="lbfgs", max_iter=3000, alpha=1e-6),
        "mlp_lbfgs_128x128":   lambda: mlp(hidden_layer_sizes=(128, 128), solver="lbfgs", max_iter=3000, alpha=1e-6),
        "mlp_lbfgs_128x128x64": lambda: mlp(hidden_layer_sizes=(128, 128, 64), solver="lbfgs", max_iter=3000, alpha=1e-6),
    },
}

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "nmos_on"
    d = load_case(case)
    rows = []
    for family, cfgs in CONFIGS.items():
        for name, make in cfgs.items():
            t0 = time.time()
            model = make().fit(d["Xtr"], d["Ttr"])
            Tp = model.predict(d["Xva"])
            m = metrics(d["Iva"], inv_symlog(Tp), d["Tva"], Tp)
            rows.append(dict(family=family, config=name, r2=m["r2"], median_err=m["median"], p95_err=m["p95"], seconds=time.time() - t0))
            print("done", name, flush=True)
    res = pd.DataFrame(rows).sort_values("median_err")
    res.to_csv(ROOT + "/results/tune_%s.csv" % case, index=False)
    print("\nVALIDATION results for", case, "(best first)")
    print(res.to_string(index=False, float_format=lambda x: "%.4g" % x))
