import sys, time
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from mlcommon import load_case, inv_symlog, metrics, SEED

def make_models():
    return {
        "poly_ridge": make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "grad_boost": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=SEED)),
        "mlp": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(128, 128, 64), early_stopping=True,
                                   max_iter=500, random_state=SEED),
            transformer=StandardScaler()),
    }

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "nmos_on"
    d = load_case(case)
    rows = []
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(d["Xtr"], d["Ttr"])
        Tp = model.predict(d["Xva"])
        m = metrics(d["Iva"], inv_symlog(Tp), d["Tva"], Tp)
        rows.append(dict(model=name, r2=m["r2"], median_err=m["median"], p95_err=m["p95"], seconds=time.time() - t0))
        print("done", name, flush=True)
    print("\nVALIDATION results for", case)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "%.4g" % x))
