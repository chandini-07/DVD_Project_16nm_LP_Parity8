import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Ensure pert is importable across layouts
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _d in [_this_dir,
           os.path.join(_this_dir, "..", "scripts"),
           os.path.join(_this_dir, "..", "..", "regenerate_code", "scripts"),
           os.path.join(_this_dir, "..", "..", "scripts", "PartA_transistor", "ml_models"),
           os.path.join(_this_dir, "..")]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))
try:
    import pert
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
RAW = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]
SEED = 42
C0 = 1e-15      # symlog scale in amps (= the solver noise level)
FLOOR = 1e-15   # relative error uses max(|true|, FLOOR) so near-zero currents don't blow it up

def features(df):
    """12 raw inputs -> 18 model inputs (adds physical hints: 1/T, log W, voltage differences)."""
    X = df[RAW].copy()
    X["logw"] = np.log10(df["w"])
    X["invT"] = 1000.0 / (df["temp"] + 273.15)
    X["vds"] = df["vd"] - df["vs"]
    X["vgs"] = df["vg"] - df["vs"]
    X["vgd"] = df["vg"] - df["vd"]
    X["vbs"] = df["vb"] - df["vs"]
    X["vbd"] = df["vb"] - df["vd"]
    return X.drop(columns=["w"])

def symlog(I):
    """signed log: keeps the sign, compresses 1e-17..1e-4 A into about 0..11"""
    return np.sign(I) * np.log10(1.0 + np.abs(I) / C0)

def inv_symlog(t):
    return np.sign(t) * C0 * (10.0 ** np.abs(t) - 1.0)

def load_case(case):
    """read one CSV, split 70/15/15 with a fixed seed, scale the inputs (scaler fitted on train only)"""
    candidates = [
        os.path.join(ROOT, "data", "PartA_transistor", case + ".csv"),
        os.path.join(ROOT, "data", case + ".csv"),
        os.path.join(ROOT, "..", "data", "PartA_transistor", case + ".csv"),
        os.path.join(ROOT, "..", "data", case + ".csv"),
        os.path.join(_this_dir, "..", "..", "data", "PartA_transistor", case + ".csv"),
    ]
    data_path = None
    for p in candidates:
        if os.path.isfile(p):
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError(f"Could not find {case}.csv in: {candidates}")
    df = pd.read_csv(data_path)
    X, I = features(df).to_numpy(), df[TARGETS].to_numpy()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED)
    sc = StandardScaler().fit(X[tr])
    d = dict(scaler=sc, n_features=X.shape[1])
    for name, ix in (("tr", tr), ("va", va), ("te", te)):
        d["X" + name] = sc.transform(X[ix])   # scaled inputs
        d["I" + name] = I[ix]                 # true currents (amps)
        d["T" + name] = symlog(I[ix])         # transformed targets: what the models learn
    return d

def metrics(I_true, I_pred, T_true, T_pred):
    """R2 on the transformed targets; median / 95th-percentile relative error on the currents in amps"""
    rel = np.abs(I_pred - I_true) / np.maximum(np.abs(I_true), FLOOR)
    out = dict(r2=r2_score(T_true, T_pred), median=np.median(rel), p95=np.percentile(rel, 95))
    for j, k in enumerate(TARGETS):
        out[k + "_median"], out[k + "_p95"] = np.median(rel[:, j]), np.percentile(rel[:, j], 95)
    return out

if __name__ == "__main__":
    d = load_case("nmos_off")
    print("sizes train/val/test:", len(d["Xtr"]), len(d["Xva"]), len(d["Xte"]), " features:", d["n_features"])
    print("scaled train mean ~0, std ~1:", d["Xtr"].mean().round(3), d["Xtr"].std().round(3))
    I = d["Itr"]
    back = inv_symlog(symlog(I))
    print("symlog round-trip max relative error:", np.max(np.abs(back - I) / np.maximum(np.abs(I), 1e-30)))
    print("transformed target range per current:", d["Ttr"].min(axis=0).round(2), d["Ttr"].max(axis=0).round(2))
    print("perfect predictor ->", {k: round(v, 3) for k, v in metrics(I, I, d["Ttr"], d["Ttr"]).items() if k in ("r2", "median", "p95")})
