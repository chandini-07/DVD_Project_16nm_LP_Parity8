"""predict_currents(): the one function the gate-level and test-circuit code call to use the four transistor models."""
import os, sys, joblib
import numpy as np, pandas as pd

# Ensure pert is importable across both canonical and regenerate layouts
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

from mlcommon import features, inv_symlog, RAW, ROOT

_CACHE = {}

def _find_model(case):
    candidates = [
        os.path.join(ROOT, "models", "PartA_transistor", f"final_{case}.joblib"),
        os.path.join(ROOT, "models", f"final_{case}.joblib"),
        os.path.join(ROOT, "..", "models", "PartA_transistor", f"final_{case}.joblib"),
        os.path.join(ROOT, "..", "models", f"final_{case}.joblib"),
        os.path.join(_this_dir, "..", "..", "models", "PartA_transistor", f"final_{case}.joblib"),
        os.path.join(_this_dir, "..", "models", f"final_{case}.joblib"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find model final_{case}.joblib in any of: {candidates}")

def _load(case):
    if case not in _CACHE:
        model_path = _find_model(case)
        _CACHE[case] = joblib.load(model_path)
    return _CACHE[case]

def predict_currents(case, vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj):
    """case: 'nmos_on' | 'nmos_off' | 'pmos_on' | 'pmos_off'.
    Inputs: volts, deg C, metres (w, toxe, xj), etc. Each may be a number or a numpy array.
    Returns [id, ig, is, ib] in amps (current INTO each terminal):
    shape (4,) for scalar inputs, shape (n, 4) for array inputs."""
    args = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in
                                 (vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj)])
    scalar = args[0].ndim == 0
    df = pd.DataFrame({k: np.atleast_1d(a) for k, a in zip(RAW, args)})
    m = _load(case)
    X = m["scaler"].transform(features(df).to_numpy())
    I = inv_symlog(m["model"].predict(X))
    return I[0] if scalar else I
