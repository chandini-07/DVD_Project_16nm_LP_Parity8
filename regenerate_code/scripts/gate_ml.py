import os, numpy as np, pandas as pd
from predict import predict_currents
from gates import GATES, CASES18, WN, WP, ROOT
from sampler import LMIN

# transistor lists: (type, drain, gate, source, body, width key) -- same connections as the netlists
def inv(i, a, o, k):
    return [("n", o, a, "0", "0", k), ("p", o, a, "vdd", "vdd", k)]
def nand(i, a, b, o, k):
    x = "x" + i
    return [("n", o, a, x, "0", k), ("n", x, b, "0", "0", k),
            ("p", o, a, "vdd", "vdd", k), ("p", o, b, "vdd", "vdd", k)]
def nor(i, a, b, o, k):
    y = "y" + i
    return [("p", y, a, "vdd", "vdd", k), ("p", o, b, y, "vdd", k),
            ("n", o, a, "0", "0", k), ("n", o, b, "0", "0", k)]
TRANS = {"INV": inv("1", "a", "out", "INV"),
         "NAND2": nand("1", "a", "b", "out", "NAND2"),
         "NOR2": nor("1", "a", "b", "out", "NOR2"),
         "AND2": nand("1", "a", "b", "n1", "NAND2") + inv("2", "n1", "out", "INV"),
         "OR2": nor("1", "a", "b", "n1", "NOR2") + inv("2", "n1", "out", "INV")}

def ml_gate_leak(g, ab, d):
    """d = SPICE rows for this gate/input (one per corner). Returns ML leakage (n,) in amps.
    Leakage = total power / VDD = sum over every transistor terminal of (terminal voltage x terminal current) / VDD.
    Nodes that are not sources cancel, so this equals the current sourced by all VDD-level sources,
    but ON devices (Vds ~ 0) contribute almost nothing, so their weak channel-current predictions do not matter.
    For an ON device we also force KCL (Id+Ig+Is+Ib = 0): channel current Ic = (Id-Is)/2, and the
    gate+body currents are shared equally between drain and source."""
    n_in = GATES[g]
    A = (ab >> (n_in - 1)) & 1
    B = ab & 1
    vdd = d["vdd"].to_numpy()
    V = {"0": 0 * vdd, "vdd": vdd, "a": A * vdd, "b": B * vdd, "out": d["spice_vout"].to_numpy()}
    for k in ("x1", "y1", "n1"):
        if "spice_" + k in d and d["spice_" + k].notna().all():
            V[k] = d["spice_" + k].to_numpy()
    P = np.zeros(len(d))
    for (t, nd, ng, ns, nb, k) in TRANS[g]:
        on = V[ng] > vdd / 2 if t == "n" else V[ng] < vdd / 2
        assert on.all() or (~on).all(), "gate state changes across corners?"
        on = bool(on.all())
        case = ("nmos_" if t == "n" else "pmos_") + ("on" if on else "off")
        pre = "n_" if t == "n" else "p_"
        w = (WN if t == "n" else WP)[k] * LMIN
        I = predict_currents(case, vdd, d["temp"].to_numpy(), w, V[nd], V[ns], V[ng], V[nb],
                             d[pre + "vth0"].to_numpy(), d[pre + "toxe"].to_numpy(), d[pre + "ndep"].to_numpy(),
                             d[pre + "u0"].to_numpy(), d[pre + "xj"].to_numpy())       # (n,4): id ig is ib
        Id, Ig, Is, Ib = I[:, 0], I[:, 1], I[:, 2], I[:, 3]
        if on:
            Ic, gb = (Id - Is) / 2, (Ig + Ib) / 2
            Id, Is = Ic - gb, -Ic - gb
        P += (V[nd] * Id + V[ng] * Ig + V[ns] * Is + V[nb] * Ib) / vdd
    return P

if __name__ == "__main__":
    sp = pd.read_csv(os.path.join(ROOT, "data", "gate_spice.csv"))
    cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
    df = sp.merge(cs.drop(columns=["vdd", "temp"]), on="corner")
    out = []
    for g, ab in CASES18:
        d = df[(df.gate == g) & (df.ab == ab)].copy()
        d["ml_leak"] = ml_gate_leak(g, ab, d)
        out.append(d)
    res = pd.concat(out)
    res["rel_err"] = (res.ml_leak - res.spice_leak) / res.spice_leak
    res.to_csv(os.path.join(ROOT, "data", "gate_ml.csv"), index=False)
    rows = []
    for (g, ab), d in res.groupby(["gate", "ab"], sort=False):
        e = d.rel_err.abs()
        rows.append({"gate": g, "ab": ab, "spice_median": d.spice_leak.median(), "ml_median": d.ml_leak.median(),
                     "bias_%": 100 * d.rel_err.median(), "median_err_%": 100 * e.median(), "p95_err_%": 100 * e.quantile(0.95),
                     "mean_acc_%": 100 * (1 - e.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ROOT, "results", "gate_results.csv"), index=False)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda x: "%.3g" % x))
    e = res.rel_err.abs()
    print("ALL 3600: mean accuracy %.2f%%  median err %.2f%%  p95 err %.2f%%  max err %.1f%%" % (100 * (1 - e.mean()), 100 * e.median(), 100 * e.quantile(0.95), 100 * e.max()))
