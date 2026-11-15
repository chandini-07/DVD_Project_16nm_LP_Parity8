import os, numpy as np, pandas as pd
from predict import predict_currents
from gate_ml import TRANS
from gates import GATES, CASES18, WN, WP, ROOT
from sampler import LMIN

UNK = {"INV": ["out"], "NAND2": ["x1", "out"], "NOR2": ["y1", "out"],
       "AND2": ["x1", "n1", "out"], "OR2": ["y1", "n1", "out"]}

def logic(g, A, B):
    """ideal logic level (0/1) of every node, from Boolean logic only"""
    nand_o, nor_o = 1 - (A & B), 1 - (A | B)
    L = {"0": 0, "vdd": 1, "a": A, "b": B}
    L.update({"INV": {"out": 1 - A}, "NAND2": {"out": nand_o, "x1": 0}, "NOR2": {"out": nor_o, "y1": 1},
              "AND2": {"n1": nand_o, "x1": 0, "out": 1 - nand_o}, "OR2": {"n1": nor_o, "y1": 1, "out": 1 - nor_o}}[g])
    return L

def dev_I(tr, V, L, d):
    """KCL-consistent terminal currents (id, ig, is, ib) of one transistor, using voltages V"""
    t, nd, ng, ns, nb, k = tr
    on = (L[ng] == 1) if t == "n" else (L[ng] == 0)
    case = ("nmos_" if t == "n" else "pmos_") + ("on" if on else "off")
    pre = "n_" if t == "n" else "p_"
    w = (WN if t == "n" else WP)[k] * LMIN
    I = predict_currents(case, d["vdd"].to_numpy(), d["temp"].to_numpy(), w, V[nd], V[ns], V[ng], V[nb],
                         *[d[pre + p].to_numpy() for p in ("vth0", "toxe", "ndep", "u0", "xj")])
    Id, Ig, Is, Ib = I[:, 0], I[:, 1], I[:, 2], I[:, 3]
    if on:
        Ic, gb = (Id - Is) / 2, (Ig + Ib) / 2
        Id, Is = Ic - gb, -Ic - gb
    return Id, Ig, Is, Ib

def node_residual(node, V, L, d, trs):
    """net current flowing INTO this node from all transistor terminals attached to it (zero at the solution)"""
    F = np.zeros(len(d))
    for tr in trs:
        if node in tr[1:5]:
            I = dev_I(tr, V, L, d)
            for c, nn in enumerate(tr[1:5]):
                if nn == node:
                    F -= I[c]          # I is current into the transistor; node loses it
    return F

def solve_nodes(g, ab, d, sweeps=6, nbis=28):
    n_in = GATES[g]; A = (ab >> (n_in - 1)) & 1; B = ab & 1
    vdd = d["vdd"].to_numpy(); L = logic(g, A, B); trs = TRANS[g]
    V = {"0": 0 * vdd, "vdd": vdd, "a": A * vdd, "b": B * vdd}
    for k in UNK[g]:
        V[k] = L[k] * vdd.copy()                       # start from ideal logic levels
    for _ in range(sweeps):
        for node in UNK[g]:
            attached = [tr for tr in trs if node in tr[1:5]]
            lo, hi = 0 * vdd, vdd.copy()
            for _ in range(nbis):
                mid = (lo + hi) / 2
                V[node] = mid
                F = node_residual(node, V, L, d, attached)
                up = F > 0                              # more current in than out -> voltage must rise
                lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
            V[node] = (lo + hi) / 2
    return V, L

def leak_from_V(g, V, L, d):
    P = np.zeros(len(d)); vdd = d["vdd"].to_numpy()
    for tr in TRANS[g]:
        I = dev_I(tr, V, L, d)
        P += sum(V[nn] * I[c] for c, nn in enumerate(tr[1:5])) / vdd
    return P

if __name__ == "__main__":
    ml = pd.read_csv(os.path.join(ROOT, "data", "gate_ml.csv"))
    out = []
    for g, ab in CASES18:
        d = ml[(ml.gate == g) & (ml.ab == ab)].copy()
        V, L = solve_nodes(g, ab, d)
        for k in UNK[g]:
            d["ml_" + k] = V[k]
        d["ml_leak_mlV"] = leak_from_V(g, V, L, d)
        out.append(d); print(g, ab, "done", flush=True)
    res = pd.concat(out)
    res.to_csv(os.path.join(ROOT, "data", "gate_nodes.csv"), index=False)
    rows = []
    for (g, ab), d in res.groupby(["gate", "ab"], sort=False):
        r = {"gate": g, "ab": ab}
        for k in ("x1", "y1", "n1", "out"):
            sk = "spice_vout" if k == "out" else "spice_" + k
            if sk in d and d[sk].notna().all():
                dv = (d["ml_" + k] - d[sk]).abs() / d.vdd
                r[k + "_med_err_%VDD"] = 100 * dv.median(); r[k + "_p95_%VDD"] = 100 * dv.quantile(.95)
        e = (d.ml_leak_mlV - d.spice_leak).abs() / d.spice_leak
        r["leak_med_err_%"] = 100 * e.median(); r["leak_p95_%"] = 100 * e.quantile(.95)
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(ROOT, "results", "gate_node_results.csv"), index=False)
    pd.set_option("display.width", 250); print(t.to_string(index=False, float_format=lambda x: "%.3g" % x))
    e = (res.ml_leak_mlV - res.spice_leak).abs() / res.spice_leak
    print("leakage from ML-solved voltages: mean accuracy %.2f%%  median err %.2f%%  p95 %.2f%%" % (100 * (1 - e.mean()), 100 * e.median(), 100 * e.quantile(.95)))
