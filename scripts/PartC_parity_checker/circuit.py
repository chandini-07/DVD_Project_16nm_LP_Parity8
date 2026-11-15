import os, re, subprocess, tempfile
import numpy as np, pandas as pd
from runner import make_card, OPTIONS, ROOT
from gates import nand, WN, WP
from sampler import LMIN

# 8-bit parity checker = XOR tree: 7 XOR2, each XOR2 = 4 NAND2  ->  28 NAND2 gates (the NAND2 of Part B, same widths)
def build_gates():
    gl, cnt = [], [0]
    def NAND(a, b):
        cnt[0] += 1
        o = "n%d" % cnt[0]
        gl.append((cnt[0], a, b, o))
        return o
    def XOR(a, b):
        m = NAND(a, b); p = NAND(a, m); q = NAND(b, m)
        return NAND(p, q)
    l1 = [XOR("i%d" % (2 * k), "i%d" % (2 * k + 1)) for k in range(4)]
    l2 = [XOR(l1[0], l1[1]), XOR(l1[2], l1[3])]
    out = XOR(l2[0], l2[1])
    return gl, out
GL, OUT = build_gates()          # list of (index, inA, inB, outnode); OUT = node holding the parity

def logic_levels(vec):
    """Boolean value of every node for input vector vec (0..255, bit k = input i<k>)"""
    L = {"i%d" % k: (vec >> k) & 1 for k in range(8)}
    for idx, a, b, o in GL:
        L[o] = 1 - (L[a] & L[b])
    return L

def circuit_netlist(vec, c, jitter=0.0):
    vdd = c["vdd"]
    nm = {"vth0": c["n_vth0"], "toxe": c["n_toxe"], "ndep": c["n_ndep"], "u0": c["n_u0"], "xj": c["n_xj"]}
    pm = {"vth0": c["p_vth0"], "toxe": c["p_toxe"], "ndep": c["p_ndep"], "u0": c["p_u0"], "xj": c["p_xj"]}
    L = logic_levels(vec)
    out = ["8-bit parity checker, input vector %d" % vec,       # first line = title
           OPTIONS, ".temp %.6f" % c["temp"], make_card("nmos", nm), make_card("pmos", pm),
           "Vdd vdd 0 dc %.9f" % vdd]
    for k in range(8):
        out.append("V%d i%d 0 dc %.9f" % (k, k, L["i%d" % k] * vdd))
    for idx, a, b, o in GL:
        out += nand("g%d" % idx, a, b, o, {"n": WN["NAND2"] * LMIN, "p": WP["NAND2"] * LMIN})
    ns = ["v(%s)=%.9f" % (o, (L[o] + jitter) * vdd) for _, _, _, o in GL]
    ns += ["v(xg%d)=0" % idx for idx, _, _, _ in GL]
    out.append(".nodeset " + " ".join(ns))
    probes = ["i(Vdd)"] + ["i(V%d)" % k for k in range(8)] + ["v(%s)" % o for _, _, _, o in GL]
    out += [".op", ".control", "set numdgt=12", "run", "print " + " ".join(probes), ".endc", ".end", ""]
    return "\n".join(out)

def run_once(vec, c, jitter):
    fd, path = tempfile.mkstemp(suffix=".cir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(circuit_netlist(vec, c, jitter))
        o = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True, timeout=120).stdout
        v = dict(re.findall(r"(?m)^(\S+)\s*=\s*(\S+)", o))
        L = logic_levels(vec)
        leak = -(float(v["i(vdd)"]) + sum(L["i%d" % k] * float(v["i(v%d)" % k]) for k in range(8)))
        nodes = {o_: float(v["v(%s)" % o_]) for _, _, _, o_ in GL}
        return {"leak": leak, "vpar": nodes[OUT], "nodes": nodes}
    except Exception:
        return None
    finally:
        if os.path.exists(path):
            os.remove(path)

def valid(r, c, vec):
    if r is None or r["leak"] <= 0:
        return False
    L, vdd = logic_levels(vec), c["vdd"]
    # every gate output must sit within 5 % of VDD of its ideal logic level
    return all(abs(x - L[k] * vdd) < 0.05 * vdd for k, x in r["nodes"].items())

def run_circuit(vec, c):
    """retry with a slightly different starting guess if the solver lands on a wrong operating point"""
    best = None
    for jit in (0.0, 0.001, 0.01, 0.003, 0.03, 0.1):
        r = run_once(vec, c, jit)
        if valid(r, c, vec):
            return r
        best = best or r
    return best

if __name__ == "__main__":
    import time
    print("gates:", len(GL), "NAND2; parity output node:", OUT)
    bad = [v for v in range(256) if logic_levels(v)[OUT] != bin(v).count("1") % 2]
    print("Boolean model vs popcount parity: %d mismatches out of 256" % len(bad))
    cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
    c = cs.iloc[0].to_dict()
    print("corner 0: vdd=%.3f temp=%.1f" % (c["vdd"], c["temp"]))
    for vec in (0, 1, 7, 255):
        t = time.time(); r = run_circuit(vec, c)
        print("vec %3d  parity=%d  v(par)=%.4f V  leak=%.4e A  (%.2f s)" % (vec, bin(vec).count("1") % 2, r["vpar"], r["leak"], time.time() - t))
