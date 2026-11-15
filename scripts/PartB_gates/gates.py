import os, re, subprocess, tempfile
import numpy as np, pandas as pd
from runner import make_card, OPTIONS, ROOT
from sampler import LMIN

# widths in units of Lmin (spec allows 1x..8x). Series stacks are made wider.
WN = {"INV": 2, "NAND2": 4, "NOR2": 2}      # NMOS width / Lmin
WP = {"INV": 4, "NAND2": 4, "NOR2": 8}      # PMOS width / Lmin
GATES = {"INV": 1, "NAND2": 2, "NOR2": 2, "AND2": 2, "OR2": 2}
CASES18 = [(g, ab) for g, n in GATES.items() for ab in range(2 ** n)]   # 2 + 4*4 = 18

def inv(i, a, o, w):
    return ["Mn%s %s %s 0 0 nmos W=%.6e L=%.6e" % (i, o, a, w["n"], LMIN),
            "Mp%s %s %s vdd vdd pmos W=%.6e L=%.6e" % (i, o, a, w["p"], LMIN)]

def nand(i, a, b, o, w):          # internal node x<i>
    x = "x" + i
    return ["Mn%sa %s %s %s 0 nmos W=%.6e L=%.6e" % (i, o, a, x, w["n"], LMIN),
            "Mn%sb %s %s 0 0 nmos W=%.6e L=%.6e" % (i, x, b, w["n"], LMIN),
            "Mp%sa %s %s vdd vdd pmos W=%.6e L=%.6e" % (i, o, a, w["p"], LMIN),
            "Mp%sb %s %s vdd vdd pmos W=%.6e L=%.6e" % (i, o, b, w["p"], LMIN)]

def nor(i, a, b, o, w):           # internal node y<i>
    y = "y" + i
    return ["Mp%sa %s %s vdd vdd pmos W=%.6e L=%.6e" % (i, y, a, w["p"], LMIN),
            "Mp%sb %s %s %s vdd pmos W=%.6e L=%.6e" % (i, o, b, y, w["p"], LMIN),
            "Mn%sa %s %s 0 0 nmos W=%.6e L=%.6e" % (i, o, a, w["n"], LMIN),
            "Mn%sb %s %s 0 0 nmos W=%.6e L=%.6e" % (i, o, b, w["n"], LMIN)]

def width(k):
    return {"n": WN[k] * LMIN, "p": WP[k] * LMIN}

def body(gate):
    """transistor lines + names of the internal nodes to record"""
    if gate == "INV":
        return inv("1", "a", "out", width("INV")), []
    if gate == "NAND2":
        return nand("1", "a", "b", "out", width("NAND2")), ["x1"]
    if gate == "NOR2":
        return nor("1", "a", "b", "out", width("NOR2")), ["y1"]
    if gate == "AND2":   # NAND2 -> INV
        return nand("1", "a", "b", "n1", width("NAND2")) + inv("2", "n1", "out", width("INV")), ["x1", "n1"]
    if gate == "OR2":    # NOR2 -> INV
        return nor("1", "a", "b", "n1", width("NOR2")) + inv("2", "n1", "out", width("INV")), ["y1", "n1"]

def gate_netlist(gate, ab, c, jitter=0.0):
    """c = one row of corners_200.csv; ab = input code (bit1 = A, bit0 = B)"""
    n = GATES[gate]
    A = (ab >> (n - 1)) & 1
    B = ab & 1
    lines, nodes = body(gate)
    vdd = c["vdd"]
    nm = {"vth0": c["n_vth0"], "toxe": c["n_toxe"], "ndep": c["n_ndep"], "u0": c["n_u0"], "xj": c["n_xj"]}
    pm = {"vth0": c["p_vth0"], "toxe": c["p_toxe"], "ndep": c["p_ndep"], "u0": c["p_u0"], "xj": c["p_xj"]}
    out = ["%s gate, inputs %s" % (gate, ab),          # first line = title
           OPTIONS, ".temp %.6f" % c["temp"],
           make_card("nmos", nm), make_card("pmos", pm),
           "Vdd vdd 0 dc %.9f" % vdd,
           "Va a 0 dc %.9f" % (A * vdd)]
    if n == 2:
        out.append("Vb b 0 dc %.9f" % (B * vdd))
    out += lines
    # starting guess = ideal logic levels (stops the solver landing on a wrong operating point)
    nand_o = 1 - (A & B); nor_o = 1 - (A | B)
    lv = {"INV": {"out": 1 - A}, "NAND2": {"out": nand_o}, "NOR2": {"out": nor_o},
          "AND2": {"n1": nand_o, "out": 1 - nand_o}, "OR2": {"n1": nor_o, "out": 1 - nor_o}}[gate]
    if "x1" in nodes: lv["x1"] = 0
    if "y1" in nodes: lv["y1"] = 1
    out.append(".nodeset " + " ".join("v(%s)=%.9f" % (k, (v + jitter) * vdd) for k, v in lv.items()))
    probes = ["i(Vdd)", "i(Va)", "i(Vb)" if n == 2 else "i(Va)", "v(out)"] + ["v(%s)" % x for x in nodes]
    out += [".op", ".control", "set numdgt=12", "run", "print " + " ".join(probes), ".endc", ".end", ""]
    return "\n".join(out), nodes

def run_gate_once(gate, ab, c, jitter):
    text, nodes = gate_netlist(gate, ab, c, jitter)
    n2 = GATES[gate] == 2
    A = (ab >> (GATES[gate] - 1)) & 1
    B = ab & 1 if n2 else 0
    fd, path = tempfile.mkstemp(suffix=".cir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        o = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True, timeout=60).stdout
        v = dict(re.findall(r"(?m)^(\S+)\s*=\s*(\S+)", o))
        # leakage = total current sourced by all VDD-level sources (VDD + any input held at 1)
        leak = -(float(v["i(vdd)"]) + A * float(v["i(va)"]) + (B * float(v["i(vb)"]) if n2 else 0.0))
        return {"leak": leak, "vout": float(v["v(out)"]), **{k: float(v["v(%s)" % k]) for k in nodes}}
    except Exception:
        return None
    finally:
        if os.path.exists(path):
            os.remove(path)

def valid(r, vdd):
    if r is None or r["leak"] <= 0:
        return False
    return all(-1e-5 <= x <= vdd + 1e-5 for k, x in r.items() if k != "leak")

def run_gate(gate, ab, c):
    """retry with a slightly different starting guess if the solver lands on a wrong operating point"""
    best = None
    for jit in (0.0, 0.001, 0.01, 0.003, 0.03, 0.1):
        r = run_gate_once(gate, ab, c, jit)
        if valid(r, c["vdd"]):
            return r
        best = best or r
    return best

if __name__ == "__main__":
    cs = pd.read_csv(os.path.join(ROOT, "data", "corners_200.csv"))
    c = cs.iloc[0].to_dict()
    print("corner 0: vdd=%.3f temp=%.1f" % (c["vdd"], c["temp"]))
    for g, ab in CASES18:
        print(g, ab, run_gate(g, ab, c))
