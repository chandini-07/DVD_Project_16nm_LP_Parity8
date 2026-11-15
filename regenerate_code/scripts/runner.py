import os, re, sys, subprocess, tempfile, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from sampler import CASES, NOM, LMIN, draw_case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"   # mandatory line
PROC = ["vth0", "toxe", "ndep", "u0", "xj"]
NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
CARDS = {d: open(os.path.join(ROOT, "model", d + "_card.pm")).read() for d in ("nmos", "pmos")}

def make_card(dev, s):
    """copy of the model block with this sample's 5 process values"""
    card = CARDS[dev]
    for p in PROC:
        card, n = re.subn(r"(?<![A-Za-z0-9_])" + p + r"(\s*=\s*)" + NUM,
                          lambda m: p + m.group(1) + "%.8e" % s[p], card, count=1)
        assert n == 1, "parameter %s not found" % p
    return card

def netlist(case, s):
    dev = CASES[case][0]
    return "\n".join([
        "single %s, case %s" % (dev, case),          # first line = title (not a command!)
        OPTIONS,
        ".temp %.6f" % s["temp"],
        make_card(dev, s),
        "Mdut d g s b %s W=%.6e L=%.6e" % (dev, s["w"], LMIN),
        "Vd d 0 dc %.9f" % s["vd"],                  # every terminal has its own source
        "Vg g 0 dc %.9f" % s["vg"],
        "Vs s 0 dc %.9f" % s["vs"],
        "Vb b 0 dc %.9f" % s["vb"],
        ".op", ".control", "set numdgt=12", "run",
        "print i(Vd) i(Vg) i(Vs) i(Vb)", ".endc", ".end", ""])

def _format_context(case, s):
    corner = s.get("corner", "N/A") if isinstance(s, dict) else getattr(s, "corner", "N/A")
    vd = s.get("vd", "N/A") if isinstance(s, dict) else getattr(s, "vd", "N/A")
    vg = s.get("vg", "N/A") if isinstance(s, dict) else getattr(s, "vg", "N/A")
    vs = s.get("vs", "N/A") if isinstance(s, dict) else getattr(s, "vs", "N/A")
    vb = s.get("vb", "N/A") if isinstance(s, dict) else getattr(s, "vb", "N/A")
    temp = s.get("temp", "N/A") if isinstance(s, dict) else getattr(s, "temp", "N/A")
    return f"case={case}, corner={corner}, vd={vd}, vg={vg}, vs={vs}, vb={vb}, temp={temp}"

def run_one(args):
    case, s = args
    fd, path = tempfile.mkstemp(suffix=".cir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(netlist(case, s))
        res = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            err = res.stderr.strip() if res.stderr else "none"
            sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | ngspice rc={res.returncode} | stderr: {err}\n")
            return [np.nan] * 4
        out = res.stdout
        v = {k: float(x) for k, x in re.findall(r"(?m)^i\((v[dgsb])\)\s*=\s*(\S+)", out)}
        if len(v) != 4:
            err = res.stderr.strip() if res.stderr else "none"
            sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | missing terminal currents (found {len(v)} of 4: {list(v.keys())}) | stderr: {err}\n")
            return [np.nan] * 4
        I = np.array([-v["vd"], -v["vg"], -v["vs"], -v["vb"]], dtype=float)
        if not np.isfinite(I).all():
            sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | non-finite currents detected: {I.tolist()}\n")
            return [np.nan] * 4
        kcl_resid = abs(np.sum(I))
        max_b = np.max(np.abs(I))
        if kcl_resid > 1e-13 and kcl_resid > 1e-3 * max_b:
            sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | KCL residual violation: |sum|={kcl_resid:.3e} > tolerance (max_branch={max_b:.3e})\n")
            return [np.nan] * 4
        return I.tolist()
    except subprocess.TimeoutExpired as e:
        sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | timeout expired after {e.timeout}s\n")
        return [np.nan] * 4
    except Exception as e:
        sys.stderr.write(f"[runner.py error] {_format_context(case, s)} | exception: {e}\n")
        return [np.nan] * 4
    finally:
        if os.path.exists(path):
            os.remove(path)

def run_batch(case, df, nproc=None):
    rows = [(case, r) for r in df.to_dict("records")]
    with Pool(nproc or os.cpu_count()) as pool:
        return np.array(pool.map(run_one, rows, chunksize=20))   # shape (n, 4): id, ig, is, ib

if __name__ == "__main__":
    # test 1: nominal sample must reproduce your Step 4 NMOS-ON result (Id = +3.426092e-05)
    s = dict(vdd=1.0, temp=27.0, w=64e-9, vd=0.5, vs=0.0, vg=1.0, vb=0.0, **NOM["nmos"])
    print("nominal NMOS ON:", run_one(("nmos_on", s)))
    # test 2: 20 random samples of each case, timed
    rng = np.random.default_rng(42)
    for case in CASES:
        df = draw_case(case, 20, rng)
        t = time.time()
        I = run_batch(case, df)
        print("== %s: %d ok / 20, %.2f s" % (case, np.isfinite(I).all(axis=1).sum(), time.time() - t))
        res = np.abs(I.sum(axis=1)) / np.abs(I).max(axis=1)
        print(pd.DataFrame(I, columns=["id", "ig", "is", "ib"]).assign(rel_resid=res).head(5).to_string())
