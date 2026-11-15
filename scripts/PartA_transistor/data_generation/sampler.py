import numpy as np, pandas as pd

LMIN = 16e-9
VDD_NOM = 1.0

# nominal process values from the model file (your table from Step 2)
NOM = {
    "nmos": dict(vth0=0.68191, toxe=1.2e-9,  ndep=7e18,   u0=0.028,  xj=5e-9),
    "pmos": dict(vth0=-0.6862, toxe=1.22e-9, ndep=4.4e18, u0=0.0075, xj=7.2e-9),
}
# case -> (device, Vg as a fraction of VDD, Vb as a fraction of VDD)
CASES = {
    "nmos_on":  ("nmos", 1, 0),
    "nmos_off": ("nmos", 0, 0),
    "pmos_on":  ("pmos", 0, 1),
    "pmos_off": ("pmos", 1, 1),
}

def draw_case(case, n, rng):
    dev, vg_f, vb_f = CASES[case]
    vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n)      # 1) VDD first
    d = {
        "vdd":  vdd,
        "temp": rng.uniform(-40, 125, n),
        "w":    rng.uniform(1, 8, n) * LMIN,                # W = 1..8 x Lmin
        "vd":   rng.uniform(0, 1, n) * vdd,                 # Vd in 0..VDD
        "vs":   rng.uniform(0, 1, n) * vdd,                 # Vs in 0..VDD
        "vg":   vg_f * vdd,                                 # fixed by the case
        "vb":   vb_f * vdd,
    }
    for p, nom in NOM[dev].items():                         # process: nominal +-10 %
        d[p] = nom * (1 + rng.uniform(-0.1, 0.1, n))
    return pd.DataFrame(d)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for case in CASES:
        df = draw_case(case, 10000, rng)
        print("==", case)
        print(df.agg(["min", "max"]).T.to_string())
