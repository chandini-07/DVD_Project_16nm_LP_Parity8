import os
from sampler import CASES, NOM, LMIN
from runner import netlist, ROOT

BIAS = {"nmos_on": (0.5, 0.0), "nmos_off": (1.0, 0.0),     # (Vd, Vs)
        "pmos_on": (0.5, 1.0), "pmos_off": (0.0, 1.0)}
for case, (dev, vg_f, vb_f) in CASES.items():
    vd, vs = BIAS[case]
    s = dict(vdd=1.0, temp=27.0, w=4 * LMIN, vd=vd, vs=vs, vg=vg_f, vb=vb_f, **NOM[dev])
    open(os.path.join(ROOT, "netlists", "fet_%s.cir" % case), "w").write(netlist(case, s))
