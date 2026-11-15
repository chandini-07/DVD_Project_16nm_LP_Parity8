import os, time
import numpy as np
from predict import predict_currents
from mlcommon import ROOT

# the 4 example netlists of Step 4 (nominal process, 1 V, 27 C, W = 64 nm) and their SPICE results [id, ig, is, ib]
NOM_N = dict(vth0=0.68191, toxe=1.2e-9, ndep=7e18, u0=0.028, xj=5e-9)
NOM_P = dict(vth0=-0.6862, toxe=1.22e-9, ndep=4.4e18, u0=0.0075, xj=7.2e-9)
EX = {
    "nmos_on":  (NOM_N, dict(vd=0.5, vs=0.0, vg=1.0, vb=0.0), [3.426092e-05, 6.616356e-10, -3.426158e-05, -7.272301e-13]),
    "nmos_off": (NOM_N, dict(vd=1.0, vs=0.0, vg=0.0, vb=0.0), [5.017788e-12, -9.421935e-31, -5.007787e-12, -1.000149e-14]),
    "pmos_on":  (NOM_P, dict(vd=0.5, vs=1.0, vg=0.0, vb=1.0), [-1.644838e-05, -2.024046e-10, 1.644858e-05, 9.188483e-13]),
    "pmos_off": (NOM_P, dict(vd=0.0, vs=1.0, vg=1.0, vb=1.0), [-1.586148e-11, 7.012169e-12, 8.839449e-12, 9.964252e-15]),
}
for case, (proc, bias, spice) in EX.items():
    if not os.path.exists("%s/models/final_%s.joblib" % (ROOT, case)):
        print(case, ": no model file, skipped"); continue
    ml = predict_currents(case, vdd=1.0, temp=27.0, w=64e-9, **bias, **proc)
    print("==", case)
    for k, s, p in zip(["id", "ig", "is", "ib"], spice, ml):
        print("  %s  SPICE %+.4e   ML %+.4e   rel.err %.1f%%" % (k, s, p, 100 * abs(p - s) / max(abs(s), 1e-15)))
# speed test: array input
n = 50000
case = [c for c in EX if os.path.exists("%s/models/final_%s.joblib" % (ROOT, c))][0]
proc, bias, _ = EX[case]
t = time.time()
out = predict_currents(case, np.full(n, 1.0), np.full(n, 27.0), np.full(n, 64e-9), np.full(n, bias["vd"]),
                       np.full(n, bias["vs"]), np.full(n, bias["vg"]), np.full(n, bias["vb"]), **{k: np.full(n, v) for k, v in proc.items()})
print("batch of %d rows -> shape %s in %.2f s" % (n, out.shape, time.time() - t))
