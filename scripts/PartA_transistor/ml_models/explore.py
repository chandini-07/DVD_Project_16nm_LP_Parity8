import os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for c in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    df = pd.read_csv(os.path.join(ROOT, "data", c + ".csv"))
    print("==", c, df.shape)
    rows = []
    for k in ["id", "ig", "is", "ib"]:
        a = df[k].abs()
        rows.append((k, "%.1e" % a.min(), "%.1e" % a.median(), "%.1e" % a.max(),
                     "%.2f" % (df[k] < 0).mean(), "%.2f" % (a < 1e-15).mean()))
    print(pd.DataFrame(rows, columns=["current", "min|I|", "median|I|", "max|I|", "frac_negative", "frac<1e-15"]).to_string(index=False))
