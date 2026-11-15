"""Step 9: how accurate are the models at the voltages gates really use (Vd, Vs exactly 0 or VDD)?"""
import os, sys, json, hashlib, tempfile, time
import numpy as np, pandas as pd
from sampler import draw_case, CASES, NOM, LMIN, VDD_NOM
from runner import run_batch, ROOT, OPTIONS
from predict import predict_currents
from mlcommon import TARGETS, FLOOR, RAW

N = 300

def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_config_fingerprint(case, n, rng_seed):
    ci = list(CASES.keys()).index(case)
    dev, vg_f, vb_f = CASES[case]
    
    # 1. Model card contents
    model_hashes = {}
    for fname in ("nmos_card.pm", "pmos_card.pm", "16nm_LP.pm"):
        fpath = os.path.join(ROOT, "model", fname)
        model_hashes[fname] = compute_file_sha256(fpath) if os.path.exists(fpath) else None

    # 2. Source code controlling generation and schemas
    source_hashes = {}
    for fname in ("sampler.py", "runner.py", "mlcommon.py", "rail_check.py"):
        fpath = os.path.join(ROOT, "scripts", fname)
        source_hashes[fname] = compute_file_sha256(fpath) if os.path.exists(fpath) else None

    config = {
        "case": case,
        "device": dev,
        "n_samples": n,
        "seed": list(rng_seed),
        "spice_options": OPTIONS,
        "environment": {
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "sampling_bounds": {
            "vdd_range": [0.9 * VDD_NOM, 1.1 * VDD_NOM],
            "temp_range": [-40.0, 125.0],
            "w_range": [1.0 * LMIN, 8.0 * LMIN],
            "lmin": LMIN,
            "process_variation_pct": 0.10,
            "nominal_process": NOM[dev],
        },
        "rail_bias_rules": {
            "vd_allowed_fractions": [0.0, 1.0],
            "vs_allowed_fractions": [0.0, 1.0],
            "vg_fraction": vg_f,
            "vb_fraction": vb_f,
        },
        "model_file_hashes": model_hashes,
        "source_file_hashes": source_hashes,
    }
    config_json = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

def validate_rail_dataset(df, case, expected_n):
    expected_cols = RAW + TARGETS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    if len(df) != expected_n:
        return False, f"Expected {expected_n} rows, found {len(df)}"

    try:
        mat = df[expected_cols].to_numpy(dtype=float)
    except Exception as e:
        return False, f"Non-numeric values in dataset: {e}"

    if not np.isfinite(mat).all():
        return False, "Dataset contains NaN or infinite values"

    vdd = df["vdd"].to_numpy()
    vd = df["vd"].to_numpy()
    vs = df["vs"].to_numpy()
    vg = df["vg"].to_numpy()
    vb = df["vb"].to_numpy()

    # Rail voltage check: Vd and Vs must be within 1 mV of 0 or VDD
    vd_on_rail = (np.abs(vd) < 1e-3) | (np.abs(vd - vdd) < 1e-3)
    vs_on_rail = (np.abs(vs) < 1e-3) | (np.abs(vs - vdd) < 1e-3)
    if not (vd_on_rail.all() and vs_on_rail.all()):
        return False, "Vd or Vs values violate rail boundary conditions"

    # Operating state bias check
    dev, vg_f, vb_f = CASES[case]
    vg_expected = vg_f * vdd
    vb_expected = vb_f * vdd
    if not (np.abs(vg - vg_expected) < 1e-3).all():
        return False, f"Gate voltage Vg violates case bias rule for {case}"
    if not (np.abs(vb - vb_expected) < 1e-3).all():
        return False, f"Bulk voltage Vb violates case bias rule for {case}"

    # KCL residual check: sum(Id + Ig + Is + Ib) == 0 within numerical tolerance
    kcl_sum = np.abs(df["id"] + df["ig"] + df["is"] + df["ib"]).to_numpy()
    max_branch = df[["id", "ig", "is", "ib"]].abs().max(axis=1).to_numpy()
    
    # Acceptance tolerance: max(1e-13 A, 1e-3 * max_branch)
    kcl_viol = (kcl_sum > 1e-13) & (kcl_sum > 1e-3 * max_branch)
    if kcl_viol.any():
        bad_idx = np.where(kcl_viol)[0][0]
        return False, f"KCL violation at row {bad_idx}: residual={kcl_sum[bad_idx]:.3e} A, max_branch={max_branch[bad_idx]:.3e} A"

    return True, ""

def check_reusable_dataset(rail_csv, meta_path, case, expected_n, expected_fp):
    if not os.path.exists(rail_csv):
        return False, None, "CSV does not exist"
    if not os.path.exists(meta_path):
        return False, None, "Metadata sidecar does not exist"

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        return False, None, f"Metadata JSON corrupted: {e}"

    if not isinstance(meta, dict):
        return False, None, "Metadata is not a valid JSON dictionary"
    if meta.get("case") != case:
        return False, None, f"Metadata case mismatch ({meta.get('case')} != {case})"
    if meta.get("rows") != expected_n:
        return False, None, f"Metadata row count mismatch ({meta.get('rows')} != {expected_n})"
    if meta.get("config_fingerprint") != expected_fp:
        return False, None, f"Configuration fingerprint mismatch"

    expected_sha256 = meta.get("csv_sha256")
    if not expected_sha256 or not isinstance(expected_sha256, str):
        return False, None, "Metadata missing valid csv_sha256"

    try:
        actual_sha256 = compute_file_sha256(rail_csv)
    except Exception as e:
        return False, None, f"Failed to compute CSV hash: {e}"

    if actual_sha256 != expected_sha256:
        return False, None, f"Cryptographic SHA-256 mismatch (meta: {expected_sha256}, actual: {actual_sha256})"

    try:
        df = pd.read_csv(rail_csv)
    except Exception as e:
        return False, None, f"Failed to parse CSV: {e}"

    val_ok, val_msg = validate_rail_dataset(df, case, expected_n)
    if not val_ok:
        return False, None, f"Dataset physical validation failed: {val_msg}"

    return True, df, "Validated reusable dataset"

def write_rail_dataset_atomically(out, rail_csv, meta_path, case, current_fp):
    target_dir = os.path.dirname(rail_csv)
    os.makedirs(target_dir, exist_ok=True)
    temp_csv_fd = None
    temp_csv_path = None
    temp_meta_fd = None
    temp_meta_path = None

    try:
        temp_csv_fd, temp_csv_path = tempfile.mkstemp(prefix=f"rail_test_{case}_", suffix=".tmp", dir=target_dir)
        with os.fdopen(temp_csv_fd, "w") as f:
            temp_csv_fd = None
            out.to_csv(f, index=False, float_format="%.10e")
            f.flush()
            os.fsync(f.fileno())

        csv_hash = compute_file_sha256(temp_csv_path)

        temp_meta_fd, temp_meta_path = tempfile.mkstemp(prefix=f"rail_meta_{case}_", suffix=".tmp", dir=target_dir)
        meta_content = {
            "case": case,
            "rows": len(out),
            "csv_sha256": csv_hash,
            "config_fingerprint": current_fp,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with os.fdopen(temp_meta_fd, "w") as f:
            temp_meta_fd = None
            json.dump(meta_content, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic commit: CSV first, metadata last
        os.replace(temp_csv_path, rail_csv)
        temp_csv_path = None

        os.replace(temp_meta_path, meta_path)
        temp_meta_path = None

    finally:
        if temp_csv_fd is not None:
            try: os.close(temp_csv_fd)
            except OSError: pass
        if temp_meta_fd is not None:
            try: os.close(temp_meta_fd)
            except OSError: pass
        if temp_csv_path is not None and os.path.exists(temp_csv_path):
            try: os.remove(temp_csv_path)
            except OSError: pass
        if temp_meta_path is not None and os.path.exists(temp_meta_path):
            try: os.remove(temp_meta_path)
            except OSError: pass

def main():
    is_baseline = "--baseline" in sys.argv
    uni_csv = os.path.join(ROOT, "results", "test_chosen_baseline.csv" if is_baseline else "test_chosen.csv")
    out_csv = os.path.join(ROOT, "results", "rail_check_baseline.csv" if is_baseline else "rail_check.csv")

    if not os.path.exists(uni_csv):
        sys.exit(f"Error: Required reference metric file not found: {uni_csv}")

    uni = pd.read_csv(uni_csv).set_index("case")
    rows = []
    for ci, case in enumerate(CASES):
        model_path = "%s/models/final_%s.joblib" % (ROOT, case)
        if not os.path.exists(model_path):
            print("skipping", case); continue

        rail_csv = "%s/data/rail_test_%s.csv" % (ROOT, case)
        meta_path = "%s/data/rail_test_%s.meta.json" % (ROOT, case)
        rng_seed = [777, ci]
        current_fp = compute_config_fingerprint(case, N, rng_seed)

        reusable, df, reason = check_reusable_dataset(rail_csv, meta_path, case, N, current_fp)
        if reusable:
            print(f"[{case}] Reusing verified dataset: {rail_csv}")
            I = df[TARGETS].to_numpy()
        else:
            print(f"[{case}] Recomputing SPICE truth ({reason})")
            rng = np.random.default_rng(rng_seed)
            df = draw_case(case, N, rng)
            r = rng.integers(0, 2, (N, 2))
            df["vd"], df["vs"] = r[:, 0] * df["vdd"], r[:, 1] * df["vdd"]
            I = run_batch(case, df)
            
            # Populate target columns directly onto df
            df[TARGETS] = I
            val_ok, val_msg = validate_rail_dataset(df, case, N)
            if not val_ok:
                raise RuntimeError(f"Generated SPICE dataset failed physical validation: {val_msg}")
            
            write_rail_dataset_atomically(df, rail_csv, meta_path, case, current_fp)

        # Diagnostic reporting of KCL residuals (unified for both generation and reuse)
        kcl_res = np.abs(df["id"] + df["ig"] + df["is"] + df["ib"]).to_numpy()
        max_branch = df[["id", "ig", "is", "ib"]].abs().max(axis=1).to_numpy()
        rel_res = kcl_res / np.maximum(max_branch, 1e-18)
        print(f"  KCL residual stats: max={np.max(kcl_res):.3e} A, median={np.median(kcl_res):.3e} A, p95={np.percentile(kcl_res, 95):.3e} A, max_rel={np.max(rel_res):.3e}")

        P = predict_currents(case, *[df[k].to_numpy() for k in RAW])
        rel = np.abs(P - I) / np.maximum(np.abs(I), FLOOR)
        row = dict(case=case, rail_median=np.median(rel), rail_p95=np.percentile(rel, 95),
                   uniform_median=uni.loc[case, "median_err"], uniform_p95=uni.loc[case, "p95_err"])
        for j, k in enumerate(TARGETS):
            row["rail_%s_median" % k] = np.median(rel[:, j])
        rows.append(row)

    res = pd.DataFrame(rows)
    target_res_dir = os.path.dirname(out_csv)
    os.makedirs(target_res_dir, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"\nWrote results to {out_csv}")
    print(res.to_string(index=False, float_format=lambda x: "%.4g" % x))

if __name__ == "__main__":
    main()
