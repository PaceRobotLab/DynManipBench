from pathlib import Path
import csv
import json
import random
import zipfile
import numpy as np
import h5py

ROOT = Path.home() / "DynManipBench"
PATH_ZIP = ROOT / "data" / "raw" / "pathData.zip"
H5_DIR = ROOT / "data" / "processed" / "dynamic_v1"
OUT = ROOT / "runs" / "trajectory_representation_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260828
SAMPLE_ENVS = [100, 125, 150, 175, 199, 200, 225, 250, 275, 299,
               300, 325, 350, 375, 399]
PATHS_PER_ENV = 20
CONT = np.array([0, 2, 4, 6], dtype=int)


def wrap_delta(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d.copy()
    d[CONT] = np.arctan2(np.sin(d[CONT]), np.cos(d[CONT]))
    return d


def wrapped_dist(a, b):
    return float(np.linalg.norm(wrap_delta(a, b)))


def parse_configs_bytes(data):
    """Parse Klampt-style .configs text robustly into numeric rows."""
    text = data.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = []
        ok = True
        for tok in line.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                ok = False
                break
        if ok and vals:
            # Common Klampt config formats may prefix dimension.
            if len(vals) >= 10 and int(round(vals[0])) == len(vals) - 1:
                vals = vals[1:]
            rows.append(vals)
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = max(map(len, rows))
    rows = [r for r in rows if len(r) == width]
    return np.asarray(rows, dtype=float)


def to7(q):
    q = np.asarray(q, float)
    if q.ndim != 2:
        return q
    if q.shape[1] == 9:
        return q[:, 1:8]
    if q.shape[1] == 7:
        return q
    return q


def h5_path(env, row):
    p = H5_DIR / f"env{env}.h5"
    with h5py.File(p, "r") as h5:
        off = h5["trajectory_offsets"][:]
        a, b = int(off[row]), int(off[row+1])
        return np.asarray(h5["waypoints_7dof"][a:b], dtype=float)


def candidate_names(env, row):
    # Historical archive indices have appeared in both 0- and 1-based forms.
    vals = [row, row + 1]
    out = []
    for i in vals:
        out += [
            f"env{env}_{i}.configs",
            f"env{env}/{i}.configs",
            f"env{env}/env{env}_{i}.configs",
        ]
    return out


def build_zip_index(z):
    idx = {}
    for n in z.namelist():
        base = Path(n).name
        idx.setdefault(base, []).append(n)
    return idx


def locate_member(idx, env, row):
    for cand in candidate_names(env, row):
        base = Path(cand).name
        for hit in idx.get(base, []):
            if f"env{env}" in hit:
                return hit
    return None


def compare(raw, proc):
    raw7, proc7 = to7(raw), to7(proc)
    result = {
        "raw_rows": int(len(raw7)),
        "processed_rows": int(len(proc7)),
        "same_row_count": int(len(raw7) == len(proc7)),
        "endpoint_start_error": None,
        "endpoint_goal_error": None,
        "raw_adjacent_median": None,
        "processed_adjacent_median": None,
        "raw_cost": None,
        "processed_cost": None,
        "raw_to_processed_row_ratio": None,
    }
    if len(raw7) and len(proc7) and raw7.shape[1] == 7 and proc7.shape[1] == 7:
        result["endpoint_start_error"] = wrapped_dist(raw7[0], proc7[0])
        result["endpoint_goal_error"] = wrapped_dist(raw7[-1], proc7[-1])
        rd = [wrapped_dist(raw7[i], raw7[i+1]) for i in range(len(raw7)-1)]
        pd = [wrapped_dist(proc7[i], proc7[i+1]) for i in range(len(proc7)-1)]
        result["raw_adjacent_median"] = float(np.median(rd)) if rd else 0.0
        result["processed_adjacent_median"] = float(np.median(pd)) if pd else 0.0
        result["raw_cost"] = float(sum(rd))
        result["processed_cost"] = float(sum(pd))
        result["raw_to_processed_row_ratio"] = len(proc7) / len(raw7)
    return result


def main():
    if not PATH_ZIP.exists():
        raise FileNotFoundError(PATH_ZIP)
    rng = random.Random(SEED)
    records = []

    print("DynManipBench trajectory representation audit v1")
    print("=" * 80)
    print("Comparing raw .configs trajectories with dynamic_v1 HDF5.")
    print("This script does NOT modify any data.")

    with zipfile.ZipFile(PATH_ZIP) as z:
        idx = build_zip_index(z)
        for env in SAMPLE_ENVS:
            hp = H5_DIR / f"env{env}.h5"
            with h5py.File(hp, "r") as h5:
                n = len(h5["trajectory_offsets"]) - 1
            rows = sorted(rng.sample(range(n), min(PATHS_PER_ENV, n)))
            print(f"\nenv{env}: auditing {len(rows)} trajectories")
            for row in rows:
                member = locate_member(idx, env, row)
                if member is None:
                    records.append({"environment":env,"path_row":row,
                                    "archive_member":"","status":"raw_not_found"})
                    continue
                raw = parse_configs_bytes(z.read(member))
                proc = h5_path(env, row)
                rec = {"environment":env,"path_row":row,
                       "archive_member":member,"status":"ok",
                       "raw_columns":int(raw.shape[1]) if raw.ndim == 2 and raw.size else 0,
                       "processed_columns":int(proc.shape[1]) if proc.ndim == 2 else 0}
                rec.update(compare(raw, proc))
                records.append(rec)

    csvp = OUT / "trajectory_representation_audit.csv"
    fields = sorted({k for r in records for k in r})
    with csvp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(records)

    ok = [r for r in records if r.get("status") == "ok"]
    def vals(k):
        return np.array([r[k] for r in ok if r.get(k) is not None], dtype=float)

    summary = {
        "sample_records": len(records),
        "matched_records": len(ok),
        "raw_rows": {},
        "processed_rows": {},
        "same_row_count_fraction": None,
        "start_endpoint_error_max": None,
        "goal_endpoint_error_max": None,
        "raw_to_processed_row_ratio": {},
        "interpretation_hint": (
            "If endpoints agree but processed_rows greatly exceed raw_rows, "
            "dynamic_v1 contains interpolation/resampling rather than canonical archived waypoints."
        ),
    }
    for key in ("raw_rows","processed_rows","raw_to_processed_row_ratio"):
        a = vals(key)
        if len(a):
            summary[key] = {"min":float(a.min()),"median":float(np.median(a)),
                            "mean":float(a.mean()),"max":float(a.max())}
    if ok:
        summary["same_row_count_fraction"] = float(np.mean(vals("same_row_count")))
        a=vals("endpoint_start_error")
        b=vals("endpoint_goal_error")
        summary["start_endpoint_error_max"] = float(a.max()) if len(a) else None
        summary["goal_endpoint_error_max"] = float(b.max()) if len(b) else None

    jp = OUT / "summary.json"
    jp.write_text(json.dumps(summary, indent=2))

    print("\nREPRESENTATION AUDIT SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nOUTPUT")
    print(csvp)
    print(jp)
    print("\nPASS — representation audit completed.")


if __name__ == "__main__":
    main()
