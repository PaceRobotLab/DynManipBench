#!/usr/bin/env python3
"""Build trajectory_geometry_v1.parquet from validated DynManipBench analysis outputs."""
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "DynManipBench"
SRC = ROOT / "runs" / "canonical_path_connectivity_postanalysis_v1" / "canonical_path_connectivity.csv"
ENV = ROOT / "runs" / "dataset_endpoint_connectivity_full_v1" / "environment_summary.csv"
OUT = ROOT / "data" / "metadata" / "trajectory_geometry_v1.parquet"

def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    df = pd.read_csv(SRC)

    required = {
        "environment","family","block","path_row","direct_free",
        "canonical_waypoints","endpoint_distance_rad",
        "canonical_path_cost_rad","canonical_cost_direct_ratio"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Stable public-facing names.
    df = df.rename(columns={
        "block":"variant",
        "path_row":"trajectory_id",
        "canonical_waypoints":"canonical_waypoint_count",
    })

    # Add first-collision fraction when available from the complete connectivity output.
    # Search per-environment CSVs and merge only the compact required columns.
    edir = ROOT / "runs" / "dataset_endpoint_connectivity_full_v1" / "environments"
    pieces=[]
    for p in sorted(edir.glob("*.csv")):
        x=pd.read_csv(p)
        cols=[c for c in ["environment","path_row","first_collision_fraction"] if c in x.columns]
        if len(cols)==3:
            pieces.append(x[cols])
    if pieces:
        fc=pd.concat(pieces, ignore_index=True).rename(columns={"path_row":"trajectory_id"})
        df=df.merge(fc, on=["environment","trajectory_id"], how="left")

    df["annotation_semantics"] = "snapshot_geometry"
    df["trajectory_representation"] = "canonical_archive"
    df["provenance"] = "derived_annotation_from_canonical_archive"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")

    print(f"Wrote {len(df):,} rows")
    print(OUT)
    print(f"Size: {OUT.stat().st_size/1024/1024:.2f} MiB")
    print("PASS — trajectory_geometry_v1.parquet created.")

if __name__ == "__main__":
    main()
