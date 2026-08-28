# DynManipBench

DynManipBench is a reconstructed and validated robot-motion benchmark for a 7-DOF Kinova Gen3 manipulator operating in dynamic environments.

## Release model

The Git repository contains documentation, schemas, validation summaries, and release-building scripts.

The dataset itself should be distributed through a versioned GitHub Release so that the release assets have stable URLs and downloadable asset counts.

Recommended first public release:

**DynManipBench v1.0**

Primary downloadable asset:

`DynManipBench-v1.0-core.zip`

The core package should contain the preserved raw environment and trajectory archives, reconstruction metadata, benchmark documentation, validation summaries, and the scripts required to regenerate derived representations.

Large derived representations such as HDF5 trajectories or precomputed supervised-learning examples may be offered as optional release assets, but they should not be required to reproduce the benchmark from the core package.

## Dataset scale

The reconstructed archive contains:

- 300 environment descriptions
- 100 reconstructed three-member environment families
- 750,000 canonical trajectories
- 5,385,741 waypoints
- 4,635,741 successive trajectory transitions
- 3,885,741 derived supervised-learning examples
- 524 reconstructed dynamic-obstacle identities
- 476 reconstructed static-obstacle identities

## Canonical and processed trajectory representations

The `.configs` trajectories in the archived `pathData.zip` are the **canonical 
archived trajectories** and are authoritative for historical waypoint counts and 
canonical trajectory geometry.

`data/processed/dynamic_v1` is a **derived, densely resampled representation** 
prepared for downstream processing and model training. It must not be interpreted as a
one-for-one HDF5 transcription of the canonical waypoint sequences.

A stratified audit of 300 trajectories found:

| Statistic | Canonical archive | `dynamic_v1` |
|---|---:|---:|
| Minimum configurations | 4 | 16 |
| Median configurations | 6 | 40 |
| Mean configurations | 7.103 | 41.877 |
| Maximum in audit sample | 23 | 87 |

No audited trajectory had the same row count in the two representations. 
Nevertheless, the processed representation preserved the canonical endpoints to 
numerical precision:
maximum wrapped start-configuration discrepancy was `1.9563e-7` rad and maximum
goal-configuration discrepancy was `1.9014e-7` rad.

See `docs/DATA_PROVENANCE.md` for interpretation rules.

## Snapshot geometric annotations

The v1.1.0 characterization adds snapshot configuration-space connectivity 
annotations for all 750,000 canonical trajectories:

- 560,069 (74.676%) have collision-free direct start-to-goal interpolation.
- 189,931 (25.324%) are directly obstructed.
- All 750,000 start and goal configurations were valid under the corresponding
- stored snapshot geometry.
- Canonical median waypoint count is 5 for direct cases and 11 for blocked cases.
- Canonical mean waypoint count is 5.427 for direct cases and 12.352 for blocked
 cases.
- Median canonical path/direct-distance ratio is 1.0000 for direct cases and
  1.043283 for blocked cases.
- Mean canonical path/direct-distance ratio is 1.000084 for direct cases and
  1.122949 for blocked cases.

These labels characterize **snapshot geometry**, not historical dynamic-time
feasibility. The historical obstacle speed law, absolute phase, and temporal sampling 
interval remain unresolved.

Run `scripts/release/build_trajectory_geometry_v1.py` on the reconstructed local
benchmark to create the trajectory-level Parquet annotation file.

## Provenance

The historical dissertation describes a 100-environment, 250,000-trajectory experiment generated in Klampt using RRT* for a Kinova Gen3 manipulator. The surviving research archive is larger and contains 300 environments and 750,000 trajectories. DynManipBench explicitly distinguishes:

- `source_verified`
- `archive_reconstructed`
- `unresolved_historical`

See `docs/PROVENANCE.md`.

## Citation

See `CITATION.cff`.

The dissertation associated with the original experimental work should be cited as:

Liang, Y. (2025). *High-dimensional Spaces Motion Planning for Robotic Arm in Dynamic Environment* [Unpublished doctoral dissertation]. Pace University.

## Downloading

For published releases, use the GitHub **Releases** page rather than cloning the repository for the dataset itself.

The paper should point readers to the repository and, preferably, to the specific versioned release used in the paper.

## Verification

After downloading a release asset:

```bash
python scripts/verify_release.py /path/to/DynManipBench-v1.0-core.zip /path/to/SHA256SUMS.txt
```

## License

DynManipBench uses separate licenses for the dataset/documentation and the software.

### Dataset and documentation

The DynManipBench dataset, metadata, benchmark documentation, and associated data files are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

https://creativecommons.org/licenses/by/4.0/

You may share and adapt the dataset, including for commercial purposes, provided that appropriate attribution is given and changes are indicated.

### Software

Source code and scripts in this repository are licensed under the **BSD 3-Clause License**.

See `LICENSE.md` for the complete license terms.

When using DynManipBench in scholarly work, please cite the DynManipBench paper and the underlying dissertation as described in `CITATION.cff`.
