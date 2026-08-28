# Data Provenance and Representation Semantics

## 1. Purpose

DynManipBench contains both source-preserved material and derived representations.
This document distinguishes them so that downstream analyses do not accidentally treat
a processing representation as historical source data.

## 2. Provenance classes

### Source-verified / canonical

The `.configs` files in `data/raw/pathData.zip` are the canonical archived trajectory
records. Use these files for:

- canonical waypoint counts;
- canonical start and goal configurations;
- canonical wrapped trajectory length/path cost;
- analyses intended to describe the surviving historical trajectory archive.

### Archive-reconstructed

Environment/family relationships, obstacle metadata, and other fields reconstructed
deterministically from surviving archive structure should be identified as reconstructed,
not as statements recovered verbatim from historical source code.

### Derived / processed

`data/processed/dynamic_v1` is a derived representation. Its trajectories are more
densely sampled than the canonical `.configs` trajectories. It is appropriate for the
downstream processing/training tasks for which it was generated, but its row counts
must not be reported as historical waypoint counts.

### Historically unresolved

The historical dynamic-obstacle speed law, absolute temporal phase, and temporal
sampling interval have not been recovered. Any modern timing regime is therefore a
benchmark convention rather than a recovered historical fact.

## 3. Representation audit

A stratified audit compared 300 canonical trajectories with their `dynamic_v1`
counterparts.

Canonical row counts: min 4, median 6, mean 7.103333, max 23.
Processed row counts: min 16, median 40, mean 41.876667, max 87.
Fraction with identical row count: 0.0.
Median processed/canonical row-count ratio: 6.8.

Despite the sampling difference, endpoints were preserved:
maximum wrapped start error = 1.9562141122811702e-7 rad;
maximum wrapped goal error = 1.9013513304406594e-7 rad.

This endpoint agreement supports use of the processed representation's endpoints in the
completed snapshot direct-connectivity test. Canonical `.configs` files remain
authoritative for waypoint-count and path-complexity statistics.

## 4. Snapshot-connectivity semantics

`direct_free = true` means that direct configuration-space interpolation from the
canonical start to canonical goal is collision-free under the corresponding stored
environment geometry and the benchmark's stated collision-checking convention.

`direct_free = false` means that this direct interpolation is obstructed.

This is deliberately called **snapshot connectivity**. It is not equivalent to
historical dynamic-time feasibility because the original obstacle timing variables
listed above are unresolved.

## 5. Canonical geometric characterization

Across 750,000 trajectories:

| Group | N | Fraction | Median waypoints | Mean waypoints | Median cost/direct | Mean cost/direct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 560,069 | 74.676% | 5 | 5.427 | 1.000000 | 1.000084 |
| Blocked | 189,931 | 25.324% | 11 | 12.352 | 1.043283 | 1.122949 |

Variant-level direct/blocked counts:

| Variant | N | Direct | Direct % | Blocked | Blocked % |
|---|---:|---:|---:|---:|---:|
| A | 250,000 | 180,109 | 72.044 | 69,891 | 27.956 |
| B | 250,000 | 200,002 | 80.001 | 49,998 | 19.999 |
| C | 250,000 | 179,958 | 71.983 | 70,042 | 28.017 |

For blocked trajectories, median cost/direct ratios are 1.0499 (A), 1.0302 (B),
and 1.0495 (C).
