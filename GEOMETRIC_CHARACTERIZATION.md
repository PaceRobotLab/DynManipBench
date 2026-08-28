# Snapshot Geometric Characterization

This document summarizes derived geometric annotations added with DynManipBench v1.1.0.

## Complete endpoint-connectivity test

All 750,000 reconstructed trajectory start-goal pairs were checked against their
corresponding stored environment geometry. All endpoints were collision-free.

- Directly connectable: 560,069 (74.676%)
- Directly obstructed: 189,931 (25.324%)

The first collision location among blocked direct interpolations had median normalized
fraction 0.341. The middle 50% lay between 0.158 and 0.568; the 5th and 95th
percentiles were 0.032 and 0.799.

## Canonical path complexity

Path statistics were recomputed from the original `.configs` archive rather than the
densely resampled `dynamic_v1` representation.

Direct cases have median/mean canonical waypoint counts of 5/5.427. Blocked cases have
11/12.352. Median path/direct-distance ratios are 1.000000 and 1.043283 respectively;
means are 1.000084 and 1.122949.

These results reveal two geometrically different populations that are obscured by
aggregate trajectory statistics.

## Interpretation limitation

The labels are snapshot-geometric annotations only. They do not reconstruct historical
dynamic-time feasibility. Historical speed, phase, and sampling-time semantics remain
unresolved.
