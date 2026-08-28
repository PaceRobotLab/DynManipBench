# `trajectory_geometry_v1.parquet` schema

| Field | Meaning |
|---|---|
| `environment` | Environment identifier |
| `family` | Three-variant family identifier |
| `variant` | Environment variant A, B, or C |
| `trajectory_id` | Trajectory row/index used by the reconstructed benchmark |
| `archive_member` | Corresponding canonical `.configs` archive member |
| `direct_free` | Whether direct start-goal interpolation is collision-free under stored snapshot geometry |
| `canonical_waypoint_count` | Number of configurations in the canonical archived trajectory |
| `endpoint_distance_rad` | Wrapped 7-DOF start-goal configuration-space distance |
| `canonical_path_cost_rad` | Sum of wrapped 7-DOF distances along canonical archived trajectory |
| `canonical_cost_direct_ratio` | Canonical path cost divided by endpoint distance |
| `first_collision_fraction` | Normalized first-collision position along blocked direct interpolation, when available |
| `annotation_semantics` | `snapshot_geometry` |
| `trajectory_representation` | `canonical_archive` |
| `provenance` | Provenance label for the derived annotation |

Continuous-joint differences are wrapped before angular-distance calculations.

`direct_free` is a snapshot-geometric annotation, not a historical dynamic-time
feasibility label.
