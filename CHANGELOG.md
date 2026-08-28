# Changelog

## v1.1.0

### Added
- Explicit documentation distinguishing canonical archived `.configs` trajectori
es from
  the densely resampled `dynamic_v1` processing representation.
- A 300-trajectory representation audit documenting endpoint preservation.
- Complete snapshot direct-connectivity characterization for 750,000 trajectorie
s.
- Canonical direct-vs-blocked path-complexity characterization.
- Reproducible validation/analysis scripts.
- Tooling and schema for `trajectory_geometry_v1.parquet`.

### Clarified
- Historical waypoint statistics must be computed from canonical `.configs` traj
ectories.
- Snapshot direct-connectivity labels do not imply historical dynamic-time feasi
bility.
- Historical obstacle speed law, absolute phase, and temporal sampling interval 
remain unresolved.

### Changed
No canonical raw trajectory or environment data are modified by this release.

## v1.0 — 2026-08-25
Initial public DynManipBench release.

### Included
- 300 environment descriptions
- 750,000 canonical trajectories
- reconstruction metadata
- validation summaries
- provenance documentation
- release verification tooling

### Notes
Historical temporal variables that cannot be recovered are explicitly marked unresolved.

