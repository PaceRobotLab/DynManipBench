# DynManipBench Dataset Card

## Summary
DynManipBench is a reconstruction of a large robot-motion corpus for a Kinova Gen3 manipulator in environments containing static and moving obstacles.

## Primary release contents
- original/surviving environment archive
- original/surviving trajectory archive
- reconstruction metadata
- validation summaries
- scripts to regenerate derived benchmark representations
- provenance and schema documentation

## Intended uses
- robot motion prediction
- next-configuration prediction
- motion-planning research
- obstacle-conditioned prediction
- representation studies
- collision-aware learning
- controlled counterfactual obstacle interventions

## Known limitations
- historical obstacle velocity law is unresolved
- historical obstacle phase is unresolved
- historical simulation delta-t is unresolved
- portions of the surviving archive contain geometric representations not explicitly described in the dissertation
- the benchmark is simulation-derived and should not be treated as equivalent to physical-robot data

## Provenance labels
Every reconstructed metadata field should, where practical, carry one of:
- source_verified
- archive_reconstructed
- unresolved_historical
