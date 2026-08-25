# DynManipBench Schema

## Canonical trajectory level
Each trajectory is associated with:
- environment identifier
- trajectory identifier
- ordered configuration sequence
- start configuration
- goal configuration
- waypoint count

## Learning representation
Reconstructed supervised examples use:
- `rbInput`: 21 values
- `constInput`: 112 values
- `obsInput`: 80 values
- `numOfObsInput`: scalar
- target: 7 values

### rbInput
The reconstructed 21-value representation is:
- current robot configuration: 7
- previous robot configuration: 7
- goal robot configuration: 7

### obsInput
The reconstructed 80-value representation stores previous and current obstacle records for up to ten obstacles.

## Angular treatment
Continuous joints q1, q3, q5, and q7 require wrapped angular differences for transition analysis and appropriate evaluation metrics.
