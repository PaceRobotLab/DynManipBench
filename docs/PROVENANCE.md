# Provenance

DynManipBench separates historical evidence from modern reconstruction.

## source_verified
Properties explicitly supported by the surviving dissertation or another primary historical source.

Examples:
- Kinova Gen3 7-DOF manipulator
- Klampt simulation environment
- RRT* used for trajectory generation
- 100 documented environments
- 2,500 trajectories per documented environment
- ten obstacles per documented base environment
- conceptual network inputs: rbInput, constInput, obsInput, numOfObsInput

## archive_reconstructed
Properties established by structural or numerical analysis of the surviving computational artifacts.

Examples:
- 300 archived environment descriptions
- 750,000 canonical trajectories
- 100 reconstructed three-member environment families
- 5,385,741 waypoints
- 524 dynamic and 476 static obstacle identities
- dynamic-obstacle spatial corridors
- 3,885,741 derived supervised-learning examples
- exact reconstructed serialized input dimensions

## unresolved_historical
Quantities that cannot be justified from the surviving evidence.

Examples:
- original dynamic-obstacle speed law
- original absolute phase
- original temporal sampling interval / delta-t
