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
