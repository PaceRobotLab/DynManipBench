# Release Process

## 1. Freeze a version
Use semantic release tags such as:

`v1.0.0`

The paper should identify the exact release tag used for its reported results.

## 2. Build the core release asset
From the DynManipBench project root:

```bash
python scripts/build_release_v1.py --root ~/DynManipBench --version 1.0.0
```

This creates a staging directory and a zip asset.

## 3. Verify checksums
```bash
sha256sum release_build/DynManipBench-v1.0.0-core.zip
```

Store the result in `SHA256SUMS.txt`.

## 4. Create the GitHub Release
Create a release from tag `v1.0.0` and upload:
- `DynManipBench-v1.0.0-core.zip`
- `SHA256SUMS.txt`
- optional derived-data assets

## 5. Paper pointer
The manuscript should contain both:
- canonical repository URL
- exact release URL/tag used for the paper

Suggested text:

> DynManipBench is available from the Pace Robot Lab GitHub repository at https://github.com/PaceRobotLab/DynManipBench. The dataset used in this paper is distributed as the versioned GitHub Release v1.0.0; users should obtain the benchmark from the Releases page so that the exact published artifact and checksum can be identified.

## 6. Do not rewrite v1.0.0
If corrections are needed after release, publish v1.0.1 or a later version rather than silently replacing a published asset.
