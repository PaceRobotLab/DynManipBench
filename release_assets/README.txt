INTENDED GITHUB RELEASE ASSETS

Required:
  DynManipBench-v1.0.0-core.zip
  SHA256SUMS.txt

Optional, if useful:
  DynManipBench-v1.0.0-hdf5.tar.gz
  DynManipBench-v1.0.0-training-examples.tar.gz

Recommended contents of the CORE zip:
  data/raw/envData.zip
  data/raw/pathData.zip
  data/metadata/
  docs/
  scripts/
  CITATION.cff
  LICENSE.md
  VERSION
  MANIFEST.csv
  SHA256SUMS.txt (inside package for internal files)

Rationale:
The preserved raw archives plus metadata and reconstruction scripts are sufficient to reproduce derived representations. Large derived formats can remain optional downloads.
