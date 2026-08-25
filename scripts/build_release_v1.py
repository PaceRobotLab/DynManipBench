\
#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, hashlib, shutil, zipfile, os

def sha256(path, chunk=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="DynManipBench project root")
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    version = args.version
    build_root = root / "release_build"
    stage = build_root / f"DynManipBench-v{version}-core"
    outzip = build_root / f"DynManipBench-v{version}-core.zip"

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    build_root.mkdir(parents=True, exist_ok=True)

    required = [
        root / "data" / "raw" / "envData.zip",
        root / "data" / "raw" / "pathData.zip",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    # Raw archives
    (stage / "data" / "raw").mkdir(parents=True)
    for p in required:
        shutil.copy2(p, stage / "data" / "raw" / p.name)

    # Metadata
    src_meta = root / "data" / "metadata"
    if src_meta.exists():
        shutil.copytree(src_meta, stage / "data" / "metadata")

    # Documentation and release metadata, if present in project root
    for name in ["README.md", "DATASET_CARD.md", "CITATION.cff", "LICENSE.md", "CHANGELOG.md"]:
        src = root / name
        if src.exists():
            shutil.copy2(src, stage / name)

    src_docs = root / "docs"
    if src_docs.exists():
        shutil.copytree(src_docs, stage / "docs", dirs_exist_ok=True)

    # Include model/reconstruction scripts but exclude caches
    src_model = root / "src" / "model"
    if src_model.exists():
        dest = stage / "src" / "model"
        dest.mkdir(parents=True, exist_ok=True)
        for p in src_model.glob("*.py"):
            shutil.copy2(p, dest / p.name)

    (stage / "VERSION").write_text(version + "\n")

    # Internal manifest
    rows = []
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            rel = p.relative_to(stage).as_posix()
            rows.append({
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            })

    manifest = stage / "MANIFEST.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)

    # Internal checksums
    sums = stage / "SHA256SUMS.txt"
    sums.write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))

    # Zip without double-compressing already-compressed formats too aggressively
    if outzip.exists():
        outzip.unlink()
    with zipfile.ZipFile(outzip, "w", allowZip64=True) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                rel = f"DynManipBench-v{version}-core/" + p.relative_to(stage).as_posix()
                if p.suffix.lower() in {".zip", ".gz", ".npz", ".parquet", ".h5", ".hdf5"}:
                    compression = zipfile.ZIP_STORED
                else:
                    compression = zipfile.ZIP_DEFLATED
                z.write(p, rel, compress_type=compression)

    external_sum = build_root / "SHA256SUMS.txt"
    external_sum.write_text(f"{sha256(outzip)}  {outzip.name}\n")

    print("Created:", outzip)
    print("Checksum:", external_sum)
    print("Size bytes:", outzip.stat().st_size)

if __name__ == "__main__":
    main()
