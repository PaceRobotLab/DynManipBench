\
#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, sys

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
    ap.add_argument("asset", type=Path)
    ap.add_argument("checksums", type=Path)
    args = ap.parse_args()

    expected = None
    for line in args.checksums.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1] == args.asset.name:
            expected = parts[0]
            break

    if expected is None:
        raise SystemExit(f"No checksum found for {args.asset.name}")

    actual = sha256(args.asset)
    print("expected:", expected)
    print("actual:  ", actual)
    if actual != expected:
        raise SystemExit("FAIL — checksum mismatch")
    print("PASS — checksum verified")

if __name__ == "__main__":
    main()
