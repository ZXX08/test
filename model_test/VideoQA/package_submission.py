#!/usr/bin/env python3
"""Package VideoQA.json alone at the root of VideoQA.zip."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.name != "VideoQA.json":
        raise ValueError(f"Unexpected submission filename: {args.input.name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.input, arcname=args.input.name)
    with zipfile.ZipFile(args.output) as archive:
        if archive.namelist() != ["VideoQA.json"] or archive.testzip() is not None:
            raise RuntimeError("ZIP structure or CRC validation failed")
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
