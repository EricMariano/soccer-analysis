"""Inspect the pinned cohort or incrementally prepare reproducible shot splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from match_analyst.data import DEFAULT_CACHE_DIR, DataError, StatsBombClient
from match_analyst.dataset import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    _write_json,
    build_catalog_manifest,
    load_dataset_config,
    prepare_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="Never access the network.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent downloads, from 1 to 16.")
    parser.add_argument("--catalog-only", action="store_true", help="Confirm match counts and dates without downloading events.")
    args = parser.parse_args()
    try:
        if args.catalog_only:
            config = load_dataset_config(args.config)
            client = StatsBombClient(args.cache_dir, revision=config["source_revision"], offline=args.offline)
            manifest = build_catalog_manifest(config, client)
            path = args.output_dir / "catalog_manifest.json"
            _write_json(path, manifest)
            result = {
                "status": "catalog_only",
                "source_revision": manifest["source_revision"],
                "selection_id": manifest["selection_id"],
                "manifest_path": str(path),
                "total_matches": len(manifest["matches"]),
                "coverage": manifest["coverage"],
                "coverage_notes": manifest["coverage_notes"],
            }
        else:
            result = prepare_dataset(
                args.config, cache_dir=args.cache_dir, output_dir=args.output_dir,
                offline=args.offline, workers=args.workers,
                progress=lambda message: print(message, flush=True),
            )
    except (DataError, OSError) as exc:
        parser.exit(1, f"Could not prepare dataset: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
