"""Explicit offline training; the Streamlit interface never calls this script."""

import argparse
from pathlib import Path

from match_analyst.xg import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a validation-selected xG model from a prepared local dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/xg_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()
    try:
        artifact = train_model(
            args.dataset_dir, args.output_dir, seed=args.seed,
            candidates=args.candidates, model_version=args.model_version,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    metadata = artifact.metadata
    metric = metadata["metrics"]["validation"]["npxg"]
    print(f"Saved {metadata['model_version']} in {args.output_dir}")
    print(f"Dataset: {metadata['dataset_id']} · Selected C: {metadata['config']['C']}")
    print(f"Validation npxG log loss: {metric['model']['log_loss']:.6f}; baseline: {metric['baseline']['log_loss']:.6f}")
    for limitation in metadata["limitations"]:
        print(f"Limitation: {limitation}")


if __name__ == "__main__":
    main()
