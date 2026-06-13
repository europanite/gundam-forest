from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gundam_rf.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Gundam character war-death Random Forest baseline."
    )
    parser.add_argument(
        "--data",
        default="data/raw/gundam_character_death_dataset_first_pass.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for model outputs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout test size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
