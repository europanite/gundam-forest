from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gundam_rf.titanic import load_titanic_csv, load_titanic_openml, run_titanic_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ordinary Titanic survival Random Forest baseline."
    )
    parser.add_argument(
        "--data",
        default="data/raw/titanic.csv",
        help="Path to a Titanic CSV with Survived/Pclass/Sex/Age/SibSp/Parch/Fare/Embarked.",
    )
    parser.add_argument(
        "--use-openml",
        action="store_true",
        help="Fetch Titanic from OpenML when --data does not exist. Requires network access.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/titanic_baseline",
        help="Directory for Titanic baseline outputs.",
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


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)

    if data_path.exists():
        df = load_titanic_csv(data_path)
    elif args.use_openml:
        df = load_titanic_openml()
    else:
        raise FileNotFoundError(
            f"Titanic CSV was not found: {data_path}. "
            "Place Kaggle-style train.csv at data/raw/titanic.csv, pass --data, "
            "or add --use-openml for a network-based fallback."
        )

    result = run_titanic_training(
        df=df,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
