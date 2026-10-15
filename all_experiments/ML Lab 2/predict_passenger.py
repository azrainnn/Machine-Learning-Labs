"""Predict Titanic survival with the model saved by ML_Lab2.

Example:
    python predict_passenger.py --pclass 2 --sex female --age 30 --fare 20
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


def find_model(name="titanic_survival.joblib"):
    """Search this folder and its parents for models/<name>."""
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        candidate = folder / "models" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find models/{name}; run ML_Lab2 first to create it.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict Titanic survival probability.")
    parser.add_argument("--pclass", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--sex", required=True, choices=["male", "female"])
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--fare", type=float, required=True)
    parser.add_argument("--sibsp", type=int, default=0)
    parser.add_argument("--parch", type=int, default=0)
    parser.add_argument("--embarked", choices=["C", "Q", "S"], default="S")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.age, args.fare, args.sibsp, args.parch) < 0:
        print("error: age, fare, sibsp and parch must be non-negative", file=sys.stderr)
        return 2
    row = pd.DataFrame([{
        "Pclass": args.pclass, "Sex": args.sex, "Age": args.age, "SibSp": args.sibsp,
        "Parch": args.parch, "Fare": args.fare, "Embarked": args.embarked,
    }])
    model = joblib.load(find_model())
    probability = float(model.predict_proba(row)[0, 1])
    verdict = "survived" if probability >= 0.5 else "did not survive"
    print(f"survival probability = {probability:.3f} -> predicted: {verdict} (threshold 0.50)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
