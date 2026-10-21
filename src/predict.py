"""Predict a grade category from the eight assessment totals.

Usage:
    python src/predict.py                       # asks for the totals one by one
    python src/predict.py 92 96 90 80 85 100 90 85   # totals given on the command line

The totals must be given in this order:
    personal_assignment_1, personal_assignment_2,
    team_assignment_1, team_assignment_2, team_assignment_3,
    quiz_1, quiz_2, final_exam
"""

import argparse
import sys
from pathlib import Path

# Make sure the modules in this folder can be imported, whatever folder the
# script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from preprocessing import ASSESSMENT_NAMES, FEATURE_COLUMNS, MODEL_FILE


def predict_grade(totals: list[float]) -> str:
    """Load the saved model and return the predicted grade for one student."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_FILE}. Run 'python src/train.py' first."
        )
    model = joblib.load(MODEL_FILE)
    student = pd.DataFrame([totals], columns=FEATURE_COLUMNS)
    return model.predict(student)[0]


def ask_for_totals() -> list[float]:
    """Ask the user for every assessment total."""
    totals = []
    for name in ASSESSMENT_NAMES:
        while True:
            answer = input(f"{name.replace('_', ' ')} total (0-100): ").strip()
            try:
                value = float(answer)
            except ValueError:
                print("Please enter a number.")
                continue
            if not 0 <= value <= 100:
                print("Please enter a number between 0 and 100.")
                continue
            totals.append(value)
            break
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a student's grade category from the assessment totals."
    )
    parser.add_argument(
        "totals",
        nargs="*",
        type=float,
        help="Eight totals in this order: " + ", ".join(ASSESSMENT_NAMES),
    )
    arguments = parser.parse_args()

    if len(arguments.totals) == len(ASSESSMENT_NAMES):
        totals = arguments.totals
    elif not arguments.totals:
        totals = ask_for_totals()
    else:
        print(
            f"Please give {len(ASSESSMENT_NAMES)} totals "
            f"({', '.join(ASSESSMENT_NAMES)}) or none at all."
        )
        sys.exit(1)

    print("\nAssessment totals used:")
    for name, value in zip(ASSESSMENT_NAMES, totals):
        print(f"  {name.replace('_', ' '):24s} {value:g}")

    grade = predict_grade(totals)
    print(f"\nPredicted grade category: {grade}")


if __name__ == "__main__":
    main()
