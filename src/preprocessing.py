"""Loading and preprocessing helpers for the student grade classification project.

Only the eight assessment totals are used as model input.
`final_score` is excluded because the grade label is calculated from it, and the
student id / record number are excluded because they are identifiers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "student_assessment_clean.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_model.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Settings ----------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_CV_FOLDS = 5

# Fixed grade order used for every metric and every chart.
GRADE_LABELS = ["A", "B", "C", "D"]

# The eight assessments, in the order used by the prediction script.
ASSESSMENT_NAMES = [
    "personal_assignment_1",
    "personal_assignment_2",
    "team_assignment_1",
    "team_assignment_2",
    "team_assignment_3",
    "quiz_1",
    "quiz_2",
    "final_exam",
]
FEATURE_COLUMNS = [f"{name}_total" for name in ASSESSMENT_NAMES]

TARGET_COLUMN = "grading_category"

# Never used as model input: the first leaks the label, the other two are ids.
EXCLUDED_COLUMNS = ["final_score", "student_id", "record_number"]

# The four models that are compared in this project.
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest"]


def load_dataset(path=DATA_FILE) -> pd.DataFrame:
    """Read the cleaned dataset and check that the needed columns are present."""
    data = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing these columns: {missing}")
    return data


def split_data(data: pd.DataFrame):
    """Split the data 80/20, keeping the grade distribution the same in both parts."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def choose_cv_folds(y: pd.Series, max_folds: int = MAX_CV_FOLDS) -> int:
    """Pick the largest number of stratified folds the rarest grade can support.

    Stratified cross-validation needs at least one training sample of the
    rarest grade in every fold, so the number of folds can never be larger
    than the number of samples of the smallest class.
    """
    smallest_class = int(y.value_counts().min())
    folds = min(max_folds, smallest_class)
    if folds < 2:
        raise ValueError(
            "Not enough samples of the rarest grade to run cross-validation."
        )
    return folds


def build_pipeline(model_name: str) -> Pipeline:
    """Return the pipeline for one model.

    Every pipeline keeps preprocessing inside it, so the scaler is fitted on
    the training data only and never sees the test data.
    """
    if model_name == "Logistic Regression":
        steps = [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    elif model_name == "Decision Tree":
        steps = [("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]
    elif model_name == "KNN":
        steps = [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    elif model_name == "Random Forest":
        steps = [
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return Pipeline(steps)
