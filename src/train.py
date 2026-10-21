"""Train the four models, choose the best one, and evaluate it once on the test set.

Steps:
1. Load the cleaned dataset and split it 80/20 (stratified, random_state=42).
2. Compare the four models with cross-validation on the training data only.
3. Pick the best model from those cross-validation scores.
4. Fit the best model on the full training data and evaluate it once on the test set.
5. Save the model, the comparison table and the confusion matrix.
"""

import sys
from pathlib import Path

# Make sure the modules in this folder can be imported, whatever folder the
# script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")  # lets the script save charts without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from preprocessing import (
    GRADE_LABELS,
    MODEL_FILE,
    MODEL_NAMES,
    RANDOM_STATE,
    RESULTS_DIR,
    build_pipeline,
    choose_cv_folds,
    load_dataset,
    split_data,
)


def compare_models(X_train, y_train, folds: int) -> pd.DataFrame:
    """Cross-validate every model on the training data and return one row per model."""
    cross_validation = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(
            f1_score, labels=GRADE_LABELS, average="macro", zero_division=0
        ),
        "weighted_f1": make_scorer(
            f1_score, labels=GRADE_LABELS, average="weighted", zero_division=0
        ),
    }

    rows = []
    for model_name in MODEL_NAMES:
        scores = cross_validate(
            build_pipeline(model_name),
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "Model": model_name,
                "CV Accuracy": np.mean(scores["test_accuracy"]),
                "CV Accuracy Std": np.std(scores["test_accuracy"], ddof=1),
                "CV Macro F1": np.mean(scores["test_macro_f1"]),
                "CV Weighted F1": np.mean(scores["test_weighted_f1"]),
            }
        )

    # The best model is the one with the highest cross-validation accuracy.
    # The macro F1 score is only used to break a tie.
    comparison = pd.DataFrame(rows).sort_values(
        ["CV Accuracy", "CV Macro F1"], ascending=False
    )
    return comparison.reset_index(drop=True)


def save_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    """Save one confusion matrix chart for the selected model."""
    matrix = confusion_matrix(y_true, y_pred, labels=GRADE_LABELS)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(GRADE_LABELS)), GRADE_LABELS)
    axis.set_yticks(range(len(GRADE_LABELS)), GRADE_LABELS)
    axis.set_xlabel("Predicted grade")
    axis.set_ylabel("Actual grade")
    axis.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            colour = "white" if value > matrix.max() / 2 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=colour)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    data = load_dataset()
    X_train, X_test, y_train, y_test = split_data(data)

    print(f"Rows: {len(data)} total -> {len(X_train)} train, {len(X_test)} test")
    print("Training grades:", y_train.value_counts().sort_index().to_dict())
    print("Test grades:    ", y_test.value_counts().sort_index().to_dict())

    folds = choose_cv_folds(y_train)
    smallest_class = int(y_train.value_counts().min())
    if folds < 5:
        print(
            f"Using {folds}-fold cross-validation instead of 5-fold: the rarest grade "
            f"has only {smallest_class} training rows, and stratified folds need at "
            f"least one row of every grade in each fold."
        )

    comparison = compare_models(X_train, y_train, folds)
    print("\nCross-validation results on the training data:")
    print(comparison.to_string(index=False))

    best_row = comparison.iloc[0]
    best_model_name = best_row["Model"]
    print(f"\nSelected model: {best_model_name}")

    # The test set is used only here, after the model has been chosen.
    best_model = build_pipeline(best_model_name)
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)
    test_macro_f1 = f1_score(
        y_test, predictions, labels=GRADE_LABELS, average="macro", zero_division=0
    )
    test_weighted_f1 = f1_score(
        y_test, predictions, labels=GRADE_LABELS, average="weighted", zero_division=0
    )

    comparison["Selected"] = comparison["Model"] == best_model_name
    comparison["Test Accuracy"] = np.where(comparison["Selected"], test_accuracy, np.nan)
    comparison["Test Macro F1"] = np.where(comparison["Selected"], test_macro_f1, np.nan)
    comparison["Test Weighted F1"] = np.where(
        comparison["Selected"], test_weighted_f1, np.nan
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_confusion_matrix(
        y_test,
        predictions,
        f"Confusion matrix - {best_model_name}",
        RESULTS_DIR / "confusion_matrix.png",
    )
    joblib.dump(best_model, MODEL_FILE)

    print("\nTest set results (evaluated once):")
    print(f"  Accuracy      : {test_accuracy:.4f}")
    print(f"  Macro F1      : {test_macro_f1:.4f}")
    print(f"  Weighted F1   : {test_weighted_f1:.4f}")
    print(f"\nSaved model      : {MODEL_FILE}")
    print(f"Saved comparison : {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Saved chart      : {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
