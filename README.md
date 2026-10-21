# Student Learning-Outcome Classification Using Machine Learning

## About the Project
This project uses machine learning to predict student grade categories (A, B, C, D) from eight assessment totals. Four classification algorithms are compared to identify the best-performing model.

## Dataset
- **Source:** [Zenodo dataset](https://doi.org/10.5281/zenodo.19871067)
- **Available records:** 171 students
- **Features:** Eight assessment totals covering assignments, quizzes, and final examination.
- **Target:** `grading_category` (A, B, C, D)
- **Preprocessing:** Student IDs, record numbers, and `final_score` are excluded to prevent data leakage.

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree

## Results

| Model | Cross-Validation Accuracy |
|---|---:|
| Logistic Regression | 92.64% |
| KNN | 82.32% |
| Random Forest | 80.14% |
| Decision Tree | 71.27% |

**Best model:** Logistic Regression

- Test accuracy: **91.43%**
- Test weighted F1-score: **91.12%**
- Test macro F1-score: **94.68%**

The test set contains 35 records and was evaluated once. Results should be interpreted cautiously because the dataset is small and some grade categories have very few examples.

## Project Outputs
- `models/best_model.joblib` — trained model
- `results/model_comparison.csv` — cross-validation results
- `results/confusion_matrix.png` — test-set confusion matrix

## Technologies
Python, pandas, NumPy, scikit-learn, Joblib, Matplotlib.

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python src/train.py
```

Predict a student's grade:

```bash
python src/predict.py
```

## Limitation
The model predicts grade categories from assessment scores. It does not independently measure a student's complete learning competency. The available dataset contains 171 records, while the reference paper reports 513 students.