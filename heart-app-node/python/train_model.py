"""
train_model.py
----------------
ONE-TIME script (run with Python) that trains and compares 3 ML models on the
Cleveland Heart Disease dataset, using the same cleaning / outlier-capping /
encoding / scaling / feature-selection steps used in
CSM3601_Heart_Disease_Prediction.ipynb (Phase 2).

It then exports the WINNING model (Logistic Regression, chosen for its
transparent math) as a plain JSON file: ../model/model_artifact.json

The Node.js / Express app reads that JSON and re-implements the exact same
scaling + one-hot-encoding + logistic-regression math in plain JavaScript,
so the running website needs ZERO Python at request time.

Run once (or whenever you retrain):
    cd python
    pip install -r requirements.txt
    python train_model.py
"""

import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

CSV_PATH = "Heart_disease_cleveland_new.csv"
OUT_PATH = os.path.join("..", "model", "model_artifact.json")

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
PASSTHROUGH_COLS = ["sex", "fbs", "exang", "ca"]
OUTLIER_COLS = ["chol", "trestbps", "oldpeak", "thalach"]
RAW_FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).replace("null", np.nan)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in NUMERIC_COLS:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    for col in OUTLIER_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

    return df


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(CSV_PATH)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)

    X = df[RAW_FEATURE_ORDER]
    y = (df["target"].astype(int) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
        ]
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("select", SelectKBest(score_func=f_classif, k=10)),
            ("model", clf),
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "cv_accuracy": round(float(cv_scores.mean()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        results.append(metrics)
        fitted[name] = pipe
        print(name, metrics)

    results_sorted = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    print("\n=== Comparison (sorted by ROC-AUC) ===")
    for r in results_sorted:
        print(r)

    # --- Deploy Logistic Regression specifically: its math (scaling + one-hot +
    # linear coefficients + sigmoid) can be re-implemented exactly in plain
    # JavaScript with no extra libraries, so the Node.js site needs no Python
    # at runtime. ---
    deploy_name = "Logistic Regression"
    pipe = fitted[deploy_name]
    pre = pipe.named_steps["preprocessor"]
    selector = pipe.named_steps["select"]
    clf = pipe.named_steps["model"]

    scaler = pre.named_transformers_["num"]
    encoder = pre.named_transformers_["cat"]

    # Full encoded feature names IN ORDER: num cols, then one-hot cat cols, then passthrough
    encoded_names = list(NUMERIC_COLS)
    cat_categories = []
    for col_idx, col in enumerate(CATEGORICAL_COLS):
        cats = list(encoder.categories_[col_idx])
        kept_cats = cats[1:]  # drop="first"
        cat_categories.append({"column": col, "all_categories": cats, "kept_categories": kept_cats})
        for c in kept_cats:
            encoded_names.append(f"{col}_{c}")
    encoded_names.extend(PASSTHROUGH_COLS)

    support_mask = selector.get_support()
    selected_names = [n for n, keep in zip(encoded_names, support_mask) if keep]

    artifact = {
        "model_name": deploy_name,
        "feature_order_raw": RAW_FEATURE_ORDER,
        "numeric_cols": NUMERIC_COLS,
        "scaler": {
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
        },
        "categorical_cols": CATEGORICAL_COLS,
        "categories": cat_categories,
        "passthrough_cols": PASSTHROUGH_COLS,
        "encoded_feature_names": encoded_names,
        "selected_feature_names": selected_names,
        "selected_mask": support_mask.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "metrics_all_models": results_sorted,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nSaved model artifact -> {OUT_PATH}")
    print(f"Selected features used by deployed model: {selected_names}")


if __name__ == "__main__":
    main()
