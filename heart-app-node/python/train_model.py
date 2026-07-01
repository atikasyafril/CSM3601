"""
Matches the EXACT preprocessing pipeline from CSM3601_Heart_Disease_Prediction.ipynb Phase 2:
  - pd.get_dummies(drop_first=True) for categorical cols
  - MinMaxScaler on numeric cols only
  - SelectKBest(f_classif, k=10)

Trains 3 models: Logistic Regression, Random Forest, Decision Tree
Exports the Random Forest as a full JSON tree structure so lib/predictor.js
can run it in pure JavaScript with no Python at runtime.
"""

import json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

CSV_PATH = "Heart_disease_cleveland_new.csv"
OUT_PATH = os.path.join("..", "model", "model_artifact.json")

CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'thal']
NUMERIC_COLS     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).replace("null", np.nan)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in NUMERIC_COLS:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    for col in ['chol','trestbps','oldpeak','thalach']:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3-Q1
        df[col] = np.clip(df[col], Q1-1.5*IQR, Q3+1.5*IQR)
    return df

def export_tree(tree, feature_names):
    t = tree.tree_
    def node(i):
        if t.feature[i] == -2:
            vals = t.value[i][0]
            return {"leaf": True, "prediction": int(np.argmax(vals))}
        return {
            "leaf": False,
            "feature": feature_names[t.feature[i]],
            "threshold": float(t.threshold[i]),
            "left":  node(t.children_left[i]),
            "right": node(t.children_right[i]),
        }
    return node(0)

def main():
    print("Loading and cleaning data...")
    df = load_and_clean(CSV_PATH)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    X = df_encoded.drop(columns=['target'])
    y = (df_encoded['target'].astype(int) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_best = selector.fit_transform(X_train_scaled, y_train)
    X_test_best  = selector.transform(X_test_scaled)

    selected_features = list(X_train_scaled.columns[selector.get_support()])
    print("Selected features:", selected_features)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted  = {}

    for name, clf in candidates.items():
        clf.fit(X_train_best, y_train)
        cv_scores = cross_val_score(clf, X_train_best, y_train, cv=cv, scoring="accuracy")
        y_pred  = clf.predict(X_test_best)
        y_proba = clf.predict_proba(X_test_best)[:,1]
        metrics = {
            "model":        name,
            "cv_accuracy":  round(float(cv_scores.mean()),4),
            "test_accuracy":round(float(accuracy_score(y_test,y_pred)),4),
            "precision":    round(float(precision_score(y_test,y_pred)),4),
            "recall":       round(float(recall_score(y_test,y_pred)),4),
            "f1_score":     round(float(f1_score(y_test,y_pred)),4),
            "roc_auc":      round(float(roc_auc_score(y_test,y_proba)),4),
        }
        results.append(metrics)
        fitted[name] = clf
        print(name, metrics)

    results_sorted = sorted(results, key=lambda r: r["roc_auc"], reverse=True)

    rf = fitted["Random Forest"]
    trees_json = [export_tree(est, selected_features) for est in rf.estimators_]

    scaler_stats = {}
    for i, col in enumerate(NUMERIC_COLS):
        scaler_stats[col] = {
            "data_min": float(scaler.data_min_[i]),
            "data_max": float(scaler.data_max_[i]),
        }

    artifact = {
        "model_name":         "Random Forest",
        "numeric_cols":       NUMERIC_COLS,
        "categorical_cols":   CATEGORICAL_COLS,
        "selected_features":  selected_features,
        "scaler_stats":       scaler_stats,
        "rf_trees":           trees_json,
        "metrics_all_models": results_sorted,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}  ({len(trees_json)} trees)")

if __name__ == "__main__":
    main()
