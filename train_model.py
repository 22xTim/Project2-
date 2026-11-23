"""Train the SleepWise model and save it to model/sleep_model.joblib.

Usage:
    python train_model.py                       # uses data/sleep_efficiency.csv
    python train_model.py --data path/to.csv    # or choose a file"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import sleepwise_core as core


def pick_dataset(arg):
    path = Path(arg) if arg else Path("data/sleep_efficiency.csv")
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}\nDownload the Kaggle 'Sleep Efficiency' "
                         "dataset and save it as data/sleep_efficiency.csv, then re-run.")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    path = pick_dataset(parser.parse_args().data)

    df = core.load_dataset(path)
    X, y = df[core.FEATURES], df["quality"]
    counts = y.value_counts()
    print(f"Loaded {len(df)} rows from {path}\nClass counts:\n{counts.to_string()}\n")

    folds = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    majority = counts.max() / len(y)
    scores = {}
    for kind in ("forest", "logistic"):
        s = cross_val_score(core.build_pipeline(kind), X, y, cv=cv, scoring="accuracy")
        scores[kind] = float(s.mean())
        print(f"{kind:9s} {folds}-fold CV accuracy: {s.mean():.3f} (+/- {s.std():.3f})")
    print(f"Always-predict-majority baseline: {majority:.3f}\n")

    best = max(scores, key=scores.get)
    print(f"Using the {best} model.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    check = core.build_pipeline(best).fit(X_tr, y_tr)
    pred = check.predict(X_te)
    holdout = float((pred == y_te).mean())
    print(f"Hold-out accuracy (20% test split): {holdout:.3f}")
    print(classification_report(y_te, pred, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_te, pred, labels=core.LABELS), "\n")

    final = core.build_pipeline(best).fit(X, y)   # final model uses all rows
    bundle = {
        "pipeline": final, "model_kind": best, "reference": core.reference_values(X),
        "cv_accuracy": scores, "holdout_accuracy": holdout, "majority_baseline": float(majority),
        "n_rows": int(len(df)), "class_counts": counts.to_dict(), "data_source": "kaggle",
    }
    Path("model").mkdir(exist_ok=True)
    joblib.dump(bundle, "model/sleep_model.joblib")
    print("Saved model/sleep_model.joblib")


if __name__ == "__main__":
    main()
