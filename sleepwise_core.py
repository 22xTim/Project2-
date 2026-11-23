"""Core logic for SleepWise: data loading, model pipeline, prediction,
explanation, and what-if suggestions. Kept separate from app.py so it can be
tested without Streamlit."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sleep efficiency (time asleep / time in bed) is turned into three categories.
# 85%+ is the commonly cited "good" benchmark.
LABELS = ["Poor", "Fair", "Good"]
POOR_BELOW = 0.70
GOOD_FROM = 0.85

NUMERIC = ["age", "bedtime", "wake_time", "caffeine_mg", "alcohol_oz", "exercise_days"]
CATEGORICAL = ["gender", "smoker"]
FEATURES = NUMERIC + CATEGORICAL

# "bedtime" is stored as hours after 6 PM (so 11:30 PM = 5.5, 1:00 AM = 7.0).
# This avoids the midnight wrap-around problem.
FRIENDLY = {
    "age": "Age", "bedtime": "Bedtime", "wake_time": "Wake-up time",
    "caffeine_mg": "Caffeine", "alcohol_oz": "Alcohol", "exercise_days": "Exercise",
    "gender": "Gender", "smoker": "Smoking",
}


# ---------- formatting helpers ----------
def fmt_clock(clock_hours):
    hh = int(clock_hours) % 24
    mm = int(round((clock_hours - int(clock_hours)) * 60))
    if mm == 60:
        hh, mm = (hh + 1) % 24, 0
    return f"{hh % 12 or 12}:{mm:02d} {'AM' if hh < 12 else 'PM'}"


def fmt_bedtime(hours_after_6pm):
    return fmt_clock((hours_after_6pm + 18) % 24)


def describe_change(feature, old, new):
    if feature == "bedtime":
        return f"Go to bed at {fmt_bedtime(new)} instead of {fmt_bedtime(old)}"
    if feature == "caffeine_mg":
        return f"Lower caffeine from {old:.0f} mg to {new:.0f} mg"
    if feature == "alcohol_oz":
        return f"Lower alcohol from {old:.0f} oz to {new:.0f} oz"
    if feature == "exercise_days":
        return f"Exercise {new:.0f} days/week instead of {old:.0f}"
    return f"Change {FRIENDLY.get(feature, feature)}"


# ---------- data ----------
def efficiency_to_label(eff):
    return pd.cut(eff, bins=[-np.inf, POOR_BELOW, GOOD_FROM, np.inf],
                  labels=LABELS, right=False).astype(str)


def _clock_hours(series):
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.hour + dt.dt.minute / 60


def _clean_text(series):
    out = series.astype(str).str.strip().str.title().astype(object)
    return out.where(out.str.lower() != "nan", np.nan)


def load_dataset(path):
    """Read the Kaggle Sleep Efficiency CSV and return model-ready columns."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    needed = ["age", "gender", "bedtime", "wakeup time", "sleep efficiency",
              "caffeine consumption", "alcohol consumption",
              "smoking status", "exercise frequency"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df = pd.DataFrame({
        "age": pd.to_numeric(raw["age"], errors="coerce"),
        "gender": _clean_text(raw["gender"]),
        "bedtime": (_clock_hours(raw["bedtime"]) - 18) % 24,
        "wake_time": _clock_hours(raw["wakeup time"]),
        "caffeine_mg": pd.to_numeric(raw["caffeine consumption"], errors="coerce"),
        "alcohol_oz": pd.to_numeric(raw["alcohol consumption"], errors="coerce"),
        "smoker": _clean_text(raw["smoking status"]),
        "exercise_days": pd.to_numeric(raw["exercise frequency"], errors="coerce"),
        "efficiency": pd.to_numeric(raw["sleep efficiency"], errors="coerce"),
    })
    df = df.dropna(subset=["efficiency", "bedtime", "wake_time"]).reset_index(drop=True)
    df["quality"] = efficiency_to_label(df["efficiency"])
    return df


# ---------- model ----------
def build_pipeline(kind="forest"):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == "logistic":
        num_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    if kind == "forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                       class_weight="balanced", random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", prep), ("model", model)])


def reference_values(X):
    ref = {c: float(X[c].median()) for c in NUMERIC}
    for c in CATEGORICAL:
        ref[c] = X[c].mode().iloc[0]
    return ref


# ---------- prediction ----------
def predict_proba(bundle, row):
    X = pd.DataFrame([row])[FEATURES]
    probs = bundle["pipeline"].predict_proba(X)[0]
    found = dict(zip(bundle["pipeline"].classes_, probs))
    return {label: float(found.get(label, 0.0)) for label in LABELS}


def predict_label(probs):
    return max(probs, key=probs.get)


def explain(bundle, row):
    """How much does each habit move P(Good) compared with a typical person?
    Each feature is swapped to the dataset's typical value, one at a time.
    Positive = this habit helps relative to typical; negative = it hurts."""
    base = predict_proba(bundle, row)["Good"]
    ref = bundle["reference"]
    out = []
    for f in FEATURES:
        if row[f] == ref[f]:
            continue
        alt = dict(row)
        alt[f] = ref[f]
        out.append((f, base - predict_proba(bundle, alt)["Good"]))
    return sorted(out, key=lambda t: abs(t[1]), reverse=True)


def suggest(bundle, row, min_gain=0.01):
    """Best single change per habit, ranked by gain in P(Good)."""
    base = predict_proba(bundle, row)["Good"]
    candidates = {
        "caffeine_mg": [float(v) for v in range(0, int(row["caffeine_mg"]), 25)],
        "alcohol_oz": [float(v) for v in range(0, int(row["alcohol_oz"]))],
        "exercise_days": [float(v) for v in range(int(row["exercise_days"]) + 1, 6)],
        "bedtime": [row["bedtime"] - d for d in (0.5, 1, 1.5, 2, 3) if row["bedtime"] - d >= 2],
    }
    results = []
    for feature, values in candidates.items():
        best = None
        for v in values:
            alt = dict(row)
            alt[feature] = v
            p = predict_proba(bundle, alt)["Good"]
            if best is None or p > best[1]:
                best = (v, p)
        if best and best[1] - base >= min_gain:
            results.append({"feature": feature, "from": row[feature], "to": best[0],
                            "gain": best[1] - base, "new_p_good": best[1],
                            "text": describe_change(feature, row[feature], best[0])})
    return sorted(results, key=lambda r: r["gain"], reverse=True)
