"""SleepWise: Streamlit app that predicts sleep quality from daily habits,
explains the prediction, and lets the user explore "what if" changes."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import sleepwise_core as core

MODEL_PATH = Path(__file__).parent / "model" / "sleep_model.joblib"
BEDTIMES = [2 + 0.5 * i for i in range(17)]    # 8:00 PM .. 4:00 AM (hours after 6 PM)
WAKETIMES = [4 + 0.5 * i for i in range(13)]   # 4:00 AM .. 10:00 AM (clock hours)
CAFFEINE = list(range(0, 201, 25))             # mg in the 24 h before bed

st.set_page_config(page_title="SleepWise", page_icon="😴", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train_model.py` in this folder first, "
             "then restart the app.")
    st.stop()
bundle = load_bundle()

st.title("😴 SleepWise")
st.caption("Predict your sleep quality from your habits, see what drives it, "
           "and test what would help.")

# ---------------- 1. INPUT FORM ----------------
with st.sidebar.form("habits"):
    st.header("1. Your habits")
    age = st.slider("Age", 18, 80, 25)
    gender = st.selectbox("Gender", ["Female", "Male"])
    bedtime = st.select_slider("Usual bedtime", options=BEDTIMES, value=5.5,
                               format_func=core.fmt_bedtime)
    wake_time = st.select_slider("Usual wake-up time", options=WAKETIMES, value=7.0,
                                 format_func=core.fmt_clock)
    caffeine = st.select_slider("Caffeine in the 24 h before bed (mg)", options=CAFFEINE,
                                value=50, help="One cup of coffee is roughly 95 mg.")
    alcohol = st.slider("Alcohol in the 24 h before bed (oz)", 0, 5, 1)
    exercise = st.slider("Exercise (days per week)", 0, 5, 2)
    smoker = st.radio("Smoker?", ["No", "Yes"], horizontal=True)
    submitted = st.form_submit_button("Predict sleep quality", type="primary")

if submitted:
    st.session_state["row"] = {
        "age": float(age), "gender": gender, "bedtime": float(bedtime),
        "wake_time": float(wake_time), "caffeine_mg": float(caffeine),
        "alcohol_oz": float(alcohol), "smoker": smoker, "exercise_days": float(exercise),
    }
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

row = st.session_state.get("row")
if row is None:
    st.info("Enter your habits in the sidebar and click **Predict sleep quality**.")
    st.stop()
run_id = st.session_state["run_id"]

# ---------------- 2. PREDICTION RESULTS ----------------
probs = core.predict_proba(bundle, row)
label = core.predict_label(probs)

st.header("2. Prediction results")
left, right = st.columns([1, 2])
with left:
    st.metric("Predicted sleep quality", label)
    st.metric("Chance of Good sleep", f"{probs['Good']:.0%}")
    st.bar_chart(pd.DataFrame({"Probability": [probs[l] for l in core.LABELS]},
                              index=core.LABELS))
with right:
    st.subheader("What affected this prediction")
    effects = core.explain(bundle, row)
    if effects:
        names = [core.FRIENDLY[f] for f, _ in effects][::-1]
        values = [v * 100 for _, v in effects][::-1]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(names, values, color=["#5BB37F" if v >= 0 else "#E8A33D" for v in values])
        ax.axvline(0, color="#33414F", lw=0.8)
        ax.set_xlabel("Change in chance of Good sleep vs. a typical person (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Each habit is swapped to the typical value in the data, one at a time. "
                   "Green bars help your sleep; orange bars hurt it.")
    else:
        st.write("Your habits match the typical values in the data.")

# ---------------- 3. WHAT-IF ANALYSIS ----------------
st.header("3. What if you changed a habit?")
w1, w2 = st.columns([1, 1])
with w1:
    st.write("Adjust the sliders to test a change:")
    new_bed = st.select_slider("Bedtime", options=BEDTIMES, value=row["bedtime"],
                               format_func=core.fmt_bedtime, key=f"wb{run_id}")
    new_caf = st.select_slider("Caffeine (mg)", options=CAFFEINE, value=int(row["caffeine_mg"]),
                               key=f"wc{run_id}")
    new_alc = st.slider("Alcohol (oz)", 0, 5, int(row["alcohol_oz"]), key=f"wa{run_id}")
    new_ex = st.slider("Exercise (days/week)", 0, 5, int(row["exercise_days"]), key=f"we{run_id}")
with w2:
    alt = {**row, "bedtime": float(new_bed), "caffeine_mg": float(new_caf),
           "alcohol_oz": float(new_alc), "exercise_days": float(new_ex)}
    alt_probs = core.predict_proba(bundle, alt)
    alt_label = core.predict_label(alt_probs)
    delta = (alt_probs["Good"] - probs["Good"]) * 100
    st.metric("Predicted sleep quality", alt_label,
              delta=None if alt_label == label else f"was {label}", delta_color="off")
    st.metric("Chance of Good sleep", f"{alt_probs['Good']:.0%}", delta=f"{delta:+.0f} pts")

st.subheader("Biggest single improvements")
tips = core.suggest(bundle, row)
if tips:
    st.dataframe(pd.DataFrame({
        "#": range(1, len(tips) + 1),
        "Suggested change": [t["text"] for t in tips],
        "Chance of Good sleep": [f"{row_p:.0%}" for row_p in [t["new_p_good"] for t in tips]],
        "Gain (pts)": [f"{t['gain'] * 100:+.0f}" for t in tips],
    }), hide_index=True, use_container_width=True)
else:
    st.write("No single habit change is predicted to help noticeably.")

with st.expander("About this model"):
    st.write(f"Model: **{bundle['model_kind']}**, trained on {bundle['n_rows']} records "
             "(Kaggle Sleep Efficiency dataset).")
    st.write(f"Cross-validated accuracy: "
             f"{', '.join(f'{k} {v:.0%}' for k, v in bundle['cv_accuracy'].items())} "
             f"(always guessing the most common class: {bundle['majority_baseline']:.0%}).")
    st.write("Sleep quality categories come from sleep efficiency: Poor < 70%, Fair 70-85%, "
             "Good 85%+. The what-if results show patterns in the data, not proven cause and "
             "effect. This tool is for education and is not medical advice.")
