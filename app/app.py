"""Streamlit dashboard for student risk triage and supportive interventions."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "student_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

st.set_page_config(page_title="Student Support Radar", page_icon="🎓", layout="wide")


def load_assets():
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Run `python data_generation.py` and `python -m src.train` before opening the dashboard.")
        st.stop()
    return pd.read_csv(DATA_PATH), joblib.load(MODEL_PATH)


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    return "Low"


def factor_labels(student: pd.Series) -> list[tuple[str, str]]:
    factors = []
    if student["attendance_rate"] < 75:
        factors.append(("Attendance", "Offer an attendance check-in and remove transport or scheduling barriers."))
    if student["avg_grade"] < 60:
        factors.append(("Academic performance", "Connect the student with tutoring and a short, achievable study plan."))
    if student["assignment_submission_rate"] < 0.65 or student["lms_logins_per_week"] < 4:
        factors.append(("Course engagement", "Use a weekly mentor check-in and help the student plan missed work."))
    if student["previous_failures"] > 1:
        factors.append(("Previous failures", "Create an academic recovery plan with an advisor and course instructor."))
    if student["financial_aid"] == 0 or student["socioeconomic_status"] == "Low":
        factors.append(("Financial context", "Offer a confidential financial-aid and emergency-support referral."))
    if student["disciplinary_incidents"] >= 2:
        factors.append(("Wellbeing and belonging", "Prioritize restorative counseling and a trusted-adult connection."))
    if not factors:
        factors.append(("Protective profile", "Maintain positive feedback, belonging, and regular light-touch check-ins."))
    return factors


def overview(data: pd.DataFrame) -> None:
    st.title("Student Support Radar")
    st.caption("A decision-support view for earlier, more compassionate outreach.")
    high_risk = data["dropout"].mean()
    cols = st.columns(4)
    cols[0].metric("Students observed", f"{len(data):,}")
    cols[1].metric("Observed dropout rate", f"{high_risk:.1%}")
    cols[2].metric("Average attendance", f"{data['attendance_rate'].mean():.1f}%")
    cols[3].metric("Average grade", f"{data['avg_grade'].mean():.1f}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(data, x="attendance_rate", color="dropout", nbins=25, barmode="overlay", title="Attendance distribution"), use_container_width=True)
    with right:
        st.plotly_chart(px.box(data, x="socioeconomic_status", y="avg_grade", color="dropout", title="Grades by socioeconomic status"), use_container_width=True)
    st.subheader("Equity monitoring")
    fairness = data.groupby(["gender", "socioeconomic_status"], as_index=False)["dropout"].mean().rename(columns={"dropout": "dropout_rate"})
    st.dataframe(fairness.style.format({"dropout_rate": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.info("Ethical use: this model is for support and resource allocation, never punishment or automatic high-stakes decisions. Review outcomes for fairness across gender and socioeconomic groups.")


def risk_explorer(data: pd.DataFrame, bundle: dict) -> None:
    st.title("Student Risk Explorer")
    st.caption("Adjust the profile to estimate risk and surface actionable signals.")
    values = {}
    left, right = st.columns(2)
    with left:
        values["age"] = st.slider("Age", 17, 30, 20)
        values["gender"] = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
        values["socioeconomic_status"] = st.selectbox("Socioeconomic status", ["Low", "Medium", "High"])
        values["attendance_rate"] = st.slider("Attendance rate", 0, 100, 82)
        values["avg_grade"] = st.slider("Average grade", 0, 100, 70)
        values["assignment_submission_rate"] = st.slider("Assignment submission rate", 0.0, 1.0, 0.75, 0.01)
    with right:
        values["lms_logins_per_week"] = st.slider("LMS logins per week", 0, 25, 8)
        values["participation_score"] = st.slider("Participation score", 0.0, 10.0, 6.0, 0.1)
        values["disciplinary_incidents"] = st.slider("Disciplinary incidents", 0, 8, 0)
        values["financial_aid"] = st.checkbox("Receiving financial aid", value=True)
        values["first_generation"] = st.checkbox("First-generation student", value=False)
        values["distance_from_school_km"] = st.slider("Distance from school (km)", 0.0, 35.0, 6.0, 0.5)
        values["previous_failures"] = st.slider("Previous course failures", 0, 6, 0)
    values["financial_aid"] = int(values["financial_aid"])
    values["first_generation"] = int(values["first_generation"])
    student = pd.DataFrame([values], columns=FEATURES)
    probability = float(bundle["model"].predict_proba(student)[0, 1])
    band = risk_band(probability)
    st.divider()
    col1, col2 = st.columns([1, 2])
    col1.metric("Estimated dropout probability", f"{probability:.1%}")
    col1.metric("Risk band", band)
    with col2:
        factors = factor_labels(student.iloc[0])
        st.subheader("Top risk signals and supports")
        for name, action in factors:
            st.markdown(f"**{name}**  \\n{action}")
    st.caption("This estimate reflects patterns in the training data, not a student's ability or worth. Confirm context with the student.")


def interventions(data: pd.DataFrame) -> None:
    st.title("Intervention Recommendations")
    st.caption("Translate common risk signals into coordinated, human-led support.")
    rules = pd.DataFrame(
        [
            ["Attendance below 75%", "Mentoring and attendance check-in", "Explore transport, health, work, or schedule barriers."],
            ["Average grade below 60", "Tutoring and academic recovery", "Set one-week goals and connect to subject support."],
            ["Low LMS or assignment engagement", "Course engagement outreach", "Plan missed work with an instructor or mentor."],
            ["Financial aid not received / low SES", "Financial aid referral", "Offer confidential aid, food, housing, or emergency support."],
            ["Two or more previous failures", "Advisor case review", "Create a realistic course load and recovery plan."],
            ["Two or more disciplinary incidents", "Counseling and restorative support", "Prioritize belonging, wellbeing, and a trusted adult."],
        ],
        columns=["Signal", "Recommended action", "First conversation"],
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)
    st.subheader("Students with observed target risk signals")
    flagged = data.assign(
        risk_signal=(data["attendance_rate"] < 75) | (data["avg_grade"] < 60) | (data["previous_failures"] > 1)
    ).query("risk_signal")[FEATURES].head(25)
    st.dataframe(flagged, use_container_width=True, hide_index=True)
    st.warning("Recommendations require consent, context, and professional judgment. Do not use this tool to penalize, exclude, or label students permanently.")


def main() -> None:
    data, bundle = load_assets()
    page = st.sidebar.radio("Dashboard", ["Overview", "Student Risk Explorer", "Intervention Recommendations"])
    if page == "Overview":
        overview(data)
    elif page == "Student Risk Explorer":
        risk_explorer(data, bundle)
    else:
        interventions(data)


if __name__ == "__main__":
    main()
