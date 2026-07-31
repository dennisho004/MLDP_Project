import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration

st.set_page_config(
    page_title="Early-Stage Diabetes Risk Screener",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
 

# Load trained model
MODEL_PATH = "best_rf_model.pkl"
 
@st.cache_resource
def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_columns"], bundle["threshold"]
 
model = None
feature_columns = None
threshold = 0.2
 
try:
    model, feature_columns, threshold = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️ Could not find the model file '{MODEL_PATH}'. "
        "Please make sure it is saved in the same folder as this app before running it again."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Something went wrong while loading the model: {e}")
    st.stop()
 
# Header

st.markdown('<div class="main-title">🩺 Early-Stage Diabetes Risk Screener</div>', unsafe_allow_html=True)
 
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This screener uses a Random Forest model trained on early-stage diabetes symptom data "
        "to estimate the likelihood of a positive diabetes risk classification."
    )
    st.markdown("---")
    st.write("**How to use it:**")
    st.write("1. Fill in the demographic details.\n2. Answer each symptom question.\n3. Click **Predict Risk**.")
 
st.write("")
 

# Input form

symptom_labels = {
    "Polyuria": "Frequent urination (Polyuria)",
    "Polydipsia": "Excessive thirst (Polydipsia)",
    "sudden weight loss": "Sudden weight loss",
    "Polyphagia": "Excessive hunger (Polyphagia)",
    "visual blurring": "Blurred vision",
    "Irritability": "Irritability",
    "partial paresis": "Partial paresis (muscle weakness)",
    "Alopecia": "Alopecia (hair loss)",
}
 
with st.form("risk_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Demographic Details")
 
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
            help="Enter an age between 1 and 120.",
        )
    with col2:
        gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
 
    st.markdown("---")
    st.subheader("Reported Symptoms")
    st.caption("Select **Yes** or **No** for each symptom.")
 
    responses = {}
    symptom_items = list(symptom_labels.items())
    left_col, right_col = st.columns(2)
    for i, (key, label) in enumerate(symptom_items):
        target_col = left_col if i % 2 == 0 else right_col
        with target_col:
            responses[key] = st.radio(label, options=["No", "Yes"], horizontal=True, key=f"symptom_{key}")
 
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    submitted = st.form_submit_button("🔍 Predict Risk")
 
# Prediction + result display

if submitted:
    # Input validation
    errors = []
    if age is None or age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years.")
    if gender not in ["Female", "Male"]:
        errors.append("Please select a valid gender.")
    for key in symptom_labels:
        if responses.get(key) not in ["Yes", "No"]:
            errors.append(f"Please answer the '{symptom_labels[key]}' question.")
 
    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    else:
        try:
            # Build the Age_Group bucket exactly as done during training
            if age <= 30:
                age_group = "Youth"
            elif age <= 60:
                age_group = "Adult"
            else:
                age_group = "Elderly"
 
            raw_input = {
                "Gender": gender,
                "Age_Group": age_group,
            }
            for key in symptom_labels:
                raw_input[key] = responses[key]
 
            df_input = pd.DataFrame([raw_input])
 
            # One-hot encode to match training pipeline, then align to model's expected columns
            df_input = pd.get_dummies(df_input, columns=["Gender", "Age_Group"] + list(symptom_labels.keys()))
            df_input = df_input.reindex(columns=feature_columns, fill_value=0)
 
            proba_positive = model.predict_proba(df_input)[0][list(model.classes_).index("Positive")]
            is_positive = proba_positive >= threshold
 
            st.write("")
            if is_positive:
                st.markdown(
                    f"""
                    <div class="result-box" style="background-color:#fdecea; border:1px solid #f5b7b1; color:#7a1f13;">
                    <strong>st⚠️ Elevated Risk Detected</strong><br>
                    Estimated likelihood of a positive diabetes risk indication: <strong>{proba_positive*100:.1f}%</strong><br>
                    Please consider consulting a healthcare professional for a proper clinical assessment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box" style="background-color:#eafaf1; border:1px solid #a9dfbf; color:#1e6b3c;">
                    <strong>✅ Lower Risk Indicated</strong><br>
                    Estimated likelihood of a positive diabetes risk indication: <strong>{proba_positive*100:.1f}%</strong><br>
                    Continue maintaining healthy habits and routine check-ups.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
 
            st.progress(min(max(proba_positive, 0.0), 1.0))
 
            with st.expander("See a breakdown of what was submitted"):
                st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
 
        except Exception as e:
            st.error(f"⚠️ We couldn't generate a prediction due to an unexpected error: {e}")