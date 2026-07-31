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
 
# Styling

st.markdown(
    """
    <style>
    :root {
        --card-bg: #ffffff;
        --card-border: #e1ecf3;
        --text-primary: #12232f;
        --text-secondary: #56707f;
        --accent: #0b6e8f;
        --accent-dark: #094f68;
        --accent-soft: #e4f3f8;
        --success-bg: #eafaf1;
        --success-border: #a9dfbf;
        --success-text: #1e6b3c;
        --danger-bg: #fdecea;
        --danger-border: #f5b7b1;
        --danger-text: #7a1f13;
        --shadow: 0 4px 18px rgba(11, 61, 92, 0.08);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: #1a232b;
            --card-border: #2c3944;
            --text-primary: #eaf2f6;
            --text-secondary: #9fb2bd;
            --accent: #3fc1e3;
            --accent-dark: #2ea1c0;
            --accent-soft: #16323d;
            --success-bg: #123322;
            --success-border: #2f6b45;
            --success-text: #8fe0ac;
            --danger-bg: #3a1613;
            --danger-border: #7a352c;
            --danger-text: #f5a89c;
            --shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        }
    }
 
    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, var(--accent) 0%, var(--accent-dark) 100%);
        padding: 1.8rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: var(--shadow);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #eaf6fb;
        margin-top: 0.4rem;
        max-width: 720px;
    }
 
    /* Section cards */
    .card {
        background-color: var(--card-bg);
        color: var(--text-primary);
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: var(--shadow);
        border: 1px solid var(--card-border);
        margin-bottom: 1rem;
    }
    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }
    .section-caption {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }
 
    /* Result banners */
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin-top: 1rem;
        font-size: 1.05rem;
        box-shadow: var(--shadow);
    }
    .result-positive {
        background-color: var(--danger-bg);
        border: 1px solid var(--danger-border);
        color: var(--danger-text);
    }
    .result-negative {
        background-color: var(--success-bg);
        border: 1px solid var(--success-border);
        color: var(--success-text);
    }
 
    .footer-note {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 2rem;
        text-align: center;
    }
 
    /* Buttons */
    div.stButton > button {
        background-color: var(--accent);
        color: #ffffff;
        border-radius: 10px;
        padding: 0.65rem 1.6rem;
        font-weight: 700;
        border: none;
        box-shadow: var(--shadow);
        transition: background-color 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: var(--accent-dark);
        color: #ffffff;
    }
 
    /* Progress bar accent */
    div[data-testid="stProgress"] > div > div {
        background-color: var(--accent) !important;
    }
 
    /* Symptom toggle rows */
    .symptom-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.35rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
                    <strong>⚠️ Elevated Risk Detected</strong><br>
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