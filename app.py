"""Streamlit app: churn probability for a customer profile.

Run `make train` first to fit the model and save models/churn_model.joblib,
then `make app` (or `streamlit run app.py`).

The factor list under the score is read from the trained model bundle:
global permutation importance ranks the drivers, and for each driver that
has a segment view, the app shows the churn rate of the segment this
profile falls into, computed from the full dataset.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn.features import ALL_FEATURES, add_features, load_data

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"

SEGMENT_LABELS = {
    "Contract": "Contract",
    "InternetService": "Internet service",
    "TechSupport": "Tech support",
    "OnlineSecurity": "Online security",
    "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing",
}


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


def tenure_band(tenure: int) -> str:
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "13-24"
    if tenure <= 48:
        return "25-48"
    return "49-72"


def main() -> None:
    st.set_page_config(page_title="Telco Churn Explorer", layout="centered")
    st.title("Telco churn: profile risk check")
    st.write(
        "Set a customer profile and get the churn probability from the "
        "trained model, with the factors that matter most for profiles "
        "like it. Model and segment rates come from the IBM Telco sample "
        "dataset (7,043 customers)."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Run `make data` then `make train` first."
        )
        return

    bundle = load_bundle()
    ref = load_reference_data()
    overall = float(bundle["overall_churn_rate"])

    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox(
            "Internet service", ["Fiber optic", "DSL", "No"]
        )
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    with col2:
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        online_security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])

    total = round(monthly * max(tenure, 1), 2)
    profile = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "No",
                "Dependents": "No",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": tech_support,
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )
    profile = add_features(profile)[ALL_FEATURES]
    prob = float(bundle["pipeline"].predict_proba(profile)[:, 1][0])

    st.subheader("Result")
    st.metric("Churn probability", f"{prob:.1%}")
    st.write(f"Dataset average churn rate: {overall:.1%}.")
    if prob >= 0.5:
        st.write("Band: high risk at the default 0.50 threshold.")
    elif prob >= 0.3:
        st.write("Band: watch zone. Flagged at a 0.30 retention threshold.")
    else:
        st.write("Band: low risk at either threshold.")

    st.subheader("Top factors for this profile")
    st.write(
        "Drivers are ranked by permutation importance on the held-out test "
        "set (drop in ROC AUC when the column is shuffled). Segment rates "
        "are the raw churn rates in the full dataset."
    )
    segment_rates = bundle["segment_rates"]
    drivers = bundle["drivers"]
    shown = 0
    profile_values = {
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "tenure_band": tenure_band(tenure),
    }
    for _, row in drivers.iterrows():
        feature = row["feature"]
        key = "tenure_band" if feature == "tenure" else feature
        if key not in profile_values or key not in segment_rates:
            continue
        value = profile_values[key]
        rates = pd.DataFrame(segment_rates[key])
        match = rates[rates[key].astype(str) == str(value)]
        if match.empty:
            continue
        rate = float(match.iloc[0]["churn_rate"])
        label = "Tenure band" if key == "tenure_band" else SEGMENT_LABELS.get(key, key)
        st.write(
            f"- {label}: {value}. Customers in this segment churn at "
            f"{rate:.1%}, versus {overall:.1%} overall."
        )
        shown += 1
        if shown >= 4:
            break

    churned = ref[ref["Churn"] == 1]
    stayed = ref[ref["Churn"] == 0]
    st.write(
        f"For context, median tenure is {int(churned['tenure'].median())} months "
        f"for customers who churned and {int(stayed['tenure'].median())} months "
        f"for customers who stayed."
    )
    st.caption(
        f"Model: {bundle['model_name']}. Sample data from IBM; results are "
        "an analysis exercise, not a production retention system."
    )


if __name__ == "__main__":
    main()
