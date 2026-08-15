import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

@st.cache_resource
def load_model():
    pipeline = joblib.load("churn_model.joblib")
    meta = joblib.load("churn_model_meta.joblib")
    return pipeline, meta

pipeline, meta = load_model()
choices = meta["choices"]

st.title("📉 Customer Churn Predictor")
st.write(
    "Predicts the probability a telecom customer will churn, using a Random Forest model "
    "trained on the top 10 features identified in the churn analysis (Contract type, tenure, "
    "monthly/total charges, and key add-on services)."
)

st.divider()
st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months with company)", min_value=0, max_value=100, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
    contract = st.selectbox("Contract Type", choices["Contract"])
    senior_citizen = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")

with col2:
    online_security = st.selectbox("Online Security", choices["OnlineSecurity"])
    tech_support = st.selectbox("Tech Support", choices["TechSupport"])
    internet_service = st.selectbox("Internet Service", choices["InternetService"])
    payment_method = st.selectbox("Payment Method", choices["PaymentMethod"])
    paperless_billing = st.selectbox("Paperless Billing", choices["PaperlessBilling"])

st.divider()

if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "SeniorCitizen": senior_citizen,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Likely to churn** — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ **Likely to stay** — estimated churn probability: {probability:.1%}")

    st.progress(min(float(probability), 1.0))

    with st.expander("See what was sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model: Random Forest (class_weight='balanced'), trained on the IBM Telco Customer Churn dataset. "
    "Test-set performance: Accuracy 76.4%, Precision 54.0%, Recall 75.9%, F1 0.631, ROC-AUC 0.842."
)
