
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("loan_prediction_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

# Page settings
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="centered"
)

st.title("Loan Eligibility Predictor")

st.markdown("""
Welcome to the Loan Eligibility Predictor.

Enter an applicant's personal and financial details below
to generate a prediction using our trained machine learning model.
""")

st.info(
    "This application is for educational purposes. "
    "Its predictions are estimates, not actual lending decisions."
)

# Applicant information form
with st.form("loan_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["No", "Yes"])
    dependents = st.selectbox(
        "Number of Dependents",
        ["0", "1", "2", "3+"]
    )
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )
    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0,
        value=2000
    )

    loan_amount = st.number_input(
        "Loan Amount (in dataset units)",
        min_value=0,
        value=120
    )

    loan_term = st.selectbox(
        "Loan Term (months)",
        [360, 180, 120, 60, 36, 12]
    )

    credit_history = st.selectbox(
        "Credit History",
        ["Good (1)", "Poor (0)"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:

    # Convert the answers into the same numerical format
    # used when training your model.
    applicant = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 0 if married == "Yes" else 1,
        "Dependents": 4 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Good (1)" else 0,
        "Property_Area": {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        }[property_area]
    }])

    # Scale the input using the saved scaler
    applicant_scaled = scaler.transform(applicant)

    # Predict the result
    prediction = model.predict(applicant_scaled)[0]
    probabilities = model.predict_proba(applicant_scaled)[0]

    approval_probability = probabilities[1] * 100

    if prediction == 1:
        st.success("Loan Prediction: APPROVED")
    else:
        st.error("Loan Prediction: REJECTED")

    st.metric(
        "Estimated Approval Probability",
        f"{approval_probability:.2f}%"
    )

    st.caption(
        "This is a machine learning estimate for educational purposes, "
        "not an actual bank lending decision."
    )
