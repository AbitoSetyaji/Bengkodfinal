import streamlit as st
import joblib
import pandas as pd

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("model_churn.pkl")

model = load_model()

st.title("📊 Telco Churn Predictor (Versi Final)")

# Buat Form dengan 19 Fitur agar tidak error "columns are missing"
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", 0, 100, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

if st.button("Prediksi Churn"):
    # Buat DataFrame dengan 19 kolom lengkap
    input_df = pd.DataFrame({
        'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner],
        'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone],
        'MultipleLines': [lines], 'InternetService': [internet], 'OnlineSecurity': [security],
        'OnlineBackup': [backup], 'DeviceProtection': [protection], 'TechSupport': [support],
        'StreamingTV': [tv], 'StreamingMovies': [movies], 'Contract': [contract],
        'PaperlessBilling': [paperless], 'PaymentMethod': [payment],
        'MonthlyCharges': [monthly], 'TotalCharges': [total]
    })
    
    res = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {'CHURN' if res == 1 or res == 'Yes' else 'TETAP'}")
