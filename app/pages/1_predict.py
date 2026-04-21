import streamlit as st
import requests
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "churn_secret_key_2024")

st.set_page_config(page_title="Predict Churn", page_icon="🔮", layout="wide")

st.title("🔮 Predict Customer Churn")
st.markdown("Fill in the customer details below to predict churn probability.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📱 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0)

st.markdown("---")

if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            headers={"X-API-Key": API_KEY}
        )
        result = response.json()

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            probability = result['churn_probability']
            risk = result['risk_level']

            color = "#2ecc71" if risk == "Low" else "#f39c12" if risk == "Medium" else "#e74c3c"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(243, 156, 18, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(231, 76, 60, 0.2)"},
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col_result2:
            st.markdown(f"### Risk Level: :{('green' if risk == 'Low' else 'orange' if risk == 'Medium' else 'red')}[{risk}]")
            st.markdown(f"**Churn Probability:** `{probability:.1%}`")
            st.markdown(f"**Prediction:** {'⚠️ Will Churn' if result['churn_prediction'] else '✅ Will Stay'}")
            st.info(result['message'])

            if risk == "High":
                st.error("🚨 Immediate retention action recommended!")
                st.markdown("""
                **Suggested actions:**
                - 📞 Call the customer
                - 💰 Offer a discount
                - 📧 Send personalized email
                - 🎁 Offer contract upgrade
                """)
            elif risk == "Medium":
                st.warning("⚠️ Monitor this customer closely.")
                st.markdown("""
                **Suggested actions:**
                - 📧 Send satisfaction survey
                - 💡 Highlight unused services
                """)
            else:
                st.success("✅ Customer is loyal. No action needed.")

    except Exception as e:
        st.error(f"❌ API Error: {e}. Make sure the API is running at http://localhost:8000")