import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customers", page_icon="👥", layout="wide")

st.title("👥 Batch Customer Analysis")
st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")
st.markdown("---")

# Sample data download
sample_data = {
    "gender": ["Male", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0],
    "Partner": ["Yes", "No", "Yes"],
    "Dependents": ["No", "No", "Yes"],
    "tenure": [12, 3, 48],
    "PhoneService": ["Yes", "Yes", "Yes"],
    "MultipleLines": ["No", "No", "Yes"],
    "InternetService": ["Fiber optic", "Fiber optic", "DSL"],
    "OnlineSecurity": ["No", "No", "Yes"],
    "OnlineBackup": ["No", "No", "Yes"],
    "DeviceProtection": ["No", "No", "Yes"],
    "TechSupport": ["No", "No", "Yes"],
    "StreamingTV": ["Yes", "Yes", "No"],
    "StreamingMovies": ["Yes", "Yes", "No"],
    "Contract": ["Month-to-month", "Month-to-month", "Two year"],
    "PaperlessBilling": ["Yes", "Yes", "No"],
    "PaymentMethod": ["Electronic check", "Electronic check", "Bank transfer (automatic)"],
    "MonthlyCharges": [85.0, 90.0, 45.0]
}
df_sample = pd.DataFrame(sample_data)

st.download_button(
    label="📥 Download Sample CSV",
    data=df_sample.to_csv(index=False),
    file_name="sample_customers.csv",
    mime="text/csv"
)

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(f"📋 Loaded {len(df)} customers")
    st.dataframe(df, use_container_width=True)

    if st.button("🔮 Predict Churn for All Customers", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, row in df.iterrows():
            customer = row.to_dict()
            try:
                response = requests.post("http://localhost:8000/predict", json=customer)
                result = response.json()
                results.append({
                    "Customer #": i + 1,
                    "Churn Probability": f"{result['churn_probability']:.1%}",
                    "Risk Level": result['risk_level'],
                    "Prediction": "⚠️ Will Churn" if result['churn_prediction'] else "✅ Will Stay",
                    "Action": result['message']
                })
            except Exception as e:
                results.append({
                    "Customer #": i + 1,
                    "Churn Probability": "Error",
                    "Risk Level": "Error",
                    "Prediction": "Error",
                    "Action": str(e)
                })

            progress.progress((i + 1) / len(df))
            status.text(f"Processing customer {i + 1} of {len(df)}...")

        status.empty()
        progress.empty()

        df_results = pd.DataFrame(results)

        st.markdown("---")
        st.subheader("📊 Results")

        # Summary metrics
        total = len(df_results)
        high_risk = len(df_results[df_results['Risk Level'] == 'High'])
        medium_risk = len(df_results[df_results['Risk Level'] == 'Medium'])
        low_risk = len(df_results[df_results['Risk Level'] == 'Low'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", total)
        col2.metric("🔴 High Risk", high_risk)
        col3.metric("🟡 Medium Risk", medium_risk)
        col4.metric("🟢 Low Risk", low_risk)

        # Pie chart
        fig_pie = px.pie(
            values=[high_risk, medium_risk, low_risk],
            names=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"],
            title="Risk Distribution"
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Results table
        st.dataframe(
            df_results.style.apply(
                lambda x: ['background-color: rgba(231,76,60,0.3)' if v == 'High'
                          else 'background-color: rgba(243,156,18,0.3)' if v == 'Medium'
                          else 'background-color: rgba(46,204,113,0.3)' if v == 'Low'
                          else '' for v in x],
                subset=['Risk Level']
            ),
            use_container_width=True
        )

        # Download results
        st.download_button(
            label="📥 Download Results CSV",
            data=df_results.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )