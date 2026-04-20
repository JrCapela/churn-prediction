import streamlit as st

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Churn Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🎯 Model AUC-ROC",
        value="0.839",
        delta="Best: Logistic Regression"
    )

with col2:
    st.metric(
        label="👥 Training Customers",
        value="7,043",
        delta="26.5% churn rate"
    )

with col3:
    st.metric(
        label="✅ Churners Caught",
        value="77%",
        delta="290 out of 374"
    )

st.markdown("---")

st.markdown("""
### 🚀 Navigation

Use the sidebar to navigate between pages:

- **🔮 Predict** — Predict churn for a single customer
- **📈 Dashboard** — Model metrics and performance
- **👥 Customers** — Batch prediction for multiple customers
""")

st.markdown("---")
st.markdown("""
### 📋 About this project

This dashboard uses a **Logistic Regression** model trained on the 
Telco Customer Churn dataset to predict which customers are likely 
to cancel their subscription.

**Tech Stack:** Python · Scikit-learn · FastAPI · Streamlit · Docker · GCS
""")