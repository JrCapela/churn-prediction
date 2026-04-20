import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("📈 Model Performance Dashboard")
st.markdown("---")

# Model comparison
st.subheader("🏆 Model Comparison")

models_data = {
    "Model": ["Logistic Regression", "XGBoost", "Random Forest"],
    "AUC-ROC": [0.839, 0.826, 0.818],
    "Color": ["#2ecc71", "#3498db", "#9b59b6"]
}
df_models = pd.DataFrame(models_data)

fig_models = go.Figure(go.Bar(
    x=df_models["AUC-ROC"],
    y=df_models["Model"],
    orientation='h',
    marker_color=df_models["Color"],
    text=df_models["AUC-ROC"],
    textposition='outside'
))
fig_models.update_layout(
    xaxis_range=[0.79, 0.86],
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=250
)
st.plotly_chart(fig_models, use_container_width=True)

st.markdown("---")

# Confusion matrix
st.subheader("🔢 Confusion Matrix — Logistic Regression")

col1, col2 = st.columns(2)

with col1:
    cm_data = [[754, 281], [84, 290]]
    fig_cm = go.Figure(go.Heatmap(
        z=cm_data,
        x=["Predicted No Churn", "Predicted Churn"],
        y=["Actually No Churn", "Actually Churn"],
        colorscale="Blues",
        text=[[str(v) for v in row] for row in cm_data],
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.markdown("### 📊 Key Metrics")
    st.metric("True Positives (Churners caught)", "290", "77% recall")
    st.metric("False Negatives (Missed churners)", "84", "-")
    st.metric("False Positives (False alarms)", "281", "-")
    st.metric("True Negatives (Loyal customers)", "754", "-")

st.markdown("---")

# SHAP feature importance
st.subheader("🔍 Top Features — SHAP Importance")

shap_data = {
    "Feature": [
        "tenure", "InternetService_No", "InternetService_Fiber optic",
        "Contract_Two year", "Contract_Month-to-month",
        "MonthlyCharges", "TechSupport", "PaperlessBilling"
    ],
    "Importance": [0.67, 0.42, 0.42, 0.31, 0.27, 0.18, 0.10, 0.16],
    "Direction": ["Decreases", "Increases", "Increases", "Decreases",
                  "Increases", "Increases", "Decreases", "Increases"]
}
df_shap = pd.DataFrame(shap_data).sort_values("Importance", ascending=True)

colors = ["#e74c3c" if d == "Increases" else "#2ecc71" for d in df_shap["Direction"]]

fig_shap = go.Figure(go.Bar(
    x=df_shap["Importance"],
    y=df_shap["Feature"],
    orientation='h',
    marker_color=colors,
    text=df_shap["Importance"],
    textposition='outside'
))
fig_shap.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=350,
    xaxis_title="Mean |SHAP Value|"
)
st.plotly_chart(fig_shap, use_container_width=True)

st.caption("🔴 Red = increases churn risk | 🟢 Green = decreases churn risk")