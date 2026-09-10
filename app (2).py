
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="AI Budget Deviation Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI-Based Financial Budget Deviation Prediction")

st.write(
    "Predict budget deviation and identify financial risk "
    "using Machine Learning and Explainable AI."
)

MODEL_FILE = "budget_deviation_model.pkl"
HISTORY_FILE = "prediction_history.csv"

model = joblib.load(MODEL_FILE)

st.sidebar.header("Budget Information")

department = st.sidebar.selectbox(
    "Department",
    [
        "IT",
        "HR",
        "Finance",
        "Marketing",
        "Operations"
    ]
)

planned_budget = st.sidebar.number_input(
    "Planned Budget",
    min_value=1000.0,
    value=200000.0,
    step=5000.0
)

employee_count = st.sidebar.number_input(
    "Employee Count",
    min_value=1,
    value=100,
    step=1
)

month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

previous_deviation = st.sidebar.number_input(
    "Previous Deviation",
    value=0.0,
    step=5000.0
)

predict_button = st.button(
    "🔮 Predict Budget Deviation"
)

if predict_button:

    input_data = pd.DataFrame({
        "Department": [department],
        "Planned_Budget": [planned_budget],
        "Employee_Count": [employee_count],
        "Month": [month],
        "Previous_Deviation": [previous_deviation]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Deviation",
            f"₹ {prediction:,.2f}"
        )

    with col2:

        if prediction > 50000:
            risk = "🔴 High Risk"
        elif prediction > 10000:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        st.metric(
            "Risk Level",
            risk
        )

    if prediction > 0:
        st.warning(
            "The model predicts potential budget overspending."
        )
    else:
        st.success(
            "The model predicts spending below the planned budget."
        )

    st.subheader("📊 Financial Recommendation")

    if prediction > 50000:
        st.error(
            "High deviation risk detected. "
            "Review departmental expenses and consider "
            "immediate budget controls."
        )
    elif prediction > 10000:
        st.warning(
            "Moderate deviation risk detected. "
            "Monitor expenses closely."
        )
    else:
        st.success(
            "Budget deviation risk appears relatively low."
        )

    history_row = pd.DataFrame({
        "Department": [department],
        "Planned_Budget": [planned_budget],
        "Employee_Count": [employee_count],
        "Month": [month],
        "Previous_Deviation": [previous_deviation],
        "Predicted_Deviation": [prediction],
        "Risk_Level": [risk]
    })

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat(
            [history, history_row],
            ignore_index=True
        )
    else:
        history = history_row

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    st.session_state["history"] = history

st.divider()

st.subheader("📜 Prediction History")

if os.path.exists(HISTORY_FILE):

    history = pd.read_csv(HISTORY_FILE)

    st.dataframe(
        history,
        use_container_width=True
    )

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No prediction history available. "
        "Make a prediction first."
    )

st.divider()

st.caption(
    "AI-Based Financial Budget Deviation Prediction | "
    "Random Forest + SHAP"
)
