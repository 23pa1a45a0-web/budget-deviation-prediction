
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="BudgetGuard",
    page_icon="💰",
    layout="wide"
)

# Load model
model = joblib.load("model.joblib")

# Load training columns
model_columns = joblib.load("model_columns.joblib")


# Risk classification
def classify_risk(deviation_percentage):

    if deviation_percentage > 10:
        return "High Risk"

    elif deviation_percentage > 5:
        return "Medium Risk"

    elif deviation_percentage >= -5:
        return "Low Risk"

    else:
        return "Underspending"


# Custom dark premium styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f0f14 0%,
        #171323 50%,
        #0d1117 100%
    );
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #a78bfa,
        #60a5fa
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #a1a1aa;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: rgba(30, 30, 45, 0.85);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">💰 BudgetGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Financial Budget Deviation Prediction'
    '</div>',
    unsafe_allow_html=True
)


# Input section
st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📊 Enter Budget Information")

col1, col2 = st.columns(2)

with col1:

    department = st.selectbox(
        "Department",
        [
            "HR",
            "IT",
            "Finance",
            "Sales",
            "Marketing",
            "Operations"
        ]
    )

    budget = st.number_input(
        "Budget (₹)",
        min_value=1.0,
        value=200000.0,
        step=1000.0
    )

    employees = st.number_input(
        "Number of Employees",
        min_value=1,
        value=50,
        step=1
    )


with col2:

    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=5
    )

    previous_deviation = st.number_input(
        "Previous Deviation (₹)",
        value=0.0,
        step=1000.0
    )


st.markdown('</div>', unsafe_allow_html=True)


# Prediction
if st.button(
    "🔮 Predict Budget Deviation",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Budget": [budget],
        "Employees": [employees],
        "Month": [month],
        "Previous_Deviation": [previous_deviation],
        "Department": [department]
    })

    # Encode department
    input_data = pd.get_dummies(
        input_data,
        columns=["Department"],
        drop_first=True
    )

    # Match model columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    input_data = input_data.astype(float)

    # Prediction
    predicted_deviation = model.predict(
        input_data
    )[0]

    predicted_percentage = (
        predicted_deviation / budget
    ) * 100

    risk_level = classify_risk(
        predicted_percentage
    )

    predicted_spending = (
        budget + predicted_deviation
    )

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Deviation",
            f"₹{predicted_deviation:,.2f}"
        )

    with col2:
        st.metric(
            "Deviation Percentage",
            f"{predicted_percentage:.2f}%"
        )

    with col3:
        st.metric(
            "Risk Level",
            risk_level
        )

    # Chart
    chart_data = pd.DataFrame({
        "Amount": [
            budget,
            predicted_spending
        ]
    }, index=[
        "Budget",
        "Predicted Spending"
    ])

    st.bar_chart(chart_data)

    # Explanation
    st.info(
        f"Predicted spending is "
        f"₹{predicted_spending:,.2f}, "
        f"compared with a budget of "
        f"₹{budget:,.2f}."
    )
