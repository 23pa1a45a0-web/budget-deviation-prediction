import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BudgetGuard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.joblib"
COLUMNS_PATH = BASE_DIR / "model_columns.joblib"


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(96,165,250,0.12),
                transparent 35%
            ),
            linear-gradient(
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
        margin-bottom: 0;
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

    .success-box {
        padding: 20px;
        border-radius: 15px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.30);
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL SAFELY
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Model columns file not found: {COLUMNS_PATH.name}"
        )

    loaded_model = joblib.load(MODEL_PATH)
    loaded_columns = joblib.load(COLUMNS_PATH)

    return loaded_model, loaded_columns


# =========================================================
# MODEL INITIALIZATION
# =========================================================

try:

    with st.spinner("Loading BudgetGuard AI model..."):
        model, model_columns = load_model()

except Exception as e:

    st.error("❌ BudgetGuard could not load the AI model.")

    st.markdown(
        """
        ### Deployment checklist

        Make sure these files are in the **same GitHub folder as `app.py`**:

        - `app.py`
        - `model.joblib`
        - `model_columns.joblib`
        - `requirements.txt`

        If the files are present, the most likely problem is a
        **scikit-learn version mismatch** between model training and deployment.
        """
    )

    st.code(str(e))

    st.stop()


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(deviation_percentage):

    if deviation_percentage > 10:
        return "High Risk"

    elif deviation_percentage > 5:
        return "Medium Risk"

    elif deviation_percentage >= -5:
        return "Low Risk"

    else:
        return "Underspending"


# =========================================================
# HEADER
# =========================================================

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


# =========================================================
# INPUT SECTION
# =========================================================

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

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Budget Deviation",
    type="primary",
    use_container_width=True
):

    try:

        # -------------------------------------------------
        # CREATE INPUT DATA
        # -------------------------------------------------

        input_data = pd.DataFrame(
            {
                "Budget": [budget],
                "Employees": [employees],
                "Month": [month],
                "Previous_Deviation": [previous_deviation],
                "Department": [department]
            }
        )


        # -------------------------------------------------
        # ENCODE DEPARTMENT
        # -------------------------------------------------

        input_data = pd.get_dummies(
            input_data,
            columns=["Department"],
            drop_first=True
        )


        # -------------------------------------------------
        # MATCH TRAINING COLUMNS
        # -------------------------------------------------

        input_data = input_data.reindex(
            columns=model_columns,
            fill_value=0
        )


        # -------------------------------------------------
        # ENSURE NUMERIC DATA
        # -------------------------------------------------

        input_data = input_data.astype(float)


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        predicted_deviation = float(
            model.predict(input_data)[0]
        )


        # -------------------------------------------------
        # CALCULATIONS
        # -------------------------------------------------

        predicted_percentage = (
            predicted_deviation / budget
        ) * 100

        risk_level = classify_risk(
            predicted_percentage
        )

        predicted_spending = (
            budget + predicted_deviation
        )


        # =================================================
        # RESULTS
        # =================================================

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


        # =================================================
        # CHART
        # =================================================

        chart_data = pd.DataFrame(
            {
                "Amount": [
                    budget,
                    predicted_spending
                ]
            },
            index=[
                "Budget",
                "Predicted Spending"
            ]
        )

        st.bar_chart(chart_data)


        # =================================================
        # EXPLANATION
        # =================================================

        st.markdown(
            f"""
            <div class="success-box">

            <b>🤖 BudgetGuard Analysis</b>

            <br><br>

            Predicted spending:
            <b>₹{predicted_spending:,.2f}</b>

            <br>

            Allocated budget:
            <b>₹{budget:,.2f}</b>

            <br>

            Expected deviation:
            <b>₹{predicted_deviation:,.2f}</b>

            <br>

            Risk classification:
            <b>{risk_level}</b>

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))

        st.info(
            "If this error mentions feature names, columns, "
            "or data types, the deployed model and "
            "model_columns.joblib are not compatible."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "BudgetGuard • AI-Based Financial Budget Deviation Prediction"
)
