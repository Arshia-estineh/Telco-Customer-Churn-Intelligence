from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEEP_BLUE = "#07111F"
PANEL_BLUE = "#0E1B2C"
BORDER_BLUE = "#24344D"
EMERALD = "#10B981"
SUNSET = "#F97316"
MUTED = "#9CA3AF"
TEXT = "#F8FAFC"

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ARTIFACT_DIR = BASE_DIR / "models"
RAW_DATA_PATH = Path(r"C:\Users\arshi\Desktop\telco_customer\WA_Fn-UseC_-Telco-Customer-Churn.csv")
LOCAL_DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {DEEP_BLUE};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background: #081526;
        border-right: 1px solid {BORDER_BLUE};
    }}
    [data-testid="stMetric"] {{
        background: {PANEL_BLUE};
        border: 1px solid {BORDER_BLUE};
        border-radius: 8px;
        padding: 18px 18px;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {MUTED};
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT};
    }}
    div[data-testid="stPlotlyChart"] {{
        background: {PANEL_BLUE};
        border: 1px solid {BORDER_BLUE};
        border-radius: 8px;
        padding: 8px;
    }}
    .section-title {{
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.4rem 0 0.6rem 0;
        color: {TEXT};
    }}
    .subtle {{
        color: {MUTED};
        font-size: 0.95rem;
    }}
    .risk-panel {{
        background: {PANEL_BLUE};
        border: 1px solid {BORDER_BLUE};
        border-radius: 8px;
        padding: 18px;
    }}
    .risk-high {{ color: {SUNSET}; font-weight: 700; }}
    .risk-low {{ color: {EMERALD}; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data and artifact loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else LOCAL_DATA_PATH
    if not data_path.exists():
        st.error("Raw dataset not found. Place the CSV next to app.py or update RAW_DATA_PATH.")
        st.stop()

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


@st.cache_resource
def load_artifacts():
    required = {
        "model": ARTIFACT_DIR / "mlp_churn_model.joblib",
        "preprocessor": ARTIFACT_DIR / "preprocessor.joblib",
        "schema": ARTIFACT_DIR / "feature_schema.joblib",
        "scaler": ARTIFACT_DIR / "scaler.joblib",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        st.error(
            "Model artifacts are missing. Run model_development.ipynb first so it creates the models folder."
        )
        st.stop()

    return {
        "model": joblib.load(required["model"]),
        "preprocessor": joblib.load(required["preprocessor"]),
        "schema": joblib.load(required["schema"]),
        "scaler": joblib.load(required["scaler"]),
    }


df = load_raw_data()
artifacts = load_artifacts()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
schema = artifacts["schema"]

# -----------------------------------------------------------------------------
# Chart builders
# -----------------------------------------------------------------------------
def apply_chart_layout(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PANEL_BLUE,
        plot_bgcolor=PANEL_BLUE,
        font=dict(color=TEXT),
        margin=dict(l=25, r=25, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def churn_distribution_chart(data: pd.DataFrame):
    churn_counts = data["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Customers"]
    fig = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.58,
        color="Churn",
        color_discrete_map={"No": EMERALD, "Yes": SUNSET},
        title="Distribution of Churn",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_chart_layout(fig)


def contract_churn_chart(data: pd.DataFrame):
    contract_churn = (
        data.groupby("Contract", as_index=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .rename(columns={"Churn": "Churn Rate (%)"})
        .sort_values("Churn Rate (%)", ascending=False)
    )
    fig = px.bar(
        contract_churn,
        x="Contract",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color="Contract",
        color_discrete_sequence=[SUNSET, "#2563EB", EMERALD],
        title="Churn Rate by Contract Type",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return apply_chart_layout(fig)


def monthly_charge_box_chart(data: pd.DataFrame):
    fig = px.box(
        data,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": EMERALD, "Yes": SUNSET},
        points="outliers",
        title="Monthly Charges vs Churn",
    )
    return apply_chart_layout(fig)


def churn_gauge(probability: float):
    percent = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=percent,
            number={"suffix": "%", "font": {"size": 42, "color": TEXT}},
            delta={"reference": 50, "suffix": "% threshold"},
            title={"text": "Churn Probability", "font": {"size": 20, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": SUNSET if percent >= 50 else EMERALD},
                "bgcolor": "#111827",
                "borderwidth": 1,
                "bordercolor": BORDER_BLUE,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.28)"},
                    {"range": [35, 65], "color": "rgba(37, 99, 235, 0.24)"},
                    {"range": [65, 100], "color": "rgba(249, 115, 22, 0.30)"},
                ],
                "threshold": {
                    "line": {"color": TEXT, "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    return apply_chart_layout(fig, height=360)

# -----------------------------------------------------------------------------
# Sidebar customer input
# -----------------------------------------------------------------------------
def options_for(field: str, fallback):
    return schema.get("category_options", {}).get(field, fallback)


def select_feature(label: str, field: str, fallback):
    options = options_for(field, fallback)
    return st.sidebar.selectbox(label, options)


def build_customer_input() -> pd.DataFrame:
    st.sidebar.title("Customer Scenario")
    st.sidebar.caption("Adjust the profile to estimate churn risk for one customer.")

    gender = select_feature("Gender", "gender", ["Female", "Male"])
    senior = st.sidebar.toggle("Senior Citizen", value=False)
    partner = select_feature("Partner", "Partner", ["No", "Yes"])
    dependents = select_feature("Dependents", "Dependents", ["No", "Yes"])

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = select_feature("Phone Service", "PhoneService", ["No", "Yes"])
    multiple_lines = select_feature("Multiple Lines", "MultipleLines", ["No", "No phone service", "Yes"])

    internet_service = select_feature("Internet Service", "InternetService", ["DSL", "Fiber optic", "No"])
    online_security = select_feature("Online Security", "OnlineSecurity", ["No", "No internet service", "Yes"])
    online_backup = select_feature("Online Backup", "OnlineBackup", ["No", "No internet service", "Yes"])
    device_protection = select_feature("Device Protection", "DeviceProtection", ["No", "No internet service", "Yes"])
    tech_support = select_feature("Tech Support", "TechSupport", ["No", "No internet service", "Yes"])
    streaming_tv = select_feature("Streaming TV", "StreamingTV", ["No", "No internet service", "Yes"])
    streaming_movies = select_feature("Streaming Movies", "StreamingMovies", ["No", "No internet service", "Yes"])

    contract = select_feature("Contract", "Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = select_feature("Paperless Billing", "PaperlessBilling", ["No", "Yes"])
    payment_method = select_feature(
        "Payment Method",
        "PaymentMethod",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    )

    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 70.0, 0.5)
    total_default = float(monthly_charges * tenure)
    total_charges = st.sidebar.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=total_default,
        step=25.0,
    )

    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior),
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
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    feature_columns = schema.get("feature_columns", list(customer.keys()))
    return pd.DataFrame([customer])[feature_columns]


customer_df = build_customer_input()

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def predict_churn_probability(customer: pd.DataFrame) -> float:
    prepared = preprocessor.transform(customer)
    positive_target_value = schema.get("positive_target_value", 1)
    positive_index = list(model.classes_).index(positive_target_value)
    return float(model.predict_proba(prepared)[0][positive_index])


churn_probability = predict_churn_probability(customer_df)
model_accuracy = schema.get("accuracy")
overall_churn_rate = (df["Churn"].eq("Yes").mean() * 100)

# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
st.title("Telco Churn Intelligence")
st.markdown(
    '<p class="subtle">Executive-ready customer retention dashboard powered by an MLP churn model.</p>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Total Customers", f"{len(df):,}")
m2.metric("Overall Churn Rate", f"{overall_churn_rate:.1f}%")
m3.metric("Model Accuracy", f"{model_accuracy:.1%}" if model_accuracy is not None else "Run notebook")

st.markdown('<div class="section-title">Business Context</div>', unsafe_allow_html=True)
c1, c2 = st.columns((1, 1))
with c1:
    st.plotly_chart(churn_distribution_chart(df), use_container_width=True)
with c2:
    st.plotly_chart(contract_churn_chart(df), use_container_width=True)
st.plotly_chart(monthly_charge_box_chart(df), use_container_width=True)

st.markdown('<div class="section-title">Interactive Prediction</div>', unsafe_allow_html=True)
gauge_col, summary_col = st.columns((1.15, 0.85))
with gauge_col:
    st.plotly_chart(churn_gauge(churn_probability), use_container_width=True)

with summary_col:
    risk_label = "High churn risk" if churn_probability >= 0.50 else "Lower churn risk"
    risk_class = "risk-high" if churn_probability >= 0.50 else "risk-low"
    retention_action = (
        "Prioritize save offer, service recovery, and proactive outreach."
        if churn_probability >= 0.50
        else "Maintain relationship quality and monitor for risk changes."
    )
    st.markdown(
        f"""
        <div class="risk-panel">
            <p class="subtle">Predicted segment</p>
            <h3 class="{risk_class}">{risk_label}</h3>
            <p style="color:{TEXT}; margin-top: 0.8rem;">{retention_action}</p>
            <p class="subtle">Probability: {churn_probability:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(customer_df.T.rename(columns={0: "Selected Customer"}), use_container_width=True)
