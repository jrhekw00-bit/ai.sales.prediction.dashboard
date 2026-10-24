import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. DEMO DATASET GENERATOR
# --------------------------------------------------

@st.cache_data
def generate_demo_data(rows=600):
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        start="2024-01-01",
        end="2025-12-31",
        periods=rows
    )

    products = rng.choice(
        ["Laptop", "Mobile", "Headphones", "Tablet", "Smartwatch"],
        size=rows
    )

    price_ranges = {
        "Laptop": (700, 1500),
        "Mobile": (200, 1000),
        "Headphones": (20, 200),
        "Tablet": (150, 700),
        "Smartwatch": (50, 400)
    }

    prices = np.array([
        rng.uniform(*price_ranges[p])
        for p in products
    ])

    advertising = rng.uniform(50, 1000, rows)
    discount = rng.uniform(0, 30, rows)
    month = pd.DatetimeIndex(dates).month

    seasonal_effect = 1 + 0.20 * np.sin(
        2 * np.pi * (month - 1) / 12
    )

    units = (
        25
        + advertising * 0.09
        - prices * 0.035
        + discount * 1.1
        + rng.normal(0, 12, rows)
    ) * seasonal_effect

    units = np.maximum(1, np.round(units)).astype(int)

    revenue = units * prices * (1 - discount / 100)

    df = pd.DataFrame({
        "Date": dates,
        "Product": products,
        "Price": np.round(prices, 2),
        "Advertising_Spend": np.round(advertising, 2),
        "Discount": np.round(discount, 2),
        "Units_Sold": units,
        "Revenue": np.round(revenue, 2)
    })

    return df


# --------------------------------------------------
# 3. LOAD OR UPLOAD DATA
# --------------------------------------------------

st.sidebar.title("⚙️ Dashboard Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            "Date",
            "Product",
            "Price",
            "Advertising_Spend",
            "Discount",
            "Units_Sold"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                "CSV mein required columns missing hain: "
                + ", ".join(missing_columns)
            )
            st.stop()

        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        )

        numeric_columns = [
            "Price",
            "Advertising_Spend",
            "Discount",
            "Units_Sold"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df = df.dropna(
            subset=required_columns
        ).copy()

        if df.empty:
            st.error("CSV mein valid sales records nahi hain.")
            st.stop()

        if (df["Price"] < 0).any():
            st.error("Price negative nahi ho sakti.")
            st.stop()

        if (df["Units_Sold"] < 0).any():
            st.error("Units_Sold negative nahi ho sakta.")
            st.stop()

        if not df["Discount"].between(0, 100).all():
            st.error("Discount 0 se 100 ke darmiyan hona chahiye.")
            st.stop()

        if (df["Advertising_Spend"] < 0).any():
            st.error("Advertising spend negative nahi ho sakta.")
            st.stop()

        if "Revenue" not in df.columns:
            df["Revenue"] = (
                df["Price"]
                * df["Units_Sold"]
                * (1 - df["Discount"] / 100)
            )

        df = df.dropna(
            subset=["Revenue", "Product"]
        ).copy()

        if df.empty or len(df) < 20:
            st.error(
                "Reliable model evaluation ke liye "
                "kam az kam 20 valid records dein."
            )
            st.stop()

        if (df["Revenue"] < 0).any():
            st.error("Revenue negative nahi ho sakti.")
            st.stop()

        data_source = "Uploaded CSV"

    except Exception as e:
        st.error(f"CSV load karne mein error: {e}")
        st.stop()

else:
    df = generate_demo_data()
    data_source = "Demo Dataset"


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

df["Month"] = df["Date"].dt.month

feature_columns = [
    "Product",
    "Price",
    "Advertising_Spend",
    "Discount",
    "Month"
]

target_column = "Revenue"

X = df[feature_columns]
y = df[target_column]


# --------------------------------------------------
# 5. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "product",
            OneHotEncoder(handle_unknown="ignore"),
            ["Product"]
        ),
        (
            "numbers",
            "passthrough",
            [
                "Price",
                "Advertising_Spend",
                "Discount",
                "Month"
            ]
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)
r2 = r2_score(y_test, test_predictions)


# --------------------------------------------------
# 6. DASHBOARD HEADER
# --------------------------------------------------

st.title("📊 AI Sales Prediction Dashboard")

st.write(
    "Analyze sales performance and predict revenue "
    "using a Machine Learning model."
)

st.caption(
    f"Data source: {data_source} | "
    f"Dataset records: {len(df):,}"
)


# --------------------------------------------------
# 7. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.subheader("Filters")

products = sorted(df["Product"].unique().tolist())

selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.info("Please select a start date and an end date.")
    st.stop()

start_date, end_date = date_range

filtered_df = df[
    df["Product"].isin(selected_products)
    & df["Date"].dt.date.between(start_date, end_date)
].copy()

if filtered_df.empty:
    st.warning("Selected filters ke liye koi data nahi mila.")
    st.stop()


# --------------------------------------------------
# 8. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_revenue = filtered_df["Revenue"].sum()
total_units = filtered_df["Units_Sold"].sum()
total_orders = len(filtered_df)
average_revenue = filtered_df["Revenue"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"${total_revenue:,.2f}"
)

col2.metric(
    "Units Sold",
    f"{total_units:,}"
)

col3.metric(
    "Sales Records",
    f"{total_orders:,}"
)

col4.metric(
    "Average Revenue",
    f"${average_revenue:,.2f}"
)


# --------------------------------------------------
# 9. SALES TREND CHART
# --------------------------------------------------

st.subheader("📈 Sales Trend Over Time")

monthly_sales = (
    filtered_df
    .set_index("Date")["Revenue"]
    .resample("MS")
    .sum()
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

ax.set_title("Monthly Revenue")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. PRODUCT PERFORMANCE
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("🏆 Revenue by Product")

    product_sales = (
        filtered_df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    product_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Product")
    ax.set_ylabel("Revenue ($)")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("📦 Units Sold by Product")

    units_by_product = (
        filtered_df.groupby("Product")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    units_by_product.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Product")
    ax.set_ylabel("Units Sold")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 11. MODEL EVALUATION
# --------------------------------------------------

st.subheader("🤖 Machine Learning Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"${mae:,.2f}")
m2.metric("RMSE", f"${rmse:,.2f}")
m3.metric("R² Score", f"{r2:.3f}")

with st.expander("What do these metrics mean?"):
    st.write(
        "**MAE:** Average absolute prediction error. "
        "Lower is better."
    )
    st.write(
        "**RMSE:** Prediction error metric that penalizes "
        "larger errors more strongly. Lower is better."
    )
    st.write(
        "**R² Score:** Measures how well predictions explain "
        "variation in the test data. Closer to 1 is generally better."
    )

st.caption(
    "Metrics test split par calculate kiye gaye hain. "
    "Demo dataset synthetic hai; iske scores real business "
    "performance ka proof nahi hain."
)


# --------------------------------------------------
# 12. INTERACTIVE SALES PREDICTION
# --------------------------------------------------

st.subheader("🔮 Predict Sales Revenue")

st.write(
    "Product aur sales conditions select karein "
    "aur model se estimated revenue hasil karein."
)

with st.form("prediction_form"):

    p1, p2 = st.columns(2)

    with p1:
        chosen_product = st.selectbox(
            "Product",
            options=products
        )

        chosen_price = st.number_input(
            "Product Price ($)",
            min_value=0.01,
            max_value=100000.0,
            value=100.0,
            step=10.0
        )

    with p2:
        chosen_advertising = st.number_input(
            "Advertising Budget ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=200.0,
            step=10.0
        )

        chosen_discount = st.slider(
            "Discount (%)",
            min_value=0,
            max_value=100,
            value=10
        )

    chosen_month = st.selectbox(
        "Sales Month",
        options=list(range(1, 13)),
        index=9,
        format_func=lambda m: pd.Timestamp(
            year=2025, month=m, day=1
        ).strftime("%B")
    )

    predict_button = st.form_submit_button(
        "Predict Revenue",
        use_container_width=True
    )

if predict_button:

    new_data = pd.DataFrame([{
        "Product": chosen_product,
        "Price": chosen_price,
        "Advertising_Spend": chosen_advertising,
        "Discount": chosen_discount,
        "Month": chosen_month
    }])

    predicted_revenue = float(
        model.predict(new_data)[0]
    )

    predicted_revenue = max(0, predicted_revenue)

    estimated_units = (
        predicted_revenue
        / (chosen_price * (1 - chosen_discount / 100))
        if chosen_discount < 100
        else 0
    )

    st.success(
        f"Predicted Revenue: ${predicted_revenue:,.2f}"
    )

    st.write(
        f"Estimated equivalent units sold: "
        f"{estimated_units:,.1f}"
    )

    st.caption(
        "Yeh model estimate hai, guaranteed future sales nahi. "
        "100% discount par revenue conversion se units estimate "
        "nahi kiye ja sakte."
    )


# --------------------------------------------------
# 13. DATA TABLE AND DOWNLOAD
# --------------------------------------------------

st.subheader("📋 Sales Data")

st.dataframe(
    filtered_df.sort_values("Date", ascending=False),
    use_container_width=True
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Sales CSV",
    data=csv_data,
    file_name="sales_dashboard_data.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Sales Prediction Dashboard | "
    "Python • Pandas • Matplotlib • Seaborn • Scikit-learn • Streamlit"
)