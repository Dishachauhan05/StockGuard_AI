
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from data_utils import (
    load_demo_data,
    validate_and_prepare,
    product_history,
    REQUIRED_COLUMNS
)

from models import (
    train_lstm,
    forecast_future,
    calculate_inventory_recommendation
)


st.set_page_config(
    page_title="StockGuard AI",
    page_icon="📦",
    layout="wide"
)

st.title("📦 StockGuard AI")

st.caption(
    "Inventory decision support for small retailers · "
    "LSTM demand forecasting"
)

st.info(
    "The bundled dataset is synthetic demo data. "
    "Forecasts are educational estimates, not guarantees."
)


@st.cache_data(show_spinner=False)
def cached_train(sales_tuple, lookback, epochs):
    return train_lstm(
        list(sales_tuple),
        lookback=int(lookback),
        epochs=int(epochs)
    )


def format_units(value):
    return f"{value:,.0f} units"


with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader(
        "Upload inventory CSV",
        type=["csv"],
        help="Required columns: date, product, daily_sales, stock, lead_time"
    )

    st.caption(
        "One row per product per date. "
        "Keep daily sales and stock in whole units."
    )

    st.divider()
    st.header("Model settings")

    lookback = st.slider(
        "Lookback window (days)",
        min_value=7,
        max_value=28,
        value=14
    )

    epochs = st.slider(
        "LSTM training epochs",
        min_value=10,
        max_value=100,
        value=35,
        step=5
    )

    horizon = st.slider(
        "Forecast horizon (days)",
        min_value=1,
        max_value=7,
        value=7
    )

    safety_factor = st.slider(
        "Safety-stock factor",
        min_value=0.5,
        max_value=3.0,
        value=1.65,
        step=0.05
    )

    target_days = st.slider(
        "Target stock coverage (days)",
        min_value=2,
        max_value=14,
        value=7
    )

    train_clicked = st.button(
        "Train / refresh LSTM model",
        type="primary",
        width='stretch'
    )


if uploaded is not None:
    try:
        df = validate_and_prepare(pd.read_csv(uploaded))
        data_source = "Uploaded CSV"

    except Exception as exc:
        st.error(f"Could not use uploaded CSV: {exc}")
        st.stop()

else:
    df = load_demo_data()
    data_source = "Synthetic demo data"


st.caption(
    f"Data source: **{data_source}** · "
    f"{len(df):,} rows · {df['product'].nunique()} products"
)


products = sorted(df["product"].unique().tolist())

if not products:
    st.error("No products found in the data.")
    st.stop()


latest = (
    df.sort_values("date")
    .groupby("product", as_index=False)
    .tail(1)
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Products",
    f"{df['product'].nunique()}"
)

c2.metric(
    "Current total stock",
    format_units(latest["stock"].sum())
)

c3.metric(
    "Products needing review",
    f"{int((latest['stock'] <= latest['lead_time'] * latest['daily_sales']).sum())}"
)

st.divider()


selected = st.selectbox(
    "Select a product",
    products
)

history = product_history(df, selected)

if len(history) < max(30, lookback + 12):
    st.warning(
        f"{selected} has {len(history)} observations. "
        f"At least {max(30, lookback + 12)} observations "
        "are recommended for this LSTM setup."
    )
    st.stop()


current_stock = int(history.iloc[-1]["stock"])
lead_time = int(history.iloc[-1]["lead_time"])

sales = history["daily_sales"].astype(float).tolist()

signature = hashlib.sha256(
    np.asarray(sales, dtype=np.float32).tobytes()
).hexdigest()

cache_key = (
    signature,
    lookback,
    epochs
)


if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}


if (
    train_clicked
    or cache_key not in st.session_state["trained_models"]
):
    with st.spinner("Training the LSTM on historical sales..."):
        try:
            result = cached_train(
                tuple(sales),
                lookback,
                epochs
            )

            st.session_state["trained_models"][cache_key] = result

        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()


trained = st.session_state["trained_models"][cache_key]

forecast = forecast_future(
    sales,
    trained,
    horizon=horizon
)


recommendation_forecast = (
    forecast
    if len(forecast) >= lead_time
    else forecast_future(
        sales,
        trained,
        max(horizon, min(lead_time, 30))
    )
)


recommendation = calculate_inventory_recommendation(
    sales,
    recommendation_forecast,
    current_stock,
    lead_time,
    safety_factor=safety_factor,
    target_days=target_days
)


st.subheader(f"Inventory Decision: {selected}")

a, b, c, d = st.columns(4)

a.metric(
    "Current Stock",
    format_units(current_stock)
)

b.metric(
    "Supplier Lead Time",
    f"{lead_time} day(s)"
)

c.metric(
    "Reorder Point",
    format_units(recommendation["reorder_point"])
)

d.metric(
    "Suggested Reorder",
    format_units(recommendation["suggested_quantity"])
)


if recommendation["is_below_reorder_point"]:
    st.warning(
        "Current stock is at or below the calculated reorder point."
    )
else:
    st.success(
        "Current stock is above the calculated reorder point."
    )


st.caption(
    "Suggested order = max(0, target stock − current stock). "
    "The target covers forecast demand for the selected target days "
    "plus safety stock."
)


st.divider()

left, right = st.columns([1.35, 1])


with left:
    st.subheader("Historical Sales")

    hist_chart = (
        history[["date", "daily_sales"]]
        .tail(60)
        .copy()
        .set_index("date")
    )

    st.line_chart(
        hist_chart.rename(
            columns={"daily_sales": "Actual Daily Sales"}
        ),
        x_label="Date",
        y_label="Units Sold"
    )


with right:
    st.subheader("Future Demand Forecast")

    future = forecast.copy()

    future_dates = pd.date_range(
        start=history["date"].max() + pd.Timedelta(days=1),
        periods=len(future),
        freq="D"
    )

    future_chart = pd.DataFrame(
        {
            "Predicted Demand": future["predicted_demand"].values
        },
        index=future_dates
    )

    future_chart.index.name = "Date"

    st.line_chart(
        future_chart,
        x_label="Date",
        y_label="Predicted Units"
    )


st.divider()
st.subheader("Reorder Calculation")

r1, r2, r3 = st.columns(3)

r1.metric(
    "Expected Demand During Lead Time",
    format_units(recommendation["lead_time_demand"])
)

r2.metric(
    "Safety Stock",
    format_units(recommendation["safety_stock"])
)

r3.metric(
    "Target Stock",
    format_units(recommendation["target_stock"])
)


st.write(
    f"**Reorder Point:** Lead-time demand "
    f"({recommendation['lead_time_demand']:.1f}) + safety stock "
    f"({recommendation['safety_stock']:.1f}) = "
    f"{recommendation['reorder_point']:.1f} units."
)

st.caption(
    "Safety stock = safety factor × recent daily-sales standard deviation "
    "× square root of lead time."
)


st.divider()
st.subheader("LSTM Model Evaluation")

st.write(
    "The test set is the final 20% of the time series. "
    "The scaler is fitted on the first 80% only; test windows use "
    "prior observations but not future target values."
)


m1, m2 = st.columns(2)

m1.metric(
    "Test MAE",
    f"{trained['mae']:.2f} units"
)

m2.metric(
    "Test RMSE",
    f"{trained['rmse']:.2f} units"
)


st.caption(
    f"Training observations: {trained['train_count']} sequences · "
    f"Held-out test observations: {trained['test_count']} sequences · "
    f"Lookback: {lookback} days"
)


eval_df = trained["evaluation"].copy()

eval_df["Test Day"] = np.arange(
    1,
    len(eval_df) + 1
)

st.line_chart(
    eval_df.set_index("Test Day")[
        ["actual", "predicted"]
    ].rename(
        columns={
            "actual": "Actual Sales",
            "predicted": "Predicted Sales"
        }
    ),
    x_label="Test Day",
    y_label="Units Sold"
)


loss_df = pd.DataFrame({
    "Epoch": np.arange(
        1,
        len(trained["losses"]) + 1
    ),
    "Training MSE Loss": trained["losses"]
}).set_index("Epoch")

st.line_chart(
    loss_df,
    x_label="Epoch",
    y_label="MSE Loss"
)


with st.expander("Data Preview and Validation Details"):
    st.dataframe(
        history.tail(20),
        width='stretch'
    )

    st.write(
        "Required CSV columns:",
        ", ".join(REQUIRED_COLUMNS)
    )

    st.write(
        "Dates are parsed, invalid or missing values are rejected, "
        "negative sales/stock are rejected, and duplicate product-date "
        "rows are reduced to the last row."
    )

    st.download_button(
        "Download Current Data as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="stockguard_data.csv",
        mime="text/csv"
    )


with st.expander("Model Assumptions and Limitations"):
    st.markdown("""
- **LSTM:** Learns patterns in historical daily sales to forecast future demand. Forecast errors can accumulate over multiple days.
- **Synthetic data:** The bundled dataset is for demonstration and does not establish real-world predictive performance.
- **Data meaning:** Stock is treated as the latest available inventory snapshot for each product.
- **Operational use:** Confirm supplier schedules, pack sizes, minimum order quantities, expiry, and actual stock before placing an order.
""")
