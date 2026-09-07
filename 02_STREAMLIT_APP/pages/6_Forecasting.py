import streamlit as st

from src.forecasting import run_forecasting
from src.streamlit_helpers import configure_page, get_dashboard_data, show_data_quality, sidebar_filters, stop_if_empty
from src.visualizations import backtest_chart, forecast_chart


configure_page("Forecasting")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Forecasting / Prediction")
st.caption("Simple monthly sales forecasting with model comparison metrics.")

target = st.sidebar.selectbox("Forecast target", ["Revenue", "Quantity"])
periods = st.sidebar.slider("Future months to forecast", min_value=1, max_value=6, value=3)

try:
    result = run_forecasting(filtered, target=target, periods=periods, test_size=3)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Best model", result.best_model)
col2.metric("Complete months used", len(result.model_monthly))
col3.metric("Partial months excluded", result.excluded_partial_months)

if result.excluded_partial_months:
    st.warning(
        "Partial boundary months are shown for context but excluded from model training and backtesting. "
        "This prevents the short December 2011 period from being treated as a full-month decline."
    )

st.plotly_chart(forecast_chart(result.monthly, result.future_forecast), width="stretch")
st.plotly_chart(backtest_chart(result.backtest, result.best_model), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Model comparison")
    st.dataframe(result.comparison, width="stretch")
with right:
    st.subheader("Future forecast")
    st.dataframe(result.future_forecast, width="stretch")

with st.expander("Monthly aggregated data"):
    st.dataframe(result.monthly, width="stretch")

st.caption("MAE and RMSE are in the selected target units. MAPE is a percentage. The best model is selected by the lowest MAE.")
