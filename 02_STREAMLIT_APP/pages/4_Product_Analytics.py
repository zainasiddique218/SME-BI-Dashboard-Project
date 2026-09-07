import streamlit as st

from src.kpi_calculator import classify_product_actions, product_performance, slow_moving_products, top_products
from src.streamlit_helpers import configure_page, get_dashboard_data, show_data_quality, sidebar_filters, stop_if_empty
from src.utils import format_currency, format_number
from src.visualizations import product_action_scatter, product_bar


configure_page("Product Analytics")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Product Analytics")
st.caption("Best-selling products, slow-moving products, high-revenue items, and product recommendations.")

include_non_merchandise = st.sidebar.toggle(
    "Include service / adjustment codes",
    value=False,
    help="Includes postage, manual entries, bank charges, and similar operational lines in product rankings.",
)
products = product_performance(filtered, include_non_merchandise=include_non_merchandise)
products = classify_product_actions(products)
best_revenue = top_products(filtered, top_n=15, include_non_merchandise=include_non_merchandise)
best_quantity = products.sort_values("Quantity", ascending=False).head(15)
slow = slow_moving_products(filtered, top_n=20)

col1, col2, col3 = st.columns(3)
col1.metric("Products analysed", format_number(products["StockCode"].nunique()))
col2.metric("Highest product revenue", format_currency(products["Revenue"].max()))
col3.metric("Highest product quantity", format_number(products["Quantity"].max()))

left, right = st.columns(2)
with left:
    st.plotly_chart(product_bar(best_revenue, value_col="Revenue", title="High-Revenue Products"), width="stretch")
with right:
    st.plotly_chart(product_bar(best_quantity, value_col="Quantity", title="Best-Selling Products by Quantity"), width="stretch")

st.subheader("Product-level recommendation logic")
st.markdown(
    """
- High revenue and high quantity: protect stock availability and prioritize promotion.
- High revenue but lower quantity: monitor premium or high-value product opportunities.
- Low revenue and low quantity: review pricing, bundling, promotion, or discontinuation.
"""
)

st.plotly_chart(product_action_scatter(products), width="stretch")

st.subheader("Slow-moving product candidates")
st.dataframe(slow, width="stretch")

with st.expander("Full product performance table"):
    st.dataframe(products, width="stretch")
