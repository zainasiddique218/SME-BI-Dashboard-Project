import streamlit as st

from src.kpi_calculator import calculate_overview_kpis, monthly_sales, sales_by_country, top_products
from src.streamlit_helpers import configure_page, get_dashboard_data, show_data_quality, sidebar_filters, stop_if_empty
from src.utils import format_currency, format_number
from src.visualizations import country_bar, product_bar, revenue_trend


configure_page("Executive Overview")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Executive Overview")
st.caption("Summary KPIs and high-level sales performance for strategic decision-making.")

kpis = calculate_overview_kpis(filtered)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total revenue", format_currency(kpis["total_revenue"]))
col2.metric("Total orders", format_number(kpis["total_orders"]))
col3.metric("Total customers", format_number(kpis["total_customers"]))
col4.metric("Average order value", format_currency(kpis["average_order_value"]))

col5, col6, col7 = st.columns(3)
col5.metric("Quantity sold", format_number(kpis["total_quantity"]))
col6.metric("Unique products", format_number(kpis["unique_products"]))
col7.metric("Countries / markets", format_number(kpis["countries"]))

monthly_df = monthly_sales(filtered)
country_df = sales_by_country(filtered, top_n=10)
product_df = top_products(filtered, top_n=10)

st.plotly_chart(revenue_trend(monthly_df), width="stretch")

left, right = st.columns(2)
with left:
    st.plotly_chart(country_bar(country_df), width="stretch")
with right:
    st.plotly_chart(product_bar(product_df, title="Top Products by Revenue"), width="stretch")

with st.expander("View top product table"):
    st.dataframe(product_df, width="stretch")
