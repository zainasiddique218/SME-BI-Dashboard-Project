"""Main Streamlit entry point for the BI decision support dashboard."""

import streamlit as st

from src.decision_support import generate_business_insights
from src.kpi_calculator import calculate_overview_kpis, monthly_sales, sales_by_country
from src.streamlit_helpers import (
    configure_page,
    get_dashboard_data,
    show_data_quality,
    sidebar_filters,
    stop_if_empty,
)
from src.utils import format_currency, format_number
from src.visualizations import country_bar, revenue_trend


configure_page("SME BI Decision Support Dashboard")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("SME Business Intelligence Dashboard")
st.caption("Retail performance, customer behaviour, product demand, and decision-support analytics.")

kpis = calculate_overview_kpis(filtered)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", format_currency(kpis["total_revenue"]))
col2.metric("Orders", format_number(kpis["total_orders"]))
col3.metric("Customers", format_number(kpis["total_customers"]))
col4.metric("Average order value", format_currency(kpis["average_order_value"]))

monthly = monthly_sales(filtered)
countries = sales_by_country(filtered, top_n=8)
left, right = st.columns([1.7, 1])
with left:
    st.plotly_chart(revenue_trend(monthly), width="stretch")
with right:
    st.plotly_chart(country_bar(countries), width="stretch")

st.subheader("Priority signals")
for item in generate_business_insights(filtered)[:3]:
    with st.container(border=True):
        st.markdown(f"**{item['title']}**")
        st.write(item["message"])
        st.caption(item["recommendation"])

st.caption(
    "Source: UCI Online Retail public transaction dataset. Customer identifiers are treated as anonymous analytical keys."
)
