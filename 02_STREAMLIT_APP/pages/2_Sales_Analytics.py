import streamlit as st

from src.kpi_calculator import cancellation_summary, customer_value_table, monthly_sales, sales_by_country, top_products
from src.streamlit_helpers import (
    apply_filter_selection,
    configure_page,
    get_dashboard_data,
    show_data_quality,
    sidebar_filters,
    stop_if_empty,
)
from src.utils import format_currency, format_number, format_percent
from src.visualizations import country_bar, product_bar, quantity_histogram, revenue_trend


configure_page("Sales Analytics")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered, selection = sidebar_filters(clean_sales, return_selection=True)
filtered_all = apply_filter_selection(prepared_all, selection)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Sales Analytics")
st.caption("Explore revenue, orders, customer contribution, quantity movement, and cancelled transactions.")

monthly_df = monthly_sales(filtered)
country_df = sales_by_country(filtered, top_n=15)
product_df = top_products(filtered, top_n=15)
customer_df = customer_value_table(filtered).head(20)
cancel_summary = cancellation_summary(filtered_all)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cancelled / return rows", format_number(cancel_summary["cancelled_rows"]))
col2.metric("Cancelled invoices", format_number(cancel_summary["cancelled_invoices"]))
col3.metric("Cancelled value", format_currency(cancel_summary["cancelled_value"]))
col4.metric("Share of rows", format_percent(cancel_summary["cancelled_share_rows"]))

st.plotly_chart(revenue_trend(monthly_df), width="stretch")

left, right = st.columns(2)
with left:
    st.plotly_chart(country_bar(country_df), width="stretch")
with right:
    st.plotly_chart(product_bar(product_df, value_col="Quantity", title="Top Products by Quantity Sold"), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Top customers by revenue")
    st.dataframe(customer_df, width="stretch")
with right:
    st.plotly_chart(quantity_histogram(filtered), width="stretch")
