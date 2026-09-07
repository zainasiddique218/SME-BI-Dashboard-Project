"""Filtered transaction explorer and reproducible data export."""

import streamlit as st

from src.streamlit_helpers import (
    configure_page,
    get_dashboard_data,
    show_data_quality,
    sidebar_filters,
    stop_if_empty,
)
from src.utils import format_currency, format_number


configure_page("Data Explorer")

prepared_all, clean_sales, quality = get_dashboard_data()
filtered = sidebar_filters(clean_sales)
show_data_quality(quality)
stop_if_empty(filtered)

st.title("Data Explorer")
st.caption("Inspect the analysis-ready transactions behind the dashboard outputs.")

col1, col2, col3 = st.columns(3)
col1.metric("Filtered rows", format_number(len(filtered)))
col2.metric("Filtered revenue", format_currency(filtered["Revenue"].sum()))
col3.metric("Date coverage", f"{filtered['InvoiceDate'].min():%Y-%m-%d} to {filtered['InvoiceDate'].max():%Y-%m-%d}")

display_columns = [
    "InvoiceNo",
    "InvoiceDate",
    "StockCode",
    "Description",
    "Quantity",
    "UnitPrice",
    "Revenue",
    "CustomerID",
    "Country",
]
st.dataframe(filtered[display_columns].head(5000), width="stretch", hide_index=True)
if len(filtered) > 5000:
    st.caption("The on-screen preview is limited to 5,000 rows; the download contains all filtered rows.")

st.download_button(
    "Download filtered transactions",
    data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
    file_name="filtered_online_retail.csv",
    mime="text/csv",
)

with st.expander("Analysis field definitions"):
    st.markdown(
        """
        - **Revenue:** Quantity multiplied by UnitPrice for a valid positive sale.
        - **IsCancelled:** Invoice number begins with C or quantity is negative.
        - **IsValidSale:** Date and description are present, quantity and price are positive, and the row is not cancelled.
        - **IsNonMerchandise:** Operational entries such as postage, manual adjustments, or bank charges.
        - **Month:** Calendar month used for trend and forecast aggregation.
        """
    )
