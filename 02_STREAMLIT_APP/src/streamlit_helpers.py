"""Small Streamlit helper functions shared by dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.data_loader import load_or_prepare_data
from src.utils import filter_dataframe


def configure_page(title: str) -> None:
    """Apply a consistent Streamlit page setup."""

    st.set_page_config(
        page_title=title,
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
        :root { --ink: #17202a; --muted: #667085; --line: #d7dde5; --accent: #e85d2a; }
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        [data-testid="stMetric"] {
            border-top: 3px solid var(--accent);
            border-bottom: 1px solid var(--line);
            padding: 0.75rem 0.25rem 0.65rem 0.25rem;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] > div {
            font-size: 1.55rem;
            line-height: 1.2;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }
        [data-testid="stMetricValue"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        [data-testid="stAlert"] { border-radius: 6px; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class DashboardFilterSelection:
    """Selected global dashboard filter values."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    countries: tuple[str, ...]


@st.cache_data(show_spinner="Loading and cleaning retail dataset...")
def get_dashboard_data():
    """Load and cache dashboard data for Streamlit."""

    return load_or_prepare_data(prefer_processed=True)


def sidebar_filters(
    clean_sales: pd.DataFrame,
    return_selection: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, DashboardFilterSelection]:
    """Render common date and country filters."""

    st.sidebar.header("Filters")
    min_date = clean_sales["InvoiceDate"].min().date()
    max_date = clean_sales["InvoiceDate"].max().date()
    selected_dates = st.sidebar.date_input(
        "Invoice date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    countries = sorted(clean_sales["Country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Countries / markets",
        options=countries,
        default=[],
        help="Leave empty to include all countries.",
    )

    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date = pd.to_datetime(selected_dates[0])
        end_date = pd.to_datetime(selected_dates[1])

    selection = DashboardFilterSelection(
        start_date=start_date,
        end_date=end_date,
        countries=tuple(selected_countries),
    )
    filtered = filter_dataframe(
        clean_sales,
        date_range=(selection.start_date, selection.end_date),
        countries=selection.countries,
    )
    return (filtered, selection) if return_selection else filtered


def apply_filter_selection(
    df: pd.DataFrame,
    selection: DashboardFilterSelection,
) -> pd.DataFrame:
    """Apply the same sidebar selection to another transaction table."""

    return filter_dataframe(
        df,
        date_range=(selection.start_date, selection.end_date),
        countries=selection.countries,
    )


def stop_if_empty(df: pd.DataFrame) -> None:
    """Stop a page cleanly when filters return no sales rows."""

    if df.empty:
        st.warning("No valid sales match the selected filters.")
        st.stop()


def show_data_quality(quality_summary: dict) -> None:
    """Display key cleaning facts in the sidebar."""

    with st.sidebar.expander("Data quality notes"):
        st.write(f"Total rows: {quality_summary['total_rows']:,}")
        st.write(f"Valid sales rows: {quality_summary['valid_sales_rows']:,}")
        st.write(f"Missing customer rows: {quality_summary['missing_customer_rows']:,}")
        st.write(f"Cancelled/return rows: {quality_summary['cancelled_or_return_rows']:,}")
        st.write(f"Zero or negative price rows: {quality_summary['zero_or_negative_price_rows']:,}")
        st.write(f"Non-merchandise sales rows: {quality_summary.get('non_merchandise_sales_rows', 0):,}")
