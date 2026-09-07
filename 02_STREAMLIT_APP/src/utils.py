"""General utility functions for the BI dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_currency(value: float, currency: str = "GBP") -> str:
    """Format a numeric value as a readable currency string."""

    if pd.isna(value):
        return f"{currency} 0"
    return f"{currency} {value:,.2f}"


def format_number(value: float) -> str:
    """Format whole numbers and large counts consistently."""

    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    """Format a decimal as a percentage."""

    if pd.isna(value):
        return "0.0%"
    return f"{value * 100:.1f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Avoid divide-by-zero errors in KPI calculations."""

    if denominator in (0, None) or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def filter_dataframe(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    countries: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply common dashboard filters."""

    filtered = df.copy()
    if date_range and "InvoiceDate" in filtered.columns:
        start, end = date_range
        filtered = filtered[
            (filtered["InvoiceDate"] >= pd.to_datetime(start))
            & (filtered["InvoiceDate"] < pd.to_datetime(end) + pd.Timedelta(days=1))
        ]
    if countries and "Country" in filtered.columns:
        selected = list(countries)
        if selected:
            filtered = filtered[filtered["Country"].isin(selected)]
    return filtered


def short_text(value: str, max_length: int = 45) -> str:
    """Shorten long product descriptions for charts."""

    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 3] + "..."
