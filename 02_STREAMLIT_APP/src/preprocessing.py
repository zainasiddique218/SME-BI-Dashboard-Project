"""Cleaning and preprocessing logic for the Online Retail dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# The UCI dataset contains operational lines that are not retail products.
# They remain part of revenue calculations but can be excluded from product
# rankings so that recommendations focus on merchandise.
NON_MERCHANDISE_CODES = {
    "AMAZONFEE",
    "BANK CHARGES",
    "C2",
    "CRUK",
    "D",
    "DOT",
    "M",
    "PADS",
    "POST",
    "S",
}


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when required dataset columns are missing."""

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _clean_identifier(value) -> str | pd.NA:
    """Convert Excel numeric IDs into readable strings without .0 suffixes."""

    if pd.isna(value):
        return pd.NA
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else pd.NA


def standardize_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, types, and basic text fields."""

    standardized = df.copy()
    standardized.columns = [str(column).strip() for column in standardized.columns]
    validate_columns(standardized)

    standardized["InvoiceNo"] = standardized["InvoiceNo"].apply(_clean_identifier)
    standardized["StockCode"] = standardized["StockCode"].apply(_clean_identifier)
    standardized["CustomerID"] = standardized["CustomerID"].apply(_clean_identifier)
    standardized["Description"] = (
        standardized["Description"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    standardized["Country"] = standardized["Country"].astype("string").str.strip()
    standardized["Quantity"] = pd.to_numeric(standardized["Quantity"], errors="coerce")
    standardized["UnitPrice"] = pd.to_numeric(standardized["UnitPrice"], errors="coerce")
    standardized["InvoiceDate"] = pd.to_datetime(standardized["InvoiceDate"], errors="coerce")

    return standardized


def prepare_retail_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create analysis-ready dataframes.

    Returns all standardized rows plus a clean positive-sales table. Cancelled
    and invalid rows are preserved in the first dataframe for data-quality and
    cancellation analysis.
    """

    prepared = standardize_raw_data(raw_df)

    prepared["IsCancelled"] = (
        prepared["InvoiceNo"].astype("string").str.startswith("C", na=False)
        | (prepared["Quantity"] < 0)
    )
    prepared["IsMissingCustomer"] = prepared["CustomerID"].isna()
    prepared["IsZeroOrNegativePrice"] = prepared["UnitPrice"].fillna(0) <= 0
    prepared["IsZeroOrNegativeQuantity"] = prepared["Quantity"].fillna(0) <= 0
    prepared["IsNonMerchandise"] = (
        prepared["StockCode"].astype("string").str.upper().isin(NON_MERCHANDISE_CODES)
        | prepared["Description"].astype("string").str.upper().str.contains(
            r"POSTAGE|BANK CHARGE|AMAZON FEE|MANUAL|DISCOUNT|CARRIAGE",
            regex=True,
            na=False,
        )
    )
    prepared["Revenue"] = prepared["Quantity"] * prepared["UnitPrice"]
    prepared["InvoiceDateOnly"] = prepared["InvoiceDate"].dt.date
    prepared["InvoiceDateOnly"] = pd.to_datetime(prepared["InvoiceDateOnly"])
    prepared["Month"] = prepared["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    prepared["Year"] = prepared["InvoiceDate"].dt.year
    prepared["YearMonth"] = prepared["InvoiceDate"].dt.to_period("M").astype("string")

    valid_sales_mask = (
        prepared["InvoiceDate"].notna()
        & prepared["Description"].notna()
        & prepared["Quantity"].gt(0)
        & prepared["UnitPrice"].gt(0)
        & ~prepared["IsCancelled"]
    )
    prepared["IsValidSale"] = valid_sales_mask
    clean_sales = prepared.loc[valid_sales_mask].copy()

    return prepared, clean_sales


def summarize_data_quality(prepared_df: pd.DataFrame) -> dict:
    """Summarize records that need cleaning or special handling."""

    return {
        "total_rows": int(len(prepared_df)),
        "missing_customer_rows": int(prepared_df["CustomerID"].isna().sum()),
        "cancelled_or_return_rows": int(prepared_df["IsCancelled"].sum()),
        "zero_or_negative_price_rows": int(prepared_df["IsZeroOrNegativePrice"].sum()),
        "zero_or_negative_quantity_rows": int(prepared_df["IsZeroOrNegativeQuantity"].sum()),
        "valid_sales_rows": int(prepared_df["IsValidSale"].sum()),
        "non_merchandise_sales_rows": int(
            (prepared_df["IsValidSale"] & prepared_df["IsNonMerchandise"]).sum()
        ),
    }


def save_processed_data(clean_sales: pd.DataFrame, output_path: str | Path) -> Path:
    """Save cleaned sales data as CSV for faster app loading."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    clean_sales.to_csv(path, index=False, compression=compression)
    return path


def save_prepared_data(prepared_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save all standardized rows, including quality flags, as compressed CSV."""

    return save_processed_data(prepared_df, output_path)
