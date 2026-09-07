"""Data loading helpers for the Online Retail BI dashboard.

The project uses the public UCI Online Retail dataset as a representative
retail transaction dataset. These helpers keep file paths predictable for local
development, Google Colab, GitHub, and Streamlit Cloud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_DATASET_NAME = "Online Retail.xlsx"
DEFAULT_PROCESSED_NAME = "cleaned_online_retail.csv"
DEFAULT_COMPRESSED_NAME = "prepared_online_retail.csv.gz"


def find_dataset_path(data_path: Optional[str | Path] = None) -> Path:
    """Find the raw Online Retail dataset.

    Parameters
    ----------
    data_path:
        Optional explicit path. If omitted, the function checks the standard
        project locations used by the app and the Colab notebook.
    """

    if data_path:
        candidate = Path(data_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset path does not exist: {candidate}")

    candidates = [
        RAW_DATA_DIR / DEFAULT_DATASET_NAME,
        PROJECT_ROOT / DEFAULT_DATASET_NAME,
        PROJECT_ROOT / "Online_Retail.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the Online Retail dataset. Checked:\n"
        f"{checked}\n\n"
        "Place Online Retail.xlsx inside data/raw/ before running the app."
    )


def load_raw_data(data_path: Optional[str | Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw Excel or CSV transaction dataset."""

    path = find_dataset_path(data_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_processed_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load a processed CSV file if one has already been generated."""

    processed_path = Path(path) if path else PROCESSED_DATA_DIR / DEFAULT_PROCESSED_NAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    return pd.read_csv(
        processed_path,
        parse_dates=["InvoiceDate", "InvoiceDateOnly", "Month"],
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "CustomerID": "string",
            "Country": "string",
            "YearMonth": "string",
        },
        low_memory=False,
    )


def load_prepared_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the deployment-friendly compressed transaction bundle."""

    prepared_path = Path(path) if path else PROCESSED_DATA_DIR / DEFAULT_COMPRESSED_NAME
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared compressed file not found: {prepared_path}")

    return pd.read_csv(
        prepared_path,
        compression="gzip",
        parse_dates=["InvoiceDate", "InvoiceDateOnly", "Month"],
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "CustomerID": "string",
            "Country": "string",
            "YearMonth": "string",
        },
        low_memory=False,
    )


def load_or_prepare_data(prefer_processed: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load processed data when available, otherwise load and clean raw data.

    Returns
    -------
    prepared_all:
        Standardized transaction table including invalid/cancelled flags.
    clean_sales:
        Positive, valid sales rows used for most dashboard analytics.
    quality_summary:
        Data-quality summary dictionary.
    """

    from src.preprocessing import prepare_retail_data, summarize_data_quality

    compressed_path = PROCESSED_DATA_DIR / DEFAULT_COMPRESSED_NAME
    if prefer_processed and compressed_path.exists():
        prepared_all = load_prepared_data(compressed_path)
        clean_sales = prepared_all.loc[prepared_all["IsValidSale"]].copy()
        return prepared_all, clean_sales, summarize_data_quality(prepared_all)

    processed_path = PROCESSED_DATA_DIR / DEFAULT_PROCESSED_NAME
    if prefer_processed and processed_path.exists():
        clean_sales = load_processed_data(processed_path)
        raw_df = load_raw_data()
        prepared_all, _ = prepare_retail_data(raw_df)
        return prepared_all, clean_sales, summarize_data_quality(prepared_all)

    raw_df = load_raw_data()
    prepared_all, clean_sales = prepare_retail_data(raw_df)
    return prepared_all, clean_sales, summarize_data_quality(prepared_all)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Return a compact profile used in docs and dashboard evidence."""

    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }
    if "InvoiceDate" in df.columns:
        dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        profile["date_min"] = dates.min()
        profile["date_max"] = dates.max()
    if "Country" in df.columns:
        profile["countries"] = int(df["Country"].nunique(dropna=True))
    if "CustomerID" in df.columns:
        profile["customers"] = int(df["CustomerID"].nunique(dropna=True))
    if "InvoiceNo" in df.columns:
        profile["orders"] = int(df["InvoiceNo"].nunique(dropna=True))
    if "StockCode" in df.columns:
        profile["products"] = int(df["StockCode"].nunique(dropna=True))
    return profile


def ensure_project_directories() -> None:
    """Create standard project folders if they do not already exist."""

    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
