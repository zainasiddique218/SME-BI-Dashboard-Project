"""Prepare cleaned CSV files for faster dashboard loading.

Run locally:
    python scripts/prepare_data.py

Run in Colab after uploading/cloning the project:
    !python scripts/prepare_data.py
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import PROCESSED_DATA_DIR, dataset_profile, ensure_project_directories, load_raw_data
from src.kpi_calculator import monthly_sales, product_performance
from src.preprocessing import prepare_retail_data, save_prepared_data, save_processed_data, summarize_data_quality


def main() -> None:
    ensure_project_directories()
    raw_df = load_raw_data()
    prepared_all, clean_sales = prepare_retail_data(raw_df)
    quality = summarize_data_quality(prepared_all)

    cleaned_path = save_processed_data(clean_sales, PROCESSED_DATA_DIR / "cleaned_online_retail.csv")
    compressed_path = save_prepared_data(
        prepared_all,
        PROCESSED_DATA_DIR / "prepared_online_retail.csv.gz",
    )
    monthly_path = PROCESSED_DATA_DIR / "monthly_sales.csv"
    products_path = PROCESSED_DATA_DIR / "product_performance.csv"
    profile_path = PROCESSED_DATA_DIR / "dataset_profile.txt"

    monthly_sales(clean_sales).to_csv(monthly_path, index=False)
    product_performance(clean_sales).to_csv(products_path, index=False)

    profile = dataset_profile(raw_df)
    lines = ["Dataset profile"]
    for key, value in profile.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Data quality summary")
    for key, value in quality.items():
        lines.append(f"{key}: {value}")
    Path(profile_path).write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved cleaned data: {cleaned_path}")
    print(f"Saved compressed deployment data: {compressed_path}")
    print(f"Saved monthly sales: {monthly_path}")
    print(f"Saved product performance: {products_path}")
    print(f"Saved profile: {profile_path}")


if __name__ == "__main__":
    main()
