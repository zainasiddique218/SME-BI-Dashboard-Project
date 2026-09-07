"""Train analytics artifacts for the BI dashboard.

This script is intentionally simple enough to run in Google Colab. It creates
RFM customer segments and monthly sales forecast outputs that can be saved,
downloaded, or committed to GitHub if required.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.data_loader import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_directories, load_raw_data
from src.forecasting import run_forecasting
from src.preprocessing import prepare_retail_data, save_processed_data
from src.segmentation import build_rfm_table, evaluate_cluster_counts, save_segmentation_result, segment_customers


def main() -> None:
    ensure_project_directories()
    raw_df = load_raw_data()
    _, clean_sales = prepare_retail_data(raw_df)
    save_processed_data(clean_sales, PROCESSED_DATA_DIR / "cleaned_online_retail.csv")

    rfm = build_rfm_table(clean_sales)
    diagnostics = evaluate_cluster_counts(rfm, minimum=2, maximum=6)
    recommended_clusters = int(diagnostics.loc[diagnostics["SilhouetteScore"].idxmax(), "Clusters"])
    segmentation = segment_customers(rfm, n_clusters=recommended_clusters)
    save_segmentation_result(segmentation, MODELS_DIR)
    segmentation.rfm.to_csv(PROCESSED_DATA_DIR / "rfm_segments.csv", index=False)
    segmentation.summary.to_csv(PROCESSED_DATA_DIR / "segment_summary.csv", index=False)

    revenue_forecast = run_forecasting(clean_sales, target="Revenue", periods=3, test_size=3)
    quantity_forecast = run_forecasting(clean_sales, target="Quantity", periods=3, test_size=3)

    revenue_forecast.comparison.to_csv(PROCESSED_DATA_DIR / "forecast_model_comparison_revenue.csv", index=False)
    revenue_forecast.future_forecast.to_csv(PROCESSED_DATA_DIR / "future_forecast_revenue.csv", index=False)
    quantity_forecast.comparison.to_csv(PROCESSED_DATA_DIR / "forecast_model_comparison_quantity.csv", index=False)
    quantity_forecast.future_forecast.to_csv(PROCESSED_DATA_DIR / "future_forecast_quantity.csv", index=False)

    joblib.dump(
        {
            "revenue": revenue_forecast,
            "quantity": quantity_forecast,
        },
        Path(MODELS_DIR) / "forecast_results.joblib",
    )

    print("Training complete.")
    print(f"Recommended clusters: {recommended_clusters}")
    print(f"Customers segmented: {segmentation.metrics['customers_segmented']}")
    print(f"Segmentation artifacts: {MODELS_DIR}")
    print(f"Processed outputs: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
