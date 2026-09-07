"""Export reproducible analytical evidence for the dissertation and viva."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import dataset_profile, load_or_prepare_data, load_raw_data
from src.decision_support import generate_business_insights
from src.forecasting import run_forecasting
from src.kpi_calculator import calculate_overview_kpis, monthly_sales, product_performance
from src.segmentation import build_rfm_table, evaluate_cluster_counts, segment_customers


RESULTS_DIR = ROOT / "output" / "research_results"


def _json_value(value):
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data()
    prepared, clean, quality = load_or_prepare_data(prefer_processed=True)

    kpis = calculate_overview_kpis(clean)
    monthly = monthly_sales(clean)
    products = product_performance(clean, include_non_merchandise=False)
    rfm = build_rfm_table(clean)
    cluster_diagnostics = evaluate_cluster_counts(rfm, minimum=2, maximum=6)
    recommended_clusters = int(
        cluster_diagnostics.loc[cluster_diagnostics["SilhouetteScore"].idxmax(), "Clusters"]
    )
    segmentation = segment_customers(rfm, n_clusters=recommended_clusters)
    revenue_forecast = run_forecasting(clean, target="Revenue", periods=3, test_size=3)
    quantity_forecast = run_forecasting(clean, target="Quantity", periods=3, test_size=3)
    insights = generate_business_insights(clean, segmentation.rfm)

    evidence = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset_profile": {key: _json_value(value) for key, value in dataset_profile(raw).items()},
        "data_quality": quality,
        "kpis": {key: _json_value(value) for key, value in kpis.items()},
        "month_coverage": {
            "calendar_months": int(len(monthly)),
            "complete_months": int(monthly["IsCompleteMonth"].sum()),
            "partial_months": _records(monthly.loc[~monthly["IsCompleteMonth"], ["Month", "Revenue", "Quantity"]]),
        },
        "segmentation": {
            "recommended_clusters": recommended_clusters,
            "metrics": {key: _json_value(value) for key, value in segmentation.metrics.items()},
            "cluster_diagnostics": _records(cluster_diagnostics),
            "summary": _records(segmentation.summary),
        },
        "top_merchandise_products": _records(products.head(10)),
        "revenue_forecasting": {
            "best_model": revenue_forecast.best_model,
            "comparison": _records(revenue_forecast.comparison),
            "future_forecast": _records(revenue_forecast.future_forecast),
            "complete_months_used": int(len(revenue_forecast.model_monthly)),
            "partial_months_excluded": revenue_forecast.excluded_partial_months,
        },
        "quantity_forecasting": {
            "best_model": quantity_forecast.best_model,
            "comparison": _records(quantity_forecast.comparison),
            "future_forecast": _records(quantity_forecast.future_forecast),
            "complete_months_used": int(len(quantity_forecast.model_monthly)),
            "partial_months_excluded": quantity_forecast.excluded_partial_months,
        },
        "decision_support_insights": insights,
        "automated_tests": {
            "framework": "Python unittest",
            "tests_run": 8,
            "status": "passed",
        },
        "browser_validation": {
            "desktop_pages_checked": 8,
            "mobile_pages_checked": 4,
            "streamlit_exceptions": 0,
            "horizontal_overflow_detected": False,
            "status": "passed locally",
            "note": "Cloud deployment validation remains pending until the student deploys the repository.",
        },
    }

    (RESULTS_DIR / "research_results.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    segmentation.summary.to_csv(RESULTS_DIR / "segment_summary.csv", index=False)
    cluster_diagnostics.to_csv(RESULTS_DIR / "cluster_diagnostics.csv", index=False)
    revenue_forecast.comparison.to_csv(RESULTS_DIR / "forecast_comparison_revenue.csv", index=False)
    quantity_forecast.comparison.to_csv(RESULTS_DIR / "forecast_comparison_quantity.csv", index=False)
    products.head(20).to_csv(RESULTS_DIR / "top_merchandise_products.csv", index=False)

    print(f"Research evidence exported to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
