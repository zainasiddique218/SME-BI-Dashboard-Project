"""Measure reproducible local execution times for the main analytics pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_or_prepare_data
from src.forecasting import run_forecasting
from src.kpi_calculator import calculate_overview_kpis
from src.segmentation import build_rfm_table, segment_customers


OUTPUT_PATH = ROOT / "output" / "research_results" / "performance_benchmark.json"


def timed(label: str, operation):
    start = perf_counter()
    result = operation()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f} seconds")
    return result, elapsed


def main() -> None:
    (_, clean, _), load_seconds = timed(
        "Load and validate deployment data",
        lambda: load_or_prepare_data(prefer_processed=True),
    )
    _, kpi_seconds = timed("Calculate overview KPIs", lambda: calculate_overview_kpis(clean))
    rfm, rfm_seconds = timed("Build RFM customer table", lambda: build_rfm_table(clean))
    _, segmentation_seconds = timed(
        "Fit three-cluster customer segmentation",
        lambda: segment_customers(rfm, n_clusters=3),
    )
    _, forecasting_seconds = timed(
        "Compare monthly revenue forecast models",
        lambda: run_forecasting(clean, target="Revenue", periods=3, test_size=3),
    )

    evidence = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset_rows": int(len(clean)),
        "timings_seconds": {
            "load_and_validate": round(load_seconds, 3),
            "overview_kpis": round(kpi_seconds, 3),
            "rfm_table": round(rfm_seconds, 3),
            "segmentation_k3": round(segmentation_seconds, 3),
            "forecast_model_comparison": round(forecasting_seconds, 3),
        },
        "interpretation": (
            "These are single-run local development measurements. They demonstrate feasibility "
            "on the test computer and are not a service-level guarantee for Streamlit Cloud."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Benchmark evidence exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
