"""Unit and integration tests for the BI analytics pipeline."""

import unittest

import pandas as pd

from src.decision_support import generate_business_insights
from src.forecasting import run_forecasting
from src.kpi_calculator import (
    calculate_overview_kpis,
    complete_monthly_sales,
    monthly_sales,
    product_performance,
)
from src.preprocessing import prepare_retail_data, summarize_data_quality
from src.segmentation import build_rfm_table, evaluate_cluster_counts, segment_customers
from src.utils import filter_dataframe


def make_raw_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["10001", "A1", "ALPHA PRODUCT", 2, "2011-01-01 09:00", 10.0, 1, "United Kingdom"],
            ["10002", "A1", "ALPHA PRODUCT", 3, "2011-01-31 09:00", 10.0, 1, "United Kingdom"],
            ["10003", "B1", "BETA PRODUCT", 4, "2011-02-01 09:00", 5.0, 2, "France"],
            ["10004", "B1", "BETA PRODUCT", 2, "2011-02-28 09:00", 5.0, 2, "France"],
            ["C10005", "A1", "ALPHA PRODUCT", -1, "2011-02-15 09:00", 10.0, 1, "United Kingdom"],
            ["10006", "DOT", "DOTCOM POSTAGE", 1, "2011-02-20 09:00", 50.0, 3, "United Kingdom"],
            ["10007", "C1", "GAMMA PRODUCT", 1, "2011-03-05 09:00", 7.0, None, "Germany"],
            ["10008", "D1", "INVALID PRICE", 1, "2011-03-05 09:00", 0.0, 4, "Germany"],
        ],
        columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ],
    )


def make_forecasting_sales() -> pd.DataFrame:
    rows = []
    invoice = 20000
    for month in pd.date_range("2011-01-01", periods=8, freq="MS"):
        month_end = month + pd.offsets.MonthEnd(0)
        value = 100 + (month.month * 10)
        for date in (month, month_end):
            invoice += 1
            rows.append(
                [
                    str(invoice),
                    "A1",
                    "ALPHA PRODUCT",
                    value / 10,
                    date,
                    10.0,
                    str((month.month % 4) + 1),
                    "United Kingdom",
                ]
            )
    rows.append(["29999", "A1", "ALPHA PRODUCT", 1, pd.Timestamp("2011-09-05"), 10.0, "1", "United Kingdom"])
    raw = pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ],
    )
    return prepare_retail_data(raw)[1]


class AnalyticsPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared, cls.clean = prepare_retail_data(make_raw_transactions())

    def test_cleaning_preserves_quality_flags(self):
        quality = summarize_data_quality(self.prepared)

        self.assertEqual(quality["total_rows"], 8)
        self.assertEqual(quality["cancelled_or_return_rows"], 1)
        self.assertEqual(quality["zero_or_negative_price_rows"], 1)
        self.assertEqual(quality["valid_sales_rows"], 6)
        self.assertEqual(quality["non_merchandise_sales_rows"], 1)
        self.assertAlmostEqual(self.clean["Revenue"].sum(), 137.0)

    def test_kpis_and_date_filter_are_consistent(self):
        january = filter_dataframe(
            self.clean,
            date_range=(pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-31")),
        )
        kpis = calculate_overview_kpis(january)

        self.assertAlmostEqual(kpis["total_revenue"], 50.0)
        self.assertEqual(kpis["total_orders"], 2)
        self.assertAlmostEqual(kpis["average_order_value"], 25.0)

    def test_product_rankings_exclude_operational_lines(self):
        merchandise = product_performance(self.clean)
        all_lines = product_performance(self.clean, include_non_merchandise=True)

        self.assertNotIn("DOT", merchandise["StockCode"].tolist())
        self.assertIn("DOT", all_lines["StockCode"].tolist())

    def test_partial_month_is_identified_and_excluded(self):
        monthly = monthly_sales(self.clean)
        complete = complete_monthly_sales(self.clean)
        partial_flag = monthly.loc[
            monthly["Month"].eq(pd.Timestamp("2011-03-01")),
            "IsCompleteMonth",
        ].item()

        self.assertFalse(bool(partial_flag))
        self.assertNotIn(pd.Timestamp("2011-03-01"), complete["Month"].tolist())

    def test_forecasting_uses_complete_months_and_returns_backtest(self):
        result = run_forecasting(make_forecasting_sales(), target="Revenue", periods=3, test_size=2)

        self.assertEqual(len(result.monthly), 9)
        self.assertEqual(len(result.model_monthly), 8)
        self.assertEqual(result.excluded_partial_months, 1)
        self.assertEqual(
            set(result.comparison["Model"]),
            {"Naive Last Value", "Moving Average (3 months)", "Linear Trend"},
        )
        self.assertEqual(result.backtest["Month"].nunique(), 2)
        self.assertEqual(result.future_forecast["Month"].min(), pd.Timestamp("2011-10-01"))

    def test_segmentation_and_cluster_diagnostics(self):
        rfm = build_rfm_table(self.clean)
        result = segment_customers(rfm, n_clusters=2)
        diagnostics = evaluate_cluster_counts(rfm, minimum=2, maximum=3)

        self.assertEqual(result.metrics["customers_segmented"], len(rfm))
        self.assertEqual(result.metrics["n_clusters"], 2)
        self.assertIn("AnnualizedValueIndicator", result.rfm.columns)
        self.assertFalse(diagnostics.empty)

    def test_segmentation_rejects_single_customer(self):
        one_customer = self.clean.loc[self.clean["CustomerID"].eq("1")]
        rfm = build_rfm_table(one_customer)

        with self.assertRaisesRegex(ValueError, "at least two customers"):
            segment_customers(rfm, n_clusters=2)

    def test_decision_rules_ignore_incomplete_month_decline(self):
        insights = generate_business_insights(self.clean)
        titles = {item["title"] for item in insights}

        self.assertNotIn("Recent revenue decline detected", titles)


if __name__ == "__main__":
    unittest.main()
