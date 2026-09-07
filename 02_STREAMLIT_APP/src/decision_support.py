"""Rule-based decision-support insights for the dashboard."""

from __future__ import annotations

import pandas as pd

from src.kpi_calculator import complete_monthly_sales, product_performance, slow_moving_products


def _insight(category: str, severity: str, title: str, message: str, recommendation: str) -> dict:
    return {
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "recommendation": recommendation,
    }


def generate_business_insights(clean_sales: pd.DataFrame, rfm_df: pd.DataFrame | None = None) -> list[dict]:
    """Generate simple business recommendations from dashboard outputs."""

    insights: list[dict] = []
    monthly = complete_monthly_sales(clean_sales)

    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        change = (latest["Revenue"] - previous["Revenue"]) / previous["Revenue"] if previous["Revenue"] else 0
        if change < -0.1:
            insights.append(
                _insight(
                    "Sales",
                    "High",
                    "Recent revenue decline detected",
                    f"Latest monthly revenue is {change:.1%} lower than the previous month.",
                    "Review recent product demand, market performance, and customer activity before planning promotions.",
                )
            )
        elif change > 0.1:
            insights.append(
                _insight(
                    "Sales",
                    "Positive",
                    "Recent revenue growth detected",
                    f"Latest monthly revenue is {change:.1%} higher than the previous month.",
                    "Identify the products and countries driving the increase and protect stock availability.",
                )
            )

    products = product_performance(clean_sales, include_non_merchandise=False)
    if not products.empty:
        top_product = products.iloc[0]
        insights.append(
            _insight(
                "Product",
                "Positive",
                "Top product deserves priority",
                f"{top_product['Description']} is the highest-revenue product in the dataset.",
                "Prioritize availability, monitor demand, and consider promotion around this product category.",
            )
        )

        slow = slow_moving_products(clean_sales, top_n=1)
        if not slow.empty:
            slow_product = slow.iloc[0]
            insights.append(
                _insight(
                    "Product",
                    "Medium",
                    "Slow-moving product review needed",
                    f"{slow_product['Description']} has low revenue and quantity movement.",
                    "Review whether this item needs bundling, discounting, repositioning, or removal.",
                )
            )

    if rfm_df is not None and not rfm_df.empty and "SegmentName" in rfm_df.columns:
        segment_revenue = rfm_df.groupby("SegmentName")["Monetary"].sum().sort_values(ascending=False)
        top_segment = segment_revenue.index[0]
        insights.append(
            _insight(
                "Customer",
                "Positive",
                "High-value customer segment identified",
                f"The {top_segment} segment contributes the largest customer revenue share.",
                "Use loyalty, retention, or targeted communication strategies for this segment.",
            )
        )

        at_risk = rfm_df[rfm_df["Recency"] > rfm_df["Recency"].quantile(0.75)]
        if not at_risk.empty:
            insights.append(
                _insight(
                    "Customer",
                    "Medium",
                    "Inactive customer group requires attention",
                    f"{len(at_risk):,} customers have relatively high recency values.",
                    "Create a re-engagement campaign for customers who have not purchased recently.",
                )
            )

    if not insights:
        insights.append(
            _insight(
                "General",
                "Info",
                "No major warning detected",
                "The current filtered dataset does not trigger a major alert.",
                "Continue monitoring KPIs, product movement, and customer segment changes.",
            )
        )

    return insights
