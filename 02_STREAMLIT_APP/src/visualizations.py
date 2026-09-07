"""Plotly visualization helpers used across dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import short_text


COLOR_SEQUENCE = ["#111111", "#FF6B35", "#7F7F7F", "#B8BCC4", "#444444"]


def revenue_trend(monthly_df: pd.DataFrame, value_col: str = "Revenue") -> go.Figure:
    if "IsCompleteMonth" not in monthly_df.columns:
        fig = px.line(monthly_df, x="Month", y=value_col, markers=True, title=f"Monthly {value_col} Trend")
        fig.update_traces(line_color="#E85D2A", line_width=3)
        fig.update_layout(template="plotly_white", hovermode="x unified")
        return fig

    complete = monthly_df.loc[monthly_df["IsCompleteMonth"]].copy()
    partial = monthly_df.loc[~monthly_df["IsCompleteMonth"]].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=complete["Month"],
            y=complete[value_col],
            mode="lines+markers",
            name="Complete month",
            line=dict(color="#E85D2A", width=3),
        )
    )
    if not partial.empty:
        fig.add_trace(
            go.Scatter(
                x=partial["Month"],
                y=partial[value_col],
                mode="markers",
                name="Partial month",
                marker=dict(color="#D92D20", size=11, symbol="diamond"),
            )
        )
    fig.update_layout(title=f"Monthly {value_col} Trend")
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def country_bar(country_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        country_df.sort_values("Revenue"),
        x="Revenue",
        y="Country",
        orientation="h",
        title="Top Countries / Markets by Revenue",
        color_discrete_sequence=["#111111"],
    )
    fig.update_layout(template="plotly_white", yaxis_title="", xaxis_title="Revenue")
    return fig


def product_bar(product_df: pd.DataFrame, value_col: str = "Revenue", title: str = "Top Products") -> go.Figure:
    data = product_df.copy()
    data["Product"] = data["Description"].apply(short_text)
    fig = px.bar(
        data.sort_values(value_col),
        x=value_col,
        y="Product",
        orientation="h",
        title=title,
        color_discrete_sequence=["#111111"],
    )
    fig.update_layout(template="plotly_white", yaxis_title="", xaxis_title=value_col)
    return fig


def customer_scatter(customer_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        customer_df,
        x="Orders",
        y="Revenue",
        size="Quantity",
        hover_name="CustomerID",
        title="Customer Revenue vs Purchase Frequency",
        color_discrete_sequence=["#FF6B35"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def rfm_segment_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        rfm_df,
        x="Frequency",
        y="Monetary",
        color="SegmentName",
        size="RFMScore",
        hover_name="CustomerID",
        title="Customer Segments by Frequency and Monetary Value",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(template="plotly_white")
    return fig


def segment_bar(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary_df.sort_values("TotalRevenue"),
        x="TotalRevenue",
        y="SegmentName",
        orientation="h",
        title="Customer Segment Revenue Contribution",
        color="Customers",
        color_continuous_scale=["#EDEDED", "#FF6B35"],
    )
    fig.update_layout(template="plotly_white", yaxis_title="", xaxis_title="Total Revenue")
    return fig


def forecast_chart(monthly_df: pd.DataFrame, future_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    complete = monthly_df.loc[monthly_df.get("IsCompleteMonth", True)].copy()
    partial = monthly_df.loc[~monthly_df.get("IsCompleteMonth", True)].copy()
    fig.add_trace(
        go.Scatter(
            x=complete["Month"],
            y=complete["Target"],
            mode="lines+markers",
            name="Actual - complete month",
            line=dict(color="#111111", width=3),
        )
    )
    if not partial.empty:
        fig.add_trace(
            go.Scatter(
                x=partial["Month"],
                y=partial["Target"],
                mode="markers",
                name="Actual - partial month",
                marker=dict(color="#D92D20", size=11, symbol="diamond"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=future_df["Month"],
            y=future_df["Forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#FF6B35", width=3, dash="dash"),
        )
    )
    fig.update_layout(template="plotly_white", title="Actual vs Forecasted Monthly Sales", hovermode="x unified")
    return fig


def backtest_chart(backtest_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Compare actual and predicted values for the selected holdout model."""

    data = backtest_df.loc[backtest_df["Model"].eq(model_name)].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Month"],
            y=data["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#111111", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["Month"],
            y=data["Predicted"],
            mode="lines+markers",
            name="Backtest prediction",
            line=dict(color="#E85D2A", width=3, dash="dash"),
        )
    )
    fig.update_layout(template="plotly_white", title="Holdout Backtest", hovermode="x unified")
    return fig


def cluster_diagnostic_chart(diagnostics: pd.DataFrame) -> go.Figure:
    """Show silhouette score for candidate customer cluster counts."""

    fig = px.line(
        diagnostics,
        x="Clusters",
        y="SilhouetteScore",
        markers=True,
        title="Cluster Count Diagnostic",
    )
    fig.update_traces(line_color="#167D6D", line_width=3)
    fig.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    return fig


def product_action_scatter(product_df: pd.DataFrame) -> go.Figure:
    """Visualize product action categories by revenue and quantity."""

    fig = px.scatter(
        product_df,
        x="Quantity",
        y="Revenue",
        color="ActionCategory",
        hover_name="Description",
        log_x=True,
        log_y=True,
        title="Product Action Matrix",
        color_discrete_map={
            "Protect and promote": "#167D6D",
            "Premium-value opportunity": "#3659A2",
            "Volume / margin review": "#E9A23B",
            "Review or rationalize": "#D92D20",
        },
    )
    fig.update_layout(template="plotly_white")
    return fig


def quantity_histogram(clean_sales: pd.DataFrame) -> go.Figure:
    capped = clean_sales[clean_sales["Quantity"] <= clean_sales["Quantity"].quantile(0.99)]
    fig = px.histogram(capped, x="Quantity", nbins=50, title="Quantity Sold Distribution")
    fig.update_traces(marker_color="#111111")
    fig.update_layout(template="plotly_white")
    return fig
