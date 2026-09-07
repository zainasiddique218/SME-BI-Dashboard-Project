"""Simple monthly sales forecasting models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.kpi_calculator import monthly_sales


@dataclass
class ForecastResult:
    monthly: pd.DataFrame
    model_monthly: pd.DataFrame
    comparison: pd.DataFrame
    backtest: pd.DataFrame
    future_forecast: pd.DataFrame
    best_model: str
    excluded_partial_months: int


def aggregate_monthly(
    clean_sales: pd.DataFrame,
    target: str = "Revenue",
    complete_months_only: bool = False,
) -> pd.DataFrame:
    """Aggregate revenue or quantity by month."""

    if target not in {"Revenue", "Quantity"}:
        raise ValueError("target must be either 'Revenue' or 'Quantity'")

    monthly = monthly_sales(clean_sales)
    if complete_months_only:
        monthly = monthly.loc[monthly["IsCompleteMonth"]].copy()
    monthly["Target"] = monthly[target]
    return monthly


def _moving_average_forecast(history: list[float], horizon: int, window: int = 3) -> np.ndarray:
    """Forecast by repeatedly averaging the latest observations."""

    values = list(history)
    preds = []
    for _ in range(horizon):
        window_values = values[-window:] if len(values) >= window else values
        pred = float(np.mean(window_values))
        preds.append(pred)
        values.append(pred)
    return np.array(preds)


def _linear_trend_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast with a simple linear trend over time index."""

    x_train = np.arange(len(history)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_train, history)
    x_future = np.arange(len(history), len(history) + horizon).reshape(-1, 1)
    return model.predict(x_future)


def _naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast by repeating the latest actual value."""

    return np.repeat(history[-1], horizon)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": _mape(actual, predicted),
    }


def evaluate_forecast_models(
    monthly: pd.DataFrame,
    test_size: int = 3,
    return_backtest: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models using a chronological holdout period."""

    if len(monthly) < 6:
        raise ValueError("At least 6 monthly observations are recommended for forecast comparison.")

    test_size = min(test_size, max(1, len(monthly) // 4))
    train = monthly["Target"].iloc[:-test_size].astype(float).to_numpy()
    test = monthly["Target"].iloc[-test_size:].astype(float).to_numpy()

    predictions = {
        "Naive Last Value": _naive_forecast(train, len(test)),
        "Moving Average (3 months)": _moving_average_forecast(train.tolist(), len(test), window=3),
        "Linear Trend": _linear_trend_forecast(train, len(test)),
    }

    rows = []
    backtest_rows = []
    for name, pred in predictions.items():
        values = _metrics(test, pred)
        rows.append({"Model": name, **values})
        for month, actual, predicted in zip(monthly["Month"].iloc[-test_size:], test, pred):
            backtest_rows.append(
                {
                    "Month": month,
                    "Model": name,
                    "Actual": float(actual),
                    "Predicted": float(predicted),
                    "AbsoluteError": float(abs(actual - predicted)),
                }
            )
    comparison = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    backtest = pd.DataFrame(backtest_rows)
    return (comparison, backtest) if return_backtest else comparison


def forecast_future(
    monthly: pd.DataFrame,
    model_name: str,
    periods: int = 3,
    forecast_after: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Forecast future months using the selected model."""

    history = monthly["Target"].astype(float).to_numpy()
    last_model_month = pd.to_datetime(monthly["Month"].max())
    forecast_after = pd.to_datetime(forecast_after or last_model_month)
    month_gap = max(
        0,
        (forecast_after.year - last_model_month.year) * 12
        + forecast_after.month
        - last_model_month.month,
    )
    full_horizon = periods + month_gap

    if model_name == "Moving Average (3 months)":
        all_predictions = _moving_average_forecast(history.tolist(), full_horizon, window=3)
    elif model_name == "Linear Trend":
        all_predictions = _linear_trend_forecast(history, full_horizon)
    else:
        all_predictions = _naive_forecast(history, full_horizon)

    pred = all_predictions[month_gap:]

    future_months = pd.date_range(forecast_after + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"Month": future_months, "Forecast": pred.clip(min=0)})


def run_forecasting(clean_sales: pd.DataFrame, target: str = "Revenue", periods: int = 3, test_size: int = 3) -> ForecastResult:
    """Run the full forecasting workflow."""

    monthly = aggregate_monthly(clean_sales, target=target, complete_months_only=False)
    model_monthly = monthly.loc[monthly["IsCompleteMonth"]].copy()
    comparison, backtest = evaluate_forecast_models(
        model_monthly,
        test_size=test_size,
        return_backtest=True,
    )
    best_model = str(comparison.iloc[0]["Model"])
    future = forecast_future(
        model_monthly,
        best_model,
        periods=periods,
        forecast_after=monthly["Month"].max(),
    )
    return ForecastResult(
        monthly=monthly,
        model_monthly=model_monthly,
        comparison=comparison,
        backtest=backtest,
        future_forecast=future,
        best_model=best_model,
        excluded_partial_months=int((~monthly["IsCompleteMonth"]).sum()),
    )
