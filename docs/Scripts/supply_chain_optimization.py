"""Supply chain optimization example with ETL, forecasting, and EOQ calculation."""
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pulp import LpMinimize, LpProblem, LpVariable, lpSum

RAW_PATH = os.path.join('raw_data', 'synthetic_supply_chain_data.csv')
CLEAN_PATH = os.path.join('raw_data', 'clean_supply_chain_data.csv')


def generate_synthetic_data(path: str, n_rows: int = 1000) -> None:
    """Generate synthetic supply chain data with basic anomalies."""
    rng = np.random.default_rng(0)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
    products = rng.choice(['A', 'B', 'C'], size=n_rows)
    demand = rng.poisson(20, size=n_rows)
    inventory = rng.integers(-5, 200, size=n_rows)  # allow negatives for anomalies
    lead_times = rng.choice([1, 2, 3, np.nan], size=n_rows, p=[0.3, 0.3, 0.3, 0.1])
    df = pd.DataFrame(
        {
            'product': products,
            'date': dates.strftime('%Y-%m-%d'),
            'demand': demand,
            'inventory': inventory,
            'lead_time_days': lead_times,
        }
    )
    df.to_csv(path, index=False)


def extract(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['demand'] = df['demand'].abs()
    df['inventory'] = df['inventory'].clip(lower=0)
    df['lead_time_days'] = df['lead_time_days'].fillna(df['lead_time_days'].median())
    return df


def load(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def forecast_monthly_demand(df: pd.DataFrame) -> dict:
    """Forecast next month's demand per product using a simple ARIMA(1,0,0)."""
    forecasts = {}
    for product, group in df.groupby('product'):
        series = group.set_index('date')['demand'].resample('ME').sum()
        model = ARIMA(series, order=(1, 0, 0)).fit()
        forecast = model.forecast(1).iloc[0]
        forecasts[product] = forecast
    return forecasts


def economic_order_quantity(annual_demand: float, order_cost: float = 100, holding_cost: float = 5) -> float:
    return np.sqrt(2 * annual_demand * order_cost / holding_cost)


def optimize_orders(forecasts: dict, budget: float = 5000) -> dict:
    unit_costs = {p: 20 + i * 5 for i, p in enumerate(sorted(forecasts))}
    prob = LpProblem('reorder_plan', LpMinimize)
    order = {p: LpVariable(f'order_{p}', lowBound=0) for p in forecasts}
    shortage = {p: LpVariable(f'short_{p}', lowBound=0) for p in forecasts}
    prob += lpSum(unit_costs[p] * order[p] + 50 * shortage[p] for p in forecasts)
    prob += lpSum(unit_costs[p] * order[p] for p in forecasts) <= budget
    for p in forecasts:
        prob += order[p] + shortage[p] >= forecasts[p]
    prob.solve()
    return {p: order[p].value() for p in forecasts}


def main() -> None:
    if not os.path.exists(RAW_PATH):
        generate_synthetic_data(RAW_PATH)
        print(f"Synthetic data written to {RAW_PATH}")

    raw = extract(RAW_PATH)
    clean = transform(raw)
    load(clean, CLEAN_PATH)
    print(f"Cleaned data written to {CLEAN_PATH}")

    forecasts = forecast_monthly_demand(clean)
    print("Next-month demand forecast:")
    for product, demand in forecasts.items():
        eoq = economic_order_quantity(annual_demand=demand * 12)
        print(f"  Product {product}: forecast={demand:.1f}, EOQ={eoq:.1f}")

    orders = optimize_orders(forecasts)
    print("Suggested order quantities (capacity constrained):")
    for product, qty in orders.items():
        print(f"  {product}: {qty:.1f} units")


if __name__ == "__main__":
    main()
