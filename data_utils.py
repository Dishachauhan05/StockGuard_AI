from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "product", "daily_sales", "stock", "lead_time"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_PATH = DATA_DIR / "demo_inventory.csv"


def generate_demo_data(path: str | Path = DEMO_PATH, seed: int = 42) -> pd.DataFrame:
    """Create reproducible synthetic daily sales and inventory records."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=150, freq="D")
    products = {
        "Milk 1L": (28, 5, 2, 0.10),
        "Bread Loaf": (18, 4, 1, 0.14),
        "Rice 5kg": (7, 2, 4, 0.08),
        "Cooking Oil 1L": (10, 3, 3, 0.12),
        "Eggs Dozen": (15, 4, 2, 0.18),
    }
    rows = []
    for product, (base, seasonal, lead_time, noise) in products.items():
        phase = rng.uniform(0, 2 * np.pi)
        sales_values = []
        for i, date in enumerate(dates):
            weekly = 1.0 + 0.18 * np.sin(2 * np.pi * i / 7 + phase)
            trend = 1.0 + 0.0008 * i
            promo = 1.35 if i in (35, 70, 105, 130) else 1.0
            expected = max(0.5, base * weekly * trend * promo)
            sale = max(0, int(round(rng.normal(expected, max(1.0, base * noise)))))
            sales_values.append(sale)
        stock = max(5, int(np.mean(sales_values[-14:]) * (lead_time + 2)))
        for date, sale in zip(dates, sales_values):
            # Daily stock is a plausible snapshot, not a fully simulated stock ledger.
            stock = max(0, stock - sale)
            if stock < max(5, base * lead_time):
                stock += int(round(base * (lead_time + 2)))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product": product,
                "daily_sales": sale,
                "stock": int(stock),
                "lead_time": lead_time,
            })
    df = pd.DataFrame(rows).sort_values(["product", "date"]).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_demo_data() -> pd.DataFrame:
    if not DEMO_PATH.exists():
        return generate_demo_data()
    return validate_and_prepare(pd.read_csv(DEMO_PATH))


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded data and return clean, consistently typed records."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing) +
                         ". Required columns are: " + ", ".join(REQUIRED_COLUMNS) + ".")
    out = df[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(out["date"].isna().sum())
    if bad_dates:
        raise ValueError(f"{bad_dates} row(s) have missing or invalid dates. Correct them and upload again.")
    out["product"] = out["product"].astype("string").str.strip()
    if out["product"].isna().any() or (out["product"] == "").any():
        raise ValueError("The product column contains blank product names.")
    for col in ["daily_sales", "stock", "lead_time"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"The {col} column contains missing or non-numeric values.")
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise ValueError(f"The {col} column contains infinite values.")
    if (out["daily_sales"] < 0).any():
        raise ValueError("daily_sales cannot be negative.")
    if (out["stock"] < 0).any():
        raise ValueError("stock cannot be negative.")
    if (out["lead_time"] < 1).any():
        raise ValueError("lead_time must be at least 1 day.")
    if not np.allclose(out["daily_sales"], np.round(out["daily_sales"])):
        raise ValueError("daily_sales must contain whole units.")
    if not np.allclose(out["stock"], np.round(out["stock"])):
        raise ValueError("stock must contain whole units.")
    if not np.allclose(out["lead_time"], np.round(out["lead_time"])):
        raise ValueError("lead_time must contain whole days.")
    out[["daily_sales", "stock", "lead_time"]] = out[["daily_sales", "stock", "lead_time"]].astype(int)
    out = out.sort_values(["product", "date"]).drop_duplicates(["product", "date"], keep="last")
    if out.empty:
        raise ValueError("No valid rows were found.")
    return out.reset_index(drop=True)


def product_history(df: pd.DataFrame, product: str) -> pd.DataFrame:
    history = df.loc[df["product"] == product].sort_values("date").copy()
    if history.empty:
        raise ValueError(f"No rows found for product: {product}")
    if history["date"].duplicated().any():
        raise ValueError("There must be no more than one row per product per date.")
    return history.reset_index(drop=True)
