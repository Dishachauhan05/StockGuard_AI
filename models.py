from __future__ import annotations

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _pgmpy_components():
    try:
        from pgmpy.models import BayesianNetwork
    except ImportError:
        try:
            from pgmpy.models import BayesianModel as BayesianNetwork
        except ImportError as exc:
            raise ImportError("pgmpy is required for Bayesian Network inference. Install dependencies with: pip install -r requirements.txt") from exc
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    return BayesianNetwork, TabularCPD, VariableElimination

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DemandLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(sales, lookback: int = 14, epochs: int = 35, learning_rate: float = 0.01):
    """Train an LSTM with chronological target split and train-only scaling."""
    set_seeds()
    values = np.asarray(sales, dtype=np.float32).reshape(-1, 1)
    n = len(values)
    if lookback < 2:
        raise ValueError("Lookback must be at least 2 days.")
    if n < max(lookback + 12, 30):
        raise ValueError(f"Need at least {max(lookback + 12, 30)} daily observations; received {n}.")
    split = int(n * 0.8)
    if split <= lookback or n - split < 2:
        raise ValueError("Not enough observations for a chronological train/test split.")

    scaler = MinMaxScaler()
    scaler.fit(values[:split])  # fit only on training-period values
    scaled = scaler.transform(values).astype(np.float32)

    X_train, y_train, X_test, y_test = [], [], [], []
    for target_idx in range(lookback, n):
        x_seq = scaled[target_idx-lookback:target_idx]
        if target_idx < split:
            X_train.append(x_seq)
            y_train.append(scaled[target_idx])
        else:
            # Test windows may use earlier observed values, but never future targets.
            X_test.append(x_seq)
            y_test.append(scaled[target_idx])
    Xtr = torch.tensor(np.array(X_train), dtype=torch.float32)
    ytr = torch.tensor(np.array(y_train), dtype=torch.float32)
    Xte = torch.tensor(np.array(X_test), dtype=torch.float32)
    yte = torch.tensor(np.array(y_test), dtype=torch.float32)

    model = DemandLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    with torch.no_grad():
        test_scaled_pred = model(Xte).cpu().numpy()
    actual = scaler.inverse_transform(yte.cpu().numpy()).reshape(-1)
    predicted = scaler.inverse_transform(test_scaled_pred).reshape(-1)
    predicted = np.maximum(0, predicted)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    evaluation = pd.DataFrame({
        "test_index": np.arange(split, n),
        "actual": actual,
        "predicted": predicted,
    })
    return {
        "model": model, "scaler": scaler, "losses": losses, "mae": mae, "rmse": rmse,
        "evaluation": evaluation, "lookback": lookback, "split_index": split,
        "observations": n, "train_count": len(X_train), "test_count": len(X_test),
    }


def forecast_future(sales, trained, horizon: int = 7) -> pd.DataFrame:
    if horizon < 1 or horizon > 30:
        raise ValueError("Forecast horizon must be between 1 and 30 days.")
    lookback = trained["lookback"]
    scaler = trained["scaler"]
    values = np.asarray(sales, dtype=np.float32).reshape(-1, 1)
    if len(values) < lookback:
        raise ValueError("Not enough sales history for the selected lookback.")
    scaled_seq = scaler.transform(values).astype(np.float32).reshape(-1).tolist()
    model = trained["model"]
    predictions = []
    model.eval()
    for _ in range(horizon):
        x = torch.tensor(scaled_seq[-lookback:], dtype=torch.float32).reshape(1, lookback, 1)
        with torch.no_grad():
            next_scaled = float(model(x).item())
        next_value = max(0.0, float(scaler.inverse_transform([[next_scaled]])[0, 0]))
        predictions.append(next_value)
        scaled_seq.append(next_scaled)
    return pd.DataFrame({"day_ahead": np.arange(1, horizon + 1), "predicted_demand": predictions})


def _risk_cpd_values():
    """Transparent illustrative probabilities derived from a simple risk score."""
    cols = []
    stock_states = ["Low", "Medium", "High"]
    demand_states = ["Low", "Medium", "High"]
    delivery_states = ["Short", "Long"]
    for stock in stock_states:
        for demand in demand_states:
            for delivery in delivery_states:
                score = {"Low": 2, "Medium": 1, "High": 0}[stock]
                score += {"Low": 0, "Medium": 1, "High": 2}[demand]
                score += {"Short": 0, "Long": 1}[delivery]
                if score <= 1:
                    probs = (0.80, 0.17, 0.03)
                elif score <= 3:
                    probs = (0.25, 0.55, 0.20)
                else:
                    probs = (0.05, 0.20, 0.75)
                cols.append(probs)
    return np.asarray(cols, dtype=float).T


def build_bayesian_network():
    """Build a valid BN. Its CPT values are demo assumptions, not learned data."""
    BayesianNetwork, TabularCPD, _ = _pgmpy_components()
    model = BayesianNetwork([
        ("StockLevel", "StockoutRisk"),
        ("LeadDemand", "StockoutRisk"),
        ("DeliveryTime", "StockoutRisk"),
    ])
    cpd_stock = TabularCPD("StockLevel", 3, [[1/3], [1/3], [1/3]],
                           state_names={"StockLevel": ["Low", "Medium", "High"]})
    cpd_demand = TabularCPD("LeadDemand", 3, [[1/3], [1/3], [1/3]],
                            state_names={"LeadDemand": ["Low", "Medium", "High"]})
    cpd_delivery = TabularCPD("DeliveryTime", 2, [[0.5], [0.5]],
                              state_names={"DeliveryTime": ["Short", "Long"]})
    cpd_risk = TabularCPD(
        "StockoutRisk", 3, _risk_cpd_values(),
        evidence=["StockLevel", "LeadDemand", "DeliveryTime"],
        evidence_card=[3, 3, 2],
        state_names={
            "StockoutRisk": ["Low", "Medium", "High"],
            "StockLevel": ["Low", "Medium", "High"],
            "LeadDemand": ["Low", "Medium", "High"],
            "DeliveryTime": ["Short", "Long"],
        },
    )
    model.add_cpds(cpd_stock, cpd_demand, cpd_delivery, cpd_risk)
    if not model.check_model():
        raise ValueError("Bayesian Network validation failed.")
    return model


def discretize_level(value: float, low_cut: float, high_cut: float) -> str:
    if value < low_cut:
        return "Low"
    if value < high_cut:
        return "Medium"
    return "High"


def estimate_stockout_risk(current_stock: float, lead_demand: float, lead_time: int,
                            safety_stock: float = 0.0) -> dict:
    """Infer stockout-risk probabilities from demo CPT assumptions."""
    _, _, VariableElimination = _pgmpy_components()
    model = build_bayesian_network()
    inference = VariableElimination(model)
    stock_level = discretize_level(current_stock, max(1.0, lead_demand * 0.75),
                                   max(2.0, lead_demand * 1.25))
    demand_level = discretize_level(lead_demand, max(1.0, current_stock * 0.75),
                                    max(2.0, current_stock * 1.25))
    delivery_state = "Long" if lead_time >= 3 else "Short"
    result = inference.query(
        variables=["StockoutRisk"],
        evidence={"StockLevel": stock_level, "LeadDemand": demand_level, "DeliveryTime": delivery_state},
        show_progress=False,
    )
    states = result.state_names["StockoutRisk"]
    probabilities = {state: float(result.values[i]) for i, state in enumerate(states)}
    category = max(probabilities, key=probabilities.get)
    return {
        "probabilities": probabilities,
        "category": category,
        "evidence": {"StockLevel": stock_level, "LeadDemand": demand_level, "DeliveryTime": delivery_state},
        "assumption_note": "Illustrative demo CPTs based on a transparent risk-score rule; not learned from real stockout labels.",
    }


def calculate_inventory_recommendation(sales, forecast, current_stock: int, lead_time: int,
                                       safety_factor: float = 1.65, target_days: int = 7) -> dict:
    """Compute reorder point and suggested quantity from forecasts and sales variability."""
    sales = np.asarray(sales, dtype=float)
    if len(sales) < 2:
        raise ValueError("At least two sales observations are required.")
    forecast_values = np.asarray(forecast["predicted_demand"], dtype=float)
    if len(forecast_values) < lead_time:
        raise ValueError("Forecast horizon must cover the supplier lead time.")
    lead_demand = float(np.sum(forecast_values[:lead_time]))
    daily_std = float(np.std(sales[-min(28, len(sales)):], ddof=1))
    safety_stock = float(max(0.0, safety_factor * daily_std * np.sqrt(max(1, lead_time))))
    reorder_point = lead_demand + safety_stock
    target_demand_days = min(max(lead_time + 1, target_days), len(forecast_values))
    target_stock = float(np.sum(forecast_values[:target_demand_days]) + safety_stock)
    suggested = max(0, int(np.ceil(target_stock - current_stock)))
    return {
        "lead_time_demand": lead_demand,
        "daily_sales_std": daily_std,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,
        "suggested_quantity": suggested,
        "is_below_reorder_point": current_stock <= reorder_point,
    }
