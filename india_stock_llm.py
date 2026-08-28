from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class StockExample:
    """Single daily stock close value."""

    date: str
    close: float


def load_india_stock_csv(path: str) -> List[StockExample]:
    """Load Indian stock prices from a CSV with Date and Close columns."""
    rows: List[StockExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Date" not in reader.fieldnames or "Close" not in reader.fieldnames:
            raise ValueError("CSV must include Date and Close columns")
        for row in reader:
            rows.append(StockExample(date=row["Date"], close=float(row["Close"])))
    if len(rows) < 3:
        raise ValueError("Need at least 3 rows of stock data")
    return rows


def build_training_windows(
    prices: Sequence[float],
    context_window: int = 5,
    horizon: int = 1,
) -> Tuple[List[List[float]], List[float]]:
    if context_window < 1:
        raise ValueError("context_window must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(prices) <= context_window + horizon - 1:
        raise ValueError("Not enough prices for requested window/horizon")

    X: List[List[float]] = []
    y: List[float] = []
    max_start = len(prices) - context_window - horizon + 1
    for start in range(max_start):
        end = start + context_window
        X.append(list(prices[start:end]))
        y.append(prices[end + horizon - 1])
    return X, y


class TinyIndiaStockLLM:
    """Tiny autoregressive sequence model for Indian stock close prediction."""

    def __init__(self, context_window: int = 5):
        if context_window < 1:
            raise ValueError("context_window must be >= 1")
        self.context_window = context_window
        self.weights = [0.0] * context_window
        self.bias = 0.0
        self._mean = 0.0
        self._std = 1.0
        self._is_trained = False

    def train(self, prices: Sequence[float], epochs: int = 300, lr: float = 0.005) -> None:
        price_mean = sum(prices) / len(prices)
        var = sum((p - price_mean) ** 2 for p in prices) / len(prices)
        price_std = var**0.5 or 1.0
        normalized_prices = [(p - price_mean) / price_std for p in prices]
        X, y = build_training_windows(normalized_prices, context_window=self.context_window)

        weights = self.weights[:]
        bias = self.bias

        for _ in range(epochs):
            grad_w = [0.0] * self.context_window
            grad_b = 0.0
            n = len(X)
            for features, target in zip(X, y):
                pred = sum(w * f for w, f in zip(weights, features)) + bias
                err = pred - target
                for i in range(self.context_window):
                    grad_w[i] += (2.0 / n) * err * features[i]
                grad_b += (2.0 / n) * err
            for i in range(self.context_window):
                weights[i] -= lr * grad_w[i]
            bias -= lr * grad_b

        self.weights = weights
        self.bias = bias
        self._mean = price_mean
        self._std = price_std
        self._is_trained = True

    def predict_next(self, recent_prices: Sequence[float]) -> float:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if len(recent_prices) != self.context_window:
            raise ValueError(f"recent_prices must contain {self.context_window} values")
        normalized = [(p - self._mean) / self._std for p in recent_prices]
        pred = sum(w * f for w, f in zip(self.weights, normalized)) + self.bias
        return pred * self._std + self._mean

    def predict_days(self, recent_prices: Sequence[float], days: int) -> List[float]:
        if days < 1:
            raise ValueError("days must be >= 1")
        window = list(recent_prices)
        if len(window) != self.context_window:
            raise ValueError(f"recent_prices must contain {self.context_window} values")

        out: List[float] = []
        for _ in range(days):
            nxt = self.predict_next(window)
            out.append(nxt)
            window = window[1:] + [nxt]
        return out


def train_india_stock_model_from_csv(
    csv_path: str,
    context_window: int = 5,
) -> TinyIndiaStockLLM:
    data = load_india_stock_csv(csv_path)
    prices = [r.close for r in data]
    model = TinyIndiaStockLLM(context_window=context_window)
    model.train(prices)
    return model
