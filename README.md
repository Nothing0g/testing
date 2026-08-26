# testing

Minimal India stock "LLM-like" predictor implementation.

## What was added

- `india_stock_llm.py`: tiny autoregressive sequence model that can be trained on Indian stock CSV data (`Date`, `Close`) and used to predict future close values.
- `tests/test_india_stock_llm.py`: focused tests validating training-window creation and next-day prediction behavior.

## Quick usage

```python
from india_stock_llm import train_india_stock_model_from_csv

model = train_india_stock_model_from_csv("nse_stock.csv", context_window=5)
next_day = model.predict_next([2450.0, 2462.5, 2470.1, 2468.9, 2475.0])
print(next_day)
```
