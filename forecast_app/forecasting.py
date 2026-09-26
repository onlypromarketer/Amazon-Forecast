"""Run TimesFM 3.0 over prepared daily series and summarise the result."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_app.series import Prepared

QUANTILE_COLUMNS = [f"p{q}" for q in range(10, 100, 10)]  # model's 9 deciles
OUTPUT_COLUMNS = ["median"] + QUANTILE_COLUMNS

# BSR spans orders of magnitude, so it is modelled as log(rank). Quantiles are
# preserved under monotonic transforms, so exp() of each quantile is exact.
_LOG_METRICS = {"bsr"}
_BOUNDS = {"rating": (1.0, 5.0), "bsr": (1.0, None)}

# Changes smaller than this (in %) are reported as "flat".
FLAT_THRESHOLD_PCT = 2.0


def load_model():
  """Load TimesFM 3.0 on Apple silicon (MLX). ~30s the first time per process."""
  from timesfm3.mlx import TimesFM3Forecaster

  return TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")


def forecast_metrics(
  model, prepared: dict[str, Prepared], horizon: int
) -> dict[str, pd.DataFrame]:
  """Forecast every usable metric in one batched model call."""
  keys = [k for k, p in prepared.items() if p.ok]
  if not keys:
    return {}

  contexts = []
  for k in keys:
    v = prepared[k].values.astype(np.float32)
    contexts.append(np.log(np.maximum(v, 1.0)) if k in _LOG_METRICS else v)

  outputs = model.predict_batch(
    contexts, horizon=horizon, return_quantiles=True, make_positive=True
  )

  results = {}
  for k, out in zip(keys, outputs):
    grid = np.column_stack([out.forecast, out.quantiles]).astype(float)
    if k in _LOG_METRICS:
      grid = np.exp(grid)
    lo, hi = _BOUNDS.get(k, (0.0, None))
    grid = np.clip(grid, lo, hi)
    grid[:, 1:] = np.sort(grid[:, 1:], axis=1)
    start = prepared[k].last_date + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=horizon, freq="D")
    results[k] = pd.DataFrame(grid, index=idx, columns=OUTPUT_COLUMNS)
  return results


def summarize(key: str, prepared: Prepared, fc: pd.DataFrame) -> dict:
  """One summary row: today vs. end of horizon, with the 80% range."""
  current = float(prepared.values[-1])
  end = fc.iloc[-1]
  forecast = float(end["median"])
  change_pct = None if current == 0 else (forecast - current) / abs(current) * 100

  if change_pct is None:
    trend = "up" if forecast > 0 else "flat"
  elif abs(change_pct) < FLAT_THRESHOLD_PCT:
    trend = "flat"
  elif key == "bsr":
    trend = "improving" if forecast < current else "worsening"
  else:
    trend = "up" if forecast > current else "down"

  return {
    "metric": key,
    "current": current,
    "forecast": forecast,
    "low": float(end["p10"]),
    "high": float(end["p90"]),
    "change_pct": change_pct,
    "trend": trend,
  }
