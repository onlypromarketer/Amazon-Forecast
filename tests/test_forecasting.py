"""Tests for the TimesFM wrapper: transforms, dates, clipping, summaries."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from forecast_app import forecasting as F
from forecast_app.series import Prepared


@dataclasses.dataclass
class _Out:
  forecast: np.ndarray
  quantiles: np.ndarray


class FakeModel:
  """Predicts a flat line at the last value, with a +/-20% band (9 deciles)."""

  def __init__(self):
    self.calls = []

  def predict_batch(self, contexts, horizon, return_quantiles, make_positive, **kw):
    self.calls.append([np.array(c) for c in contexts])
    for c in contexts:
      last = float(c[-1])
      fc = np.full(horizon, last, dtype=np.float32)
      mult = np.linspace(0.8, 1.2, 9)
      q = np.tile(last * mult, (horizon, 1)).astype(np.float32)
      yield _Out(forecast=fc, quantiles=q)


def prep(values, start="2024-01-01"):
  v = np.asarray(values, dtype=np.float32)
  return Prepared(ok=True, values=v, dates=pd.date_range(start, periods=len(v)))


def test_future_dates_start_day_after_history():
  out = F.forecast_metrics(FakeModel(), {"price": prep([10.0] * 20)}, horizon=5)
  fc = out["price"]
  assert fc.index[0] == pd.Timestamp("2024-01-21")
  assert len(fc) == 5
  assert list(fc.columns) == F.OUTPUT_COLUMNS


def test_all_metrics_go_through_one_batch_call():
  m = FakeModel()
  F.forecast_metrics(m, {"price": prep([1.0] * 20), "reviews": prep([5.0] * 20)}, 3)
  assert len(m.calls) == 1 and len(m.calls[0]) == 2


def test_bsr_is_modelled_in_log_space_and_returned_in_rank_space():
  m = FakeModel()
  out = F.forecast_metrics(m, {"bsr": prep([1000.0] * 20)}, horizon=3)
  sent = m.calls[0][0]
  assert sent[-1] == pytest.approx(np.log(1000.0), rel=1e-5)
  assert out["bsr"]["median"].iloc[0] == pytest.approx(1000.0, rel=1e-4)
  # Band is +/-20% in *log* space -> asymmetric in rank space, still ordered.
  row = out["bsr"].iloc[0]
  assert row["p10"] < row["median"] < row["p90"]


def test_rating_is_clipped_to_five_stars():
  out = F.forecast_metrics(FakeModel(), {"rating": prep([4.8] * 20)}, horizon=2)
  assert out["rating"]["p90"].max() <= 5.0


def test_prices_never_negative():
  class Neg(FakeModel):
    def predict_batch(self, contexts, horizon, **kw):
      for _ in contexts:
        yield _Out(np.full(horizon, -1.0), np.full((horizon, 9), -1.0))

  out = F.forecast_metrics(Neg(), {"price": prep([1.0] * 20)}, horizon=2)
  assert (out["price"].to_numpy() >= 0).all()


def test_quantile_columns_are_monotonic():
  out = F.forecast_metrics(FakeModel(), {"price": prep([10.0] * 20)}, horizon=4)
  q = out["price"][["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]]
  assert (np.diff(q.to_numpy(), axis=1) >= 0).all()


def test_not_ok_series_are_skipped():
  bad = Prepared(ok=False, reason="too short")
  out = F.forecast_metrics(FakeModel(), {"price": bad}, horizon=3)
  assert out == {}


def test_nothing_to_forecast_makes_no_model_call():
  m = FakeModel()
  assert F.forecast_metrics(m, {}, horizon=3) == {}
  assert m.calls == []


# ------------------------------------------------------------------- summarize


def _fc(median_end, low_end, high_end, n=3):
  idx = pd.date_range("2024-02-01", periods=n)
  df = pd.DataFrame(0.0, index=idx, columns=F.OUTPUT_COLUMNS)
  df.iloc[-1, df.columns.get_loc("median")] = median_end
  df.iloc[-1, df.columns.get_loc("p10")] = low_end
  df.iloc[-1, df.columns.get_loc("p90")] = high_end
  return df


def test_summary_change_percent():
  row = F.summarize("price", prep([20.0] * 20), _fc(25.0, 22.0, 28.0))
  assert row["current"] == 20.0
  assert row["forecast"] == 25.0
  assert row["change_pct"] == pytest.approx(25.0)
  assert row["low"] == 22.0 and row["high"] == 28.0
  assert row["trend"] == "up"


def test_summary_zero_current_gives_no_percent_not_infinity():
  row = F.summarize("monthly_sold", prep([0.0] * 20), _fc(50.0, 0.0, 100.0))
  assert row["change_pct"] is None


def test_summary_bsr_going_down_is_improving():
  row = F.summarize("bsr", prep([5000.0] * 20), _fc(4000.0, 3000.0, 6000.0))
  assert row["trend"] == "improving"


def test_summary_bsr_going_up_is_worsening():
  row = F.summarize("bsr", prep([5000.0] * 20), _fc(6000.0, 5000.0, 8000.0))
  assert row["trend"] == "worsening"


def test_summary_small_change_is_flat():
  row = F.summarize("price", prep([20.0] * 20), _fc(20.1, 19.0, 21.0))
  assert row["trend"] == "flat"
