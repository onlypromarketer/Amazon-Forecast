"""Tests for turning raw Keepa product data into clean daily series."""

import numpy as np
import pandas as pd
import pytest

from forecast_app import series as S

# Keepa time 0 == 2011-01-01 00:00 UTC. One day == 1440 Keepa minutes.
DAY = 1440


def km(day_offset, minute=0):
  """Keepa minutes for 2011-01-01 + day_offset days (+ minutes)."""
  return day_offset * DAY + minute


def empty_csv():
  return [None] * 36


# --------------------------------------------------------------------------- time


def test_keepa_minutes_to_utc_epoch():
  assert S.keepa_minutes_to_utc(0) == pd.Timestamp("2011-01-01 00:00", tz="UTC")


def test_keepa_minutes_to_utc_known_date():
  # 2024-01-01 is 4748 days after 2011-01-01.
  assert S.keepa_minutes_to_utc(4748 * DAY + 90) == pd.Timestamp(
    "2024-01-01 01:30", tz="UTC"
  )


# ----------------------------------------------------------------- raw extraction


def test_price_is_cents_to_dollars_and_negative_is_nan():
  csv = empty_csv()
  # Buy Box is a triplet stream: time, price, shipping.
  csv[S.IDX_BUY_BOX] = [km(0), 1999, 0, km(1), -1, -1, km(2), 2499, 100]
  raw = S.extract_raw_series({"csv": csv})
  s = raw["price"]
  assert s.iloc[0] == pytest.approx(19.99)
  assert np.isnan(s.iloc[1])  # no Buy Box / out of stock
  assert s.iloc[2] == pytest.approx(25.99)  # price + shipping


def test_buy_box_price_minus_one_with_zero_shipping_is_nan():
  # -1 price means "no buy box" even if shipping field is 0; must not become $-0.01.
  csv = empty_csv()
  csv[S.IDX_BUY_BOX] = [km(0), -1, 0]
  raw = S.extract_raw_series({"csv": csv})
  assert np.isnan(raw["price"].iloc[0])


def test_rating_scaled_to_stars():
  csv = empty_csv()
  csv[S.IDX_RATING] = [km(0), 45, km(3), 43]
  raw = S.extract_raw_series({"csv": csv})
  assert list(raw["rating"]) == [4.5, 4.3]


def test_bsr_and_reviews_are_integers_not_scaled():
  csv = empty_csv()
  csv[S.IDX_SALES_RANK] = [km(0), 12000, km(1), -1]
  csv[S.IDX_REVIEWS] = [km(0), 350]
  raw = S.extract_raw_series({"csv": csv})
  assert raw["bsr"].iloc[0] == 12000
  assert np.isnan(raw["bsr"].iloc[1])
  assert raw["reviews"].iloc[0] == 350


def test_monthly_sold_history_pairs():
  product = {"csv": empty_csv(), "monthlySoldHistory": [km(0), 100, km(10), 200]}
  raw = S.extract_raw_series(product)
  assert list(raw["monthly_sold"]) == [100, 200]
  assert raw["monthly_sold"].index[1] == pd.Timestamp("2011-01-11", tz="UTC")


def test_missing_streams_are_simply_absent():
  raw = S.extract_raw_series({"csv": empty_csv()})
  assert raw == {}


def test_price_falls_back_to_new_price_when_no_buy_box_stream():
  csv = empty_csv()
  csv[S.IDX_NEW] = [km(0), 1500]
  raw = S.extract_raw_series({"csv": csv})
  assert raw["price"].iloc[0] == pytest.approx(15.0)


def test_csv_none_does_not_crash():
  assert S.extract_raw_series({"csv": None}) == {}


# -------------------------------------------------------------------- to_daily


def _events(pairs):
  idx = [S.keepa_minutes_to_utc(t) for t, _ in pairs]
  return pd.Series([v for _, v in pairs], index=idx, dtype=float)


def test_to_daily_holds_value_until_next_change():
  # Keepa only logs changes: value 10 on day 0, 20 on day 3.
  s = _events([(km(0), 10), (km(3), 20)])
  d = S.to_daily(s, end=pd.Timestamp("2011-01-06", tz="UTC"))
  assert list(d.values) == [10, 10, 10, 20, 20, 20]
  assert d.index[0] == pd.Timestamp("2011-01-01")


def test_to_daily_uses_last_value_of_day():
  s = _events([(km(0, 60), 10), (km(0, 600), 12)])
  d = S.to_daily(s, end=pd.Timestamp("2011-01-01", tz="UTC"))
  assert list(d.values) == [12]


def test_to_daily_keeps_out_of_stock_gap_as_nan():
  s = _events([(km(0), 10), (km(2), np.nan), (km(4), 11)])
  d = S.to_daily(s, end=pd.Timestamp("2011-01-06", tz="UTC"))
  assert list(d.values[:2]) == [10, 10]
  assert np.isnan(d.values[2]) and np.isnan(d.values[3])
  assert list(d.values[4:]) == [11, 11]


def test_to_daily_extends_last_value_to_fetch_date():
  s = _events([(km(0), 5)])
  d = S.to_daily(s, end=pd.Timestamp("2011-01-10", tz="UTC"))
  assert len(d) == 10 and (d == 5).all()


def test_to_daily_empty():
  d = S.to_daily(pd.Series(dtype=float), end=pd.Timestamp("2011-01-10", tz="UTC"))
  assert d.empty


# --------------------------------------------------------------- prepare / quality


def test_prepare_trims_window_and_reports_gaps():
  idx = pd.date_range("2024-01-01", periods=100, freq="D")
  vals = np.arange(100, dtype=float)
  vals[50:60] = np.nan
  daily = pd.Series(vals, index=idx)
  prep = S.prepare_for_model(daily, max_days=80, min_days=14)
  assert prep.ok
  assert len(prep.values) == 80
  assert prep.missing_days == 10
  assert not np.isnan(prep.values).any()  # gaps filled for the model
  assert prep.last_date == pd.Timestamp("2024-04-09")


def test_prepare_drops_leading_nans():
  idx = pd.date_range("2024-01-01", periods=40, freq="D")
  vals = np.full(40, np.nan)
  vals[20:] = 3.0
  prep = S.prepare_for_model(pd.Series(vals, index=idx), max_days=365, min_days=14)
  assert prep.ok and len(prep.values) == 20 and prep.missing_days == 0


def test_prepare_rejects_too_short_history():
  idx = pd.date_range("2024-01-01", periods=10, freq="D")
  prep = S.prepare_for_model(pd.Series(1.0, index=idx), max_days=365, min_days=14)
  assert not prep.ok
  assert "10 days" in prep.reason


def test_prepare_rejects_all_missing():
  idx = pd.date_range("2024-01-01", periods=30, freq="D")
  prep = S.prepare_for_model(pd.Series(np.nan, index=idx), max_days=365, min_days=14)
  assert not prep.ok


def test_prepare_rejects_mostly_missing_window():
  idx = pd.date_range("2024-01-01", periods=100, freq="D")
  vals = np.full(100, np.nan)
  vals[0] = 1.0
  vals[-1] = 2.0
  prep = S.prepare_for_model(pd.Series(vals, index=idx), max_days=365, min_days=14)
  assert not prep.ok
  assert "missing" in prep.reason


# ------------------------------------------------- listing merge / split breaks


def _d(vals, start="2024-01-01"):
  return pd.Series(np.asarray(vals, dtype=float),
                   index=pd.date_range(start, periods=len(vals)))


def test_find_breaks_detects_merge_jump_and_split_drop():
  vals = [3] * 10 + [2500 + i for i in range(20)] + [40 + i for i in range(20)]
  breaks = S.find_level_breaks(_d(vals))
  assert breaks == [pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-31")]


def test_find_breaks_ignores_normal_growth_and_small_counts():
  vals = [100 + 3 * i for i in range(60)] + [1, 2, 5, 9]  # 1 -> 5 is tiny
  assert S.find_level_breaks(_d(vals[:60])) == []
  assert S.find_level_breaks(_d([1, 2, 5, 9, 12])) == []


def test_find_breaks_ignores_gaps():
  vals = [100.0] * 5 + [np.nan] * 3 + [101.0] * 5
  assert S.find_level_breaks(_d(vals)) == []


def test_prepare_break_aware_uses_only_data_after_last_break():
  vals = [3] * 30 + [2500 + i for i in range(40)]
  prep = S.prepare_for_model(_d(vals), max_days=365, min_days=14, break_aware=True)
  assert prep.ok and len(prep.values) == 40
  assert prep.trimmed_at == pd.Timestamp("2024-01-31")


def test_prepare_break_aware_rejects_when_too_little_after_break():
  vals = [3] * 60 + [2500] * 5
  prep = S.prepare_for_model(_d(vals), max_days=365, min_days=14, break_aware=True)
  assert not prep.ok
  assert "merge" in prep.reason and "2024-03-01" in prep.reason


def test_prepare_without_break_awareness_keeps_everything():
  vals = [3] * 30 + [2500 + i for i in range(40)]
  prep = S.prepare_for_model(_d(vals), max_days=365, min_days=14)
  assert len(prep.values) == 70 and prep.trimmed_at is None


# ------------------------------------------------------------ recent change


def test_days_since_last_change():
  assert S.days_since_last_change(_d([10] * 20 + [20] * 3)) == 2
  assert S.days_since_last_change(_d([10] * 20)) is None
  assert S.days_since_last_change(_d([10] * 5 + [np.nan] * 3 + [10] * 2)) is None


# ---------------------------------------------------------------- glitch blips


def test_remove_blips_one_day_glitch_becomes_missing():
  vals = [5000.0] * 10 + [2452.0] + [5080.0] * 10
  cleaned, n = S.remove_blips(_d(vals))
  assert n == 1 and np.isnan(cleaned.iloc[10])
  assert S.find_level_breaks(cleaned) == []


def test_remove_blips_two_day_glitch():
  vals = [100.0] * 10 + [1400.0, 1460.0] + [104.0] * 10
  cleaned, n = S.remove_blips(_d(vals))
  assert n == 2 and cleaned.isna().sum() == 2


def test_remove_blips_keeps_real_level_shift():
  vals = [100.0] * 10 + [2500.0] * 10
  cleaned, n = S.remove_blips(_d(vals))
  assert n == 0 and not cleaned.isna().any()


def test_break_aware_prepare_ignores_glitch():
  vals = [5000.0 + i for i in range(60)] + [2452.0] + [5080.0 + i for i in range(20)]
  prep = S.prepare_for_model(_d(vals), max_days=365, min_days=14, break_aware=True)
  assert prep.ok and prep.trimmed_at is None and len(prep.values) == 81
  assert prep.missing_days == 1
