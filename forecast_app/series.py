"""Turn raw Keepa product data into clean daily time series.

Keepa encoding rules this module handles:
  * Timestamps are "Keepa minutes": minutes since 2011-01-01 00:00 UTC.
  * History streams only log *changes*; a value holds until the next entry.
  * -1 / -2 mean "no data" (out of stock, no Buy Box, no rank) -> NaN.
  * Prices are integer cents. Ratings are 0-50 (45 == 4.5 stars).
  * The Buy Box stream is triplets (time, price, shipping); others are pairs.
All daily buckets are UTC calendar days.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

KEEPA_EPOCH = pd.Timestamp("2011-01-01", tz="UTC")

# Indices into product["csv"] (Keepa Product object docs).
IDX_AMAZON = 0
IDX_NEW = 1
IDX_SALES_RANK = 3
IDX_RATING = 16
IDX_REVIEWS = 17
IDX_BUY_BOX = 18

# Metric key -> human label. Order is the display order.
METRICS = {
  "monthly_sold": "Units bought / month (Amazon badge)",
  "bsr": "Best Seller Rank (lower is better)",
  "price": "Buy Box price",
  "reviews": "Review count",
  "rating": "Star rating",
}


def keepa_minutes_to_utc(minutes) -> pd.Timestamp | pd.DatetimeIndex:
  if np.ndim(minutes) == 0:
    return KEEPA_EPOCH + pd.Timedelta(minutes=int(minutes))
  return KEEPA_EPOCH + pd.to_timedelta(np.asarray(minutes, dtype=np.int64), unit="m")


def _stream(values, stride: int, scale: float, add_shipping: bool = False):
  """Decode a flat Keepa stream into an event-indexed float Series."""
  arr = np.asarray(values, dtype=np.int64)
  n = len(arr) // stride
  if n == 0:
    return None
  arr = arr[: n * stride].reshape(n, stride)
  times, raw = arr[:, 0], arr[:, 1].astype(float)
  missing = raw < 0
  if add_shipping:
    shipping = np.where(arr[:, 2] < 0, 0, arr[:, 2])
    raw = raw + shipping
  vals = raw / scale
  vals[missing] = np.nan
  return pd.Series(vals, index=keepa_minutes_to_utc(times), dtype=float)


def extract_raw_series(product: dict) -> dict[str, pd.Series]:
  """Pull the metrics we forecast out of a raw Keepa product dict."""
  csv = product.get("csv") or []

  def get(i):
    return csv[i] if i < len(csv) and csv[i] else None

  out: dict[str, pd.Series] = {}

  if get(IDX_BUY_BOX):
    s = _stream(get(IDX_BUY_BOX), 3, 100.0, add_shipping=True)
  elif get(IDX_NEW):
    s = _stream(get(IDX_NEW), 2, 100.0)
  else:
    s = None
  if s is not None:
    out["price"] = s

  for key, idx, scale in (
    ("bsr", IDX_SALES_RANK, 1.0),
    ("reviews", IDX_REVIEWS, 1.0),
    ("rating", IDX_RATING, 10.0),
  ):
    if get(idx):
      s = _stream(get(idx), 2, scale)
      if s is not None:
        out[key] = s

  msh = product.get("monthlySoldHistory")
  if msh:
    s = _stream(msh, 2, 1.0)
    if s is not None:
      out["monthly_sold"] = s

  return out


_SENTINEL = -9.87654321e300


def to_daily(events: pd.Series, end: pd.Timestamp) -> pd.Series:
  """Step-function events -> one value per UTC day, from first event to `end`.

  Each day takes the last value logged that day; days with no entry carry the
  previous value forward. A logged "no data" (NaN) is carried forward as NaN
  too, so out-of-stock stretches stay visible as gaps.
  """
  if events is None or events.empty:
    return pd.Series(dtype=float)
  events = events.sort_index()
  end_day = pd.Timestamp(end).tz_convert("UTC").floor("D").tz_localize(None)
  days = events.index.tz_convert("UTC").floor("D").tz_localize(None)
  marked = events.fillna(_SENTINEL)
  per_day = marked.groupby(days).last()
  per_day = per_day[per_day.index <= end_day]
  if per_day.empty:
    return pd.Series(dtype=float)
  full = pd.date_range(per_day.index[0], end_day, freq="D")
  daily = per_day.reindex(full).ffill()
  return daily.replace(_SENTINEL, np.nan).astype(float)


def _valid_steps(daily: pd.Series) -> pd.DataFrame:
  """Consecutive non-missing values as (prev, cur) pairs indexed by cur's date."""
  v = daily.dropna()
  return pd.DataFrame({"prev": v.shift(1), "cur": v}).iloc[1:]


# A day-over-day move this large in a count (e.g. reviews) is not organic
# growth; it is Amazon merging or splitting the listing's variations.
BREAK_REL_CHANGE = 0.5
BREAK_ABS_CHANGE = 50


def find_level_breaks(daily: pd.Series) -> list[pd.Timestamp]:
  steps = _valid_steps(daily)
  delta = (steps["cur"] - steps["prev"]).abs()
  rel = delta / steps["prev"].abs().clip(lower=1)
  hits = steps[(rel > BREAK_REL_CHANGE) & (delta >= BREAK_ABS_CHANGE)]
  return list(hits.index)


BLIP_MAX_DAYS = 2
BLIP_RETURN_TOLERANCE = 0.1


def _is_break(prev: float, cur: float) -> bool:
  delta = abs(cur - prev)
  return delta / max(abs(prev), 1) > BREAK_REL_CHANGE and delta >= BREAK_ABS_CHANGE


def remove_blips(daily: pd.Series) -> tuple[pd.Series, int]:
  """Blank out 1-2 day Keepa glitches that jump away and snap straight back.

  Returns the cleaned series and how many days were blanked.
  """
  out = daily.copy()
  v = daily.dropna()
  vals, idx = v.to_numpy(), v.index
  blanked, i = 0, 1
  while i < len(vals):
    prev = vals[i - 1]
    if _is_break(prev, vals[i]):
      for j in range(i + 1, min(i + 1 + BLIP_MAX_DAYS, len(vals))):
        if abs(vals[j] - prev) <= BLIP_RETURN_TOLERANCE * max(abs(prev), 1):
          out.loc[idx[i:j]] = np.nan
          blanked += j - i
          i = j
          break
    i += 1
  return out, blanked


def days_since_last_change(daily: pd.Series) -> int | None:
  steps = _valid_steps(daily)
  changed = steps[steps["cur"] != steps["prev"]]
  if changed.empty:
    return None
  return int((daily.dropna().index[-1] - changed.index[-1]).days)


@dataclasses.dataclass
class Prepared:
  ok: bool
  values: np.ndarray | None = None
  dates: pd.DatetimeIndex | None = None
  missing_days: int = 0
  reason: str = ""
  # Set when history before a listing merge/split was dropped.
  trimmed_at: pd.Timestamp | None = None

  @property
  def last_date(self):
    return self.dates[-1] if self.dates is not None and len(self.dates) else None


MAX_MISSING_FRACTION = 0.5


def prepare_for_model(
  daily: pd.Series, max_days: int, min_days: int = 14, break_aware: bool = False
) -> Prepared:
  """Trim to the recent window, check quality, and fill gaps for the model.

  With break_aware=True, history before the last listing merge/split jump is
  dropped so the model does not learn from a different listing's counts.
  """
  if daily is None or daily.empty or daily.isna().all():
    return Prepared(ok=False, reason="no usable history")
  window = daily.iloc[-max_days:]
  first_valid = window.first_valid_index()
  if first_valid is None:
    return Prepared(ok=False, reason="no usable history in the selected window")
  window = window.loc[first_valid:]
  trimmed_at = None
  if break_aware:
    window, _ = remove_blips(window)
    breaks = find_level_breaks(window)
    if breaks:
      trimmed_at = breaks[-1]
      window = window.loc[trimmed_at:]
      if len(window) < min_days:
        return Prepared(
          ok=False,
          trimmed_at=trimmed_at,
          reason=(
            f"the count jumped on {trimmed_at:%Y-%m-%d} (likely a listing "
            f"variation merge/split), leaving only {len(window)} days of "
            f"comparable history (need {min_days}+)"
          ),
        )
  if len(window) < min_days:
    return Prepared(
      ok=False, reason=f"only {len(window)} days of history (need {min_days}+)"
    )
  missing = int(window.isna().sum())
  if missing / len(window) > MAX_MISSING_FRACTION:
    return Prepared(
      ok=False,
      reason=f"{missing} of {len(window)} days missing (out of stock / no data)",
    )
  filled = window.ffill().to_numpy(dtype=np.float32)
  return Prepared(
    ok=True, values=filled, dates=window.index, missing_days=missing,
    trimmed_at=trimmed_at,
  )


AMAZON_IMAGE_BASE = "https://m.media-amazon.com/images/I/"


def product_images(product: dict) -> list[str]:
  """Listing image URLs (largest available size), MAIN image first."""
  names = []
  imgs = product.get("images") or []
  imgs = sorted(imgs, key=lambda i: i.get("variant") != "MAIN")  # stable
  for img in imgs:
    name = img.get("l") or img.get("m")
    if name:
      names.append(name)
  if not names and product.get("imagesCSV"):
    names = [n.strip() for n in product["imagesCSV"].split(",") if n.strip()]
  return [AMAZON_IMAGE_BASE + n for n in dict.fromkeys(names)]
