"""Fetch product history from Keepa, with a local JSON cache to save tokens."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

CURRENCY = {
  "US": "USD", "CA": "CAD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR",
  "ES": "EUR", "JP": "JPY", "MX": "MXN", "IN": "INR", "BR": "BRL",
}

DEFAULT_MAX_AGE = timedelta(hours=24)
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


class MissingKeyError(RuntimeError):
  pass


def normalize_asin(raw: str) -> str:
  asin = raw.strip().upper()
  if not _ASIN_RE.match(asin):
    raise ValueError(f"'{raw.strip()}' is not a valid ASIN (10 letters/digits)")
  return asin


def parse_asin_list(text: str) -> list[str]:
  seen = []
  for part in re.split(r"[\s,;]+", text):
    if part:
      asin = normalize_asin(part)
      if asin not in seen:
        seen.append(asin)
  return seen


@dataclasses.dataclass
class Fetched:
  asin: str
  found: bool
  product: dict | None
  fetched_at: datetime | None
  from_cache: bool


def _default_api_factory(key: str):
  import keepa

  return keepa.Keepa(key)


class KeepaFetcher:
  def __init__(
    self,
    api_key: str,
    cache_dir: Path | str,
    domain: str = "US",
    max_age: timedelta = DEFAULT_MAX_AGE,
    api_factory: Callable = _default_api_factory,
  ):
    self.api_key = (api_key or "").strip()
    self.cache_dir = Path(cache_dir)
    self.domain = domain
    self.max_age = max_age
    self._api_factory = api_factory
    self._api = None
    self.tokens_left: int | None = None

  def _path(self, asin: str) -> Path:
    return self.cache_dir / f"{self.domain}_{asin}.json"

  def _read_cache(self, asin: str) -> Fetched | None:
    p = self._path(asin)
    if not p.exists():
      return None
    blob = json.loads(p.read_text())
    return Fetched(
      asin=asin,
      found=True,
      product=blob["product"],
      fetched_at=datetime.fromisoformat(blob["fetched_at"]),
      from_cache=True,
    )

  def _write_cache(self, asin: str, product: dict, fetched_at: datetime) -> None:
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = self._path(asin).with_suffix(".tmp")
    tmp.write_text(
      json.dumps(
        {"fetched_at": fetched_at.isoformat(), "domain": self.domain, "product": product}
      )
    )
    tmp.replace(self._path(asin))  # atomic: never leaves a half-written file

  def get(
    self, asins: list[str], now: datetime | None = None, force: bool = False
  ) -> dict[str, Fetched]:
    now = now or datetime.now(timezone.utc)
    results: dict[str, Fetched] = {}
    to_fetch = []
    for asin in asins:
      cached = None if force else self._read_cache(asin)
      if cached and now - cached.fetched_at < self.max_age:
        results[asin] = cached
      else:
        to_fetch.append(asin)

    if to_fetch:
      if not self.api_key:
        raise MissingKeyError(
          "KEEPA_API_KEY is not set. Add it to amazon-forecast/.env and restart."
        )
      if self._api is None:
        self._api = self._api_factory(self.api_key)
      responses = self._api.query(
        to_fetch,
        domain=self.domain,
        history=True,
        rating=True,
        buybox=True,
        stats=90,
        raw=True,
        progress_bar=False,
      )
      self.tokens_left = getattr(self._api, "tokens_left", None)
      by_asin = {}
      for resp in responses:
        for p in resp.json().get("products") or []:
          by_asin[p.get("asin")] = p
      for asin in to_fetch:
        p = by_asin.get(asin)
        if not p or (not p.get("title") and not p.get("csv")):
          results[asin] = Fetched(asin, False, None, now, from_cache=False)
          continue
        self._write_cache(asin, p, now)
        results[asin] = Fetched(asin, True, p, now, from_cache=False)

    return {a: results[a] for a in asins}
