"""Tests for the Keepa fetch + local cache layer (no real API calls)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from forecast_app import keepa_source as K

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
  def __init__(self, payload):
    self._payload = payload

  def json(self):
    return self._payload


class FakeApi:
  def __init__(self, products, tokens_left=500):
    self.products = products
    self.tokens_left = tokens_left
    self.queries = []

  def query(self, asins, **kwargs):
    self.queries.append((list(asins), kwargs))
    return [FakeResponse({"products": [self.products[a] for a in asins]})]


def product(asin, title="Thing"):
  return {"asin": asin, "title": title, "csv": [None] * 36}


def fetcher(tmp_path, api):
  return K.KeepaFetcher(api_key="k", cache_dir=tmp_path, api_factory=lambda key: api)


def test_normalize_asin():
  assert K.normalize_asin(" b0ggf6s2yj ") == "B0GGF6S2YJ"
  with pytest.raises(ValueError):
    K.normalize_asin("not-an-asin")


def test_parse_asin_list_dedupes_and_splits():
  assert K.parse_asin_list("B0GGF6S2YJ, b0ggf6s2yj\nB0B3S7HJ9L") == [
    "B0GGF6S2YJ",
    "B0B3S7HJ9L",
  ]


def test_first_fetch_calls_api_and_writes_cache(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  res = fetcher(tmp_path, api).get(["B0GGF6S2YJ"], now=NOW)
  assert len(api.queries) == 1
  assert res["B0GGF6S2YJ"].from_cache is False
  assert (tmp_path / "US_B0GGF6S2YJ.json").exists()


def test_query_requests_history_rating_and_buybox(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  fetcher(tmp_path, api).get(["B0GGF6S2YJ"], now=NOW)
  kw = api.queries[0][1]
  assert kw["history"] and kw["rating"] and kw["buybox"] and kw["raw"]
  assert kw["domain"] == "US"


def test_fresh_cache_is_used_without_spending_tokens(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  res = f.get(["B0GGF6S2YJ"], now=NOW + timedelta(hours=2))
  assert len(api.queries) == 1
  assert res["B0GGF6S2YJ"].from_cache is True


def test_stale_cache_refetches(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  f.get(["B0GGF6S2YJ"], now=NOW + timedelta(hours=25))
  assert len(api.queries) == 2


def test_force_refresh_refetches(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  f.get(["B0GGF6S2YJ"], now=NOW, force=True)
  assert len(api.queries) == 2


def test_only_uncached_asins_are_queried(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ"), "B0B3S7HJ9L": product("B0B3S7HJ9L")})
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  f.get(["B0GGF6S2YJ", "B0B3S7HJ9L"], now=NOW)
  assert api.queries[1][0] == ["B0B3S7HJ9L"]


def test_refetch_overwrites_single_cache_file(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ", title="v1")})
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  api.products["B0GGF6S2YJ"] = product("B0GGF6S2YJ", title="v2")
  f.get(["B0GGF6S2YJ"], now=NOW, force=True)
  files = list(tmp_path.glob("*.json"))
  assert len(files) == 1
  assert json.loads(files[0].read_text())["product"]["title"] == "v2"


def test_unknown_asin_is_flagged_and_not_cached(tmp_path):
  # Keepa returns a stub product with no title/history for unknown ASINs.
  api = FakeApi({"B000000000": {"asin": "B000000000", "title": None, "csv": None}})
  res = fetcher(tmp_path, api).get(["B000000000"], now=NOW)
  assert res["B000000000"].found is False
  assert not list(tmp_path.glob("*.json"))


def test_fetched_at_is_recorded(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  res = fetcher(tmp_path, api).get(["B0GGF6S2YJ"], now=NOW)
  assert res["B0GGF6S2YJ"].fetched_at == NOW


def test_tokens_left_reported(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")}, tokens_left=123)
  f = fetcher(tmp_path, api)
  f.get(["B0GGF6S2YJ"], now=NOW)
  assert f.tokens_left == 123


def test_missing_api_key_gives_clear_error(tmp_path):
  with pytest.raises(K.MissingKeyError):
    K.KeepaFetcher(api_key="", cache_dir=tmp_path).get(["B0GGF6S2YJ"], now=NOW)


def test_missing_key_is_fine_when_everything_is_cached(tmp_path):
  api = FakeApi({"B0GGF6S2YJ": product("B0GGF6S2YJ")})
  fetcher(tmp_path, api).get(["B0GGF6S2YJ"], now=NOW)
  res = K.KeepaFetcher(api_key="", cache_dir=tmp_path).get(["B0GGF6S2YJ"], now=NOW)
  assert res["B0GGF6S2YJ"].from_cache


def test_currency_label():
  assert K.CURRENCY["US"] == "USD" and K.CURRENCY["CA"] == "CAD"
