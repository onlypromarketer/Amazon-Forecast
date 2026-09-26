"""Tests for building Amazon image URLs from Keepa product data."""

from forecast_app import series as S

BASE = "https://m.media-amazon.com/images/I/"


def test_images_list_main_first_large_size():
  p = {"images": [
    {"l": "b.jpg", "m": "b_m.jpg", "variant": "PT01"},
    {"l": "a.jpg", "m": "a_m.jpg", "variant": "MAIN"},
  ]}
  assert S.product_images(p) == [BASE + "a.jpg", BASE + "b.jpg"]


def test_images_fall_back_to_medium_when_no_large():
  p = {"images": [{"m": "x_m.jpg", "variant": "MAIN"}]}
  assert S.product_images(p) == [BASE + "x_m.jpg"]


def test_images_from_legacy_csv_field():
  p = {"imagesCSV": "one.jpg,two.jpg"}
  assert S.product_images(p) == [BASE + "one.jpg", BASE + "two.jpg"]


def test_images_deduped_and_empty_safe():
  assert S.product_images({}) == []
  assert S.product_images({"images": None, "imagesCSV": ""}) == []
  p = {"images": [{"l": "a.jpg"}, {"l": "a.jpg"}]}
  assert S.product_images(p) == [BASE + "a.jpg"]
