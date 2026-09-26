"""Amazon Forecast: Keepa history + TimesFM 3.0 forecasts, in the browser."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from forecast_app import forecasting as F
from forecast_app import keepa_source as K
from forecast_app import series as S

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Amazon Forecast", page_icon="📈", layout="wide")

AMAZON_TLD = {"US": "com", "CA": "ca", "GB": "co.uk", "DE": "de", "FR": "fr",
              "IT": "it", "ES": "es", "JP": "co.jp", "MX": "com.mx", "IN": "in",
              "BR": "com.br"}
HISTORY_OPTIONS = {"90 days": 90, "6 months": 180, "1 year": 365, "2 years": 730,
                   "All history": 15000}
METRIC_NOTES = {
  "monthly_sold": "Amazon's public 'X+ bought in past month' badge, which Keepa "
  "records in buckets (50+, 100+, 200+ …). A rough demand signal, not exact units.",
  "bsr": "Main-category Best Seller Rank. Lower is better. Forecast in log space.",
  "price": "Buy Box price including shipping. Gaps = no Buy Box / out of stock.",
  "reviews": "Total review + rating count shown on the listing.",
  "rating": "Average star rating (1–5).",
}


@st.cache_resource(show_spinner="Loading TimesFM 3.0 (first time ~30s)…")
def get_model():
  return F.load_model()


STEP_METRICS = {"monthly_sold", "rating"}  # values that move rarely, in steps
RECENT_DAYS = 7


def just_changed(key: str, daily: pd.Series) -> int | None:
  """Days since a step metric last moved, if that was within the past week."""
  if key not in STEP_METRICS:
    return None
  d = S.days_since_last_change(daily)
  return d if d is not None and d <= RECENT_DAYS else None


def fmt(key: str, v: float | None, currency: str) -> str:
  if v is None or pd.isna(v):
    return "—"
  if key == "price":
    return f"{v:,.2f} {currency}"
  if key == "rating":
    return f"{v:.2f} ★"
  return f"{v:,.0f}"


def value_format(key: str, currency: str) -> tuple[str, str]:
  """(plotly number format, prefix/suffix template) for hover labels."""
  if key == "price":
    sym = "$" if currency in ("USD", "CAD", "MXN") else ""
    tail = "" if sym else f" {currency}"
    return f"{sym}%{{y:,.2f}}{tail}", f"{sym}{{:,.2f}}{tail}"
  if key == "rating":
    return "%{y:.2f} ★", "{:.2f} ★"
  if key == "bsr":
    return "#%{y:,.0f}", "#{:,.0f}"
  return "%{y:,.0f}", "{:,.0f}"


ZOOM_BUTTONS = [("1M", 30), ("3M", 91), ("6M", 182), ("1Y", 365), ("All", None)]


def chart(key, daily, prep, fc, currency):
  """Stock-chart style: crosshair hover, zoom buttons, range slider."""
  yfmt, pyfmt = value_format(key, currency)
  fig = go.Figure()

  if fc is not None:
    fig.add_trace(go.Scatter(
      x=fc.index, y=fc["p90"], name="High (p90)", mode="lines",
      line=dict(width=0.5, color="rgba(249,115,22,0.5)"), showlegend=False,
      hovertemplate=f"High (p90): {yfmt}<extra></extra>"))
    fig.add_trace(go.Scatter(
      x=fc.index, y=fc["p10"], name="80% range (p10–p90)", mode="lines",
      fill="tonexty", fillcolor="rgba(249,115,22,0.15)",
      line=dict(width=0.5, color="rgba(249,115,22,0.5)"),
      hovertemplate=f"Low (p10): {yfmt}<extra></extra>"))
    fig.add_trace(go.Scatter(
      x=fc.index, y=fc["p70"], mode="lines", line=dict(width=0),
      showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
      x=fc.index, y=fc["p30"], name="40% range (p30–p70)", mode="lines",
      fill="tonexty", fillcolor="rgba(249,115,22,0.30)", line=dict(width=0),
      hoverinfo="skip"))
    bridge_x = [prep.last_date] + list(fc.index)
    bridge_y = [float(prep.values[-1])] + list(fc["median"])
    fig.add_trace(go.Scatter(
      x=bridge_x, y=bridge_y, name="Forecast (median)", mode="lines",
      line=dict(color="#F97316", width=2.5, dash="dash"),
      hovertemplate=f"Forecast: {yfmt}<extra></extra>"))

  fig.add_trace(go.Scatter(
    x=daily.index, y=daily.values, name="Actual", mode="lines",
    line=dict(color="#3B82F6", width=2), connectgaps=False,
    hovertemplate=f"Actual: {yfmt}<extra></extra>"))

  # "Today" divider + latest-value tag, like a stock ticker's last price.
  last_valid = daily.dropna()
  today = daily.index[-1]
  fig.add_shape(type="line", x0=today, x1=today, y0=0, y1=1, yref="paper",
                line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot"))
  fig.add_annotation(x=today, y=0, yref="paper", text="Today", showarrow=False,
                     xanchor="left", yanchor="bottom", xshift=4,
                     font=dict(size=11, color="#94A3B8"))
  if not last_valid.empty:
    fig.add_annotation(x=last_valid.index[-1], y=float(last_valid.iloc[-1]),
                       text=pyfmt.format(float(last_valid.iloc[-1])),
                       showarrow=True, arrowhead=0, ax=-50, ay=-28,
                       bgcolor="#3B82F6", font=dict(color="white", size=12),
                       borderpad=3)

  # Zoom buttons set an explicit window: N days back from today + the forecast.
  end = fc.index[-1] if fc is not None else today
  first = daily.index[0]
  log_y = key == "bsr"

  def y_range(lo):
    """Fit the y-axis to what is visible from `lo` to the forecast end."""
    parts = [daily.loc[lo:]]
    if fc is not None:
      parts += [fc["p10"], fc["p90"]]
    vals = pd.concat(parts).dropna()
    vals = vals[vals > 0] if log_y else vals
    if vals.empty:
      return None
    ymin, ymax = float(vals.min()), float(vals.max())
    if log_y:
      a, b = np.log10(ymin), np.log10(ymax)
      pad = max((b - a) * 0.08, 0.05)
      return [b + pad, a - pad]  # reversed: best rank on top
    pad = max((ymax - ymin) * 0.08, abs(ymax) * 0.02, 0.01)
    return [ymin - pad, ymax + pad]

  buttons = []
  for label, days in ZOOM_BUTTONS:
    lo = first if days is None else max(first, today - pd.Timedelta(days=days))
    buttons.append(dict(label=label, method="relayout",
                        args=[{"xaxis.range": [lo, end], "yaxis.range": y_range(lo)}]))
  default_lo = max(first, today - pd.Timedelta(days=182))

  ytitle = S.METRICS[key] + (f" ({currency})" if key == "price" else "")
  fig.update_layout(
    height=480, margin=dict(l=10, r=10, t=50, b=10),
    hovermode="x unified", dragmode="zoom",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    updatemenus=[dict(type="buttons", direction="left", buttons=buttons,
                      x=0, xanchor="left", y=1.02, yanchor="bottom",
                      pad=dict(r=4, t=0), showactive=True, active=2,
                      bgcolor="#CBD5E1", bordercolor="#64748B",
                      font=dict(color="#0F172A", size=12))],
    hoverlabel=dict(font_size=13),
  )
  fig.update_xaxes(
    range=[default_lo, end], hoverformat="%a %b %d, %Y",
    showspikes=True, spikemode="across", spikesnap="cursor",
    spikethickness=1, spikedash="dot", spikecolor="#94A3B8",
    rangeslider=dict(visible=True, thickness=0.08))
  fig.update_yaxes(
    title=ytitle, showspikes=True, spikemode="across", spikesnap="cursor",
    spikethickness=1, spikedash="dot", spikecolor="#94A3B8",
    tickformat=",.2f" if key in ("price", "rating") else ",.0f")
  if log_y:
    fig.update_yaxes(type="log", tickformat=",.0f")
  fig.update_yaxes(range=y_range(default_lo), autorange=False)
  return fig


PLOTLY_CONFIG = {"scrollZoom": False, "displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def run(asins, domain, horizon, max_days, force):
  fetcher = K.KeepaFetcher(os.getenv("KEEPA_API_KEY", ""), ROOT / "cache", domain)
  fetched = fetcher.get(asins, force=force)
  model = get_model()
  results = []
  for asin, f in fetched.items():
    if not f.found:
      results.append({"asin": asin, "found": False})
      continue
    raw = S.extract_raw_series(f.product)
    daily = {k: S.to_daily(v, end=pd.Timestamp(f.fetched_at)) for k, v in raw.items()}
    prepared = {k: S.prepare_for_model(d, max_days=max_days, break_aware=(k == "reviews"))
                for k, d in daily.items()}
    fcs = F.forecast_metrics(model, prepared, horizon)
    results.append({"asin": asin, "found": True, "fetch": f, "daily": daily,
                    "prepared": prepared, "forecasts": fcs})
  return results, fetcher.tokens_left


# ----------------------------------------------------------------------- sidebar

st.sidebar.title("📈 Amazon Forecast")
st.sidebar.caption("Keepa history → TimesFM 3.0 forecast")
asin_text = st.sidebar.text_area("ASINs (one per line or comma-separated)",
                                 "B0GGF6S2YJ\nB0B3S7HJ9L", height=110)
domain = st.sidebar.selectbox("Marketplace", list(K.CURRENCY), index=0)
horizon = st.sidebar.slider("Forecast days ahead", 7, 180, 30, step=1)
hist_label = st.sidebar.selectbox("History the model learns from",
                                  list(HISTORY_OPTIONS), index=4)
force = st.sidebar.checkbox("Force fresh Keepa pull (spends tokens)", value=False,
                            help="Otherwise pulls younger than 24h are reused.")
go_btn = st.sidebar.button("Run forecast", type="primary", width="stretch")
st.sidebar.divider()
st.sidebar.caption(
  "Model: TimesFM 3.0 (MLX, Apple silicon). **Non-commercial license** — "
  "personal/testing use only. Days are UTC calendar days."
)
if not os.getenv("KEEPA_API_KEY"):
  st.sidebar.warning("KEEPA_API_KEY missing in .env — only cached ASINs will work.")

# -------------------------------------------------------------------------- main

if go_btn:
  try:
    asins = K.parse_asin_list(asin_text)
    if not asins:
      st.error("Enter at least one ASIN.")
      st.stop()
    with st.spinner(f"Pulling Keepa history for {len(asins)} ASIN(s) and forecasting…"):
      res, tokens = run(asins, domain, horizon, HISTORY_OPTIONS[hist_label], force)
    st.session_state["res"] = dict(results=res, tokens=tokens, domain=domain,
                                   horizon=horizon, ran_at=datetime.now(timezone.utc))
  except ValueError as e:
    st.error(str(e))
  except K.MissingKeyError as e:
    st.error(str(e))
  except Exception as e:  # Keepa auth / token / network errors
    st.error(f"Keepa request failed: {e}")

state = st.session_state.get("res")
if not state:
  st.title("Amazon Forecast")
  st.write("Enter ASINs in the sidebar and press **Run forecast**. "
           "The app pulls daily price, BSR, reviews, rating and 'bought per month' "
           "history from Keepa and forecasts each one with Google's TimesFM 3.0.")
  st.stop()

currency = K.CURRENCY.get(state["domain"], state["domain"])
horizon = state["horizon"]
if state["tokens"] is not None:
  st.caption(f"Keepa tokens left after this pull: **{state['tokens']:,}**")

def note(r, k) -> str:
  parts = []
  recent = just_changed(k, r["daily"][k])
  if recent is not None:
    parts.append(f"⚠️ new level {recent}d ago — low confidence")
  if r["prepared"][k].trimmed_at is not None:
    parts.append(f"learned since {r['prepared'][k].trimmed_at:%Y-%m-%d} merge/split")
  if r["prepared"][k].missing_days:
    parts.append(f"{r['prepared'][k].missing_days} gap days filled")
  return "; ".join(parts)


summary_rows, export = [], []
for r in state["results"]:
  if not r["found"]:
    continue
  for k, fc in r["forecasts"].items():
    row = F.summarize(k, r["prepared"][k], fc)
    imgs = S.product_images(r["fetch"].product)
    summary_rows.append({"Image": imgs[0] if imgs else None,
                         "ASIN": r["asin"], "Metric": S.METRICS[k],
                         "Today": fmt(k, row["current"], currency),
                         f"In {horizon} days (median)": fmt(k, row["forecast"], currency),
                         "80% range": f"{fmt(k, row['low'], currency)} – "
                                      f"{fmt(k, row['high'], currency)}",
                         "Change": "—" if row["change_pct"] is None
                                   else f"{row['change_pct']:+.1f}%",
                         "Trend": row["trend"],
                         "Note": note(r, k)})
    e = fc.copy()
    e.insert(0, "metric", k)
    e.insert(0, "asin", r["asin"])
    e["unit"] = currency if k == "price" else ""
    export.append(e.rename_axis("date").reset_index())

st.title("Amazon Forecast")
if summary_rows:
  st.subheader(f"Summary — next {horizon} days")
  st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch",
               row_height=48,
               column_config={"Image": st.column_config.ImageColumn("", width="small")})
  skipped = [f"{r['asin']} · {S.METRICS[k].split(' (')[0]}: {p.reason}"
             for r in state["results"] if r["found"]
             for k, p in r["prepared"].items() if not p.ok]
  if skipped:
    st.warning("Not forecast:\n\n" + "\n\n".join(f"- {x}" for x in skipped))
  csv = pd.concat(export).to_csv(index=False).encode()
  st.download_button("Download all forecasts (CSV)", csv,
                     file_name=f"amazon_forecast_{datetime.now():%Y%m%d}.csv",
                     mime="text/csv")

for r in state["results"]:
  st.divider()
  if not r["found"]:
    st.error(f"**{r['asin']}** — Keepa has no product for this ASIN in "
             f"{state['domain']}. Check the ASIN and marketplace.")
    continue
  p, f = r["fetch"].product, r["fetch"]
  images = S.product_images(p)
  col_img, col_info = st.columns([1, 3])
  with col_img:
    if images:
      st.image(images[0], width="stretch")
    else:
      st.caption("No image in Keepa data")
  with col_info:
    st.header(f"{p.get('brand') or ''} · {r['asin']}")
    st.markdown(f"**{p.get('title') or ''}**")
    st.caption(f"Keepa data as of {f.fetched_at:%Y-%m-%d %H:%M} UTC "
               f"({'saved copy' if f.from_cache else 'fresh pull'}) · "
               f"[View on Amazon](https://www.amazon.{AMAZON_TLD.get(state['domain'], 'com')}"
               f"/dp/{r['asin']})")
  if len(images) > 1:
    with st.expander(f"All {len(images)} listing images"):
      cols = st.columns(5)
      for i, url in enumerate(images):
        cols[i % 5].image(url, width="stretch")

  keys = [k for k in S.METRICS if k in r["daily"]]
  if not keys:
    st.warning("Keepa returned no history for this product yet.")
    continue
  tabs = st.tabs([S.METRICS[k].split(" (")[0] for k in keys])
  for tab, k in zip(tabs, keys):
    with tab:
      prep = r["prepared"][k]
      fc = r["forecasts"].get(k)
      st.plotly_chart(chart(k, r["daily"][k], prep, fc, currency),
                      width="stretch", key=f"{r['asin']}_{k}",
                      config=PLOTLY_CONFIG)
      notes = [METRIC_NOTES[k]]
      if not prep.ok:
        st.warning(f"Not forecast: {prep.reason}.")
      else:
        notes.append(f"Model learned from {len(prep.values)} days "
                     f"({prep.dates[0]:%Y-%m-%d} → {prep.last_date:%Y-%m-%d}).")
        if prep.trimmed_at is not None:
          st.info(f"⚠️ The count jumped on {prep.trimmed_at:%Y-%m-%d} (likely a listing "
                  f"variation merge/split). The model only learned from data after it.")
        recent = just_changed(k, r["daily"][k])
        if recent is not None:
          st.info(f"⚠️ This value changed only {recent} day(s) ago. The model has "
                  f"little evidence about the new level, so treat the forecast with care.")
        if prep.missing_days:
          st.info(f"⚠️ {prep.missing_days} day(s) had no data (e.g. out of stock / "
                  f"no Buy Box) and were filled with the last known value.")
      st.caption(" ".join(notes))
