# --- Setup ---
# pip install requests pandas plotly python-dateutil

import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import tz
import plotly.express as px
import plotly.io as pio

# Force Plotly to render in your default browser (so .py scripts actually show a window)
pio.renderers.default = "browser"

# -----------------------------
# Config (tweak as you like)
# -----------------------------
SYMBOL = "BTCUSDT"          # Binance spot
INTERVAL = "1m"             # granularity for intraday vol ("1m","3m","5m","15m","30m","1h" supported below)
LOOKBACK_DAYS = 30          # "past month" window; use 30 calendar days
LOCAL_TZ = "Europe/Lisbon"  # time-of-day computed in this timezone
ANNUALIZE_ON = True         # set False to view per-interval std dev (non-annualized)

# How many intervals per day (for annualization); extend if you change INTERVAL
BARS_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24
}
if INTERVAL not in BARS_PER_DAY:
    raise ValueError(f"Unsupported INTERVAL={INTERVAL}. Choose one of {list(BARS_PER_DAY)}.")

PERIODS_PER_YEAR = 365 * BARS_PER_DAY[INTERVAL]

# -----------------------------
# Binance helpers
# -----------------------------
def to_millis(dt):
    return int(dt.timestamp() * 1000)

def get_klines(symbol, interval, start_time_ms=None, end_time_ms=None, limit=1000):
    """Fetch up to `limit` klines from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_klines_range(symbol, interval, start_dt_utc, end_dt_utc):
    """Paginate from start to end (Binance returns max 1000 candles per call)."""
    all_rows = []
    fetch_start = start_dt_utc
    while True:
        chunk = get_klines(
            symbol=symbol,
            interval=interval,
            start_time_ms=to_millis(fetch_start),
            end_time_ms=to_millis(end_dt_utc),
            limit=1000
        )
        if not chunk:
            break
        all_rows.extend(chunk)
        # Next fetch starts just after the last returned close time
        last_close_time_ms = chunk[-1][6]
        next_start = datetime.utcfromtimestamp(last_close_time_ms / 1000.0).replace(tzinfo=timezone.utc) + timedelta(milliseconds=1)
        if next_start >= end_dt_utc or len(chunk) < 1000:
            break
        fetch_start = next_start
        time.sleep(0.15)  # be polite to the API
    return all_rows

# -----------------------------
# Compute intraday volatility profile (avg over last month)
# -----------------------------
def compute_intraday_vol(df, local_tz, annualize=True):
    """
    df: DataFrame with columns closeTime (UTC tz-aware), close (float)
    Returns a DF with minute-of-day, clock labels, and volatility measures.
    """
    # Use close-to-close returns at chosen interval
    prices = df.set_index("closeTime")["close"].sort_index()
    returns = np.log(prices).diff().dropna()  # log returns

    # Convert timestamps to local time-of-day for grouping
    local_zone = tz.gettz(local_tz)
    local_times = returns.index.tz_convert(local_zone)

    # Minute-of-day index for stable sorting
    minute_of_day = local_times.hour * 60 + local_times.minute

    r_df = pd.DataFrame({
        "r": returns.values,
        "minute_of_day": minute_of_day
    })

    # Std dev across all days for each minute-of-day => "average vol over the past month"
    per_interval_std = r_df.groupby("minute_of_day")["r"].std()

    if annualize:
        y = per_interval_std * math.sqrt(PERIODS_PER_YEAR) * 100.0
        y_name = "Annualized Volatility (%)"
    else:
        y = per_interval_std * 100.0
        y_name = f"Per-Interval Std Dev (%) ({INTERVAL})"

    # Build plotting frame
    out = pd.DataFrame({
        "minute_of_day": per_interval_std.index,
        y_name: y.values
    }).sort_values("minute_of_day")

    # Pretty clock labels HH:MM
    hh = (out["minute_of_day"] // 60).astype(int)
    mm = (out["minute_of_day"] % 60).astype(int)
    out["Clock Time"] = hh.map("{:02d}".format) + ":" + mm.map("{:02d}".format)

    return out, y_name

def main():
    # Time window
    end_dt_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt_utc = end_dt_utc - timedelta(days=LOOKBACK_DAYS)

    print(f"Downloading {INTERVAL} candles for {SYMBOL} from {start_dt_utc} to {end_dt_utc} (UTC)…")
    rows = fetch_klines_range(SYMBOL, INTERVAL, start_dt_utc, end_dt_utc)
    if not rows:
        raise SystemExit("No data returned. Check network, symbol, interval, or date window.")

    # Binance kline spec: [openTime, open, high, low, close, volume, closeTime, ...]
    df = pd.DataFrame(rows, columns=[
        "openTime","open","high","low","close","volume","closeTime",
        "qav","numTrades","takerBaseVol","takerQuoteVol","ignore"
    ])
    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["closeTime","close"]].dropna()

    plot_df, y_name = compute_intraday_vol(df, LOCAL_TZ, annualize=ANNUALIZE_ON)

    title = (
        f"{SYMBOL} Intraday Volatility by Time-of-Day\n"
        f"Avg over last {LOOKBACK_DAYS} days • Interval={INTERVAL} • Timezone={LOCAL_TZ}"
    )

    fig = px.line(
        plot_df,
        x="Clock Time",
        y=y_name,
        title=title,
        labels={"Clock Time": "Time of Day", y_name: y_name},
    )
    # Make sure X shows in chronological order
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=plot_df["Clock Time"].tolist())
    fig.update_layout(hovermode="x unified", yaxis_title=y_name)

    # Show & save
    fig.show()
    out_html = f"btc_intraday_vol_{INTERVAL}_{LOOKBACK_DAYS}d.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive chart to {out_html}")

if __name__ == "__main__":
    main()
