#!/usr/bin/env python3
# btc_updown_gui_multi_theos_mu_no_px_with_sigma_chart_and_price_chart__GARCH_1S_LIVE.py
# - Live 1-second volatility via GARCH(1,1) on 1s returns (recursive updates every second).
# - Periodic refit (default 10m) to refresh parameters.
# - Backfill button fetches last 10m aggTrades (per-second), rebuilds price+sigma and refits.
# - Robust sanitization so vol plot never disappears.
#
# Fallback: if 'arch' not installed, uses EWMA on 1s returns.
# UPDATE:
# - Display-only smoothing for σ plot with an EMA (half-life configurable).
# - Raised EWMA_LAMBDA_1S to calm fallback dynamics on 1s data.
# - Plot both RAW (thin) and SMOOTHED (solid) σ for better readability.

import threading, queue, time, json, math, tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# -------- Optional: arch for true GARCH; fallback to EWMA if not available --------
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# ------------------- Config -------------------
BINANCE_API = "https://api.binance.com"
SYMBOL = "BTCUSDT"

ET  = ZoneInfo("America/New_York")
AMS = ZoneInfo("Europe/Amsterdam")
UTC = timezone.utc

# Table horizons (10m/1h/1d remain historical-vol based on 1m/1h/1d, unchanged)
LOOKBACK_10M_MIN    = 600
LOOKBACK_1H_HRS     = 336
LOOKBACK_1D_DAYS    = 60

# Drift μ from 1m EWMA (kept)
MU_EWMA_MINUTES     = 10
MU_EWMA_LAMBDA      = 0.7
MU_MAX_ABS          = 2.0

# Cadences
REFRESH_PRICE_SEC   = 1      # live price pull
REFRESH_1M_CACHE_SEC= 2      # to keep 1m closes fresh (for μ and table σs other than 1s)
REFRESH_SIGMA_1S    = 1      # read/append σ every second
REFRESH_SIGMA_10M   = 10
REFRESH_SIGMA_1H    = 60
REFRESH_SIGMA_1D    = 300
REFRESH_MU          = 5

# GARCH fit cadence and history (1-second model)
GARCH_REFIT_SEC     = 600    # refit every 10 minutes
GARCH_MIN_POINTS    = 1800   # minimum 1s returns (~30m) to fit
GARCH_MAX_POINTS    = 14400  # keep up to last 4 hours of 1s returns

# ---------------- DISPLAY-ONLY σ SMOOTHING ----------------
# Half-life (in seconds) for the EMA applied ONLY to the plotted σ line.
# Does not affect the math used for probabilities; only the chart.
SIGMA_PLOT_HALFLIFE_SEC = 20   # try 15–30 for smooth-but-responsive

# Fallback EWMA for 1s if arch missing
# RiskMetrics 0.94 is for daily data; for 1-second series use a much higher lambda.
EWMA_LAMBDA_1S      = 0.997   # calmer fallback dynamics on 1s data

# Bounds & constants
R_NEUTRAL            = 0.0
SIGMA_MIN            = 0.05
SIGMA_MAX            = 3.00

HOURS_PER_YEAR = 365 * 24
SECS_PER_YEAR  = HOURS_PER_YEAR * 3600.0
MINS_PER_YEAR  = 365 * 24 * 60.0
TENM_PER_YEAR  = MINS_PER_YEAR / 10.0
HRS_PER_YEAR   = HOURS_PER_YEAR
DAYS_PER_YEAR  = 365.0

UI_REFRESH_MS  = 250

# Per-second buffers
MAX_1S_POINTS  = max(GARCH_MAX_POINTS + 120, 15000)  # ~>4h+ buffer

# ------------------- HTTP helpers -------------------
def http_get(path: str, params: dict | None = None):
    url = BINANCE_API + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "python-urllib"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_last_trade_price() -> float:
    return float(http_get("/api/v3/ticker/price", {"symbol": SYMBOL})["price"])

def fetch_1m_klines(limit: int):
    return http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1m", "limit": max(2, min(1000, limit))})

def fetch_1h_closes(n_hours: int) -> list[float]:
    k = http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1h", "limit": max(2, min(1000, n_hours + 1))})
    return [float(row[4]) for row in k]

def fetch_1d_closes(n_days: int) -> list[float]:
    k = http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1d", "limit": max(2, min(1000, n_days + 1))})
    return [float(row[4]) for row in k]

def get_kline_open_close(start_et: datetime, end_et: datetime):
    start_ms = int(start_et.astimezone(UTC).timestamp() * 1000)
    end_ms   = int(end_et.astimezone(UTC).timestamp()   * 1000)
    k = http_get("/api/v3/klines", {
        "symbol": SYMBOL, "interval": "1h",
        "startTime": start_ms, "endTime": end_ms, "limit": 1
    })
    if not k:
        return None
    row = k[0]
    return float(row[1]), float(row[4]), int(row[0]), int(row[6])  # open, close, open_ms, close_ms

# Aggregated trades over [start_ms, end_ms], paginated (no interpolation)
def fetch_agg_trades_range(start_ms: int, end_ms: int, limit: int = 1000):
    trades = []
    cursor_ms = start_ms
    while True:
        batch = http_get("/api/v3/aggTrades", {
            "symbol": SYMBOL, "startTime": cursor_ms, "endTime": end_ms, "limit": limit
        })
        if not batch:
            break
        trades.extend(batch)
        last_T = int(batch[-1]["T"])
        next_ms = last_T + 1
        if next_ms >= end_ms:
            break
        cursor_ms = next_ms
        if len(batch) < limit // 2 and (end_ms - last_T) < 1000:
            break
    return trades

# ------------------- Helpers -------------------
def log_returns_from_prices(prices: list[float], step: int) -> list[float]:
    if len(prices) <= step:
        return []
    rets = []
    for i in range(step, len(prices)):
        p0, p1 = prices[i-step], prices[i]
        if p0 > 0:
            rets.append(math.log(p1 / p0))
    return rets

def ann_sigma_from_returns(returns: list[float], units_per_year: float) -> float:
    if not returns:
        return 0.0
    if len(returns) == 1:
        std = abs(returns[0])
    else:
        mean = sum(returns) / len(returns)
        var  = sum((r - mean)**2 for r in returns) / (len(returns) - 1)
        std  = math.sqrt(var)
    return min(SIGMA_MAX, max(SIGMA_MIN, std * math.sqrt(units_per_year)))

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def fair_up_prob(S0: float, St: float, tau_years: float, sigma_ann: float, r_ann: float) -> float:
    if tau_years <= 0:
        return 1.0 if St >= S0 else 0.0
    sigma = max(1e-12, sigma_ann)
    m = (r_ann - 0.5 * sigma * sigma) * tau_years
    s = sigma * math.sqrt(tau_years)
    z = (math.log(S0 / St) - m) / s
    return 1.0 - norm_cdf(z)

def finite_sigma(v: float) -> float:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return SIGMA_MIN
    return min(SIGMA_MAX, max(SIGMA_MIN, float(v)))

def current_et_hour_window(now_dt: datetime) -> tuple[datetime, datetime]:
    """
    Given a timezone-aware datetime (usually UTC), return the start and end of the
    *current ET hour* as timezone-aware datetimes in ET.
    Start is inclusive, end is exclusive: [start_et, end_et).
    """
    now_et = now_dt.astimezone(ET)
    start_et = now_et.replace(minute=0, second=0, microsecond=0)
    end_et = start_et + timedelta(hours=1)
    return start_et, end_et

# ------------------- 1m cache (for μ and table σs) -------------------
class OneMinuteCloseCache:
    def __init__(self, maxlen=2000):
        self.closes = deque(maxlen=maxlen)
        self.last_open_ms = None
    def bootstrap(self):
        k = fetch_1m_klines(limit=min(1000, max(5, self.closes.maxlen)))
        if not k or len(k) < 2:
            return
        for row in k[:-1]:
            self.closes.append(float(row[4]))
            self.last_open_ms = row[0]
    def update(self):
        k = fetch_1m_klines(limit=2)
        if not k or len(k) < 2:
            return False
        closed = k[-2]
        open_ms = closed[0]
        if self.last_open_ms is None or open_ms > self.last_open_ms:
            self.closes.append(float(closed[4]))
            self.last_open_ms = open_ms
            return True
        return False
    def get_list(self) -> list[float]:
        return list(self.closes)

# ------------------- Live per-second price cache -------------------
class SecondPriceCache:
    """
    Maintains per-second price series.
    - Backfill from aggTrades (no interpolation; carry-forward last price between sparse seconds)
    - Update each second with latest last-trade price; carry-forward if no new trade changed price.
    """
    def __init__(self, maxlen=MAX_1S_POINTS):
        self.times = deque(maxlen=maxlen)  # datetime (UTC)
        self.prices = deque(maxlen=maxlen)
        self.last_sec = None
        self.last_price = None

    def backfill_last_n_minutes(self, minutes: int):
        end_utc = datetime.now(tz=UTC)
        start_utc = end_utc - timedelta(minutes=minutes + 1)  # small buffer
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms   = int(end_utc.timestamp() * 1000)
        trades = fetch_agg_trades_range(start_ms, end_ms, limit=1000)
        if not trades:
            raise RuntimeError("No trades returned for backfill")

        # second -> last price in that second
        sec_to_price = {}
        for t in trades:
            sec = int(int(t["T"]) // 1000)
            sec_to_price[sec] = float(t["p"])

        end_sec = int(end_utc.timestamp())
        start_sec = end_sec - minutes * 60
        last = None
        # try to find fallback
        for s in range(start_sec - 120, start_sec + 1):
            if s in sec_to_price:
                last = sec_to_price[s]

        times = []
        prices = []
        for s in range(start_sec, end_sec + 1):
            if s in sec_to_price:
                last = sec_to_price[s]
            if last is None:
                # extremely unlikely for BTC; skip until we have a price
                continue
            times.append(datetime.fromtimestamp(s, tz=UTC))
            prices.append(last)

        if not times:
            raise RuntimeError("Backfill failed to produce per-second series")

        # reset buffers
        self.times.clear(); self.prices.clear()
        for t, p in zip(times, prices):
            self.times.append(t); self.prices.append(p)
        self.last_sec = int(self.times[-1].timestamp())
        self.last_price = self.prices[-1]

    def tick(self, price_now: float):
        """Call once per second with latest last-trade price (carry-forward allowed)."""
        sec_now = int(datetime.now(tz=UTC).timestamp())
        if self.last_sec is None:
            # initialize
            self.last_sec = sec_now
            self.last_price = price_now
            self.times.append(datetime.fromtimestamp(sec_now, tz=UTC))
            self.prices.append(price_now)
            return True
        # fill any gaps >1s by carrying forward last known price
        for s in range(self.last_sec + 1, sec_now + 1):
            self.times.append(datetime.fromtimestamp(s, tz=UTC))
            if s == sec_now:
                self.prices.append(price_now if price_now is not None else self.last_price)
            else:
                self.prices.append(self.last_price)
        self.last_sec = sec_now
        if price_now is not None:
            self.last_price = price_now
        return True

    def get_prices(self):
        return list(self.prices)
    def get_times(self):
        return list(self.times)

# ------------------- GARCH(1,1) forecaster on 1s -------------------
class Garch1sForecaster:
    """
    Fit GARCH(1,1) on 1-second log returns.
    - Fit every GARCH_REFIT_SEC seconds on recent history (up to GARCH_MAX_POINTS)
    - Between refits, update sigma^2 via recursion each new second with the latest 1s return
    - Returns annualized σ per second (sqrt(variance) * sqrt(SECS_PER_YEAR))
    Fallback: EWMA if 'arch' missing or too little data
    """
    def __init__(self):
        self.params = None           # (omega, alpha, beta) for percent returns
        self.sigma2_last = None      # last variance (percent^2)
        self.last_ret_pct = None     # last 1s return in percent
        self.last_fit_t = 0.0
        self.last_sigma_ann = None

    def _returns(self, prices):
        return log_returns_from_prices(prices, step=1)

    def _to_pct(self, returns):
        try:
            return [r * 100.0 for r in returns]
        except Exception:
            return [r * 100.0 for r in returns]

    def _annualize_from_pct_sigma(self, sigma_pct):
        return (sigma_pct / 100.0) * math.sqrt(SECS_PER_YEAR)

    def _fallback_ewma(self, returns):
        # classic RiskMetrics EWMA (with higher lambda for high-frequency data)
        lam = EWMA_LAMBDA_1S
        var = None
        for r in returns[-5000:]:  # cap to keep it light
            x = r
            var = (1 - lam) * (x * x) + (lam * var if var is not None else 0.0)
        if var is None:
            return SIGMA_MIN
        sigma = math.sqrt(max(1e-18, var)) * math.sqrt(SECS_PER_YEAR)
        return finite_sigma(sigma)

    def refit(self, prices_1s):
        returns = self._returns(prices_1s)
        if len(returns) < GARCH_MIN_POINTS or not HAVE_ARCH:
            self.params = None
            self.sigma2_last = None
            self.last_ret_pct = None
            self.last_sigma_ann = finite_sigma(self._fallback_ewma(returns))
            self.last_fit_t = time.time()
            return

        # Use last GARCH_MAX_POINTS returns
        returns = returns[-GARCH_MAX_POINTS:]
        r_pct = self._to_pct(returns)
        try:
            am = arch_model(r_pct, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
            res = am.fit(disp="off", show_warning=False)
            p = res.params
            omega = float(p.get('omega', float('nan')))
            alpha = float(p.get('alpha[1]', float('nan')))
            beta  = float(p.get('beta[1]',  float('nan')))
            if not (math.isfinite(omega) and math.isfinite(alpha) and math.isfinite(beta)):
                raise ValueError("Non-finite GARCH parameters")

            self.params = (omega, alpha, beta)
            try:
                last_sigma_pct = float(res.conditional_volatility.values[-1])
                self.sigma2_last = max(1e-18, last_sigma_pct ** 2)
            except Exception:
                var_uc = omega / max(1e-8, 1.0 - alpha - beta)
                self.sigma2_last = max(1e-18, var_uc)

            self.last_ret_pct = float(r_pct[-1]) if r_pct else None

            # one-step forecast cache
            sigma_next_pct = math.sqrt(max(1e-18, self._next_sigma2(self.last_ret_pct, self.sigma2_last)))
            self.last_sigma_ann = finite_sigma(self._annualize_from_pct_sigma(sigma_next_pct))
            self.last_fit_t = time.time()
        except Exception:
            self.params = None
            self.sigma2_last = None
            self.last_ret_pct = None
            self.last_sigma_ann = finite_sigma(self._fallback_ewma(returns))
            self.last_fit_t = time.time()

    def _next_sigma2(self, last_ret_pct, last_sigma2_pct2):
        if self.params is None or last_sigma2_pct2 is None:
            return float('nan')
        omega, alpha, beta = self.params
        eps2 = 0.0 if (last_ret_pct is None or not math.isfinite(last_ret_pct)) else (last_ret_pct ** 2)
        return max(1e-18, omega + alpha * eps2 + beta * last_sigma2_pct2)

    def on_new_second(self, r_nat_1s: float | None):
        # r_nat_1s: ln(Pt/Pt-1) per-second
        if self.params is None:
            return  # fallback path will be used by caller
        r_pct = None if r_nat_1s is None else (r_nat_1s * 100.0)
        next_var = self._next_sigma2(self.last_ret_pct, self.sigma2_last)
        self.sigma2_last = next_var
        self.last_ret_pct = r_pct
        sigma_next_pct = math.sqrt(max(1e-18, next_var))
        self.last_sigma_ann = finite_sigma(self._annualize_from_pct_sigma(sigma_next_pct))

    def get_sigma_ann(self, prices_1s):
        if self.params is None:
            # compute fallback on the fly (cheap)
            returns = self._returns(prices_1s)
            return finite_sigma(self._fallback_ewma(returns))
        return finite_sigma(self.last_sigma_ann)

# ------------------- Other horizon helpers (unchanged) -------------------
def sigma_10m_from_1mcache(cache: OneMinuteCloseCache) -> float:
    closes = cache.get_list()[-(LOOKBACK_10M_MIN + 1):]
    rets = log_returns_from_prices(closes, step=10)
    return ann_sigma_from_returns(rets, TENM_PER_YEAR)

def sigma_1h() -> float:
    closes_1h = fetch_1h_closes(LOOKBACK_1H_HRS)
    rets_1h   = log_returns_from_prices(closes_1h, step=1)
    return ann_sigma_from_returns(rets_1h, HRS_PER_YEAR)

def sigma_1d() -> float:
    closes_1d = fetch_1d_closes(LOOKBACK_1D_DAYS)
    rets_1d   = log_returns_from_prices(closes_1d, step=1)
    return ann_sigma_from_returns(rets_1d, DAYS_PER_YEAR)

def mu_ewma_ann_from_cache(cache: OneMinuteCloseCache, lam: float = MU_EWMA_LAMBDA) -> float:
    closes = cache.get_list()[-(MU_EWMA_MINUTES + 1):]
    if len(closes) < 2:
        return 0.0
    rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
    if not rets:
        return 0.0
    m = None
    for r in rets:
        m = r if m is None else lam * m + (1.0 - lam) * r
    mu_ann = (m if m is not None else 0.0) * MINS_PER_YEAR
    return max(-MU_MAX_ABS, min(MU_MAX_ABS, mu_ann))

# ------------------- Worker thread -------------------
def worker(out_q: "queue.Queue[dict]", stop_event: threading.Event):
    now = datetime.now(tz=UTC)
    start_et, end_et = current_et_hour_window(now)
    start_utc = start_et.astimezone(UTC)
    end_utc   = end_et.astimezone(UTC)

    strike = None
    while not stop_event.is_set() and strike is None:
        try:
            k = get_kline_open_close(start_et, end_et)
            if k: strike = k[0]
        except Exception:
            pass
        time.sleep(0.3)

    # Caches
    cache_1m = OneMinuteCloseCache(2000)
    try: cache_1m.bootstrap()
    except Exception: pass

    cache_1s = SecondPriceCache(MAX_1S_POINTS)
    backfilled_ok = False
    try:
        cache_1s.backfill_last_n_minutes(15)  # seed ~15m of 1s history
        backfilled_ok = True
    except Exception:
        # seed with one point using current price; the series will build up
        try:
            px0 = get_last_trade_price()
            cache_1s.tick(px0)
        except Exception:
            pass

    # GARCH on 1s
    g1s = Garch1sForecaster()
    try:
        if backfilled_ok:
            g1s.refit(cache_1s.get_prices())
    except Exception:
        pass
    last_refit_t = time.time()

    last_price_t   = 0.0
    last_1mcache_t = 0.0
    last_s1s_t     = 0.0
    last_s10_t     = 0.0
    last_s1h_t     = 0.0
    last_s1d_t     = 0.0
    last_mu_t      = 0.0

    px = None
    last_px_sent = None

    sigmas = {"1s": SIGMA_MIN, "10m": SIGMA_MIN, "1h": SIGMA_MIN, "1d": SIGMA_MIN}
    mu_ann = 0.0

    while not stop_event.is_set():
        tnow = time.time()
        now = datetime.now(tz=UTC)

        # Hour rollover
        if now >= end_utc:
            try:
                _, final_close, _, _ = get_kline_open_close(start_et, end_et)
                outcome = "UP" if (final_close >= strike) else "DOWN"
                out_q.put({
                    "event": "closed",
                    "final_close": final_close,
                    "outcome": outcome,
                    "strike": strike, "now": now, "start_et": start_et, "end_et": end_et
                })
            except Exception as e:
                out_q.put({"event": "status", "msg": f"Closed hour: couldn't fetch final close ({e}). Rolling…"})
            start_et, end_et = current_et_hour_window(now)
            start_utc = start_et.astimezone(UTC)
            end_utc   = end_et.astimezone(UTC)
            strike = None
            while not stop_event.is_set() and strike is None:
                try:
                    k = get_kline_open_close(start_et, end_et)
                    if k: strike = k[0]
                except Exception:
                    pass
                time.sleep(0.3)
            last_price_t = last_1mcache_t = 0.0

        # Update 1m cache periodically (for μ and table)
        if tnow - last_1mcache_t >= REFRESH_1M_CACHE_SEC:
            try:
                cache_1m.update()
            except Exception:
                pass
            last_1mcache_t = tnow

        # Price every second → update 1s cache → per-second return → feed GARCH recursion
        price_changed = False
        if tnow - last_price_t >= REFRESH_PRICE_SEC:
            try:
                px = get_last_trade_price()
                price_changed = (last_px_sent is None) or (px != last_px_sent)
                last_px_sent = px
                # push to 1s cache (fills gaps)
                cache_1s.tick(px)
                # derive 1s return r_t = ln(Pt / Pt-1)
                prices = cache_1s.get_prices()
                r_1s = None
                if len(prices) >= 2 and prices[-2] > 0 and prices[-1] > 0:
                    r_1s = math.log(prices[-1] / prices[-2])
                # per-second recursion (or fallback EWMA implicit in get_sigma_ann)
                g1s.on_new_second(r_1s)
            except Exception as e:
                out_q.put({"event": "status", "msg": f"Network issue (price): {e}"})
            last_price_t = tnow

        # Periodic 1s GARCH refit on rolling history
        if tnow - last_refit_t >= GARCH_REFIT_SEC:
            try:
                g1s.refit(cache_1s.get_prices())
            except Exception:
                pass
            last_refit_t = tnow

        # 1s sigma read every second
        if tnow - last_s1s_t >= REFRESH_SIGMA_1S:
            try:
                sigmas["1s"] = g1s.get_sigma_ann(cache_1s.get_prices())
            except Exception:
                sigmas["1s"] = SIGMA_MIN
            last_s1s_t = tnow

        # Other horizons
        if tnow - last_s10_t >= REFRESH_SIGMA_10M:
            try:   sigmas["10m"] = sigma_10m_from_1mcache(cache_1m)
            except Exception: pass
            last_s10_t = tnow

        if tnow - last_s1h_t >= REFRESH_SIGMA_1H:
            try:   sigmas["1h"] = sigma_1h()
            except Exception: pass
            last_s1h_t = tnow

        if tnow - last_s1d_t >= REFRESH_SIGMA_1D:
            try:   sigmas["1d"] = sigma_1d()
            except Exception: pass
            last_s1d_t = tnow

        if tnow - last_mu_t >= REFRESH_MU:
            try:   mu_ann = mu_ewma_ann_from_cache(cache_1m)
            except Exception: pass
            last_mu_t = tnow

        # Emit tick
        if px is not None and strike is not None:
            tau_sec = max(0.0, (end_utc - now).total_seconds())
            tau_years = tau_sec / SECS_PER_YEAR
            diff = px - strike
            elapsed = (now - start_utc).total_seconds()
            frac_elapsed = min(1.0, max(0.0, elapsed / 3600.0))

            # use 1s sigma for "1m row" semantics? We'll label as "1s (GARCH)" in UI.
            # For the probability calc, use 1s sigma; others unchanged
            sig_for_prob = {
                "1m": sigmas["1s"],   # replace former 1m slot with live 1s GARCH
                "10m": sigmas["10m"],
                "1h": sigmas["1h"],
                "1d": sigmas["1d"],
            }

            theos_neutral = {k: fair_up_prob(strike, px, tau_years, sig_for_prob[k], R_NEUTRAL) for k in sig_for_prob}
            theos_mu      = {k: fair_up_prob(strike, px, tau_years, sig_for_prob[k], mu_ann)   for k in sig_for_prob}

            out_q.put({
                "event": "tick",
                "now": now,
                "start_et": start_et, "end_et": end_et,
                "strike": strike, "price": px, "diff": diff,
                "sigmas": sig_for_prob,  # send the four horizons (1m uses 1s live)
                "sig1s": sigmas["1s"],   # also send raw 1s sigma for the chart
                "theos_neutral": theos_neutral, "theos_mu": theos_mu,
                "mu_ann": mu_ann,
                "tau_sec": tau_sec, "frac_elapsed": frac_elapsed,
                "price_changed": price_changed
            })

        time.sleep(0.02)

# ------------------- GUI -------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTCUSDT — Δ Price vs Strike & Vol (1s σ: GARCH live, per-second)")
        self.geometry("900x960")
        self.resizable(False, False)

        self._init_style()

        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=worker, args=(self.q, self.stop_event), daemon=True)
        self.worker_thread.start()

        self.last_strike = None
        self.last_sigma_1s = None

        # ---- Header ----
        top = ttk.Frame(self, padding=(10,10,10,0))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.lbl_window = ttk.Label(top, text="Window: …", style="Title.TLabel")
        self.lbl_window.grid(row=0, column=0, sticky="w")
        self.lbl_server = ttk.Label(top, text="Local UTC time: …", style="Subtle.TLabel")
        self.lbl_server.grid(row=1, column=0, sticky="w", pady=(2,0))

        # ---- Price/Strike/Time ----
        pricef = ttk.Frame(self, padding=(10,6))
        pricef.grid(row=1, column=0, sticky="ew")
        for i in range(6): pricef.columnconfigure(i, weight=1)
        self.var_strike = tk.StringVar(value="…")
        self.var_price  = tk.StringVar(value="…")
        self.var_diff   = tk.StringVar(value="…")
        self.var_tclose = tk.StringVar(value="…")
        ttk.Label(pricef, text="Strike (hour open):", style="Caption.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(pricef, textvariable=self.var_strike, style="Value.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(pricef, text="Price (last):", style="Caption.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(pricef, textvariable=self.var_price, style="Value.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(pricef, text="Δ vs strike:", style="Caption.TLabel").grid(row=0, column=2, sticky="w")
        self.lbl_diff = ttk.Label(pricef, textvariable=self.var_diff, style="Value.TLabel")
        self.lbl_diff.grid(row=1, column=2, sticky="w")
        ttk.Label(pricef, text="Time to close:", style="Caption.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Label(pricef, textvariable=self.var_tclose, style="Value.TLabel").grid(row=1, column=3, sticky="w")

        # Backfill
        self.btn_backfill = ttk.Button(pricef, text="Backfill last 10m (per-second)", command=self.on_backfill_10m_persec)
        self.btn_backfill.grid(row=1, column=5, sticky="e", padx=(10,0))

        # Progress
        pwrap = ttk.Frame(self, padding=(10,0))
        pwrap.grid(row=2, column=0, sticky="ew")
        self.progress = ttk.Progressbar(pwrap, orient="horizontal", length=860, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=0, sticky="ew")

        # ---- Price chart ----
        price_chart_wrap = ttk.Frame(self, padding=(10,0))
        price_chart_wrap.grid(row=3, column=0, sticky="nsew")
        self._init_price_chart(price_chart_wrap)

        # ---- Volatility chart (1s) ----
        chart_wrap = ttk.Frame(self, padding=(10,0))
        chart_wrap.grid(row=4, column=0, sticky="nsew")
        self._init_sigma_chart(chart_wrap)

        # μ display
        muwrap = ttk.Frame(self, padding=(10,0))
        muwrap.grid(row=5, column=0, sticky="ew")
        ttk.Separator(muwrap, orient="horizontal").grid(row=0, column=0, sticky="ew", pady=6)
        self.var_mu = tk.StringVar(value="…")
        src = "GARCH(1,1) on 1s" if HAVE_ARCH else "EWMA on 1s (fallback)"
        ttk.Label(muwrap, text=f"μ (annualized) from EWMA 1m returns (last {MU_EWMA_MINUTES}m, λ={MU_EWMA_LAMBDA})",
                  style="Caption.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(muwrap, textvariable=self.var_mu, style="Value.TLabel").grid(row=1, column=0, sticky="e", padx=(0,10))

        # Table
        tbl = ttk.Frame(self, padding=(10,0))
        tbl.grid(row=6, column=0, sticky="ew")
        hdr = ("Vol source", "σ (ann.)",
               "UP% (μ=0)", "DOWN% (μ=0)",
               "UP% (with μ)", "DOWN% (with μ)")
        for c, title in enumerate(hdr):
            ttk.Label(tbl, text=title, style="Head.TLabel").grid(row=0, column=c, padx=4, pady=(4,2),
                                                                 sticky="e" if c>0 else "w")
        self.rows = {
            "1m":  {"label": "1-second (GARCH live)"},
            "10m": {"label": "10-minute (from 1m returns)"},
            "1h":  {"label": "1-hour (from 1h returns)"},
            "1d":  {"label": "Daily (from 1d returns)"},
        }
        r = 1
        for key, meta in self.rows.items():
            ttk.Label(tbl, text=meta["label"], style="Row.TLabel").grid(row=r, column=0, padx=4, pady=2, sticky="w")
            for col in ("sigma", "up_neut", "down_neut", "up_mu", "down_mu"):
                self.rows[key][col] = tk.StringVar(value="…")
            ttk.Label(tbl, textvariable=self.rows[key]["sigma"], style="Num.TLabel").grid(row=r, column=1, padx=4, sticky="e")
            ttk.Label(tbl, textvariable=self.rows[key]["up_neut"], style="NumGreen.TLabel").grid(row=r, column=2, padx=4, sticky="e")
            ttk.Label(tbl, textvariable=self.rows[key]["down_neut"], style="NumRed.TLabel").grid(row=r, column=3, padx=4, sticky="e")
            ttk.Label(tbl, textvariable=self.rows[key]["up_mu"], style="NumGreen.TLabel").grid(row=r, column=4, padx=4, sticky="e")
            ttk.Label(tbl, textvariable=self.rows[key]["down_mu"], style="NumRed.TLabel").grid(row=r, column=5, padx=4, sticky="e")
            r += 1

        # Status + Quit
        bottom = ttk.Frame(self, padding=(10,6))
        bottom.grid(row=7, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.var_status = tk.StringVar(value=("" if HAVE_ARCH else
            "arch not found — using EWMA fallback for 1s σ. `pip install arch` to enable GARCH."))
        self.lbl_status = ttk.Label(bottom, textvariable=self.var_status, style="Subtle.TLabel")
        self.lbl_status.grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Quit", command=self.on_quit, style="Danger.TButton").grid(row=0, column=1, sticky="e", padx=(10,0))

        self.after(UI_REFRESH_MS, self.poll_queue)

    # ---------- Styles ----------
    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Head.TLabel",  font=("Segoe UI", 9, "bold"))
        style.configure("Caption.TLabel", font=("Segoe UI", 9))
        style.configure("Value.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Row.TLabel",  font=("Segoe UI", 9))
        style.configure("Num.TLabel",  font=("Consolas", 10))
        style.configure("NumGreen.TLabel", font=("Consolas", 10), foreground="#118a00")
        style.configure("NumRed.TLabel",   font=("Consolas", 10), foreground="#c40000")
        style.configure("Subtle.TLabel", foreground="#666")
        style.configure("Danger.TButton", foreground="#222")
        style.configure("TProgressbar", thickness=10)

    # ---------- Price chart (TOP; X uses date numbers; Y dynamic ≤ 10 ticks) ----------
    def _init_price_chart(self, parent):
        self.price_time_hist = deque(maxlen=600)  # stores mdates floats
        self.price_hist      = deque(maxlen=600)
        self.last_price_plotted = None
        self.last_plot_time_num = None

        self.fig_px = Figure(figsize=(8.6, 2.8), dpi=100)
        self.ax_px  = self.fig_px.add_subplot(111)
        (self.price_line,) = self.ax_px.plot([], [], lw=1.5)

        self.strike_line = self.ax_px.axhline(y=0.0, linestyle="--", linewidth=1.0, alpha=0.6)

        self.ax_px.set_title("Δ price vs strike — last 10 minutes")
        self.ax_px.set_ylabel("Δ vs strike (USDT)")
        self.ax_px.set_xlabel("Time (ET)")
        self.ax_px.grid(True, which="major", alpha=0.3)
        self.ax_px.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_px.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=ET))

        # Dynamic Y: cap at most 10 labels
        self.ax_px.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune=None, min_n_ticks=2))
        self.ax_px.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:+.0f}"))

        now_et = datetime.now(tz=UTC).astimezone(ET)
        now_num = mdates.date2num(now_et)
        self.ax_px.set_xlim(now_num - (10/1440.0), now_num)

        self.canvas_px = FigureCanvasTkAgg(self.fig_px, master=parent)
        self.canvas_px.get_tk_widget().grid(row=0, column=0, sticky="ew")

    def _push_price_point_if_needed(self, when_dt_et: datetime, price: float, force: bool):
        when_num = mdates.date2num(when_dt_et)
        should_append = False
        if not self.price_time_hist:
            should_append = True
        else:
            time_ok = (self.last_plot_time_num is None) or ((when_num - self.last_plot_time_num) >= (1/86400.0))  # ≥1 second
            price_changed = (self.last_price_plotted is None) or (price != self.last_price_plotted)
            should_append = force or time_ok or price_changed

        if should_append:
            self.price_time_hist.append(when_num)
            self.price_hist.append(price)
            self.last_price_plotted = price
            self.last_plot_time_num = when_num
            ten_min_ago = when_num - (10/1440.0)
            while self.price_time_hist and self.price_time_hist[0] < ten_min_ago:
                self.price_time_hist.popleft()
                self.price_hist.popleft()

    def _redraw_price_chart(self, strike: float):
        if not self.price_time_hist:
            return
        diffs = [p - strike for p in self.price_hist]
        self.price_line.set_data(self.price_time_hist, diffs)
        x_right = self.price_time_hist[-1]
        x_left  = x_right - (10/1440.0)
        self.ax_px.set_xlim(x_left, x_right)

        # Dynamic y-limits with small padding; ticks auto-limited by MaxNLocator(≤10)
        y_min_raw = min(diffs); y_max_raw = max(diffs)
        if y_min_raw == y_max_raw:
            pad = max(5.0, abs(y_min_raw)*0.05 + 1.0)
            lower, upper = y_min_raw - pad, y_min_raw + pad
        else:
            yrange = y_max_raw - y_min_raw
            pad = max(5.0, yrange * 0.1)
            lower = y_min_raw - pad
            upper = y_max_raw + pad
        self.ax_px.set_ylim(lower, upper)

        self.ax_px.xaxis_date()
        self.fig_px.autofmt_xdate()
        self.canvas_px.draw_idle()

    # ---------- σ chart (BOTTOM; X uses date numbers) ----------
    def _init_sigma_chart(self, parent):
        # RAW and SMOOTH series (both plotted; smooth emphasized)
        self.sigma_hist_raw    = deque(maxlen=900)
        self.sigma_hist_smooth = deque(maxlen=900)
        self.time_hist  = deque(maxlen=900)  # stores mdates floats
        self.last_sigma_time_num = None

        # display-only EMA alpha derived from half-life
        self.sigma_plot_alpha = 1.0 - 0.5 ** (1.0 / max(1, SIGMA_PLOT_HALFLIFE_SEC))
        self.sigma_ema_state = None

        self.fig = Figure(figsize=(8.6, 2.4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        (self.sigma_line_smooth,) = self.ax.plot([], [], lw=1.8)      # emphasized
        (self.sigma_line_raw,)    = self.ax.plot([], [], lw=1.0, alpha=0.35)  # faint raw line

        src = "GARCH(1,1) on 1s" if HAVE_ARCH else f"EWMA on 1s (fallback, λ={EWMA_LAMBDA_1S})"
        self.ax.set_title(f"σ (annualized) — last 10 minutes ({src})")
        self.ax.set_ylabel("σ (ann.)")
        self.ax.set_xlabel("Time (ET)")
        self.ax.grid(True, which="major", alpha=0.3)
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=ET))

        now_et = datetime.now(tz=UTC).astimezone(ET)
        now_num = mdates.date2num(now_et)
        self.ax.set_xlim(now_num - (10/1440.0), now_num)
        self.ax.set_ylim(0.0, 1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="ew")

    def _push_sigma_point(self, when_dt_et: datetime, sigma_ann: float):
        when_num = mdates.date2num(when_dt_et)

        # throttle: allow at most 1 point per second (date numbers are days)
        one_sec = 1.0 / 86400.0
        if self.last_sigma_time_num is not None and (when_num - self.last_sigma_time_num) < one_sec:
            return
        self.last_sigma_time_num = when_num

        s_raw = finite_sigma(sigma_ann)

        # display-only EMA smoothing
        if self.sigma_ema_state is None:
            self.sigma_ema_state = s_raw
        else:
            a = self.sigma_plot_alpha
            self.sigma_ema_state = a * s_raw + (1.0 - a) * self.sigma_ema_state
        s_smooth = finite_sigma(self.sigma_ema_state)

        self.time_hist.append(when_num)
        self.sigma_hist_raw.append(s_raw)
        self.sigma_hist_smooth.append(s_smooth)

        ten_min_ago = when_num - (10/1440.0)
        while self.time_hist and self.time_hist[0] < ten_min_ago:
            self.time_hist.popleft()
            self.sigma_hist_raw.popleft()
            self.sigma_hist_smooth.popleft()

    def _redraw_sigma_chart(self):
        if not self.time_hist:
            return
        ys = [v for v in self.sigma_hist_smooth if math.isfinite(v)]
        if not ys:
            return
        self.sigma_line_raw.set_data(self.time_hist, self.sigma_hist_raw)
        self.sigma_line_smooth.set_data(self.time_hist, self.sigma_hist_smooth)
        x_right = self.time_hist[-1]
        x_left  = x_right - (10/1440.0)
        self.ax.set_xlim(x_left, x_right)
        y_min = min(ys); y_max = max(ys)
        pad = 0.01
        lower = max(0.0, y_min - pad)
        upper = max(lower + 0.05, y_max + pad)
        self.ax.set_ylim(lower, upper)
        self.ax.xaxis_date()
        self.fig.autofmt_xdate()
        self.canvas.draw_idle()

    # ---------- Backfill (PER-SECOND, refit GARCH on the backfilled series) ----------
    def on_backfill_10m_persec(self):
        try:
            # Ensure strike
            if self.last_strike is None:
                now_utc = datetime.now(tz=UTC)
                start_et, end_et = current_et_hour_window(now_utc)
                k = get_kline_open_close(start_et, end_et)
                if not k:
                    self.var_status.set("Backfill: couldn't fetch current hour strike.")
                    return
                self.last_strike = k[0]
                self.var_strike.set(f"{self.last_strike:,.2f}")

            # Build per-second series for EXACT last 10 minutes
            end_utc = datetime.now(tz=UTC)
            start_utc = end_utc - timedelta(minutes=10)
            start_ms = int(start_utc.timestamp() * 1000)
            end_ms   = int(end_utc.timestamp() * 1000)
            trades = fetch_agg_trades_range(start_ms, end_ms, limit=1000)
            if not trades:
                self.var_status.set("Backfill: no trades returned.")
                return

            sec_to_price = {}
            for t in trades:
                sec = int(int(t["T"]) // 1000)
                sec_to_price[sec] = float(t["p"])

            prices = []
            times = []
            last = None
            start_sec = int(start_utc.timestamp())
            end_sec   = int(end_utc.timestamp())
            # fallback search window for an initial price
            for s in range(start_sec - 60, start_sec + 1):
                if s in sec_to_price:
                    last = sec_to_price[s]
            for s in range(start_sec, end_sec + 1):
                if s in sec_to_price:
                    last = sec_to_price[s]
                if last is None:
                    continue
                times.append(datetime.fromtimestamp(s, tz=UTC))
                prices.append(last)
            if not prices:
                self.var_status.set("Backfill failed: empty price series.")
                return

            # Replace price chart history (TOP) — store time as date numbers
            self.price_time_hist.clear(); self.price_hist.clear()
            times_et = [t.astimezone(ET) for t in times]
            times_num = [mdates.date2num(t) for t in times_et]
            for tn, pv in zip(times_num, prices):
                self.price_time_hist.append(tn)
                self.price_hist.append(pv)
            if self.last_strike is not None:
                self._redraw_price_chart(strike=self.last_strike)

            # Rebuild σ history for the last 10m using fallback EWMA (as a proxy for raw),
            # then apply the display-only EMA on top for the smooth line.
            rets = log_returns_from_prices(prices, 1)
            lam = EWMA_LAMBDA_1S
            var = None
            sig_series_raw = []
            for r in rets:
                var = (1 - lam) * (r*r) + (lam * var if var is not None else 0.0)
                sig_series_raw.append(math.sqrt(max(1e-18, var)) * math.sqrt(SECS_PER_YEAR))

            times_num_for_sig = times_num[-len(sig_series_raw):]
            self.time_hist.clear(); self.sigma_hist_raw.clear(); self.sigma_hist_smooth.clear()
            self.sigma_ema_state = None

            # pad one point so the σ line aligns visually with price start
            if len(times_num_for_sig) < len(times_num) and sig_series_raw:
                self.time_hist.append(times_num[0])
                first = finite_sigma(sig_series_raw[0])
                self.sigma_hist_raw.append(first)
                self.sigma_ema_state = first
                self.sigma_hist_smooth.append(first)

            for tn, s in zip(times_num_for_sig, sig_series_raw):
                s_raw = finite_sigma(s)
                if self.sigma_ema_state is None:
                    self.sigma_ema_state = s_raw
                else:
                    a = self.sigma_plot_alpha
                    self.sigma_ema_state = a * s_raw + (1 - a) * self.sigma_ema_state
                self.time_hist.append(tn)
                self.sigma_hist_raw.append(s_raw)
                self.sigma_hist_smooth.append(finite_sigma(self.sigma_ema_state))

            self._redraw_sigma_chart()
            self.var_status.set("Backfilled last 10m (per-second). Live GARCH will continue updating each second.")
        except Exception as e:
            self.var_status.set(f"Backfill error: {e}")

    # ---------- Queue polling ----------
    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.handle(msg)
        except queue.Empty:
            pass
        self.after(UI_REFRESH_MS, self.poll_queue)

    # ---------- Message handler ----------
    def handle(self, msg: dict):
        evt = msg.get("event", "")
        if evt == "tick":
            start_et = msg["start_et"]; end_et = msg["end_et"]; now = msg["now"]
            self.lbl_window.config(
                text=f"ET window: {start_et.strftime('%Y-%m-%d %H:%M')} → {end_et.strftime('%H:%M')}  "
                     f"(AMS: {start_et.astimezone(AMS).strftime('%H:%M')}→{end_et.astimezone(AMS).strftime('%H:%M')}, "
                     f"UTC: {start_et.astimezone(UTC).strftime('%H:%M')}→{end_et.astimezone(UTC).strftime('%H:%M')})"
            )
            self.lbl_server.config(text=f"Local UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

            strike = msg["strike"]; px = msg["price"]; diff = msg["diff"]
            self.last_strike = strike
            self.var_strike.set(f"{strike:,.2f}")
            self.var_price.set(f"{px:,.2f}")
            self.var_diff.set(f"{diff:+,.2f} ({'UP' if diff>=0 else 'DOWN'})")
            self.lbl_diff.config(foreground=("#118a00" if diff >= 0 else "#c40000"))

            tau = int(msg["tau_sec"])
            mm, ss = divmod(tau, 60); hh, mm = divmod(mm, 60)
            self.var_tclose.set(f"{hh:02d}:{mm:02d}:{ss:02d}")
            self.progress["value"] = int(1000 * msg["frac_elapsed"])

            # price chart
            now_et = now.astimezone(ET)
            self._push_price_point_if_needed(now_et, px, force=bool(msg.get("price_changed", False)))
            self._redraw_price_chart(strike=strike)

            # sigma chart — use raw 1s sigma (msg['sig1s']); we plot RAW + SMOOTH
            sigma_1s_current = finite_sigma(msg.get("sig1s", msg["sigmas"]["1m"]))
            self.last_sigma_1s = sigma_1s_current
            self._push_sigma_point(now_et, sigma_1s_current)
            self._redraw_sigma_chart()

            # μ and table (we map "1m row" to 1s σ for live responsiveness)
            mu_ann = msg["mu_ann"]
            self.var_mu.set(f"{mu_ann:+.3f}")
            sigmas = msg["sigmas"]
            theos_neutral = msg["theos_neutral"]
            theos_mu      = msg["theos_mu"]
            for k in ["1m", "10m", "1h", "1d"]:
                self.rows[k]["sigma"].set(f"{finite_sigma(sigmas[k]):.3f}")
                self.rows[k]["up_neut"].set(f"{theos_neutral[k]*100:5.2f}%")
                self.rows[k]["down_neut"].set(f"{(1.0-theos_neutral[k])*100:5.2f}%")
                self.rows[k]["up_mu"].set(f"{theos_mu[k]*100:5.2f}%")
                self.rows[k]["down_mu"].set(f"{(1.0-theos_mu[k])*100:5.2f}%")

            if not HAVE_ARCH and (self.var_status.get() == "" or "arch not found" in self.var_status.get()):
                self.var_status.set("arch not found — using EWMA fallback for 1s σ. `pip install arch` for GARCH.")

        elif evt == "closed":
            start_et = msg.get("start_et"); end_et = msg.get("end_et")
            if "final_close" in msg:
                fc = msg["final_close"]; strike = msg["strike"]; outcome = msg["outcome"]
                self.var_status.set(
                    f"Closed {start_et.strftime('%H:%M')}→{end_et.strftime('%H:%M')} ET. "
                    f"Strike {strike:,.2f} | Close {fc:,.2f} | Outcome {outcome}. Rolling…"
                )
            else:
                self.var_status.set("Hour closed. Rolling to the new hour…")

        elif evt == "status":
            self.var_status.set(msg.get("msg", ""))

    def on_quit(self):
        self.stop_event.set()
        self.destroy()

# ------------------- Run -------------------
if __name__ == "__main__":
    App().mainloop()
