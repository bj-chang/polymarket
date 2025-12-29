#!/usr/bin/env python3
# btc_updown_gui__JUMP_1S_LIVE_FIXED.py
# Live 1-second Merton jump-diffusion with corrected UP/DOWN probabilities.
# - Diffusive variance via EWMA on NON-jump returns
# - Jump intensity via EWMA of jump indicator
# - Jump size mean/var via EWMA on jump returns
# - UP/DOWN probabilities from Poisson mixture of normals with:
#     * correct hurdle log(px/K)
#     * drift includes jump compensator: mu_eff = mu_ann - lambda_year * (exp(muJ + 0.5*sigmaJ^2)-1)
# - Backfill button warms the model using aggTrades
#
# NOTE (fixes for "Backfill last 10 minutes"):
# 1) After injecting backfilled data we now ALSO update the incremental plot cursors
#    (last_plot_time_num/last_price_plotted and last_sigma_time_num) so the next
#    live points append smoothly instead of being skipped or double-plotted.
# 2) Warm start for JumpDiffusion now uses the FULL backfilled return set
#    (previously only ~1/3), which produced under-initialized σ and odd shapes.

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

# ------------------- Config -------------------
BINANCE_API = "https://api.binance.com"
SYMBOL = "BTCUSDT"

ET  = ZoneInfo("America/New_York")
AMS = ZoneInfo("Europe/Amsterdam")
UTC = timezone.utc

# Table horizons (10m/1h/1d remain historical-vol based on 1m/1h/1d)
LOOKBACK_10M_MIN    = 600
LOOKBACK_1H_HRS     = 336
LOOKBACK_1D_DAYS    = 60

# Drift μ from 1m EWMA (kept)
MU_EWMA_MINUTES     = 10
MU_EWMA_LAMBDA      = 0.7
MU_MAX_ABS          = 2.0

# Cadences
REFRESH_PRICE_SEC   = 1
REFRESH_1M_CACHE_SEC= 2
REFRESH_SIGMA_1S    = 1
REFRESH_SIGMA_10M   = 10
REFRESH_SIGMA_1H    = 60
REFRESH_SIGMA_1D    = 300
REFRESH_MU          = 5

# DISPLAY-ONLY σ SMOOTHING
SIGMA_PLOT_HALFLIFE_SEC = 20  # EMA half-life for plotted total σ only

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
MAX_1S_POINTS  = 15000

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

def finite_sigma(v: float) -> float:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return SIGMA_MIN
    return min(SIGMA_MAX, max(SIGMA_MIN, float(v)))

def current_et_hour_window(now_dt: datetime) -> tuple[datetime, datetime]:
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
    def __init__(self, maxlen=MAX_1S_POINTS):
        self.times = deque(maxlen=maxlen)
        self.prices = deque(maxlen=maxlen)
        self.last_sec = None
        self.last_price = None
    def backfill_last_n_minutes(self, minutes: int):
        end_utc = datetime.now(tz=UTC)
        start_utc = end_utc - timedelta(minutes=minutes + 1)
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms   = int(end_utc.timestamp() * 1000)
        trades = fetch_agg_trades_range(start_ms, end_ms, limit=1000)
        if not trades:
            raise RuntimeError("No trades returned for backfill")
        sec_to_price = {}
        for t in trades:
            sec = int(int(t["T"]) // 1000)
            sec_to_price[sec] = float(t["p"])
        end_sec = int(end_utc.timestamp())
        start_sec = end_sec - minutes * 60
        last = None
        # look back a bit to seed "last"
        for s in range(start_sec - 120, start_sec + 1):
            if s in sec_to_price:
                last = sec_to_price[s]
        times = []
        prices = []
        for s in range(start_sec, end_sec + 1):
            if s in sec_to_price: last = sec_to_price[s]
            if last is None: continue
            times.append(datetime.fromtimestamp(s, tz=UTC))
            prices.append(last)
        if not times:
            raise RuntimeError("Backfill failed to produce per-second series")
        self.times.clear(); self.prices.clear()
        for t, p in zip(times, prices):
            self.times.append(t); self.prices.append(p)
        self.last_sec   = int(self.times[-1].timestamp())
        self.last_price = self.prices[-1]
    def tick(self, price_now: float):
        sec_now = int(datetime.now(tz=UTC).timestamp())
        if self.last_sec is None:
            self.last_sec = sec_now
            self.last_price = price_now
            self.times.append(datetime.fromtimestamp(sec_now, tz=UTC))
            self.prices.append(price_now)
            return True
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

# ------------------- Jump-Diffusion (Merton) 1s estimator -------------------
class JumpDiffusion1s:
    """
    Online Merton jump-diffusion parameter estimation on 1-second log-returns.
    r_t = μ dt + σ dW_t + J_t,   with N_t ~ Poisson(λ dt),  J ~ N(μ_J, σ_J^2) in log-space.

    Estimation (recursive):
      - Identify jumps when |r_t| > K * σ_diff_per_sec
      - σ_diff^2 (per-second) via EWMA on NON-jump r_t^2
      - λ (per-second) via EWMA of jump indicator 1{jump}
      - μ_J, σ_J^2 via EWMA on jump returns r_t (log space)
    """
    def __init__(self,
                 k_thresh: float = 4.0,
                 hl_sigma_sec: int = 300,   # half-life for diffusive variance (e.g., 5 min)
                 hl_lambda_sec: int = 120,  # half-life for jump intensity (e.g., 2 min)
                 hl_jump_sec: int = 300):   # half-life for jump size stats
        self.K = k_thresh
        self.alpha_sigma  = 1.0 - 0.5 ** (1.0 / max(1, hl_sigma_sec))
        self.alpha_lambda = 1.0 - 0.5 ** (1.0 / max(1, hl_lambda_sec))
        self.alpha_jump   = 1.0 - 0.5 ** (1.0 / max(1, hl_jump_sec))
        self.var_diff_per_sec = None
        self.lambda_per_sec   = 0.0
        self.muJ              = 0.0
        self.varJ             = 1e-10
        self.total_sigma_ann  = None

    def _sigma_per_sec(self):
        if self.var_diff_per_sec is None:
            return None
        return math.sqrt(max(1e-18, self.var_diff_per_sec))

    def _sigma_ann_from_persec(self, s_per_sec):
        return s_per_sec * math.sqrt(SECS_PER_YEAR)

    def warm_start(self, returns):
        """Initialize EWMA states from a history of per-second returns."""
        if not returns: return
        lam0 = 0.997
        v = None
        for r in returns[-5000:]:
            v = (1-lam0)*(r*r) + (lam0*v if v is not None else 0.0)
        if v is None or v <= 0:
            v = 1e-10
        s = math.sqrt(v)
        var_diff = None
        lambda_sec = 0.0
        muJ = None
        varJ = None
        for r in returns[-5000:]:
            is_jump = abs(r) > self.K * s
            if is_jump:
                muJ  = r if muJ is None else (1-self.alpha_jump)*muJ + self.alpha_jump*r
                dev  = 0.0 if muJ is None else (r - muJ)
                varJ = dev*dev if varJ is None else (1-self.alpha_jump)*varJ + self.alpha_jump*(dev*dev)
                lambda_sec = (1-self.alpha_lambda)*lambda_sec + self.alpha_lambda*1.0
            else:
                x2 = r*r
                var_diff = x2 if var_diff is None else (1-self.alpha_sigma)*var_diff + self.alpha_sigma*x2
                lambda_sec = (1-self.alpha_lambda)*lambda_sec + self.alpha_lambda*0.0
        self.var_diff_per_sec = var_diff if var_diff is not None else v
        self.lambda_per_sec   = lambda_sec
        self.muJ  = 0.0 if muJ  is None else muJ
        self.varJ = 1e-10 if varJ is None else max(1e-12, varJ)
        self._update_total_sigma_ann()

    def on_new_second(self, r_nat_1s: float | None):
        if r_nat_1s is None:
            self._update_total_sigma_ann()
            return
        s_per_sec = self._sigma_per_sec()
        if s_per_sec is None:
            self.var_diff_per_sec = r_nat_1s*r_nat_1s
            self._update_total_sigma_ann()
            return
        is_jump = abs(r_nat_1s) > self.K * s_per_sec
        self.lambda_per_sec = (1 - self.alpha_lambda)*self.lambda_per_sec + self.alpha_lambda*(1.0 if is_jump else 0.0)
        if is_jump:
            self.muJ = (1 - self.alpha_jump)*self.muJ + self.alpha_jump*r_nat_1s
            dev = r_nat_1s - self.muJ
            self.varJ = (1 - self.alpha_jump)*self.varJ + self.alpha_jump*(dev*dev)
        else:
            x2 = r_nat_1s * r_nat_1s
            self.var_diff_per_sec = (1 - self.alpha_sigma)*self.var_diff_per_sec + self.alpha_sigma*x2
        self._update_total_sigma_ann()

    def params_ann(self):
        """
        Returns:
          sigma_diff_ann, lambda_year, muJ, sigmaJ, sigma_total_ann
        """
        s_per_sec = self._sigma_per_sec()
        sigma_diff_ann = 0.0 if s_per_sec is None else self._sigma_ann_from_persec(s_per_sec)
        lambda_year = max(0.0, self.lambda_per_sec) * SECS_PER_YEAR
        sigmaJ = math.sqrt(max(1e-12, self.varJ))
        return (
            finite_sigma(sigma_diff_ann),
            lambda_year,
            float(self.muJ),
            float(sigmaJ),
            finite_sigma(self.total_sigma_ann if self.total_sigma_ann is not None else sigma_diff_ann)
        )

    def _update_total_sigma_ann(self):
        s_per_sec = self._sigma_per_sec()
        if s_per_sec is None:
            self.total_sigma_ann = SIGMA_MIN
            return
        sigma_diff_ann2 = (s_per_sec * math.sqrt(SECS_PER_YEAR))**2
        sigmaJ2 = max(1e-12, self.varJ)
        jump_var_ann = max(0.0, self.lambda_per_sec) * SECS_PER_YEAR * (sigmaJ2 + self.muJ*self.muJ)
        self.total_sigma_ann = math.sqrt(max(1e-18, sigma_diff_ann2 + jump_var_ann))

# --------- UP probability under Merton jump-diffusion (Poisson mixture) ----------
def up_prob_merton_pxK(px: float, K: float, tau_years: float,
                       sigma_diff_ann: float,    # diffusion-only annualized σ
                       lambda_per_year: float,   # jump intensity per year
                       mu_ann: float,            # annual drift you want to use
                       muJ: float,               # mean of log jump size per jump
                       sigmaJ: float,            # std of log jump size per jump
                       max_terms: int = 80,
                       tail_tol: float = 1e-10) -> float:
    """
    P(S_T >= K) under Merton jump-diffusion using Poisson mixture of normals.

    Log mixture for N=n:
      m_n = log(px/K) + (mu_eff - 0.5*sigma^2)*tau + n*muJ
      v_n = (sigma^2)*tau + n*(sigmaJ^2)
      where mu_eff = mu_ann - lambda * (exp(muJ + 0.5*sigmaJ^2)-1)

    Returns value in [0,1].
    """
    if tau_years <= 0:
        return 1.0 if px >= K else 0.0

    sigma = max(1e-12, float(sigma_diff_ann))
    lam   = max(0.0,   float(lambda_per_year))
    muJ   = float(muJ)
    sJ2   = max(1e-18, float(sigmaJ)**2)

    kappa  = math.exp(muJ + 0.5*sJ2) - 1.0
    mu_eff = float(mu_ann) - lam * kappa

    m_base = math.log(max(1e-18, px) / max(1e-18, K)) + (mu_eff - 0.5*sigma*sigma) * tau_years
    v_base = (sigma * sigma) * tau_years

    lam_tau = lam * tau_years
    w_n   = math.exp(-lam_tau)  # n = 0
    prob  = 0.0
    cum_w = 0.0

    for n in range(max_terms):
        m_n = m_base + n * muJ
        v_n = v_base + n * sJ2
        s_n = math.sqrt(max(1e-18, v_n))
        z   = - m_n / s_n  # threshold is 0 after embedding log(px/K)
        cond_up = 1.0 - norm_cdf(z)
        prob  += w_n * cond_up
        cum_w += w_n
        if 1.0 - cum_w < tail_tol and n >= 5:
            break
        w_n = w_n * lam_tau / (n + 1.0)

    return min(1.0, max(0.0, prob))

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

    cache_1m = OneMinuteCloseCache(2000)
    try: cache_1m.bootstrap()
    except Exception: pass

    cache_1s = SecondPriceCache(MAX_1S_POINTS)
    backfilled_ok = False
    try:
        cache_1s.backfill_last_n_minutes(15)
        backfilled_ok = True
    except Exception:
        try:
            px0 = get_last_trade_price()
            cache_1s.tick(px0)
        except Exception:
            pass

    jd1s = JumpDiffusion1s(k_thresh=4.0, hl_sigma_sec=300, hl_lambda_sec=120, hl_jump_sec=300)
    try:
        if backfilled_ok:
            rets = log_returns_from_prices(cache_1s.get_prices(), step=1)
            jd1s.warm_start(rets)
    except Exception:
        pass

    last_price_t   = 0.0
    last_1mcache_t = 0.0
    last_s1s_t     = 0.0
    last_s10_t     = 0.0
    last_s1h_t     = 0.0
    last_s1d_t     = 0.0
    last_mu_t      = 0.0

    px = None
    last_px_sent = None

    sigmas = {"1s_total": SIGMA_MIN, "10m": SIGMA_MIN, "1h": SIGMA_MIN, "1d": SIGMA_MIN}
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

        # Update 1m cache
        if tnow - last_1mcache_t >= REFRESH_1M_CACHE_SEC:
            try:
                cache_1m.update()
            except Exception:
                pass
            last_1mcache_t = tnow

        # Price & returns each second → feed jump-diffusion
        price_changed = False
        if tnow - last_price_t >= REFRESH_PRICE_SEC:
            try:
                px = get_last_trade_price()
                price_changed = (last_px_sent is None) or (px != last_px_sent)
                last_px_sent = px
                cache_1s.tick(px)
                prices = cache_1s.get_prices()
                r_1s = None
                if len(prices) >= 2 and prices[-2] > 0 and prices[-1] > 0:
                    r_1s = math.log(prices[-1] / prices[-2])
                jd1s.on_new_second(r_1s)
            except Exception as e:
                out_q.put({"event": "status", "msg": f"Network issue (price): {e}"})
            last_price_t = tnow

        # 1s sigma (TOTAL, includes jumps)
        if tnow - last_s1s_t >= REFRESH_SIGMA_1S:
            try:
                sigma_diff_ann, lambda_year, muJ, sigmaJ, sigma_total_ann = jd1s.params_ann()
                sigmas["1s_total"] = sigma_total_ann
            except Exception:
                sigmas["1s_total"] = SIGMA_MIN
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

            sigma_diff_ann, lambda_year, muJ, sigmaJ, sigma_total_ann = jd1s.params_ann()

            # Jump-diffusion probabilities (fixed)
            up_neut_jump = up_prob_merton_pxK(
                px, strike, tau_years,
                sigma_diff_ann=sigma_diff_ann,
                lambda_per_year=lambda_year,
                mu_ann=R_NEUTRAL, muJ=muJ, sigmaJ=sigmaJ
            )
            up_mu_jump   = up_prob_merton_pxK(
                px, strike, tau_years,
                sigma_diff_ann=sigma_diff_ann,
                lambda_per_year=lambda_year,
                mu_ann=mu_ann, muJ=muJ, sigmaJ=sigmaJ
            )

            # For the other rows, keep diffusion-only semantics with their historical vols
            def theo_up_for_sigma(s_ann, mu):
                s = max(1e-12, s_ann)
                m = (mu - 0.5*s*s) * tau_years + math.log(px / strike)
                v = (s*s) * tau_years
                z = - m / math.sqrt(max(1e-18, v))
                return 1.0 - norm_cdf(z)

            up10_neut = theo_up_for_sigma(sigmas["10m"], R_NEUTRAL)
            up10_mu   = theo_up_for_sigma(sigmas["10m"], mu_ann)
            up1h_neut = theo_up_for_sigma(sigmas["1h"],  R_NEUTRAL)
            up1h_mu   = theo_up_for_sigma(sigmas["1h"],  mu_ann)
            up1d_neut = theo_up_for_sigma(sigmas["1d"],  R_NEUTRAL)
            up1d_mu   = theo_up_for_sigma(sigmas["1d"],  mu_ann)

            out_q.put({
                "event": "tick",
                "now": now,
                "start_et": start_et, "end_et": end_et,
                "strike": strike, "price": px, "diff": diff,
                "sigmas": {
                    "1s_total": sigma_total_ann,
                    "10m": sigmas["10m"], "1h": sigmas["1h"], "1d": sigmas["1d"]
                },
                "jd_params": {
                    "sigma_diff_ann": sigma_diff_ann,
                    "lambda_year": lambda_year,
                    "muJ": muJ,
                    "sigmaJ": sigmaJ
                },
                "theos": {
                    "1s_up_neut": up_neut_jump, "1s_up_mu": up_mu_jump,
                    "10m_up_neut": up10_neut, "10m_up_mu": up10_mu,
                    "1h_up_neut": up1h_neut, "1h_up_mu": up1h_mu,
                    "1d_up_neut": up1d_neut, "1d_up_mu": up1d_mu
                },
                "mu_ann": mu_ann,
                "tau_sec": tau_sec, "frac_elapsed": frac_elapsed,
                "price_changed": price_changed
            })

        time.sleep(0.02)

# ------------------- GUI -------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTCUSDT — Δ Price vs Strike & Vol (1s: Jump-Diffusion live, FIXED)")
        self.geometry("900x960")
        self.resizable(False, False)

        self._init_style()

        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=worker, args=(self.q, self.stop_event), daemon=True)
        self.worker_thread.start()

        self.last_strike = None
        self.last_sigma_total_1s = None

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

        # ---- Volatility chart (1s TOTAL) ----
        chart_wrap = ttk.Frame(self, padding=(10,0))
        chart_wrap.grid(row=4, column=0, sticky="nsew")
        self._init_sigma_chart(chart_wrap)

        # μ display
        muwrap = ttk.Frame(self, padding=(10,0))
        muwrap.grid(row=5, column=0, sticky="ew")
        ttk.Separator(muwrap, orient="horizontal").grid(row=0, column=0, sticky="ew", pady=6)
        self.var_mu = tk.StringVar(value="…")
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
            "1s":  {"label": "1-second (Jump-Diffusion live, FIXED)"},
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
        self.var_status = tk.StringVar(value="Jump-Diffusion live with corrected probabilities (compensator + px/K).")
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

    # ---------- Price chart ----------
    def _init_price_chart(self, parent):
        self.price_time_hist = deque(maxlen=600)  # mdates floats
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
            time_ok = (self.last_plot_time_num is None) or ((when_num - self.last_plot_time_num) >= (1/86400.0))
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

    # ---------- σ chart (TOTAL σ: RAW + SMOOTH for jump-diffusion) ----------
    def _init_sigma_chart(self, parent):
        self.sigma_hist_raw    = deque(maxlen=900)
        self.sigma_hist_smooth = deque(maxlen=900)
        self.time_hist  = deque(maxlen=900)
        self.last_sigma_time_num = None

        self.sigma_plot_alpha = 1.0 - 0.5 ** (1.0 / max(1, SIGMA_PLOT_HALFLIFE_SEC))
        self.sigma_ema_state = None

        self.fig = Figure(figsize=(8.6, 2.4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        (self.sigma_line_smooth,) = self.ax.plot([], [], lw=1.8)
        (self.sigma_line_raw,)    = self.ax.plot([], [], lw=1.0, alpha=0.35)

        self.ax.set_title("σ (annualized, TOTAL) — last 10 minutes (Jump-Diffusion, FIXED)")
        self.ax.set_ylabel("σ_total (ann.)")
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

    def _push_sigma_point(self, when_dt_et: datetime, sigma_ann_total: float):
        when_num = mdates.date2num(when_dt_et)
        one_sec = 1.0 / 86400.0
        if self.last_sigma_time_num is not None and (when_num - self.last_sigma_time_num) < one_sec:
            return
        self.last_sigma_time_num = when_num

        s_raw = finite_sigma(sigma_ann_total)
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

    # ---------- Backfill (PER-SECOND; warm start jump model) ----------
    def on_backfill_10m_persec(self):
        try:
            if self.last_strike is None:
                now_utc = datetime.now(tz=UTC)
                start_et, end_et = current_et_hour_window(now_utc)
                k = get_kline_open_close(start_et, end_et)
                if not k:
                    self.var_status.set("Backfill: couldn't fetch current hour strike.")
                    return
                self.last_strike = k[0]
                self.var_strike.set(f"{self.last_strike:,.2f}")

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
            # seed 'last' using a short lookback so the first seconds aren't dropped
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

            # Replace price chart history
            self.price_time_hist.clear(); self.price_hist.clear()
            times_et = [t.astimezone(ET) for t in times]
            times_num = [mdates.date2num(t) for t in times_et]
            for tn, pv in zip(times_num, prices):
                self.price_time_hist.append(tn)
                self.price_hist.append(pv)

            # >>> FIX 1: keep plot cursors in sync so the next live tick appends cleanly
            self.last_plot_time_num = self.price_time_hist[-1]
            self.last_price_plotted = self.price_hist[-1]

            if self.last_strike is not None:
                self._redraw_price_chart(strike=self.last_strike)

            # Build per-second returns and warm-start jump estimator
            rets = log_returns_from_prices(prices, 1)
            jd = JumpDiffusion1s()
            # >>> FIX 2: warm start with the FULL backfilled history (not just ~1/3)
            jd.warm_start(rets)

            # Reset σ history and rebuild from backfilled returns for a smooth trace
            self.time_hist.clear(); self.sigma_hist_raw.clear(); self.sigma_hist_smooth.clear()
            self.sigma_ema_state = None

            tn_iter = times_num[-len(rets):]
            for tn, r in zip(tn_iter, rets):
                jd.on_new_second(r)
                _, _, _, _, sigma_total_ann = jd.params_ann()
                s_raw = finite_sigma(sigma_total_ann)
                if self.sigma_ema_state is None:
                    self.sigma_ema_state = s_raw
                else:
                    a = self.sigma_plot_alpha
                    self.sigma_ema_state = a * s_raw + (1 - a) * self.sigma_ema_state
                self.time_hist.append(tn)
                self.sigma_hist_raw.append(s_raw)
                self.sigma_hist_smooth.append(finite_sigma(self.sigma_ema_state))

            # >>> FIX 3: set σ plot last time so next live point is spaced correctly
            if self.time_hist:
                self.last_sigma_time_num = self.time_hist[-1]

            self._redraw_sigma_chart()
            self.var_status.set("Backfilled last 10m (per-second). Jump-Diffusion live is running (fixed probabilities).")
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

            # sigma chart — TOTAL σ (jump-diffusion)
            sigma_total = finite_sigma(msg["sigmas"]["1s_total"])
            self.last_sigma_total_1s = sigma_total
            self._push_sigma_point(now_et, sigma_total)
            self._redraw_sigma_chart()

            # μ and table
            self.var_mu.set(f"{msg['mu_ann']:+.3f}")

            # Row 1: jump-diffusion
            self.rows["1s"]["sigma"].set(f"{sigma_total:.3f}")
            up_neut = msg["theos"]["1s_up_neut"]
            up_mu   = msg["theos"]["1s_up_mu"]
            self.rows["1s"]["up_neut"].set(f"{up_neut*100:5.2f}%")
            self.rows["1s"]["down_neut"].set(f"{(1.0-up_neut)*100:5.2f}%")
            self.rows["1s"]["up_mu"].set(f"{up_mu*100:5.2f}%")
            self.rows["1s"]["down_mu"].set(f"{(1.0-up_mu)*100:5.2f}%")

            # Other rows unchanged
            for key, upn_key, upm_key in [
                ("10m","10m_up_neut","10m_up_mu"),
                ("1h", "1h_up_neut", "1h_up_mu"),
                ("1d", "1d_up_neut", "1d_up_mu")
            ]:
                self.rows[key]["sigma"].set(f"{finite_sigma(msg['sigmas'][key]):.3f}")
                upn = msg["theos"][upn_key]; upm = msg["theos"][upm_key]
                self.rows[key]["up_neut"].set(f"{upn*100:5.2f}%")
                self.rows[key]["down_neut"].set(f"{(1.0-upn)*100:5.2f}%")
                self.rows[key]["up_mu"].set(f"{upm*100:5.2f}%")
                self.rows[key]["down_mu"].set(f"{(1.0-upm)*100:5.2f}%")

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
