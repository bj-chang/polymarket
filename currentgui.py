# btc_updown_gui_1h_only.py
# Fixed to 1-hour windows only.
# - Strike = open of the current ET hour
# - Rollover every hour on the hour (ET)
# - Progress/time-to-close reflect the 1-hour window
# - Backfill button still builds per-second series for last 10 minutes
# - σ (volatility) chart is throttled to ~1 Hz so it retains ~10 minutes of data.

import threading, queue, time, json, math, tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError
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

# Vol lookbacks for table (unchanged)
LOOKBACK_1M_MIN     = 180
LOOKBACK_10M_MIN    = 600
LOOKBACK_1H_HRS     = 336
LOOKBACK_1D_DAYS    = 60

# Short-term drift μ (EWMA of 1m returns)
MU_EWMA_MINUTES     = 10
MU_EWMA_LAMBDA      = 0.7
MU_MAX_ABS          = 2.0

# Recompute cadences (seconds)
REFRESH_PRICE_SEC   = 1
REFRESH_1M_CACHE_SEC= 2
REFRESH_SIGMA_1M    = 2
REFRESH_SIGMA_10M   = 10
REFRESH_SIGMA_1H    = 60
REFRESH_SIGMA_1D    = 300
REFRESH_MU          = 5

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
MAX_1M_CACHE   = max(LOOKBACK_1M_MIN + 12, 200)

# Per-second vol backfill settings
ROLLING_SEC_FOR_VOL = 60  # 60s rolling window on 1s returns

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

def get_kline_open_close_hour(start_et: datetime, end_et: datetime):
    """Return open, close, open_ms, close_ms for the 1h kline covering [start_et, end_et)."""
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

# Aggregated trades over [start_ms, end_ms], paginated
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

# ------------------- Time helpers -------------------
def current_et_hour_window(now_utc: datetime):
    now_et = now_utc.astimezone(ET)
    start_et = now_et.replace(minute=0, second=0, microsecond=0)
    end_et   = start_et + timedelta(hours=1)
    return start_et, end_et

# ------------------- Math helpers -------------------
def log_returns_from_prices(prices: list[float], step: int) -> list[float]:
    if len(prices) <= step: return []
    rets = []
    for i in range(step, len(prices)):
        p0, p1 = prices[i-step], prices[i]
        if p0 > 0: rets.append(math.log(p1 / p0))
    return rets

def ann_sigma_from_returns(returns: list[float], units_per_year: float) -> float:
    if not returns: return 0.0
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

# ------------------- Incremental 1m cache (for table σs & μ) -------------------
class OneMinuteCloseCache:
    def __init__(self, maxlen=MAX_1M_CACHE):
        self.closes = deque(maxlen=maxlen)
        self.last_open_ms = None
    def bootstrap(self):
        k = fetch_1m_klines(limit=min(1000, max(5, self.closes.maxlen)))
        if not k or len(k) < 2: return
        for row in k[:-1]:
            self.closes.append(float(row[4]))
            self.last_open_ms = row[0]
    def update(self):
        k = fetch_1m_klines(limit=2)
        if not k or len(k) < 2: return False
        closed = k[-2]
        open_ms = closed[0]
        if self.last_open_ms is None or open_ms > self.last_open_ms:
            self.closes.append(float(closed[4]))
            self.last_open_ms = open_ms
            return True
        return False
    def get_list(self) -> list[float]:
        return list(self.closes)

def sigma_1m_from_cache(cache: OneMinuteCloseCache) -> float:
    closes = cache.get_list()[-(LOOKBACK_1M_MIN + 1):]
    rets = log_returns_from_prices(closes, step=1)
    return ann_sigma_from_returns(rets, MINS_PER_YEAR)

def sigma_10m_from_cache(cache: OneMinuteCloseCache) -> float:
    closes = cache.get_list()[-(LOOKBACK_10M_MIN + 1):]
    rets = log_returns_from_prices(closes, step=10)
    return ann_sigma_from_returns(rets, TENM_PER_YEAR)

def mu_ewma_ann_from_cache(cache: OneMinuteCloseCache, lam: float = MU_EWMA_LAMBDA) -> float:
    closes = cache.get_list()[-(MU_EWMA_MINUTES + 1):]
    if len(closes) < 2: return 0.0
    rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
    if not rets: return 0.0
    m = None
    for r in rets:
        m = r if m is None else lam * m + (1.0 - lam) * r
    mu_ann = (m if m is not None else 0.0) * MINS_PER_YEAR
    return max(-MU_MAX_ABS, min(MU_MAX_ABS, mu_ann))

# ------------------- Worker thread -------------------
def worker(out_q: "queue.Queue[dict]", stop_event: threading.Event):
    # Initialize 1h window/strike
    now = datetime.now(tz=UTC)
    start_et, end_et = current_et_hour_window(now)
    start_utc = start_et.astimezone(UTC); end_utc = end_et.astimezone(UTC)

    strike = None
    while not stop_event.is_set() and strike is None:
        try:
            k = get_kline_open_close_hour(start_et, end_et)
            if k: strike = k[0]
        except Exception:
            pass
        time.sleep(0.3)

    cache_1m = OneMinuteCloseCache(MAX_1M_CACHE)
    try: cache_1m.bootstrap()
    except Exception: pass

    last_price_t = last_cache_t = last_s1m_t = last_s10_t = last_s1h_t = last_s1d_t = last_mu_t = 0.0
    px = None
    sigmas = {"1m": 0.0, "10m": 0.0, "1h": 0.0, "1d": 0.0}
    mu_ann = 0.0
    last_px_sent = None

    while not stop_event.is_set():
        tnow = time.time()
        now = datetime.now(tz=UTC)

        # Rollover on window end (hourly)
        if now >= end_utc:
            try:
                _, final_close, _, _ = get_kline_open_close_hour(start_et, end_et)
                outcome = "UP" if (final_close >= strike) else "DOWN"
                out_q.put({
                    "event": "closed",
                    "final_close": final_close,
                    "outcome": outcome,
                    "strike": strike, "now": now, "start_et": start_et, "end_et": end_et
                })
            except Exception as e:
                out_q.put({"event": "status", "msg": f"Closed 1h: couldn't fetch final close ({e}). Rolling…"})
            start_et, end_et = current_et_hour_window(now)
            start_utc = start_et.astimezone(UTC); end_utc = end_et.astimezone(UTC)
            strike = None
            while not stop_event.is_set() and strike is None:
                try:
                    k = get_kline_open_close_hour(start_et, end_et)
                    if k: strike = k[0]
                except Exception:
                    pass
                time.sleep(0.3)
            last_price_t = last_cache_t = 0.0

        # Update 1m cache for table σs and μ
        if tnow - last_cache_t >= REFRESH_1M_CACHE_SEC:
            try: cache_1m.update()
            except Exception: pass
            last_cache_t = tnow

        if tnow - last_s1m_t >= REFRESH_SIGMA_1M:
            try:   sigmas["1m"] = min(SIGMA_MAX, max(SIGMA_MIN, sigma_1m_from_cache(cache_1m)))
            except Exception: pass
            last_s1m_t = tnow

        if tnow - last_s10_t >= REFRESH_SIGMA_10M:
            try:   sigmas["10m"] = min(SIGMA_MAX, max(SIGMA_MIN, sigma_10m_from_cache(cache_1m)))
            except Exception: pass
            last_s10_t = tnow

        if tnow - last_s1h_t >= REFRESH_SIGMA_1H:
            try:
                closes_1h = fetch_1h_closes(LOOKBACK_1H_HRS)
                rets_1h   = log_returns_from_prices(closes_1h, step=1)
                sigmas["1h"] = min(SIGMA_MAX, max(SIGMA_MIN, ann_sigma_from_returns(rets_1h, HRS_PER_YEAR)))
            except Exception:
                pass
            last_s1h_t = tnow

        if tnow - last_s1d_t >= REFRESH_SIGMA_1D:
            try:
                closes_1d = fetch_1d_closes(LOOKBACK_1D_DAYS)
                rets_1d   = log_returns_from_prices(closes_1d, step=1)
                sigmas["1d"] = min(SIGMA_MAX, max(SIGMA_MIN, ann_sigma_from_returns(rets_1d, DAYS_PER_YEAR)))
            except Exception:
                pass
            last_s1d_t = tnow

        if tnow - last_mu_t >= REFRESH_MU:
            try:    mu_ann = mu_ewma_ann_from_cache(cache_1m)
            except Exception: pass
            last_mu_t = tnow

        # Price (every 1s)
        price_changed = False
        if tnow - last_price_t >= REFRESH_PRICE_SEC:
            try:
                px = get_last_trade_price()
                price_changed = (last_px_sent is None) or (px != last_px_sent)
                last_px_sent = px
            except Exception as e:
                out_q.put({"event": "status", "msg": f"Network issue (price): {e}"})
            last_price_t = tnow

        # Emit tick
        if px is not None and strike is not None:
            tau_sec = max(0.0, (end_utc - now).total_seconds())
            tau_years = tau_sec / SECS_PER_YEAR
            diff = px - strike
            elapsed = (now - start_utc).total_seconds()
            frac_elapsed = min(1.0, max(0.0, elapsed / 3600.0))

            theos_neutral = {k: fair_up_prob(strike, px, tau_years, sigmas[k], R_NEUTRAL) for k in sigmas}
            theos_mu      = {k: fair_up_prob(strike, px, tau_years, sigmas[k], mu_ann)   for k in sigmas}

            out_q.put({
                "event": "tick",
                "now": now,
                "start_et": start_et, "end_et": end_et,
                "mode": "1h",
                "strike": strike, "price": px, "diff": diff,
                "sigmas": sigmas,
                "theos_neutral": theos_neutral, "theos_mu": theos_mu,
                "mu_ann": mu_ann,
                "tau_sec": tau_sec, "frac_elapsed": frac_elapsed,
                "price_changed": price_changed
            })

        time.sleep(0.05)

# ------------------- GUI -------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTCUSDT — ET 1h Window | Δ Price vs Strike & Vol (per-second backfill)")
        self.geometry("900x960")
        self.resizable(False, False)

        self._init_style()

        # Shared state & queues
        self.q = queue.Queue()       # worker -> UI
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=worker, args=(self.q, self.stop_event), daemon=True)
        self.worker_thread.start()

        # Keep last knowns
        self.last_strike = None
        self.last_sigma_1m = None

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
        self.var_strike_caption = tk.StringVar(value="Strike (hour open):")
        self.var_strike = tk.StringVar(value="…")
        self.var_price  = tk.StringVar(value="…")
        self.var_diff   = tk.StringVar(value="…")
        self.var_tclose = tk.StringVar(value="…")
        ttk.Label(pricef, textvariable=self.var_strike_caption, style="Caption.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(pricef, textvariable=self.var_strike, style="Value.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(pricef, text="Price (last):", style="Caption.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(pricef, textvariable=self.var_price, style="Value.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(pricef, text="Δ vs strike:", style="Caption.TLabel").grid(row=0, column=2, sticky="w")
        self.lbl_diff = ttk.Label(pricef, textvariable=self.var_diff, style="Value.TLabel")
        self.lbl_diff.grid(row=1, column=2, sticky="w")
        ttk.Label(pricef, text="Time to close:", style="Caption.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Label(pricef, textvariable=self.var_tclose, style="Value.TLabel").grid(row=1, column=3, sticky="w")

        # Backfill button (per-second)
        self.btn_backfill = ttk.Button(pricef, text="Backfill last 10m (per-second)", command=self.on_backfill_10m_persec)
        self.btn_backfill.grid(row=1, column=5, sticky="e", padx=(10,0))

        # Progress
        pwrap = ttk.Frame(self, padding=(10,0))
        pwrap.grid(row=2, column=0, sticky="ew")
        self.progress = ttk.Progressbar(pwrap, orient="horizontal", length=860, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=0, sticky="ew")

        # ---- Price chart (TOP; Y shows Δ vs strike) ----
        price_chart_wrap = ttk.Frame(self, padding=(10,0))
        price_chart_wrap.grid(row=3, column=0, sticky="nsew")
        self._init_price_chart(price_chart_wrap)

        # ---- Volatility chart (BOTTOM) ----
        chart_wrap = ttk.Frame(self, padding=(10,0))
        chart_wrap.grid(row=4, column=0, sticky="nsew")
        self._init_sigma_chart(chart_wrap)

        # μ display
        muwrap = ttk.Frame(self, padding=(10,0))
        muwrap.grid(row=5, column=0, sticky="ew")
        ttk.Separator(muwrap, orient="horizontal").grid(row=0, column=0, sticky="ew", pady=6)
        self.var_mu = tk.StringVar(value="…")
        ttk.Label(muwrap, text=f"μ (annualized) from EWMA 1m returns (last {MU_EWMA_MINUTES}m, λ={MU_EWMA_LAMBDA}):",
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
            "1m":  {"label": "1-minute (from 1m returns)"},
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
        self.var_status = tk.StringVar(value="")
        self.lbl_status = ttk.Label(bottom, textvariable=self.var_status, style="Subtle.TLabel")
        self.lbl_status.grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Quit", command=self.on_quit, style="Danger.TButton").grid(row=0, column=1, sticky="e", padx=(10,0))

        self.after(UI_REFRESH_MS, self.poll_queue)

    # ---------- Styles ----------
    def _init_style(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
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

    # ---------- Price chart (TOP; Y = Δ vs strike, ticks at 10s with +/− labels) ----------
    def _init_price_chart(self, parent):
        self.price_time_hist = deque(maxlen=600)
        self.price_hist      = deque(maxlen=600)
        self.last_price_plotted = None
        self.last_plot_time = None

        self.fig_px = Figure(figsize=(8.6, 2.8), dpi=100)
        self.ax_px  = self.fig_px.add_subplot(111)
        (self.price_line,) = self.ax_px.plot([], [], lw=1.5)

        self.strike_line = self.ax_px.axhline(y=0.0, linestyle="--", linewidth=1.0, alpha=0.6)

        self.ax_px.set_title("Δ price vs strike — last 10 minutes")
        self.ax_px.set_ylabel("Δ vs strike (USDT)")
        self.ax_px.set_xlabel("Time (ET)")
        self.ax_px.grid(True, which="major", alpha=0.3)
        self.ax_px.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        self.ax_px.xaxis.set_major_locator(mdates.AutoDateLocator())

        now = datetime.now(tz=UTC).astimezone(ET)
        self.ax_px.set_xlim(now - timedelta(minutes=10), now)
        self.ax_px.set_ylim(-10.0, 10.0)

        self.canvas_px = FigureCanvasTkAgg(self.fig_px, master=parent)
        self.canvas_px.get_tk_widget().grid(row=0, column=0, sticky="ew")

    def _push_price_point_if_needed(self, when_dt_et: datetime, price: float, force: bool):
        should_append = False
        if not self.price_time_hist:
            should_append = True
        else:
            time_ok = (self.last_plot_time is None) or ((when_dt_et - self.last_plot_time).total_seconds() >= 1.0)
            price_changed = (self.last_price_plotted is None) or (price != self.last_price_plotted)
            should_append = force or time_ok or price_changed

        if should_append:
            self.price_time_hist.append(when_dt_et)
            self.price_hist.append(price)
            self.last_price_plotted = price
            self.last_plot_time = when_dt_et
            ten_min_ago = when_dt_et - timedelta(minutes=10)
            while self.price_time_hist and self.price_time_hist[0] < ten_min_ago:
                self.price_time_hist.popleft()
                self.price_hist.popleft()

    def _redraw_price_chart(self, strike: float):
        if not self.price_time_hist:
            return

        diffs = [p - strike for p in self.price_hist]
        self.price_line.set_data(self.price_time_hist, diffs)

        # X-axis: 10-minute rolling window
        x_right = self.price_time_hist[-1]
        x_left = x_right - timedelta(minutes=10)
        self.ax_px.set_xlim(x_left, x_right)

        # Y-axis dynamic range calculation
        y_min_raw = min(diffs)
        y_max_raw = max(diffs)
        pad = 10.0
        lower = math.floor((y_min_raw - pad) / 10.0) * 10.0
        upper = math.ceil((y_max_raw + pad) / 10.0) * 10.0
        if upper - lower < 20.0:
            center = 0.5 * (upper + lower)
            lower = center - 10.0
            upper = center + 10.0

        # ✅ Dynamically choose tick spacing so <=10 ticks
        range_size = upper - lower
        spacing = 10
        while range_size / spacing > 10:
            spacing += 10

        # Apply new limits and tick spacing
        self.ax_px.set_ylim(lower, upper)
        self.ax_px.yaxis.set_major_locator(mticker.MultipleLocator(spacing))
        self.ax_px.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:+.0f}"))

        self.fig_px.autofmt_xdate()
        self.canvas_px.draw_idle()


    # ---------- σ chart (BOTTOM; ±0.01 pad) ----------
    def _init_sigma_chart(self, parent):
        self.sigma_hist = deque(maxlen=600)
        self.time_hist  = deque(maxlen=600)
        self.last_sigma_plot_time = None   # <-- throttle anchor for ~1 Hz σ appends

        self.fig = Figure(figsize=(8.6, 2.4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        (self.sigma_line,) = self.ax.plot([], [], lw=1.5)

        self.ax.set_title("σ (annualized, from 1s returns; 60s window) — last 10 minutes (±0.01 pad)")
        self.ax.set_ylabel("σ (ann.)")
        self.ax.set_xlabel("Time (ET)")
        self.ax.grid(True, which="major", alpha=0.3)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        now = datetime.now(tz=UTC).astimezone(ET)
        self.ax.set_xlim(now - timedelta(minutes=10), now)
        self.ax.set_ylim(0.0, 1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="ew")

    def _push_sigma_point(self, when_dt_et: datetime, sigma_1m: float):
        # Throttle σ appends to ~1 Hz or when value changes to avoid evicting history too fast.
        should_append = (
            self.last_sigma_plot_time is None
            or (when_dt_et - self.last_sigma_plot_time).total_seconds() >= 1.0
            or (not self.sigma_hist or sigma_1m != self.sigma_hist[-1])
        )
        if not should_append:
            return

        self.time_hist.append(when_dt_et)
        self.sigma_hist.append(sigma_1m)
        self.last_sigma_plot_time = when_dt_et

        ten_min_ago = when_dt_et - timedelta(minutes=10)
        while self.time_hist and self.time_hist[0] < ten_min_ago:
            self.time_hist.popleft()
            self.sigma_hist.popleft()

    def _redraw_sigma_chart(self):
        if not self.time_hist:
            return
        self.sigma_line.set_data(self.time_hist, self.sigma_hist)

        x_right = self.time_hist[-1]
        x_left  = x_right - timedelta(minutes=10)
        self.ax.set_xlim(x_left, x_right)

        y_min = min(self.sigma_hist)
        y_max = max(self.sigma_hist)
        lower = max(0.0, y_min - 0.01)
        upper = y_max + 0.01
        if upper - lower < 0.02:
            center = 0.5 * (upper + lower)
            lower  = max(0.0, center - 0.01)
            upper  = center + 0.01
        self.ax.set_ylim(lower, upper)

        self.fig.autofmt_xdate()
        self.canvas.draw_idle()

    # ---------- Backfill (PER-SECOND, no interpolation) ----------
    def on_backfill_10m_persec(self):
        """
        Build per-second series for the last 10 minutes from aggregated trades:
        - price at each second = last trade price within that second; if no trade, carry forward previous second
        - volatility at each second = rolling std of 1s log returns over last 60s, annualized
        """
        try:
            # Ensure strike for current 1h window
            if self.last_strike is None:
                now_utc = datetime.now(tz=UTC)
                start_et, end_et = current_et_hour_window(now_utc)
                k = get_kline_open_close_hour(start_et, end_et)
                if not k:
                    self.var_status.set("Backfill: couldn't fetch current window strike.")
                    return
                self.last_strike = k[0]
                self.var_strike.set(f"{self.last_strike:,.2f}")

            # Fetch aggTrades for last 11 minutes (buffer)
            end_utc = datetime.now(tz=UTC)
            start_utc = end_utc - timedelta(minutes=11)
            start_ms = int(start_utc.timestamp() * 1000)
            end_ms   = int(end_utc.timestamp() * 1000)

            trades = fetch_agg_trades_range(start_ms, end_ms, limit=1000)
            if not trades:
                self.var_status.set("Backfill: no trades returned.")
                return

            # Map second -> last price in that second
            sec_to_price = {}
            for t in trades:
                sec = int(int(t["T"]) // 1000)
                price = float(t["p"])
                sec_to_price[sec] = price

            # Build per-second series for EXACT last 10 minutes
            end_sec = int(end_utc.timestamp())
            start_sec = end_sec - 600
            times_utc = list(range(start_sec, end_sec + 1))
            prices = []
            last_price = None
            fallback_price = None
            for s in range(start_sec - 60, start_sec + 1):
                if s in sec_to_price:
                    fallback_price = sec_to_price[s]
            for s in times_utc:
                if s in sec_to_price:
                    last_price = sec_to_price[s]
                elif last_price is None:
                    last_price = fallback_price
                if last_price is None:
                    continue
                prices.append(last_price)

            if len(prices) < len(times_utc):
                times_utc = times_utc[-len(prices):]

            times_et = [datetime.fromtimestamp(s, tz=UTC).astimezone(ET) for s in times_utc]

            # 1s returns
            rets_1s = []
            for i in range(1, len(prices)):
                p0, p1 = prices[i-1], prices[i]
                rets_1s.append(math.log(p1 / p0) if p0 > 0 else 0.0)

            # Rolling 60s std, annualized
            sigmas = []
            for i in range(len(prices)):
                if i < ROLLING_SEC_FOR_VOL:
                    sigmas.append(sigmas[-1] if sigmas else SIGMA_MIN)
                else:
                    window = rets_1s[i-ROLLING_SEC_FOR_VOL:i]
                    s = ann_sigma_from_returns(window, SECS_PER_YEAR)
                    sigmas.append(min(SIGMA_MAX, max(SIGMA_MIN, s)) if s > 0 else SIGMA_MIN)

            # Replace chart histories
            self.price_time_hist.clear(); self.price_hist.clear()
            for ts, pv in zip(times_et, prices):
                self.price_time_hist.append(ts)
                self.price_hist.append(pv)

            self.time_hist.clear(); self.sigma_hist.clear()
            for ts, sv in zip(times_et, sigmas):
                self.time_hist.append(ts)
                self.sigma_hist.append(sv)
            self.last_sigma_plot_time = self.time_hist[-1] if self.time_hist else None  # keep throttle in sync

            # Redraw both charts
            self._redraw_price_chart(strike=self.last_strike)
            self._redraw_sigma_chart()

            self.var_status.set("Backfilled last 10 minutes (per-second trades; 60s rolling vol).")
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
                text=f"ET window (1h): {start_et.strftime('%Y-%m-%d %H:%M')} → {end_et.strftime('%H:%M')}  "
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

            # Live appends (price & σ)
            now_et = now.astimezone(ET)
            self._push_price_point_if_needed(now_et, px, force=bool(msg.get("price_changed", False)))
            self._redraw_price_chart(strike=strike)

            sigma_1m_current = msg["sigmas"]["1m"]
            self.last_sigma_1m = sigma_1m_current
            self._push_sigma_point(now_et, sigma_1m_current)  # throttled to ~1 Hz
            self._redraw_sigma_chart()

            # μ and table
            mu_ann = msg["mu_ann"]
            self.var_mu.set(f"{mu_ann:+.3f}")
            sigmas = msg["sigmas"]
            theos_neutral = msg["theos_neutral"]
            theos_mu      = msg["theos_mu"]
            for k in ["1m", "10m", "1h", "1d"]:
                self.rows[k]["sigma"].set(f"{sigmas[k]:.3f}")
                self.rows[k]["up_neut"].set(f"{theos_neutral[k]*100:5.2f}%")
                self.rows[k]["down_neut"].set(f"{(1.0-theos_neutral[k])*100:5.2f}%")
                self.rows[k]["up_mu"].set(f"{theos_mu[k]*100:5.2f}%")
                self.rows[k]["down_mu"].set(f"{(1.0-theos_mu[k])*100:5.2f}%")

            self.var_status.set("")

        elif evt == "closed":
            start_et = msg.get("start_et"); end_et = msg.get("end_et")
            if "final_close" in msg:
                fc = msg["final_close"]; strike = msg["strike"]; outcome = msg["outcome"]
                self.var_status.set(
                    f"Closed {start_et.strftime('%H:%M')}→{end_et.strftime('%H:%M')} ET. "
                    f"Strike {strike:,.2f} | Close {fc:,.2f} | Outcome {outcome}. Rolling…"
                )
            else:
                self.var_status.set("Window closed. Rolling to the new window…")

        elif evt == "status":
            self.var_status.set(msg.get("msg", ""))

    def on_quit(self):
        self.stop_event.set()
        self.destroy()

# ------------------- Run -------------------
if __name__ == "__main__":
    App().mainloop()
