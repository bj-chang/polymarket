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
UTC = timezone.utc

# Timeframe Lookbacks
LOOKBACK_1M_MIN     = 180
LOOKBACK_10M_MIN    = 600
LOOKBACK_1H_HRS     = 336
LOOKBACK_1D_DAYS    = 60

# --- CALIBRATION: 15-Minute Rolling Window ---
ROLLING_SEC_FOR_VOL = 900  # 15 minutes (900s) instead of 60s

# --- CALIBRATION: Mu (Drift) Dampening ---
MU_EWMA_LAMBDA      = 0.98 # Very high smoothing to prevent jumpy probabilities
MU_MAX_ABS          = 0.15 # Cap drift at 15% annualized (realistic for 1h window)

REFRESH_PRICE_SEC    = 1
R_NEUTRAL            = 0.0
SIGMA_MIN            = 0.05
SIGMA_MAX            = 3.00

SECS_PER_YEAR  = 365 * 24 * 3600.0
MINS_PER_YEAR  = 365 * 24 * 60.0
TENM_PER_YEAR  = MINS_PER_YEAR / 10.0
HRS_PER_YEAR   = 365 * 24
DAYS_PER_YEAR  = 365.0

UI_REFRESH_MS  = 250
MAX_1M_CACHE   = max(LOOKBACK_1M_MIN + 12, 200)

# ------------------- Data Helpers -------------------
def http_get(path: str, params: dict | None = None):
    url = BINANCE_API + path
    if params: url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "python-urllib"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_last_trade_price() -> float:
    return float(http_get("/api/v3/ticker/price", {"symbol": SYMBOL})["price"])

def fetch_1h_closes(n_hours: int) -> list[float]:
    k = http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1h", "limit": n_hours + 1})
    return [float(row[4]) for row in k]

def fetch_1d_closes(n_days: int) -> list[float]:
    k = http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1d", "limit": n_days + 1})
    return [float(row[4]) for row in k]

def fetch_agg_trades_range(start_ms: int, end_ms: int):
    trades = []
    cursor_ms = start_ms
    while True:
        batch = http_get("/api/v3/aggTrades", {"symbol": SYMBOL, "startTime": cursor_ms, "endTime": end_ms, "limit": 1000})
        if not batch: break
        trades.extend(batch)
        cursor_ms = int(batch[-1]["T"]) + 1
        if cursor_ms >= end_ms or len(batch) < 500: break
    return trades

def get_kline_open_close_hour(start_et: datetime, end_et: datetime):
    start_ms = int(start_et.astimezone(UTC).timestamp() * 1000)
    end_ms   = int(end_et.astimezone(UTC).timestamp()   * 1000)
    k = http_get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1h", "startTime": start_ms, "endTime": end_ms, "limit": 1})
    return (float(k[0][1]), float(k[0][4])) if k else (None, None)

# ------------------- Math -------------------
def log_returns_from_prices(prices, step=1):
    if len(prices) <= step: return []
    return [math.log(prices[i] / prices[i-step]) for i in range(step, len(prices)) if prices[i-step] > 0]

def ann_sigma_from_returns(returns, units_per_year):
    if not returns or len(returns) < 2: return SIGMA_MIN
    mean = sum(returns) / len(returns)
    var  = sum((r - mean)**2 for r in returns) / (len(returns) - 1)
    return min(SIGMA_MAX, max(SIGMA_MIN, math.sqrt(var) * math.sqrt(units_per_year)))

def fair_up_prob(S0, St, tau_yrs, sigma, r_ann):
    if tau_yrs <= 0: return 1.0 if St >= S0 else 0.0
    s = max(1e-12, sigma)
    # Drift adjusted probability
    z = (math.log(S0 / St) - (r_ann - 0.5 * s**2) * tau_yrs) / (s * math.sqrt(tau_yrs))
    return 1.0 - (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

# ------------------- Worker -------------------
def worker(out_q, stop_event):
    now = datetime.now(tz=UTC)
    start_et = now.astimezone(ET).replace(minute=0, second=0, microsecond=0)
    end_et = start_et + timedelta(hours=1)
    start_utc, end_utc = start_et.astimezone(UTC), end_et.astimezone(UTC)

    strike = None
    while not stop_event.is_set() and strike is None:
        res = get_kline_open_close_hour(start_et, end_et)
        if res[0]: strike = res[0]
        time.sleep(1)

    live_1s_prices = deque(maxlen=ROLLING_SEC_FOR_VOL + 1)
    last_tick_t = 0.0
    mu_ema = 0.0

    while not stop_event.is_set():
        tnow = time.time()
        now = datetime.now(tz=UTC)
        
        if now >= end_utc: # Rollover
            start_et = now.astimezone(ET).replace(minute=0, second=0, microsecond=0)
            end_et = start_et + timedelta(hours=1)
            start_utc, end_utc = start_et.astimezone(UTC), end_et.astimezone(UTC)
            res = get_kline_open_close_hour(start_et, end_et)
            if res[0]: strike = res[0]

        if tnow - last_tick_t >= REFRESH_PRICE_SEC:
            try:
                px = get_last_trade_price()
                if live_1s_prices:
                    # Update Mu (Drift) with EMA
                    instant_ret = math.log(px / live_1s_prices[-1])
                    mu_ema = (MU_EWMA_LAMBDA * mu_ema) + ((1 - MU_EWMA_LAMBDA) * instant_ret)
                live_1s_prices.append(px)
                
                # Annualized Mu (cap it for stability)
                mu_ann = max(-MU_MAX_ABS, min(MU_MAX_ABS, mu_ema * SECS_PER_YEAR))
                
                # Volatility Calculation (The shared 15m window)
                chart_sigma = ann_sigma_from_returns(log_returns_from_prices(list(live_1s_prices), 1), SECS_PER_YEAR)
                
                # Table Sigmas (Traditional windows)
                sigmas = {
                    "15m": chart_sigma,
                    "1h": ann_sigma_from_returns(log_returns_from_prices(fetch_1h_closes(LOOKBACK_1H_HRS), 1), HRS_PER_YEAR),
                    "1d": ann_sigma_from_returns(log_returns_from_prices(fetch_1d_closes(LOOKBACK_1D_DAYS), 1), DAYS_PER_YEAR)
                }

                tau_yrs = max(0.0, (end_utc - now).total_seconds()) / SECS_PER_YEAR
                theos_n = {k: fair_up_prob(strike, px, tau_yrs, sigmas[k], R_NEUTRAL) for k in sigmas}
                theos_m = {k: fair_up_prob(strike, px, tau_yrs, sigmas[k], mu_ann) for k in sigmas}

                out_q.put({
                    "event": "tick", "now": now, "start_et": start_et, "end_et": end_et,
                    "strike": strike, "price": px, "diff": px - strike, 
                    "sigmas": sigmas, "chart_sigma": chart_sigma,
                    "theos_neutral": theos_n, "theos_mu": theos_m, "mu_ann": mu_ann,
                    "tau_sec": (end_utc - now).total_seconds(), "frac_elapsed": (now - start_utc).total_seconds() / 3600.0
                })
                last_tick_t = tnow
            except: pass
        time.sleep(0.1)

# ------------------- GUI -------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTCUSDT 1h Strategy Dashboard")
        self.geometry("900x980")
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=worker, args=(self.q, self.stop_event), daemon=True)
        self.worker_thread.start()
        self._setup_ui()
        self._init_charts()
        self.after(UI_REFRESH_MS, self.poll_queue)

    def _setup_ui(self):
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Value.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Head.TLabel", font=("Segoe UI", 8, "bold"))
        
        f = ttk.Frame(self, padding=10); f.grid(row=0, column=0, sticky="ew")
        self.lbl_window = ttk.Label(f, text="Window: ...", style="Title.TLabel")
        self.lbl_window.grid(row=0, column=0, sticky="w")
        
        s = ttk.Frame(self, padding=10); s.grid(row=1, column=0, sticky="ew")
        self.var_strike = tk.StringVar(value="…"); self.var_price = tk.StringVar(value="…")
        self.var_diff = tk.StringVar(value="…"); self.var_tclose = tk.StringVar(value="…")
        
        ttk.Label(s, text="STRIKE").grid(row=0, column=0); ttk.Label(s, textvariable=self.var_strike, style="Value.TLabel").grid(row=1, column=0, padx=10)
        ttk.Label(s, text="PRICE").grid(row=0, column=1); ttk.Label(s, textvariable=self.var_price, style="Value.TLabel").grid(row=1, column=1, padx=10)
        ttk.Label(s, text="DELTA").grid(row=0, column=2); self.lbl_diff = ttk.Label(s, textvariable=self.var_diff, style="Value.TLabel")
        self.lbl_diff.grid(row=1, column=2, padx=10)
        ttk.Label(s, text="TIME").grid(row=0, column=3); ttk.Label(s, textvariable=self.var_tclose, style="Value.TLabel").grid(row=1, column=3, padx=10)
        
        ttk.Button(s, text="Backfill (10m - 15m Window)", command=self.on_backfill_10m_persec).grid(row=1, column=4, padx=20)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=860, mode="determinate", maximum=1000)
        self.progress.grid(row=2, column=0, pady=5)

        # TABLE: Restored DOWN% Columns
        self.tbl_frame = ttk.Frame(self, padding=10); self.tbl_frame.grid(row=5, column=0, sticky="ew")
        headers = ["Source", "σ (Ann.)", "UP%(Neut)", "DWN%(Neut)", "UP%(Mu)", "DWN%(Mu)"]
        for c, h in enumerate(headers):
            ttk.Label(self.tbl_frame, text=h, style="Head.TLabel").grid(row=0, column=c, padx=10, sticky="e" if c>0 else "w")

        self.rows = {"15m": "15m Rolling", "1h": "1h History", "1d": "Daily History"}
        self.ui_rows = {}
        for i, (k, label) in enumerate(self.rows.items(), 1):
            ttk.Label(self.tbl_frame, text=label).grid(row=i, column=0, sticky="w")
            self.ui_rows[k] = {
                "sigma": tk.StringVar(value="…"), "up_n": tk.StringVar(value="…"), "dn_n": tk.StringVar(value="…"),
                "up_m": tk.StringVar(value="…"), "dn_m": tk.StringVar(value="…")
            }
            ttk.Label(self.tbl_frame, textvariable=self.ui_rows[k]["sigma"]).grid(row=i, column=1, sticky="e")
            ttk.Label(self.tbl_frame, textvariable=self.ui_rows[k]["up_n"], foreground="green").grid(row=i, column=2, sticky="e")
            ttk.Label(self.tbl_frame, textvariable=self.ui_rows[k]["dn_n"], foreground="red").grid(row=i, column=3, sticky="e")
            ttk.Label(self.tbl_frame, textvariable=self.ui_rows[k]["up_m"], foreground="blue").grid(row=i, column=4, sticky="e")
            ttk.Label(self.tbl_frame, textvariable=self.ui_rows[k]["dn_m"], foreground="purple").grid(row=i, column=5, sticky="e")

        self.var_status = tk.StringVar(value="System Active"); ttk.Label(self, textvariable=self.var_status).grid(row=6, column=0)

    def _init_charts(self):
        self.price_time_hist = deque(maxlen=600); self.price_hist = deque(maxlen=600)
        self.sigma_hist = deque(maxlen=600); self.time_hist = deque(maxlen=600)
        self.fig_px = Figure(figsize=(8.5, 2.5), dpi=100); self.ax_px = self.fig_px.add_subplot(111)
        self.canvas_px = FigureCanvasTkAgg(self.fig_px, master=self); self.canvas_px.get_tk_widget().grid(row=3, column=0)
        self.fig_sig = Figure(figsize=(8.5, 2.1), dpi=100); self.ax_sig = self.fig_sig.add_subplot(111)
        self.canvas_sig = FigureCanvasTkAgg(self.fig_sig, master=self); self.canvas_sig.get_tk_widget().grid(row=4, column=0)

    def on_backfill_10m_persec(self):
        self.var_status.set("Backfilling (using 15m Sigma)...")
        def run():
            try:
                end_ms = int(time.time() * 1000)
                start_ms = end_ms - (26 * 60 * 1000) # Fetch 26m to compute 15m vol for a 10m chart
                trades = fetch_agg_trades_range(start_ms, end_ms)
                sec_map = {int(t['T']//1000): float(t['p']) for t in trades}
                t_end = int(end_ms // 1000); t_start = t_end - 600
                prices, times, vols = [], [], []
                last_p = next(iter(sec_map.values()))
                
                # Reconstruct second-by-second history
                all_prices = []
                for s in range(t_end - 1500, t_end + 1):
                    last_p = sec_map.get(s, last_p); all_prices.append(last_p)

                # Compute 15m Vol (900s) for the last 10m (600s)
                for i in range(len(all_prices) - 600, len(all_prices)):
                    window = all_prices[i-900:i]
                    rets = [math.log(window[j]/window[j-1]) for j in range(1, len(window))]
                    vols.append(ann_sigma_from_returns(rets, SECS_PER_YEAR))
                    prices.append(all_prices[i])
                    times.append(datetime.fromtimestamp(t_end - (len(all_prices)-1-i), tz=UTC).astimezone(ET))

                self.price_time_hist.clear(); self.price_hist.clear()
                self.price_time_hist.extend(times); self.price_hist.extend(prices)
                self.time_hist.clear(); self.sigma_hist.clear()
                self.time_hist.extend(times); self.sigma_hist.extend(vols)
                self.var_status.set("Backfill synchronized.")
            except: self.var_status.set("Backfill Error.")
        threading.Thread(target=run, daemon=True).start()

    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg["event"] == "tick":
                    self.lbl_window.config(text=f"ET Window: {msg['start_et'].strftime('%H:%M')} -> {msg['end_et'].strftime('%H:%M')}")
                    self.var_strike.set(f"{msg['strike']:,.2f}"); self.var_price.set(f"{msg['price']:,.2f}")
                    self.var_diff.set(f"{msg['diff']:+,.2f}"); self.lbl_diff.config(foreground="green" if msg['diff'] >= 0 else "red")
                    self.var_tclose.set(time.strftime('%H:%M:%S', time.gmtime(msg['tau_sec'])))
                    self.progress["value"] = int(msg['frac_elapsed'] * 1000)
                    for k in self.ui_rows:
                        u, m = msg['theos_neutral'][k], msg['theos_mu'][k]
                        self.ui_rows[k]["sigma"].set(f"{msg['sigmas'][k]:.3f}")
                        self.ui_rows[k]["up_n"].set(f"{u*100:.1f}%"); self.ui_rows[k]["dn_n"].set(f"{(1-u)*100:.1f}%")
                        self.ui_rows[k]["up_m"].set(f"{m*100:.1f}%"); self.ui_rows[k]["dn_m"].set(f"{(1-m)*100:.1f}%")
                    now_et = msg['now'].astimezone(ET)
                    self.price_time_hist.append(now_et); self.price_hist.append(msg['price'])
                    self.time_hist.append(now_et); self.sigma_hist.append(msg['chart_sigma'])
                    self._draw_charts(msg['strike'])
        except queue.Empty: pass
        self.after(UI_REFRESH_MS, self.poll_queue)

    def _draw_charts(self, strike):
        self.ax_px.clear(); self.ax_sig.clear()
        self.ax_px.plot(list(self.price_time_hist), [p - strike for p in self.price_hist], color='#0055ff')
        self.ax_px.axhline(0, color='red', lw=1, ls='--')
        self.ax_sig.plot(list(self.time_hist), list(self.sigma_hist), color='orange')
        self.canvas_px.draw_idle(); self.canvas_sig.draw_idle()

    def on_quit(self): self.stop_event.set(); self.destroy()

if __name__ == "__main__": App().mainloop()