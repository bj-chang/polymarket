#!/usr/bin/env python3
import json
import requests
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from zoneinfo import ZoneInfo
import threading

# ======================== API CONSTANTS ========================
GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

DEPTH = 3  # number of levels to show for asks and bids

# ======================== UTILITIES ============================
def current_et_slug():
    now = datetime.now(ZoneInfo("America/New_York"))
    month = MONTHS[now.month - 1]
    day = now.day
    hour12 = (now.hour % 12) or 12
    ampm = "am" if now.hour < 12 else "pm"
    return f"bitcoin-up-or-down-{month}-{day}-{hour12}{ampm}-et"

def normalize_array(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        return [s]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]

def get_event_by_slug(slug):
    r = requests.get(f"{GAMMA}/events/slug/{slug}", timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def get_book_single(token_id):
    r = requests.get(
        f"{CLOB}/book",
        params={"token_id": str(token_id)},
        headers={"User-Agent": "pm-orderbook/1.0"},
        timeout=20
    )
    r.raise_for_status()
    return r.json()

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _level_size(entry):
    for k in ("size", "amount", "quantity", "qty", "remaining", "baseSize"):
        if k in entry:
            return _safe_float(entry.get(k))
    return None

def levels_from_book(book, depth=3):
    raw_bids = book.get("bids") or []
    raw_asks = book.get("asks") or []

    def pr(e): return _safe_float(e.get("price"))
    def vol(e): return _level_size(e)

    bids_sorted = sorted([e for e in raw_bids if pr(e) is not None], key=lambda e: pr(e), reverse=True)
    asks_sorted = sorted([e for e in raw_asks if pr(e) is not None], key=lambda e: pr(e))

    bids = [(pr(e), vol(e)) for e in bids_sorted[:depth]]
    asks = [(pr(e), vol(e)) for e in asks_sorted[:depth]]
    return bids, asks

def best_bid_ask_spread_mid(book):
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    def price(x):
        try:
            return float(x.get("price"))
        except Exception:
            return None

    bb = max((b for b in bids if price(b) is not None), key=price, default=None)
    ba = min((a for a in asks if price(a) is not None), key=price, default=None)

    best_bid = price(bb) if bb else None
    best_ask = price(ba) if ba else None

    spread = book.get("spread")
    if spread is None and best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid

    mid = book.get("mid")
    if mid is None and best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2

    return best_bid, best_ask, spread, mid

def fmt(x, ndp=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndp}f}"
    except Exception:
        return str(x)

def fmt_vol(x, ndp=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndp}f}"
    except Exception:
        return str(x)

# ======================== GUI APP =============================
class OrderBookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polymarket BTC Hourly — Clean Ladder View")
        self.geometry("820x620")
        self.minsize(780, 580)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#101418")

        # Palette
        self.light_bg = "#f4f6f8"
        self.dark_bg  = "#101418"
        self.card_bg_light = "#ffffff"
        self.card_bg_dark  = "#151a1f"
        self.text_light = "#0b0f14"
        self.text_dark  = "#e6e9ed"
        self.accent     = "#4e9cff"

        # Row colors
        self.ask_bg = "#b8d1ff"
        self.bid_bg = "#ffd8a0"
        self.spread_bg = "#e9e9e9"

        # Header
        self.header = ttk.Frame(self, padding=(12, 10, 12, 6))
        self.header.pack(fill="x")

        self.slug_var = tk.StringVar(value="Slug: …")
        self.question_var = tk.StringVar(value="Market: …")
        self.updated_var = tk.StringVar(value="")

        ttk.Label(self.header, textvariable=self.slug_var, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self.header, textvariable=self.question_var, font=("Segoe UI", 11)).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # Controls
        self.controls = ttk.Frame(self.header)
        self.controls.grid(row=0, column=3, sticky="e")
        ttk.Label(self.controls, text="Refresh (s):").grid(row=0, column=0, padx=(0,6))
        self.refresh_entry = ttk.Entry(self.controls, width=5)
        self.refresh_entry.insert(0, "1")
        self.refresh_entry.grid(row=0, column=1)

        self.theme_mode = tk.StringVar(value="dark")
        ttk.Checkbutton(
            self.controls, text="Light theme", command=self.toggle_theme,
            variable=self.theme_mode, onvalue="light", offvalue="dark"
        ).grid(row=0, column=2, padx=(10,0))

        # Main layout
        self.main = ttk.Frame(self, padding=12)
        self.main.pack(fill="both", expand=True, padx=12, pady=8)

        self.left_card  = self._make_outcome_card(self.main, title="UP")
        self.separator  = ttk.Separator(self.main, orient="vertical")
        self.right_card = self._make_outcome_card(self.main, title="DOWN")

        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0,0))
        self.separator.grid(row=0, column=1, sticky="ns", padx=8)
        self.right_card.grid(row=0, column=2, sticky="nsew", padx=(0,0))

        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(2, weight=1)

        # Status bar
        self.status = ttk.Frame(self, padding=(12, 6, 12, 12))
        self.status.pack(fill="x")
        ttk.Label(self.status, textvariable=self.updated_var, font=("Segoe UI", 9)).pack(side="left")

        self._refresh_lock = threading.Lock()
        self._fetch_in_flight = False

        self.apply_theme()
        self.after(100, self.refresh_loop)

    # ---------- outcome card ----------
    def _make_outcome_card(self, parent, title="BOOK"):
        card = ttk.Frame(parent, padding=12, style="Card.TFrame")

        hdr = ttk.Frame(card, padding=(0,0,0,8), style="Card.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text=title, font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Label(hdr, text="Asks (high→low) • Spread • Bids (high→low)", font=("Segoe UI", 9)).pack(side="right")

        container = ttk.Frame(card)
        container.pack(fill="both", expand=True)

        # --- Price Tree ---
        price_tree = ttk.Treeview(container, columns=("Price",), show="headings", height=DEPTH*2 + 1)
        price_tree.heading("Price", text="Price")

        # --- Divider ---
        divider = tk.Frame(container, width=1, bg="#aaaaaa")

        # --- Volume Tree ---
        volume_tree = ttk.Treeview(container, columns=("Volume",), show="headings", height=DEPTH*2 + 1)
        volume_tree.heading("Volume", text="Volume")

        # Configure columns (compact + no padding)
        price_tree.column("Price", anchor="center", width=160, stretch=False, minwidth=0)
        volume_tree.column("Volume", anchor="center", width=160, stretch=False, minwidth=0)
        price_tree.configure(padding=0)
        volume_tree.configure(padding=0)

        # Grid layout
        price_tree.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        divider.grid(row=0, column=1, sticky="ns", padx=0, pady=0)
        volume_tree.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)

        container.columnconfigure(0, weight=1)
        container.columnconfigure(2, weight=1)

        # Tag styles
        for tree in (price_tree, volume_tree):
            tree.tag_configure("askrow", background=self.ask_bg, foreground="#0b0f14", font=("Segoe UI", 14, "bold"))
            tree.tag_configure("bidrow", background=self.bid_bg, foreground="#0b0f14", font=("Segoe UI", 14, "bold"))
            tree.tag_configure("spreadrow", background=self.spread_bg, foreground="#000000", font=("Segoe UI", 14, "bold"))

        card.price_tree = price_tree
        card.volume_tree = volume_tree
        return card

    # ---------- theming ----------
    def apply_theme(self):
        dark = self.theme_mode.get() == "dark"
        bg  = self.dark_bg if dark else self.light_bg
        fg  = self.text_dark if dark else self.text_light
        card_bg = self.card_bg_dark if dark else self.card_bg_light

        self.configure(bg=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("Card.TFrame", background=card_bg)
        self.style.configure("TLabel", background=bg, foreground=fg)

        # Add subtle horizontal separators
        self.style.configure(
            "Treeview",
            background=card_bg,
            fieldbackground=card_bg,
            foreground=fg,
            font=("Segoe UI", 14),
            rowheight=38,
            borderwidth=1,
            relief="solid",
            highlightthickness=0
        )
        self.style.map("Treeview", background=[("selected", "#4e9cff")], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", background=card_bg, foreground=fg, font=("Segoe UI", 12, "bold"), padding=[0,0,0,0])

    def toggle_theme(self):
        self.apply_theme()

    # ---------- data refresh ----------
    def refresh_loop(self):
        if not self._fetch_in_flight:
            threading.Thread(target=self._load_and_render_bg, daemon=True).start()

        try:
            secs = max(1, int(self.refresh_entry.get()))
        except ValueError:
            secs = 1
            self.refresh_entry.delete(0, tk.END)
            self.refresh_entry.insert(0, "1")
        self.after(secs * 1000, self.refresh_loop)

    def _load_and_render_bg(self):
        with self._refresh_lock:
            self._fetch_in_flight = True
            try:
                slug = current_et_slug()
                ev = get_event_by_slug(slug)
                if not ev:
                    self._post_header(slug, "Event not found for this hour.")
                    self._clear_books()
                    self._post_timestamp("No event")
                    return

                market = next((m for m in (ev.get("markets") or []) if m.get("enableOrderBook")), None)
                if not market:
                    self._post_header(slug, "No CLOB-enabled market on this event.")
                    self._clear_books()
                    self._post_timestamp("No market")
                    return

                question = market.get("question") or "—"
                outcomes = normalize_array(market.get("shortOutcomes") or market.get("outcomes"))
                token_ids = normalize_array(market.get("clobTokenIds"))

                up_idx = down_idx = None
                lowered = [o.lower() for o in outcomes]
                if "up" in lowered:   up_idx = lowered.index("up")
                if "down" in lowered: down_idx = lowered.index("down")
                if up_idx is None and len(outcomes) >= 1:
                    up_idx = 0
                if down_idx is None and len(outcomes) >= 2:
                    down_idx = 1 if up_idx != 1 else 0

                def fetch_levels(idx):
                    if idx is None or idx >= len(token_ids): return None
                    tid = token_ids[idx]
                    try:
                        book = get_book_single(tid)
                        bids, asks = levels_from_book(book, depth=DEPTH)
                        bb, ba, spread, _ = best_bid_ask_spread_mid(book)
                        while len(asks) < DEPTH: asks.append((None, None))
                        while len(bids) < DEPTH: bids.append((None, None))
                        return (bids, asks, spread)
                    except Exception:
                        return None

                up_data = fetch_levels(up_idx)
                down_data = fetch_levels(down_idx)

                self.after(0, lambda: self._apply_update(slug, question, up_data, down_data))
            except Exception as e:
                self.after(0, lambda: self.question_var.set(f"Error: {e}"))
            finally:
                self._fetch_in_flight = False

    def _apply_update(self, slug, question, up_data, down_data):
        self.slug_var.set(f"Slug: {slug}")
        self.question_var.set(question)
        self._render_single_column(self.left_card, up_data)
        self._render_single_column(self.right_card, down_data)
        self._post_timestamp("Updated")

    def _render_single_column(self, card, data_tuple):
        card.price_tree.delete(*card.price_tree.get_children())
        card.volume_tree.delete(*card.volume_tree.get_children())

        if not data_tuple:
            for _ in range(DEPTH * 2 + 1):
                card.price_tree.insert("", "end", values=("—",))
                card.volume_tree.insert("", "end", values=("—",))
            return

        bids, asks, spread = data_tuple

        # Asks (reversed)
        for px, vol in reversed(asks):
            card.price_tree.insert("", "end", values=(fmt(px),), tags=("askrow",))
            card.volume_tree.insert("", "end", values=(fmt_vol(vol),), tags=("askrow",))

        # Spread row
        card.price_tree.insert("", "end", values=(f"Spread: {fmt(spread)}",), tags=("spreadrow",))
        card.volume_tree.insert("", "end", values=(" ",), tags=("spreadrow",))

        # Bids
        for px, vol in bids:
            card.price_tree.insert("", "end", values=(fmt(px),), tags=("bidrow",))
            card.volume_tree.insert("", "end", values=(fmt_vol(vol),), tags=("bidrow",))

    def _clear_books(self):
        for card in (self.left_card, self.right_card):
            card.price_tree.delete(*card.price_tree.get_children())
            card.volume_tree.delete(*card.volume_tree.get_children())

    def _post_header(self, slug, msg):
        self.after(0, lambda: (self.slug_var.set(f"Slug: {slug}"), self.question_var.set(msg)))

    def _post_timestamp(self, prefix):
        self.after(0, lambda: self.updated_var.set(self._timestamp_et(prefix)))

    @staticmethod
    def _timestamp_et(prefix):
        ts = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M:%S %p ET")
        return f"{prefix}: {ts}"

# ======================== RUN ================================
if __name__ == "__main__":
    app = OrderBookApp()
    app.mainloop()
