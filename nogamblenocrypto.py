# polymarket_tk.py
import json
import re
import time
from datetime import datetime, timezone
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import requests
import html  # for unescaping titles

# ---- Timezone (ET) handling -------------------------------------------------
# On Windows you may need "pip install tzdata" for zoneinfo to have ET.
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    ET_NAME = "US/Eastern"
except Exception:
    ET = timezone.utc
    ET_NAME = "UTC (tzdata missing)"

# ---- Endpoints --------------------------------------------------------------
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec",
}

# ---- Helpers (API + formatting) --------------------------------------------
def extract_slug(event_url: str) -> str:
    return event_url.rstrip("/").split("/")[-1]

def get_event_by_slug(slug: str):
    # slug is array-typed in docs; pass as repeated param
    resp = requests.get(f"{GAMMA}/events", params=[("slug", slug)], timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None  # returns full event incl. markets

def get_user_trades_last_hour(
    user_address: str,
    event_url: Optional[str] = None,
    side: Optional[str] = None,
    limit: int = 500
) -> list[dict]:
    """
    Returns the user's trades from the last hour (most recent first).
    Uses: https://data-api.polymarket.com/activity
    """
    now = int(time.time())
    start = now - 60 * 60

    params = {
        "user": user_address,
        "start": start,
        "end": now,
        "limit": limit,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }

    if event_url:
        slug = extract_slug(event_url)
        ev = get_event_by_slug(slug)
        if ev and ev.get("id") is not None:
            params["eventId"] = ev["id"]
        else:
            raise RuntimeError(f"Could not resolve eventId for slug: {slug}")

    if side:
        s = side.upper()
        if s not in ("BUY", "SELL"):
            raise ValueError("side must be 'BUY' or 'SELL'")
        params["side"] = s

    r = requests.get(f"{DATA_API}/activity", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Keep only trades
    trades = [row for row in data if row.get("type") in (None, "TRADE", "trade")]
    return trades

def get_last_trade_time_any(user_address: str, event_url: Optional[str], all_time: bool) -> Optional[str]:
    """
    Returns the most recent trade time as a formatted string in ET.
    - all_time=False: looks only in the last hour
    - all_time=True: looks across all history (no time window)
    """
    if all_time:
        params = {
            "user": user_address,
            "limit": 1,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        if event_url:
            slug = extract_slug(event_url)
            ev = get_event_by_slug(slug)
            if ev and ev.get("id") is not None:
                params["eventId"] = ev["id"]
            else:
                raise RuntimeError(f"Could not resolve eventId for slug: {slug}")
        r = requests.get(f"{DATA_API}/activity", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        ts = data[0].get("timestamp") or data[0].get("time")
    else:
        trades = get_user_trades_last_hour(user_address, event_url=event_url, limit=1)
        if not trades:
            return None
        ts = trades[0].get("timestamp") or trades[0].get("time")

    if not ts:
        return None
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(ET)
    return dt.strftime(f"%Y-%m-%d %H:%M:%S {ET_NAME}")

def _first_nonempty(*vals) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _extract_market_title(row: dict) -> str:
    flat = _first_nonempty(
        row.get("marketQuestion"),
        row.get("market_question"),
        row.get("marketTitle"),
        row.get("market_title"),
        row.get("marketName"),
        row.get("market_name"),
        row.get("market"),
        row.get("question"),
        row.get("title"),
        row.get("questionTitle"),
    )
    if flat:
        return html.unescape(flat)

    m = row.get("market") or row.get("market_obj") or {}
    if isinstance(m, dict):
        nested = _first_nonempty(
            m.get("question"),
            m.get("title"),
            m.get("name"),
            m.get("marketQuestion"),
            m.get("marketTitle"),
        )
        if nested:
            return html.unescape(nested)

    alt = _first_nonempty(
        row.get("collectionName"),
        row.get("collection_name"),
        row.get("eventTitle"),
        row.get("event_title"),
    )
    return html.unescape(alt)

def _extract_outcome(row: dict) -> str:
    return _first_nonempty(
        row.get("outcome"),
        row.get("tokenName"),
        row.get("token_name"),
        row.get("outcomeName"),
        row.get("outcome_name"),
    )

def format_trade_row(row: dict) -> dict:
    ts = row.get("timestamp") or row.get("time") or 0
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(ET)

    market_title = _extract_market_title(row)
    outcome = _extract_outcome(row)

    return {
        "time_et": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "eventTitle": _first_nonempty(row.get("eventTitle"), row.get("event_title")),
        "market": market_title,
        "outcome": outcome,
        "side": row.get("side") or "",
        "price": row.get("price"),
        "size": row.get("size") or row.get("amount"),
        "cost": row.get("value") or row.get("cost"),
        "txHash": row.get("transactionHash") or row.get("txHash") or row.get("tx_hash") or "",
        "_raw": row,
    }

# ---- GUI --------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polymarket – Last-Hour Trades")
        self.geometry("900x680")

        # State
        self._auto_job = None

        # Inputs frame
        frm = ttk.LabelFrame(self, text="Inputs")
        frm.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm, text="User wallet address").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.addr_var = tk.StringVar(value="0x88712ac5d0f65592fcccb4708523c8fa6ee5830a")
        ttk.Entry(frm, textvariable=self.addr_var, width=52).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frm, text="Event URL (optional)").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.event_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.event_var, width=52).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frm, text="Side").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.side_var = tk.StringVar(value="All")
        ttk.Combobox(frm, textvariable=self.side_var, values=["All", "BUY", "SELL"], width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frm, text="Auto-refresh (sec)").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.refresh_var = tk.StringVar(value="30")
        ttk.Entry(frm, textvariable=self.refresh_var, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        self.fetch_btn = ttk.Button(frm, text="Fetch Now", command=self.fetch_and_render)
        self.fetch_btn.grid(row=0, column=4, padx=10, pady=5)

        self.toggle_auto_btn = ttk.Button(frm, text="Start Auto-Refresh", command=self.toggle_auto_refresh)
        self.toggle_auto_btn.grid(row=1, column=4, padx=10, pady=5)

        # Metrics frame
        mfrm = ttk.LabelFrame(self, text="Summary")
        mfrm.pack(fill="x", padx=10, pady=5)

        self.count_var = tk.StringVar(value="—")
        self.last_trade_hour_var = tk.StringVar(value="—")
        self.last_trade_all_var = tk.StringVar(value="—")
        self.filter_var = tk.StringVar(value="All")
        self.tz_var = tk.StringVar(value=f"Timestamps shown in {ET_NAME}")

        self._add_metric(mfrm, "Trades (last hour):", self.count_var, 0)
        self._add_metric(mfrm, "Most recent trade (last hour):", self.last_trade_hour_var, 1)
        self._add_metric(mfrm, "Most recent trade (all-time):", self.last_trade_all_var, 2)
        ttk.Label(mfrm, textvariable=self.tz_var).grid(row=0, column=6, padx=10, sticky="w")
        self._add_metric(mfrm, "Side filter:", self.filter_var, 3)

        # Table
        tfrm = ttk.Frame(self)
        tfrm.pack(fill="both", expand=True, padx=10, pady=10)

        # Removed: eventTitle, cost, txHash
        cols = ("time_et","market","outcome","side","price","size")
        self.tree = ttk.Treeview(tfrm, columns=cols, show="headings")
        headings = {
            "time_et":"Time (ET)",
            "market":"Market",
            "outcome":"Outcome",
            "side":"Side",
            "price":"Price",
            "size":"Size",
        }
        for c in cols:
            self.tree.heading(c, text=headings[c])
            # adjust widths since we have fewer columns now
            width = 200 if c == "market" else (140 if c in ("time_et","outcome") else 90)
            self.tree.column(c, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")

        vsb = ttk.Scrollbar(tfrm, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        # Tag styles for Up/Down highlighting (row-level coloring)
        self.tree.tag_configure("outcome_up", foreground="green")
        self.tree.tag_configure("outcome_down", foreground="red")

        # Raw JSON toggle
        self.raw_frame = ttk.LabelFrame(self, text="Raw API Payload")
        self.raw_text = tk.Text(self.raw_frame, height=10)
        self.raw_scroll = ttk.Scrollbar(self.raw_frame, orient="vertical", command=self.raw_text.yview)
        self.raw_text.configure(yscrollcommand=self.raw_scroll.set)

        # Buttons under table
        bfrm = ttk.Frame(self)
        bfrm.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(bfrm, text="Show Raw JSON", command=self.show_raw).pack(side="left")
        ttk.Button(bfrm, text="Hide Raw JSON", command=self.hide_raw).pack(side="left")

        # Initial fetch
        self.fetch_and_render()

    def _add_metric(self, parent, label, var, idx):
        col = idx * 2
        ttk.Label(parent, text=label).grid(row=0, column=col, padx=(10,2), pady=5, sticky="e")
        ttk.Label(parent, textvariable=var, font=("Segoe UI", 10, "bold")).grid(row=0, column=col+1, padx=(0,10), pady=5, sticky="w")

    # --- Actions -------------------------------------------------------------
    def fetch_and_render(self):
        addr = self.addr_var.get().strip()
        event_url = self.event_var.get().strip() or None
        side = self.side_var.get()
        side = None if side == "All" else side

        if not addr:
            messagebox.showerror("Input error", "Please enter a wallet address.")
            return

        try:
            # Latest times
            last_hour_time = get_last_trade_time_any(addr, event_url, all_time=False)
            all_time_time = get_last_trade_time_any(addr, event_url, all_time=True)
            self.last_trade_hour_var.set(last_hour_time or "No trades in last hour")
            self.last_trade_all_var.set(all_time_time or "No trades found")

            # Trades list (last hour)
            trades = get_user_trades_last_hour(addr, event_url=event_url, side=side)
            formatted = [format_trade_row(t) for t in trades]

            # Summary metrics
            self.count_var.set(str(len(formatted)))
            self.filter_var.set(side or "All")

            # Populate table
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            for row in formatted:
                # Build values matching the reduced columns
                values = (
                    row["time_et"],
                    row["market"],
                    row["outcome"],
                    row["side"],
                    row["price"],
                    row["size"],
                )
                # Tag for outcome highlighting (row-level)
                outcome_text = (row["outcome"] or "").strip().lower()
                tags = ()
                if outcome_text == "up":
                    tags = ("outcome_up",)
                elif outcome_text == "down":
                    tags = ("outcome_down",)
                self.tree.insert("", "end", values=values, tags=tags)

            # Stash raw payload
            self._last_raw = trades

        except requests.HTTPError as e:
            body = getattr(e, "response", None)
            body_text = getattr(body, "text", "") if body is not None else ""
            messagebox.showerror("HTTP error", f"{e}\n\n{body_text[:1000]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_raw(self):
        self.raw_frame.pack(fill="both", expand=False, padx=10, pady=(0,10))
        self.raw_text.pack(side="left", fill="both", expand=True, padx=(5,0), pady=5)
        self.raw_scroll.pack(side="right", fill="y", padx=(0,5), pady=5)
        self.raw_text.delete("1.0", "end")
        try:
            txt = json.dumps(getattr(self, "_last_raw", []), indent=2)
        except Exception:
            txt = "[]"
        self.raw_text.insert("1.0", txt)

    def hide_raw(self):
        self.raw_frame.forget()

    def toggle_auto_refresh(self):
        if self._auto_job is None:
            try:
                secs = int(self.refresh_var.get())
                if secs < 5:
                    raise ValueError
            except Exception:
                messagebox.showerror("Input error", "Auto-refresh seconds must be an integer ≥ 5.")
                return
            self.toggle_auto_btn.configure(text="Stop Auto-Refresh")
            self._schedule_auto(secs)
        else:
            self.after_cancel(self._auto_job)
            self._auto_job = None
            self.toggle_auto_btn.configure(text="Start Auto-Refresh")

    def _schedule_auto(self, secs: int):
        self.fetch_and_render()
        self._auto_job = self.after(secs * 1000, lambda: self._schedule_auto(secs))


if __name__ == "__main__":
    app = App()
    app.mainloop()
