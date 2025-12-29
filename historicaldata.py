# pm_hourly_btc_midpoints.py
import re
import csv
import json
import requests
from datetime import datetime, timedelta, timezone

# ---------------- CONFIG ----------------
N_DAYS = 5   # <-- how many days back you want
OUTFILE = "btc_hourly_midpoints.csv"

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
UTC = timezone.utc

# Recognize the hourly BTC slugs (e.g., bitcoin-up-or-down-september-21-1pm-et)
HOURLY_SLUG_RE = re.compile(r"^bitcoin-up-or-down-.*-(\d{1,2})(am|pm)-et$")

# ---------------- HELPERS ----------------
def get_events(start_dt: datetime, end_dt: datetime, limit=250, offset=0):
    """
    Fetch events whose event.startDate is in [start_dt, end_dt].
    """
    params = {
        "limit": limit,
        "offset": offset,
        "order": "startDate",
        "ascending": True,
        "start_date_min": start_dt.isoformat().replace("+00:00", "Z"),
        "start_date_max": end_dt.isoformat().replace("+00:00", "Z"),
    }
    r = requests.get(f"{GAMMA}/events", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

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
            except Exception:
                pass
        return [s]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]

def map_up_down_indices(outcomes):
    lowered = [o.lower() for o in outcomes]
    up_idx = lowered.index("up") if "up" in lowered else (0 if outcomes else None)
    down_idx = lowered.index("down") if "down" in lowered else (1 if len(outcomes) > 1 else None)
    if down_idx == up_idx and len(outcomes) >= 2:
        down_idx = 1 if up_idx == 0 else 0
    return up_idx, down_idx

def prices_history_minute(token_id: str, start_ts: int, end_ts: int):
    """
    Returns minute-fidelity midpoint history for a token in [start_ts, end_ts].
    """
    if end_ts <= start_ts:
        end_ts = start_ts + 60  # nudge to avoid empty/invalid range
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1}
    r = requests.get(f"{CLOB}/prices-history", params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    return payload.get("history", payload)

# ---------------- MAIN ----------------
def fetch_btc_hourly_midpoints(days=N_DAYS, outfile=OUTFILE):
    now = datetime.now(tz=UTC)
    start_window = now - timedelta(days=days)

    # Fetch events where the EVENT's startDate is within the window
    # (we'll still filter by each MARKET's endDate below)
    events = []
    offset = 0
    while True:
        batch = get_events(start_window, now, limit=250, offset=offset)
        if not batch:
            break
        events.extend(batch)
        if len(batch) < 250:
            break
        offset += 250

    # Keep only hourly-looking BTC events; exclude daily "on-..." and explicit 15m variants
    btc_hourly_events = []
    for e in events:
        slug = str(e.get("slug", ""))
        if HOURLY_SLUG_RE.match(slug) and "15m" not in slug and "-on-" not in slug:
            btc_hourly_events.append(e)

    rows = []
    scanned_events = 0
    considered_markets = 0
    kept_markets = 0

    for ev in btc_hourly_events:
        scanned_events += 1
        slug = ev.get("slug")

        # fetch full event (to get all markets)
        try:
            ev_full = requests.get(f"{GAMMA}/events/slug/{slug}", timeout=30)
            if ev_full.status_code == 404:
                continue
            ev_full.raise_for_status()
        except Exception:
            continue

        ev_data = ev_full.json()
        markets = ev_data.get("markets") or []
        for market in markets:
            if not market.get("enableOrderBook"):
                continue

            # parse market window
            try:
                sdt_raw = market.get("startDate")
                edt_raw = market.get("endDate")
                if not sdt_raw or not edt_raw:
                    continue
                sdt = datetime.fromisoformat(sdt_raw.replace("Z", "+00:00"))
                edt = datetime.fromisoformat(edt_raw.replace("Z", "+00:00"))
            except Exception:
                continue

            # we only want markets that have ended in the past N days
            if not (start_window <= edt <= now):
                continue

            considered_markets += 1

            # UP-only
            outcomes = normalize_array(market.get("shortOutcomes") or market.get("outcomes"))
            token_ids = normalize_array(market.get("clobTokenIds"))
            up_idx, _ = map_up_down_indices(outcomes)
            if up_idx is None or up_idx >= len(token_ids):
                continue

            tok = token_ids[up_idx]

            # ---- Only the final hour before the strike (strike = market end) ----
            strike_ts = int(edt.timestamp())
            window_start_ts = strike_ts - 3600  # last hour only

            try:
                hist = prices_history_minute(tok, window_start_ts, strike_ts)  # midpoint series in final hour
                if not isinstance(hist, list) or not hist:
                    continue
                kept_markets += 1
                strike_iso = edt.strftime("%Y-%m-%d %H:%M:%S")
                for pt in hist:
                    if "t" not in pt or "p" not in pt:
                        continue
                    ts = int(pt["t"])
                    iso_str = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
                    rows.append([slug, "UP", tok, strike_ts, strike_iso, ts, iso_str, float(pt["p"])])
                print(f"{slug} | UP | strike={strike_iso} | rows={len(hist)}")
            except Exception as e:
                print(f"{slug} UP error: {e}")

    # Sort by strike time, then by row time
    rows.sort(key=lambda r: (r[3], r[5]))  # (strike_ts, timestamp)

    # Write CSV: one row per minute point in the last hour for each market
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_slug",
            "leg",
            "token_id",
            "strike_ts",
            "strike_iso_utc",
            "timestamp",
            "iso_time_utc",
            "midpoint"
        ])
        w.writerows(rows)

    print(f"\nScanned hourly events: {scanned_events}")
    print(f"Considered markets (ended within {days}d): {considered_markets}")
    print(f"Markets with data (final hour captured): {kept_markets}")
    print(f"Wrote {len(rows)} rows to {outfile}")

if __name__ == "__main__":
    fetch_btc_hourly_midpoints()
