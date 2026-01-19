import os
import json
import math
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo
import pandas as pd

from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

# =========================
# ENV + CONFIG (no argparse)
# =========================
load_dotenv()

def env_int(k: str, default: int) -> int:
    try:
        return int(str(os.getenv(k, default)).strip())
    except Exception:
        return default

def env_float(k: str, default: float) -> float:
    try:
        return float(str(os.getenv(k, default)).strip())
    except Exception:
        return default

def env_bool(k: str, default: bool) -> bool:
    v = str(os.getenv(k, "1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

API_KEY = os.getenv("KITE_API_KEY", "").strip()
API_SECRET = os.getenv("KITE_API_SECRET", "").strip()
TOKEN_CACHE_PATH = os.getenv("KITE_TOKEN_PATH", "./kite_access_token.json").strip()

TICKER_CSV_PATH = os.getenv("TICKER_CSV_PATH", "./data/ticker_list.csv").strip()

STRIKES_EACH_SIDE = env_int("STRIKES_EACH_SIDE", 20)
WS_MODE = os.getenv("WS_MODE", "full").strip()
REFRESH_SECONDS = env_int("REFRESH_SECONDS", 5)

RISK_FREE_RATE = env_float("RISK_FREE_RATE", 0.0)
MIN_OPTION_PRICE = env_float("MIN_OPTION_PRICE", 0.05)

EXPIRY_HOUR_IST = env_int("EXPIRY_HOUR_IST", 15)
EXPIRY_MINUTE_IST = env_int("EXPIRY_MINUTE_IST", 30)

CACHE_NFO_CSV = os.getenv("CACHE_NFO_CSV", "./cache_instruments_NFO.csv").strip()
CACHE_VALIDITY_HOURS = env_int("CACHE_VALIDITY_HOURS", 12)

REST_QUOTE_CHUNK = env_int("REST_QUOTE_CHUNK", 200)
MAX_TOKENS_PER_WS = env_int("MAX_TOKENS_PER_WS", 3000)
MAX_WS_CONNECTIONS = env_int("MAX_WS_CONNECTIONS", 3)

WRITE_CHAIN_CSVS = env_bool("WRITE_CHAIN_CSVS", True)
OUT_CHAIN_DIR = os.getenv("OUT_CHAIN_DIR", "./out_chains").strip()
OUT_TRADES_DIR = os.getenv("OUT_TRADES_DIR", "./out_trades").strip()

COMBINED_LONG_CSV = os.path.join(OUT_CHAIN_DIR, "all_underlyings_chain_long.csv")
COMBINED_WIDE_CSV = os.path.join(OUT_CHAIN_DIR, "all_underlyings_chain_wide.csv")

PRINT_SAMPLE_CHAIN_ONCE = env_bool("PRINT_SAMPLE_CHAIN_ONCE", True)
SAMPLE_STRIKES_AROUND_ATM = env_int("SAMPLE_STRIKES_AROUND_ATM", 12)

# Spread logic
CLUSTER_WINDOW = env_int("CLUSTER_WINDOW", 3)
MIN_OI = env_int("MIN_OI", 1)
OFFSET_STEPS = env_int("OFFSET_STEPS", 2)
WIDTH_STEPS = env_int("WIDTH_STEPS", 1)
ACCEPT_MINUTES = env_int("ACCEPT_MINUTES", 5)
BAND_STEPS = env_int("BAND_STEPS", 1)

# Scheduling IST
START_TRACK_TIME = (env_int("START_TRACK_HH", 10), env_int("START_TRACK_MM", 0))
EOD_EXIT_TIME = (env_int("EOD_EXIT_HH", 15), env_int("EOD_EXIT_MM", 25))

LEDGER_CSV = os.path.join(OUT_TRADES_DIR, "trade_ledger.csv")
PNL_CSV = os.path.join(OUT_TRADES_DIR, "pnl_timeseries.csv")
STATE_JSON = os.path.join(OUT_TRADES_DIR, "state.json")

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing KITE_API_KEY / KITE_API_SECRET in .env")

TZ = ZoneInfo("Asia/Kolkata")

# =========================
# TIME HELPERS
# =========================
def now_ist() -> datetime:
    return datetime.now(TZ)

def dt_at(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=TZ)

def is_weekday(d: date) -> bool:
    return d.weekday() < 5

def expiry_dt_ist(expiry_d: date) -> datetime:
    return datetime(expiry_d.year, expiry_d.month, expiry_d.day, EXPIRY_HOUR_IST, EXPIRY_MINUTE_IST, tzinfo=TZ)

def time_to_expiry_years(expiry_d: date) -> float:
    t = (expiry_dt_ist(expiry_d) - now_ist()).total_seconds()
    return max(0.0, t / (365.0 * 24.0 * 3600.0))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def after_start(now: datetime) -> bool:
    return now >= dt_at(now.date(), *START_TRACK_TIME)

def eod_reached(now: datetime) -> bool:
    return now >= dt_at(now.date(), *EOD_EXIT_TIME)

# =========================
# TOKEN LOAD
# =========================
def load_access_token() -> str:
    if not os.path.exists(TOKEN_CACHE_PATH):
        raise RuntimeError(
            f"Missing token file: {TOKEN_CACHE_PATH}\n"
            "Run: python3 scripts/kite_login.py (then restart service)"
        )
    with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
        j = json.load(f)
    tok = (j.get("access_token") or "").strip()
    if not tok:
        raise RuntimeError(f"{TOKEN_CACHE_PATH} exists but has no access_token.")
    return tok

def make_kite() -> KiteConnect:
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(load_access_token())
    return kite

# =========================
# NORMAL DIST
# =========================
SQRT_2PI = math.sqrt(2.0 * math.pi)

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# =========================
# BLACK-76
# =========================
def black76_price(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    D = math.exp(-r * T)
    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    if is_call:
        return D * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return D * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> Dict[str, float]:
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {k: float("nan") for k in ["delta","gamma","theta_per_year","theta_per_day","vega_per_1vol","vega_per_1pct"]}
    D = math.exp(-r * T)
    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
    pdf_d1 = norm_pdf(d1)

    delta = D * norm_cdf(d1) if is_call else D * (norm_cdf(d1) - 1.0)
    gamma = D * pdf_d1 / (F * vol_sqrtT)
    vega = D * F * pdf_d1 * math.sqrt(T)
    theta = -D * (F * pdf_d1 * sigma) / (2.0 * math.sqrt(T))

    return {
        "delta": delta,
        "gamma": gamma,
        "theta_per_year": theta,
        "theta_per_day": theta / 365.0,
        "vega_per_1vol": vega,
        "vega_per_1pct": vega / 100.0,
    }

def implied_vol_black76(price: float, F: float, K: float, T: float, r: float, is_call: bool, sigma0: float = 0.30) -> float:
    if price is None or not (price > 0) or F <= 0 or K <= 0 or T <= 0:
        return float("nan")

    D = math.exp(-r * T)
    intrinsic = D * max(0.0, F - K) if is_call else D * max(0.0, K - F)
    upper = D * F if is_call else D * K
    if price < intrinsic - 1e-6 or price > upper + 1e-6:
        return float("nan")
    if price <= intrinsic + 1e-6:
        return 1e-6

    sigma = max(1e-6, sigma0)
    for _ in range(25):
        p = black76_price(F, K, T, r, sigma, is_call)
        diff = p - price
        if abs(diff) < 1e-5:
            return max(1e-6, sigma)

        vol_sqrtT = sigma * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
        vega = D * F * norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma = min(max(sigma, 1e-6), 10.0)

    lo, hi = 1e-6, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = black76_price(F, K, T, r, mid, is_call)
        if math.isnan(pm):
            return float("nan")
        if abs(pm - price) < 1e-5:
            return mid
        if pm < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# =========================
# TICK HELPERS (fixed mid)
# =========================
def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def best_bid_ask_from_tick(t: dict) -> Tuple[Optional[float], Optional[float]]:
    d = t.get("depth") or {}
    buy = d.get("buy") or []
    sell = d.get("sell") or []

    def first_valid(levels):
        for lv in levels:
            try:
                p = float(lv.get("price") or 0.0)
                q = lv.get("quantity", None)
                q = float(q) if q is not None else None
                if p > 0 and (q is None or q > 0):
                    return p
            except Exception:
                continue
        return None

    return first_valid(buy), first_valid(sell)

def price_from_tick(t: dict) -> Optional[float]:
    bid, ask = best_bid_ask_from_tick(t)
    if bid is not None and ask is not None and ask >= bid:
        return 0.5 * (bid + ask)
    return _safe_float(t.get("last_price"))

def oi_from_tick(t: dict) -> Optional[float]:
    return _safe_float(t.get("oi") if t.get("oi") is not None else t.get("open_interest"))

def vol_from_tick(t: dict) -> Optional[float]:
    return _safe_float(t.get("volume") if t.get("volume") is not None else t.get("volume_traded"))

# =========================
# INSTRUMENT SELECTION
# =========================
INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

@dataclass
class ContractRef:
    token: int
    tradingsymbol: str
    strike: float
    opt_type: str  # CE/PE
    lot_size: int

@dataclass
class UnderlyingUniverse:
    name: str
    expiry_opt: date
    step: float
    fut_token: Optional[int]
    fut_tradingsymbol: Optional[str]
    fut_expiry: Optional[date]
    strikes: List[float]
    ce: Dict[float, ContractRef]
    pe: Dict[float, ContractRef]

def strip_ns_suffix(x: str) -> str:
    x = str(x).strip()
    return x[:-3] if x.upper().endswith(".NS") else x

def load_tickers(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    raw = [strip_ns_suffix(v) for v in df[col].dropna().astype(str).tolist()]
    seen, out = set(), []
    for t in raw:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def load_nfo_instruments(kite: KiteConnect) -> pd.DataFrame:
    if os.path.exists(CACHE_NFO_CSV):
        age_sec = time.time() - os.path.getmtime(CACHE_NFO_CSV)
        if age_sec < CACHE_VALIDITY_HOURS * 3600:
            df = pd.read_csv(CACHE_NFO_CSV)
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            return df

    inst = kite.instruments("NFO")
    df = pd.DataFrame(inst)
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df.to_csv(CACHE_NFO_CSV, index=False)
    return df

def detect_step(strikes: List[float]) -> float:
    s = sorted(set(float(x) for x in strikes))
    diffs = [s[i+1] - s[i] for i in range(len(s)-1)]
    diffs = [d for d in diffs if d > 0]
    return min(diffs) if diffs else 1.0

def nearest_strike(strikes: List[float], x: float) -> float:
    return min(strikes, key=lambda k: abs(k - x))

def pick_nearest_expiry(opt_df: pd.DataFrame) -> date:
    today = now_ist().date()
    expiries = sorted({e for e in opt_df["expiry"].dropna().tolist() if e >= today})
    if not expiries:
        raise RuntimeError("No future expiries found.")
    return expiries[0]

def pick_future_for_underlying(inst_nfo: pd.DataFrame, underlying: str, opt_expiry: date) -> Tuple[Optional[int], Optional[str], Optional[date]]:
    today = now_ist().date()
    fut = inst_nfo[(inst_nfo["segment"] == "NFO-FUT") & (inst_nfo["name"] == underlying)].copy()
    if fut.empty:
        return None, None, None
    fut = fut[fut["expiry"].notna()]
    fut = fut[fut["expiry"] >= today]
    if fut.empty:
        return None, None, None

    if underlying not in INDEX_UNDERLYINGS:
        same = fut[fut["expiry"] == opt_expiry]
        if not same.empty:
            r = same.sort_values("expiry").iloc[0]
            return int(r["instrument_token"]), str(r["tradingsymbol"]), r["expiry"]

    r = fut.sort_values("expiry").iloc[0]
    return int(r["instrument_token"]), str(r["tradingsymbol"]), r["expiry"]

def build_universe_for_underlying(kite: KiteConnect, inst_nfo: pd.DataFrame, underlying: str) -> Optional[UnderlyingUniverse]:
    opt = inst_nfo[(inst_nfo["segment"] == "NFO-OPT") & (inst_nfo["name"] == underlying)].copy()
    if opt.empty:
        return None

    expiry_opt = pick_nearest_expiry(opt)
    opt = opt[opt["expiry"] == expiry_opt].copy()
    if opt.empty:
        return None

    strikes_all = sorted(set(opt["strike"].dropna().astype(float).tolist()))
    if not strikes_all:
        return None
    step = detect_step(strikes_all)

    fut_token, fut_sym, fut_expiry = pick_future_for_underlying(inst_nfo, underlying, expiry_opt)

    ref_price = None
    try:
        if fut_sym:
            q = kite.ltp([f"NFO:{fut_sym}"])
            ref_price = float(q[f"NFO:{fut_sym}"]["last_price"])
        else:
            q = kite.ltp([f"NSE:{underlying}"])
            ref_price = float(q[f"NSE:{underlying}"]["last_price"])
    except Exception:
        ref_price = strikes_all[len(strikes_all)//2]

    atm = nearest_strike(strikes_all, ref_price)
    below = [k for k in strikes_all if k <= atm]
    above = [k for k in strikes_all if k >= atm]
    window_strikes = sorted(set(below[-STRIKES_EACH_SIDE:] + above[:STRIKES_EACH_SIDE + 1]))

    ce, pe = {}, {}
    for _, r in opt.iterrows():
        K = float(r["strike"])
        if K not in window_strikes:
            continue
        t = str(r["instrument_type"]).upper()
        cref = ContractRef(
            token=int(r["instrument_token"]),
            tradingsymbol=str(r["tradingsymbol"]),
            strike=K,
            opt_type=t,
            lot_size=int(r.get("lot_size") or 1)
        )
        if t == "CE":
            ce.setdefault(K, cref)
        elif t == "PE":
            pe.setdefault(K, cref)

    if not ce and not pe:
        return None

    return UnderlyingUniverse(
        name=underlying,
        expiry_opt=expiry_opt,
        step=step,
        fut_token=fut_token,
        fut_tradingsymbol=fut_sym,
        fut_expiry=fut_expiry,
        strikes=window_strikes,
        ce=ce,
        pe=pe
    )

# =========================
# REST SEED
# =========================
def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def seed_ticks_with_rest_quote(kite: KiteConnect, ins_keys: List[str], tick_store: Dict[int, dict], lock: threading.Lock):
    if not ins_keys:
        return
    for part in chunked(ins_keys, REST_QUOTE_CHUNK):
        try:
            q = kite.quote(part)
            with lock:
                for _, v in q.items():
                    tok = v.get("instrument_token")
                    if tok is None:
                        continue
                    tick_store[int(tok)] = {
                        "instrument_token": int(tok),
                        "last_price": v.get("last_price"),
                        "oi": v.get("oi") if v.get("oi") is not None else v.get("open_interest"),
                        "volume": v.get("volume"),
                        "depth": v.get("depth") or {},
                        "timestamp": v.get("timestamp"),
                    }
        except Exception:
            pass

# =========================
# WEBSOCKET
# =========================
class WSConn:
    def __init__(self, idx: int, api_key: str, access_token: str, tokens: List[int], tick_store: Dict[int, dict], lock: threading.Lock):
        self.idx = idx
        self.tokens = tokens
        self.tick_store = tick_store
        self.lock = lock
        self.connected = False
        self.kws = KiteTicker(api_key, access_token)

        def on_connect(ws, response):
            self.connected = True
            ws.subscribe(self.tokens)
            if WS_MODE.lower() == "full":
                ws.set_mode(ws.MODE_FULL, self.tokens)
            elif WS_MODE.lower() == "quote":
                ws.set_mode(ws.MODE_QUOTE, self.tokens)
            else:
                ws.set_mode(ws.MODE_LTP, self.tokens)
            print(f"[WS{self.idx}] Connected. tokens={len(self.tokens)} mode={WS_MODE}")

        def on_ticks(ws, ticks):
            with self.lock:
                for t in ticks:
                    tok = t.get("instrument_token")
                    if tok is not None:
                        self.tick_store[int(tok)] = t

        def on_close(ws, code, reason):
            self.connected = False
            print(f"[WS{self.idx}] Closed. code={code} reason={reason}")

        def on_error(ws, code, reason):
            print(f"[WS{self.idx}] Error. code={code} reason={reason}")

        self.kws.on_connect = on_connect
        self.kws.on_ticks = on_ticks
        self.kws.on_close = on_close
        self.kws.on_error = on_error

    def start(self):
        self.kws.connect(threaded=True)

class MultiWS:
    def __init__(self):
        self.conns: List[WSConn] = []

    def start(self, api_key: str, access_token: str, token_lists: List[List[int]], tick_store: Dict[int, dict], lock: threading.Lock):
        for i, tokens in enumerate(token_lists[:MAX_WS_CONNECTIONS], start=1):
            c = WSConn(i, api_key, access_token, tokens, tick_store, lock)
            c.start()
            self.conns.append(c)

    def ready(self) -> bool:
        return any(c.connected for c in self.conns)

# =========================
# CHAIN COMPUTATION (long)
# =========================
def compute_chain_for_underlying(
    u: UnderlyingUniverse,
    ticks: Dict[int, dict],
    iv_seed_by_key: Dict[Tuple[str, float], float],
    kite: KiteConnect,
    forward_cache: Dict[Tuple[str, date], Tuple[float, float]],
) -> pd.DataFrame:
    T = time_to_expiry_years(u.expiry_opt)

    F = None
    forward_method = "none"
    parity_candidates = 0

    # 1) FUT tick
    if u.fut_token is not None:
        tf = ticks.get(u.fut_token)
        if tf:
            F = price_from_tick(tf)
            if F and F > 0:
                forward_method = "fut"

    # 2) parity from WS ticks
    if F is None or not (F and F > 0):
        cands = []
        if T > 0:
            for K in u.strikes:
                ce = u.ce.get(K); pe = u.pe.get(K)
                if not ce or not pe:
                    continue
                tc = ticks.get(ce.token); tp = ticks.get(pe.token)
                if not tc or not tp:
                    continue
                C = price_from_tick(tc); P = price_from_tick(tp)
                if C is None or P is None or C < MIN_OPTION_PRICE or P < MIN_OPTION_PRICE:
                    continue
                cands.append(K + math.exp(RISK_FREE_RATE * T) * (C - P))

        parity_candidates = len(cands)
        if cands:
            cands.sort()
            F = cands[len(cands)//2]
            if F and F > 0:
                forward_method = "parity_ws"

    # 3) cache
    cache_key = (u.name, u.expiry_opt)
    if (F is None or not (F and F > 0)) and cache_key in forward_cache:
        last_F, ts = forward_cache[cache_key]
        if time.time() - ts < 10.0 and last_F and last_F > 0:
            F = last_F
            forward_method = "cache"

    # 4) REST parity fallback
    if F is None or not (F and F > 0):
        try:
            K0 = u.strikes[len(u.strikes)//2]
            ce0 = u.ce.get(K0); pe0 = u.pe.get(K0)
            if ce0 and pe0 and T > 0:
                q = kite.quote([f"NFO:{ce0.tradingsymbol}", f"NFO:{pe0.tradingsymbol}"])
                C = q.get(f"NFO:{ce0.tradingsymbol}", {}).get("last_price")
                P = q.get(f"NFO:{pe0.tradingsymbol}", {}).get("last_price")
                if C is not None and P is not None and float(C) > MIN_OPTION_PRICE and float(P) > MIN_OPTION_PRICE:
                    F = K0 + math.exp(RISK_FREE_RATE * T) * (float(C) - float(P))
                    if F and F > 0:
                        forward_method = "parity_rest"
        except Exception:
            pass

    if F is None:
        F = float("nan")
    if F == F and F > 0:
        forward_cache[cache_key] = (float(F), time.time())

    # IV per strike from OTM option
    iv_by_strike: Dict[float, float] = {}
    for K in u.strikes:
        if T <= 0 or not (F == F and F > 0):
            iv_by_strike[K] = float("nan")
            continue

        otm_type = "PE" if F >= K else "CE"
        cref = u.pe.get(K) if otm_type == "PE" else u.ce.get(K)
        if cref is None:
            cref = u.ce.get(K) or u.pe.get(K)
            otm_type = cref.opt_type if cref else otm_type
        if cref is None:
            iv_by_strike[K] = float("nan")
            continue

        t = ticks.get(cref.token)
        if not t:
            iv_by_strike[K] = float("nan")
            continue

        px = price_from_tick(t)
        if px is None or px < MIN_OPTION_PRICE:
            iv_by_strike[K] = float("nan")
            continue

        seed = iv_seed_by_key.get((u.name, K), 0.30)
        iv = implied_vol_black76(px, F, K, T, RISK_FREE_RATE, is_call=(otm_type == "CE"), sigma0=seed)
        iv_by_strike[K] = iv
        if iv == iv and iv > 0:
            iv_seed_by_key[(u.name, K)] = iv

    rows = []
    ts_str = now_ist().strftime("%Y-%m-%d %H:%M:%S")
    for K in u.strikes:
        ivK = iv_by_strike.get(K, float("nan"))
        for side, mp in [("CE", u.ce), ("PE", u.pe)]:
            cref = mp.get(K)
            if not cref:
                continue
            t = ticks.get(cref.token)
            if not t:
                continue

            bid, ask = best_bid_ask_from_tick(t)
            mid = (0.5*(bid+ask)) if (bid is not None and ask is not None and ask >= bid) else None
            ltp = _safe_float(t.get("last_price"))
            px_used = mid if mid is not None else ltp

            greeks = {k: float("nan") for k in ["delta","gamma","theta_per_year","theta_per_day","vega_per_1vol","vega_per_1pct"]}
            if ivK == ivK and ivK > 0 and T > 0 and (F == F and F > 0):
                greeks = black76_greeks(F, K, T, RISK_FREE_RATE, ivK, is_call=(side == "CE"))

            rows.append({
                "ts_ist": ts_str,
                "underlying": u.name,
                "opt_expiry": u.expiry_opt.isoformat(),
                "fut_expiry": u.fut_expiry.isoformat() if u.fut_expiry else None,
                "T_years": T,
                "forward_F": F,
                "forward_method": forward_method,
                "parity_candidates": parity_candidates,
                "strike": K,
                "type": side,
                "tradingsymbol": cref.tradingsymbol,
                "instrument_token": cref.token,
                "lot_size": cref.lot_size,
                "price_used": px_used,
                "ltp": ltp,
                "bid": bid, "ask": ask, "mid": mid,
                "oi": oi_from_tick(t),
                "volume": vol_from_tick(t),
                "iv_strike": ivK,
                **greeks
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["strike", "type"])
    return df

# =========================
# LONG -> WIDE
# =========================
WIDE_VALUE_COLS = ["ltp","bid","ask","mid","oi","volume","iv_strike","delta","gamma","theta_per_day","vega_per_1pct"]

def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return df_long

    base_cols = ["ts_ist","underlying","opt_expiry","fut_expiry","T_years","forward_F","forward_method","parity_candidates"]
    meta = df_long[base_cols].iloc[0].to_dict()

    d = df_long[["strike","type"] + [c for c in WIDE_VALUE_COLS if c in df_long.columns]].copy()

    wide = d.pivot_table(index="strike", columns="type", values=[c for c in WIDE_VALUE_COLS if c in d.columns], aggfunc="last")
    wide.columns = [f"{c}_{t}" for (c, t) in wide.columns]
    wide = wide.reset_index().sort_values("strike")

    for k, v in meta.items():
        wide[k] = v

    for side in ["CE","PE"]:
        for c in ["ltp","bid","ask","mid","oi"]:
            col = f"{c}_{side}"
            if col not in wide.columns:
                wide[col] = float("nan")

    front = ["strike"] + base_cols
    rest = [c for c in wide.columns if c not in front]
    return wide[front + rest]

# =========================
# PnL tracker pieces
# =========================
def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def append_csv_row(path: str, row: dict):
    ensure_dir(os.path.dirname(path) or ".")
    df = pd.DataFrame([row])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def load_state() -> dict:
    if not os.path.exists(STATE_JSON):
        return {"open_trades": {}, "last_trade_day": None, "price_hist": {}}
    with open(STATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(st: dict):
    ensure_dir(OUT_TRADES_DIR)
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)

def best_bid_ask(bid: Optional[float], ask: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    b = bid if (bid is not None and bid > 0) else None
    a = ask if (ask is not None and ask > 0) else None
    return b, a

def price_used_for_leg(row: pd.Series, side: str, action: str) -> Optional[float]:
    bid = _f(row.get(f"bid_{side}"))
    ask = _f(row.get(f"ask_{side}"))
    mid = _f(row.get(f"mid_{side}"))
    ltp = _f(row.get(f"ltp_{side}"))
    bid, ask = best_bid_ask(bid, ask)
    if mid is None and bid is not None and ask is not None and ask >= bid:
        mid = 0.5 * (bid + ask)
    if action == "CLOSE_SHORT":  # BUY
        return ask if ask is not None else (ltp if ltp is not None and ltp > 0 else mid)
    if action == "CLOSE_LONG":   # SELL
        return bid if bid is not None else (ltp if ltp is not None and ltp > 0 else mid)
    return ltp if ltp is not None else mid

def infer_step(strikes: List[float]) -> float:
    s = sorted(set([float(x) for x in strikes if x is not None and not math.isnan(x)]))
    if len(s) < 2:
        return 1.0
    diffs = [s[i+1] - s[i] for i in range(len(s)-1)]
    diffs = [d for d in diffs if d > 0]
    return min(diffs) if diffs else 1.0

def strongest_cluster_strike(df_wide: pd.DataFrame, oi_col: str) -> Tuple[Optional[float], Optional[float]]:
    d = df_wide[["strike", oi_col]].copy()
    d[oi_col] = pd.to_numeric(d[oi_col], errors="coerce").fillna(0.0)
    d = d.sort_values("strike")
    if (d[oi_col] >= MIN_OI).sum() == 0:
        return None, None
    d["cluster"] = d[oi_col].rolling(window=CLUSTER_WINDOW, center=True, min_periods=1).sum()
    d2 = d[d[oi_col] >= MIN_OI]
    idx = d2["cluster"].idxmax()
    return float(d2.loc[idx, "strike"]), float(d2.loc[idx, "cluster"])

def pick_strike_le(strikes: List[float], target: float) -> Optional[float]:
    c = [k for k in strikes if k <= target]
    return max(c) if c else None

def pick_strike_ge(strikes: List[float], target: float) -> Optional[float]:
    c = [k for k in strikes if k >= target]
    return min(c) if c else None

def get_forward_price(df_wide: pd.DataFrame) -> Optional[float]:
    for col in ["forward_F", "F", "fut_price", "underlying_price"]:
        if col in df_wide.columns:
            v = pd.to_numeric(df_wide[col], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[0])
    return None

def accepted_over_window(hist: List[Tuple[datetime, float]], wall: float, direction: str, minutes: int) -> bool:
    if wall is None or not hist:
        return False
    now = now_ist()
    cutoff = now - timedelta(minutes=minutes)
    pts = [(t, p) for (t, p) in hist if t >= cutoff]
    if len(pts) == 0:
        return False
    if direction == "above":
        return all(p >= wall for (_, p) in pts)
    return all(p <= wall for (_, p) in pts)

def decide_trade(df_wide: pd.DataFrame, price_hist: List[Tuple[datetime, float]]) -> Optional[dict]:
    strikes = pd.to_numeric(df_wide["strike"], errors="coerce").dropna().astype(float).tolist()
    if not strikes:
        return None
    step = infer_step(strikes)

    put_wall, put_cluster = strongest_cluster_strike(df_wide, "oi_PE")
    call_wall, call_cluster = strongest_cluster_strike(df_wide, "oi_CE")
    F = get_forward_price(df_wide)
    if F is None or put_wall is None or call_wall is None:
        return None

    bull_ok = accepted_over_window(price_hist, put_wall, "above", ACCEPT_MINUTES) or (F >= (put_wall + BAND_STEPS * step))
    bear_ok = accepted_over_window(price_hist, call_wall, "below", ACCEPT_MINUTES) or (F <= (call_wall - BAND_STEPS * step))
    if not bull_ok and not bear_ok:
        return None

    if bull_ok and bear_ok:
        dist_put = abs(F - put_wall)
        dist_call = abs(call_wall - F)
        if dist_put < dist_call:
            side = "BULL_PUT"
        elif dist_call < dist_put:
            side = "BEAR_CALL"
        else:
            side = "BULL_PUT" if (put_cluster or 0) >= (call_cluster or 0) else "BEAR_CALL"
    else:
        side = "BULL_PUT" if bull_ok else "BEAR_CALL"

    if side == "BULL_PUT":
        short_target = put_wall - OFFSET_STEPS * step
        shortK = pick_strike_le(strikes, short_target)
        if shortK is None:
            return None
        longK = pick_strike_le(strikes, shortK - WIDTH_STEPS * step)
        if longK is None or longK >= shortK:
            return None
        return {"side":"BULL_PUT","step":step,"F":F,"wall_strike":put_wall,"wall_cluster":put_cluster,
                "short_strike":float(shortK),"long_strike":float(longK),"exit_level":float(longK)}
    else:
        short_target = call_wall + OFFSET_STEPS * step
        shortK = pick_strike_ge(strikes, short_target)
        if shortK is None:
            return None
        longK = pick_strike_ge(strikes, shortK + WIDTH_STEPS * step)
        if longK is None or longK <= shortK:
            return None
        return {"side":"BEAR_CALL","step":step,"F":F,"wall_strike":call_wall,"wall_cluster":call_cluster,
                "short_strike":float(shortK),"long_strike":float(longK),"exit_level":float(longK)}

@dataclass
class Trade:
    trade_id: str
    trade_date: str
    underlying: str
    expiry: str
    side: str
    wall_strike: float
    step: float
    short_strike: float
    long_strike: float
    lot_size: int
    entry_time_ist: str
    entry_short: float
    entry_long: float
    entry_credit: float
    status: str
    exit_time_ist: Optional[str] = None
    exit_short: Optional[float] = None
    exit_long: Optional[float] = None
    exit_debit: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    max_pnl: float = 0.0
    min_pnl: float = 0.0

def new_trade_id(u: str, ts: datetime) -> str:
    return f"{u}-{ts.strftime('%Y%m%d-%H%M%S')}"

def chain_row_for_strike(df_wide: pd.DataFrame, K: float) -> Optional[pd.Series]:
    d = df_wide[pd.to_numeric(df_wide["strike"], errors="coerce") == float(K)]
    if d.empty:
        return None
    return d.iloc[0]

def mtm_pnl(df_wide: pd.DataFrame, tr: Trade) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    r_short = chain_row_for_strike(df_wide, tr.short_strike)
    r_long  = chain_row_for_strike(df_wide, tr.long_strike)
    if r_short is None or r_long is None:
        return None, None, None

    if tr.side == "BULL_PUT":
        close_short = price_used_for_leg(r_short, "PE", "CLOSE_SHORT")
        close_long  = price_used_for_leg(r_long,  "PE", "CLOSE_LONG")
    else:
        close_short = price_used_for_leg(r_short, "CE", "CLOSE_SHORT")
        close_long  = price_used_for_leg(r_long,  "CE", "CLOSE_LONG")

    if close_short is None or close_long is None:
        return None, close_short, close_long

    pnl = (tr.entry_short - close_short) * tr.lot_size + (close_long - tr.entry_long) * tr.lot_size
    return float(pnl), float(close_short), float(close_long)

def lot_size_from_universe(u: UnderlyingUniverse, side: str, strike: float) -> Optional[int]:
    cref = u.pe.get(strike) if side == "BULL_PUT" else u.ce.get(strike)
    if cref:
        return int(cref.lot_size or 1)
    cref = u.ce.get(strike) or u.pe.get(strike)
    return int(cref.lot_size or 1) if cref else None

# =========================
# MAIN LOOP
# =========================
def main():
    ensure_dir(OUT_CHAIN_DIR)
    ensure_dir(OUT_TRADES_DIR)

    kite = make_kite()
    prof = kite.profile()
    print("[INFO] Logged in as:", prof.get("user_name"), "| user_id:", prof.get("user_id"))

    tickers = load_tickers(TICKER_CSV_PATH)
    print(f"[INFO] Loaded {len(tickers)} tickers from {TICKER_CSV_PATH}")
    if not tickers:
        raise RuntimeError("No tickers found in ticker_list.csv")

    inst_nfo = load_nfo_instruments(kite)

    universes: List[UnderlyingUniverse] = []
    skipped: List[str] = []
    for t in tickers:
        u = build_universe_for_underlying(kite, inst_nfo, t)
        if u is None:
            skipped.append(t)
        else:
            universes.append(u)

    print(f"[INFO] Universes: {len(universes)} | skipped: {len(skipped)}")
    if not universes:
        raise RuntimeError("No option universes found.")

    tokens: List[int] = []
    for u in universes:
        if u.fut_token is not None:
            tokens.append(u.fut_token)
        tokens += [c.token for c in u.ce.values()]
        tokens += [p.token for p in u.pe.values()]
    tokens = sorted(set(tokens))

    print(f"[INFO] Total tokens: {len(tokens)} (max={MAX_TOKENS_PER_WS*MAX_WS_CONNECTIONS})")
    if len(tokens) > MAX_TOKENS_PER_WS * MAX_WS_CONNECTIONS:
        raise RuntimeError("Too many tokens. Reduce STRIKES_EACH_SIDE or ticker count.")

    token_lists = [tokens[i:i+MAX_TOKENS_PER_WS] for i in range(0, len(tokens), MAX_TOKENS_PER_WS)]
    for i, tl in enumerate(token_lists, 1):
        print(f"  WS{i}: {len(tl)} tokens")

    tick_store: Dict[int, dict] = {}
    lock = threading.Lock()

    first_u = universes[0]
    first_keys = []
    if first_u.fut_tradingsymbol:
        first_keys.append(f"NFO:{first_u.fut_tradingsymbol}")
    first_keys += [f"NFO:{c.tradingsymbol}" for c in first_u.ce.values()]
    first_keys += [f"NFO:{p.tradingsymbol}" for p in first_u.pe.values()]
    seed_ticks_with_rest_quote(kite, first_keys, tick_store, lock)

    ws = MultiWS()
    ws.start(API_KEY, load_access_token(), token_lists, tick_store, lock)

    iv_seed_by_key: Dict[Tuple[str, float], float] = {}
    forward_cache: Dict[Tuple[str, date], Tuple[float, float]] = {}
    sample_printed = False

    st = load_state()
    price_hist_runtime: Dict[str, List[Tuple[datetime, float]]] = {}
    for u, pts in st.get("price_hist", {}).items():
        out = []
        for (ts_iso, p) in pts:
            try:
                out.append((datetime.fromisoformat(ts_iso), float(p)))
            except Exception:
                pass
        price_hist_runtime[u] = out

    open_trades: Dict[str, Trade] = {}
    for u, tdict in st.get("open_trades", {}).items():
        try:
            open_trades[u] = Trade(**tdict)
        except Exception:
            pass

    last_day = st.get("last_trade_day")

    while True:
        if not ws.ready():
            print("[INFO] Waiting for WS connect...")
            time.sleep(1)
            continue

        now = now_ist()
        d = now.date()

        if last_day != str(d):
            last_day = str(d)
            st["last_trade_day"] = last_day
            price_hist_runtime = {}
            st["price_hist"] = {}
            save_state(st)
            print(f"\n[DAY] {last_day} IST. Price history reset. Open trades carried: {len(open_trades)}")

        t0 = time.time()
        with lock:
            snap = dict(tick_store)

        frames_long: List[pd.DataFrame] = []
        frames_wide: List[pd.DataFrame] = []
        weekday = is_weekday(d)

        for u in universes:
            df_long = compute_chain_for_underlying(u, snap, iv_seed_by_key, kite, forward_cache)
            if df_long.empty:
                continue
            df_wide = long_to_wide(df_long)

            frames_long.append(df_long)
            frames_wide.append(df_wide)

            if WRITE_CHAIN_CSVS:
                df_long.to_csv(os.path.join(OUT_CHAIN_DIR, f"{u.name}_chain_long.csv"), index=False)
                df_wide.to_csv(os.path.join(OUT_CHAIN_DIR, f"{u.name}_chain_wide.csv"), index=False)

            if PRINT_SAMPLE_CHAIN_ONCE and (not sample_printed) and (u.name == first_u.name):
                print(df_long.head(5).to_string(index=False))
                sample_printed = True

            F = get_forward_price(df_wide)
            if F is None or not (F > 0):
                continue

            ph = price_hist_runtime.get(u.name, [])
            ph.append((now, float(F)))
            cutoff = now - timedelta(minutes=15)
            ph = [(t, p) for (t, p) in ph if t >= cutoff]
            price_hist_runtime[u.name] = ph
            st.setdefault("price_hist", {})
            st["price_hist"][u.name] = [(t.isoformat(), p) for (t, p) in ph[-400:]]

            # Exits / MTM
            if u.name in open_trades and open_trades[u.name].status == "OPEN":
                tr = open_trades[u.name]
                pnl, close_short, close_long = mtm_pnl(df_wide, tr)
                if pnl is not None:
                    tr.max_pnl = max(tr.max_pnl, pnl)
                    tr.min_pnl = min(tr.min_pnl, pnl)
                    append_csv_row(PNL_CSV, {
                        "ts_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "trade_id": tr.trade_id,
                        "underlying": tr.underlying,
                        "expiry": tr.expiry,
                        "side": tr.side,
                        "F": float(F),
                        "short_strike": tr.short_strike,
                        "long_strike": tr.long_strike,
                        "close_short": close_short,
                        "close_long": close_long,
                        "pnl": pnl,
                        "max_pnl": tr.max_pnl,
                        "min_pnl": tr.min_pnl,
                    })

                exit_now = False
                reason = None
                if tr.side == "BULL_PUT" and F <= tr.long_strike:
                    exit_now = True
                    reason = f"UNDERLYING_HIT_HEDGE (F<=long_put {tr.long_strike})"
                elif tr.side == "BEAR_CALL" and F >= tr.long_strike:
                    exit_now = True
                    reason = f"UNDERLYING_HIT_HEDGE (F>=long_call {tr.long_strike})"
                if not exit_now and eod_reached(now):
                    exit_now = True
                    reason = "EOD_EXIT"

                if exit_now:
                    pnl2, cs2, cl2 = mtm_pnl(df_wide, tr)
                    tr.status = "CLOSED"
                    tr.exit_time_ist = now.strftime("%Y-%m-%d %H:%M:%S")
                    tr.exit_short = cs2 if cs2 is not None else close_short
                    tr.exit_long = cl2 if cl2 is not None else close_long
                    if tr.exit_short is not None and tr.exit_long is not None:
                        tr.exit_debit = float(tr.exit_short) - float(tr.exit_long)
                    tr.pnl = pnl2 if pnl2 is not None else pnl
                    tr.exit_reason = reason

                    append_csv_row(LEDGER_CSV, {**asdict(tr), "event":"EXIT", "ts_ist": tr.exit_time_ist})
                    print(f"[EXIT] {u.name} {tr.side} pnl={tr.pnl if tr.pnl is not None else float('nan'):.2f} reason={tr.exit_reason}")

            # Entries
            if weekday and ((u.name not in open_trades) or (open_trades[u.name].status != "OPEN")):
                if not after_start(now) or eod_reached(now):
                    continue

                plan = decide_trade(df_wide, price_hist_runtime.get(u.name, []))
                if plan is None:
                    continue

                expiry = str(df_wide["opt_expiry"].iloc[0]) if "opt_expiry" in df_wide.columns else u.expiry_opt.isoformat()
                r_short = chain_row_for_strike(df_wide, plan["short_strike"])
                r_long = chain_row_for_strike(df_wide, plan["long_strike"])
                if r_short is None or r_long is None:
                    continue

                if plan["side"] == "BULL_PUT":
                    entry_short = price_used_for_leg(r_short, "PE", "CLOSE_LONG")   # sell ~ bid
                    entry_long  = price_used_for_leg(r_long,  "PE", "CLOSE_SHORT")  # buy  ~ ask
                else:
                    entry_short = price_used_for_leg(r_short, "CE", "CLOSE_LONG")
                    entry_long  = price_used_for_leg(r_long,  "CE", "CLOSE_SHORT")

                if entry_short is None or entry_long is None:
                    continue

                lot = lot_size_from_universe(u, plan["side"], float(plan["short_strike"])) or 0
                if lot <= 0:
                    continue

                tr = Trade(
                    trade_id=new_trade_id(u.name, now),
                    trade_date=str(d),
                    underlying=u.name,
                    expiry=expiry,
                    side=plan["side"],
                    wall_strike=float(plan["wall_strike"]),
                    step=float(plan["step"]),
                    short_strike=float(plan["short_strike"]),
                    long_strike=float(plan["long_strike"]),
                    lot_size=int(lot),
                    entry_time_ist=now.strftime("%Y-%m-%d %H:%M:%S"),
                    entry_short=float(entry_short),
                    entry_long=float(entry_long),
                    entry_credit=float(entry_short - entry_long),
                    status="OPEN",
                    max_pnl=0.0,
                    min_pnl=0.0,
                )
                open_trades[u.name] = tr
                append_csv_row(LEDGER_CSV, {**asdict(tr), "event":"ENTRY", "ts_ist": tr.entry_time_ist, "F_at_entry": float(plan["F"])})
                print(f"[ENTRY] {u.name} {tr.side} short={tr.short_strike} long={tr.long_strike} credit≈{tr.entry_credit:.2f} lot={tr.lot_size}")

        if WRITE_CHAIN_CSVS and frames_long:
            pd.concat(frames_long, ignore_index=True).to_csv(COMBINED_LONG_CSV, index=False)
        if WRITE_CHAIN_CSVS and frames_wide:
            pd.concat(frames_wide, ignore_index=True).to_csv(COMBINED_WIDE_CSV, index=False)

        st["open_trades"] = {u: asdict(t) for u, t in open_trades.items() if t.status == "OPEN"}
        save_state(st)

        dt = time.time() - t0
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S IST')}] chains={len(frames_long)} ticks={len(snap)} dt={dt:.2f}s open={len(st['open_trades'])}")
        time.sleep(max(0.0, REFRESH_SECONDS - dt))

if __name__ == "__main__":
    main()
