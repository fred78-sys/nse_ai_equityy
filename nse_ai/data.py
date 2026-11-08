"""Data collection and loading for the Nairobi Securities Exchange (NSE).

Everything here is fail-soft: a missing ticker, macro file or news feed never stops
the pipeline, it just removes that information source and the system says so.

Sources
-------
* ``collect_yahoo``      Yahoo Finance chart API (suffix ``.NR``) -> per-ticker CSV store.
* ``load_csv_store``     Reads ``data/csv/<TICKER>.csv`` (date, close, volume). You can drop in
                         your own files (NSE daily price lists, broker exports, etc.).
* ``make_synthetic_nse`` Simulated NSE-like market for demos and tests ONLY.
* macro / fundamentals / sentiment loaders: optional CSV inputs with publication lags so no
  information is used before it was public.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------------------
@dataclass
class MarketData:
    close: pd.DataFrame                      # dividend-adjusted close, NaN when no quote yet / delisted
    volume: pd.DataFrame                     # shares traded (0 on no-trade days)
    meta: pd.DataFrame                       # index=ticker, columns name, sector
    macro: pd.DataFrame | None = None        # daily, columns = series (already publication-lagged)
    fundamentals: pd.DataFrame | None = None # raw point-in-time records
    sentiment: pd.DataFrame | None = None    # date,ticker,score
    notes: list[str] = field(default_factory=list)
    is_synthetic: bool = False

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


# --------------------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------------------
def load_universe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df.drop_duplicates("ticker").set_index("ticker")


# --------------------------------------------------------------------------------------
# Yahoo collector (requests only; no yfinance dependency)
# --------------------------------------------------------------------------------------
def fetch_yahoo_chart(symbol: str, start: str, end: str | None = None, session=None,
                      timeout: int = 25) -> pd.DataFrame:
    """Daily bars for one Yahoo symbol. Returns columns close (adjusted), raw_close, volume."""
    import requests
    session = session or requests.Session()
    t0 = int(pd.Timestamp(start).timestamp())
    t1 = int(pd.Timestamp(end or pd.Timestamp.utcnow().tz_localize(None)).timestamp()) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": t0, "period2": t1, "interval": "1d", "events": "div,splits",
              "includeAdjustedClose": "true"}
    r = session.get(url, params=params, headers={"User-Agent": "nse-ai-research/0.1"}, timeout=timeout)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=["close", "raw_close", "volume"])
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    idx = (pd.to_datetime(ts, unit="s", utc=True).tz_convert("Africa/Nairobi")
           .normalize().tz_localize(None))
    df = pd.DataFrame({"close": adj, "raw_close": q["close"], "volume": q["volume"]}, index=idx)
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    return df


def collect_yahoo(universe: pd.DataFrame, csv_dir: str | Path, start: str = "2012-01-01",
                  fx_symbol: str | None = "KES=X", pause: float = 0.4) -> dict:
    """Download every ticker in the universe into the CSV store. Returns a status report."""
    import requests
    out = Path(csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    report = {"ok": [], "failed": {}}
    for tkr, row in universe.iterrows():
        sym = row.get("yahoo", f"{tkr}.NR")
        try:
            df = fetch_yahoo_chart(sym, start, session=sess)
            if len(df) < 60:
                report["failed"][tkr] = f"only {len(df)} rows"
                continue
            df.index.name = "date"
            df[["close", "volume"]].to_csv(out / f"{tkr}.csv")
            report["ok"].append(tkr)
        except Exception as exc:  # network, 404, parse ...
            report["failed"][tkr] = f"{type(exc).__name__}: {exc}"[:120]
        time.sleep(pause)
    if fx_symbol:
        try:
            fx = fetch_yahoo_chart(fx_symbol, start, session=sess)
            fx = fx[["close"]].rename(columns={"close": "value"})
            fx["series"] = "usdkes"
            fx.index.name = "date"
            fx.reset_index().to_csv(out.parent / "macro_usdkes.csv", index=False)
            report["fx"] = "ok"
        except Exception as exc:
            report["fx"] = f"failed: {type(exc).__name__}"
    return report


def find_yahoo_symbols(universe: pd.DataFrame, session=None, pause: float = 0.3) -> pd.DataFrame:
    """Ask Yahoo's search endpoint which symbols exist for each company name (discovery only)."""
    import requests
    sess = session or requests.Session()
    rows = []
    for tkr, r in universe.iterrows():
        for q in (r["name"], tkr):
            try:
                resp = sess.get("https://query2.finance.yahoo.com/v1/finance/search",
                                params={"q": q, "quotesCount": 6, "newsCount": 0},
                                headers={"User-Agent": "nse-ai-research/0.1"}, timeout=20)
                resp.raise_for_status()
                for h in resp.json().get("quotes", []):
                    rows.append({"ticker": tkr, "query": q, "symbol": h.get("symbol"),
                                 "name": h.get("shortname") or h.get("longname"), "exchange": h.get("exchDisp") or h.get("exchange")})
            except Exception as exc:
                rows.append({"ticker": tkr, "query": q, "symbol": None, "name": f"{type(exc).__name__}", "exchange": None})
            time.sleep(pause)
    return pd.DataFrame(rows)


def import_prices(src: str | Path, out_dir: str | Path, ticker: str | None = None, price_col: str | None = None) -> dict:
    """Normalise price files you downloaded elsewhere into the CSV store (merging across files).

    * a single-stock file  -> pass ``ticker`` (or it is taken from the file name)
    * a long multi-stock file with columns date, stock code/ticker/symbol, price, volume
      (this is the layout of the public Mendeley "NSE Kenya All Stocks Prices" files)
    * a folder of either; several files for the same ticker (e.g. one per year) are joined.
    ``price_col`` forces a specific price column (e.g. ``adjusted_price``).
    """
    src, out = Path(src), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    frames: dict[str, list] = {}
    notes: list[str] = []
    for f in files:
        try:
            raw = pd.read_csv(f)
            raw.columns = _clean_cols(raw.columns)
            tcol = next((c for c in TICKER_COLS if c in raw.columns), None)
            if tcol and ticker is None:
                if {"day_finalprice", "adjusted_price"} <= set(raw.columns):
                    a_, b_ = _num(raw["day_finalprice"]), _num(raw["adjusted_price"])
                    share = float(((a_ - b_).abs() > 1e-6).mean())
                    notes.append(f"{f.name}: 'Adjusted price' differs from 'Day FinalPrice' on {share:.0%} of rows "
                                 f"(default uses Day FinalPrice; use --price-col adjusted_price to switch)")
                for t, g in raw.groupby(tcol):
                    d = _parse_price_df(g.drop(columns=[tcol]), price_col, f.name)
                    if len(d):
                        frames.setdefault(str(t).strip().upper(), []).append(d)
            else:
                d = _parse_price_df(raw, price_col, f.name)
                frames.setdefault((ticker or f.stem).strip().upper(), []).append(d)
        except Exception as exc:
            notes.append(f"{f.name}: {exc}")
    done = []
    for t, parts in frames.items():
        path = out / f"{t}.csv"
        if path.exists():
            parts = [_read_price_csv(path)] + parts            # keep what is already in the store
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if len(d) >= 60:
            d.index.name = "date"
            d.to_csv(path)
            done.append(t)
        else:
            notes.append(f"{t}: only {len(d)} usable rows, not saved")
    return {"imported": sorted(done), "notes": notes}


# --------------------------------------------------------------------------------------
# CSV store
# --------------------------------------------------------------------------------------
def _num(series: pd.Series) -> pd.Series:
    """Parse numbers like '1,234.5', '1.2M', '350K', '-' ."""
    t = series.astype(str).str.replace(",", "").str.strip()
    mult = t.str[-1].map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    base = pd.to_numeric(t.str.rstrip("KMBkmb"), errors="coerce")
    return base * mult


CLOSE_PRIORITY = ("adj_close", "adjclose", "close", "day_finalprice", "day_final_price", "price",
                  "closing_price", "adjusted_price")
TICKER_COLS = ("ticker", "code", "symbol", "stock", "stock_code")
VOL_COLS = ("volume", "vol", "shares_traded", "volume_traded")


def _clean_cols(cols) -> list[str]:
    return [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in cols]


def _parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if d.isna().mean() > 0.2:                                  # maybe day-first like 02/01/2013
        d2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if d2.isna().mean() < d.isna().mean():
            d = d2
    return d


def _parse_price_df(df: pd.DataFrame, price_col: str | None = None, name: str = "") -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_cols(df.columns)
    dcol = next((c for c in ("date", "day", "trade_date") if c in df.columns), df.columns[0])
    df[dcol] = _parse_dates(df[dcol])
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    want = _clean_cols([price_col])[0] if price_col else None
    ccol = want if want in df.columns else (None if want else next((c for c in CLOSE_PRIORITY if c in df.columns), None))
    if ccol is None:
        raise ValueError(f"{name}: no price column found (have: {', '.join(df.columns)})")
    vcol = next((c for c in VOL_COLS if c in df.columns), None)
    out = pd.DataFrame({"close": _num(df[ccol])})
    out["volume"] = _num(df[vcol]).fillna(0) if vcol else np.nan
    out = out[~out.index.duplicated(keep="last")]
    return out[out["close"] > 0]


def _read_price_csv(path: Path, price_col: str | None = None) -> pd.DataFrame:
    return _parse_price_df(pd.read_csv(path), price_col, Path(path).name)


def load_csv_store(csv_dir: str | Path, universe: pd.DataFrame | None = None) -> MarketData:
    csv_dir = Path(csv_dir)
    closes, vols, notes = {}, {}, []
    for p in sorted(csv_dir.glob("*.csv")):
        tkr = p.stem.upper()
        if universe is not None and tkr not in universe.index:
            continue
        try:
            d = _read_price_csv(p)
        except Exception as exc:
            notes.append(f"skipped {p.name}: {exc}")
            continue
        if len(d) < 60:
            notes.append(f"skipped {tkr}: only {len(d)} rows")
            continue
        closes[tkr], vols[tkr] = d["close"], d["volume"]
    if not closes:
        raise RuntimeError(f"No usable price files in {csv_dir}. Run `collect`, or drop CSVs there, "
                           f"or use `demo` for the synthetic market.")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(vols).reindex(close.index)
    meta = pd.DataFrame(index=close.columns, columns=["name", "sector"])
    if universe is not None:
        meta["name"] = universe["name"].reindex(close.columns)
        meta["sector"] = universe["sector"].reindex(close.columns)
    meta = meta.fillna({"name": "", "sector": "Other"})
    if volume.isna().all().all():
        notes.append("no volume column in any file -> liquidity filters degrade to price-only")
    return MarketData(close=close, volume=volume.fillna(0.0) if not volume.isna().all().all() else volume,
                      meta=meta, notes=notes)


# --------------------------------------------------------------------------------------
# optional alternative data (all with publication lags -> no look-ahead)
# --------------------------------------------------------------------------------------
DEFAULT_LAG_DAYS = {"usdkes": 0, "tbill91": 1, "cbk_rate": 1, "inflation": 20, "gdp_growth": 60}


def load_macro(paths: list[str | Path], index: pd.DatetimeIndex, lags: dict | None = None) -> pd.DataFrame | None:
    frames = []
    for p in paths:
        p = Path(p)
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["date"]))
    if not frames:
        return None
    df = pd.concat(frames)
    lags = {**DEFAULT_LAG_DAYS, **(lags or {})}
    cols = {}
    for name, g in df.groupby("series"):
        s = g.set_index("date")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.index = s.index + pd.Timedelta(days=int(lags.get(name, 1)))   # only known after publication
        cols[name] = s.reindex(s.index.union(index)).ffill().reindex(index)
    return pd.DataFrame(cols, index=index)


def load_fundamentals(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["period_end", "publish_date"])
    df["ticker"] = df["ticker"].str.upper()
    return df.sort_values(["ticker", "period_end"])


def fundamentals_daily(fund: pd.DataFrame, price: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Point-in-time valuation/quality/growth frames aligned to ``price`` (T x N).

    A record only becomes visible on its ``publish_date`` -- never on ``period_end``.
    """
    idx, cols = price.index, price.columns
    raw = {k: pd.DataFrame(np.nan, index=idx, columns=cols) for k in ("eps", "dps", "bvps", "eps_g", "rev_g")}
    for tkr, g in fund.groupby("ticker"):
        if tkr not in cols:
            continue
        g = g.sort_values("period_end").copy()
        g["eps_g"] = g["eps"].pct_change(fill_method=None) if "eps" in g else np.nan
        g["rev_g"] = g["revenue"].pct_change(fill_method=None) if "revenue" in g else np.nan
        g = g.sort_values("publish_date")
        for k in raw:
            if k in g.columns:
                s = g.set_index("publish_date")[k]
                s = s[~s.index.duplicated(keep="last")]
                raw[k][tkr] = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    eps, dps, bv = raw["eps"], raw["dps"], raw["bvps"]
    safe_px = price.where(price > 0)
    return {
        "ey": (eps / safe_px), "dy": (dps / safe_px), "bp": (bv / safe_px),
        "roe": (eps / bv.where(bv > 0)), "eps_g": raw["eps_g"].clip(-1, 3), "rev_g": raw["rev_g"].clip(-1, 3),
    }


_POS = set("profit profits surge surges rise rises rose growth gain gains record dividend dividends upgrade "
           "beat beats strong expands expansion bullish rally recover recovery boost wins awarded".split())
_NEG = set("loss losses fall falls fell drop drops decline declines slump warning warns fraud probe suspend "
           "suspended downgrade weak cut cuts bearish default debt crisis lawsuit penalty fine fined".split())


def lexicon_score(text: str) -> float:
    w = re.findall(r"[a-z]+", text.lower())
    p, n = sum(x in _POS for x in w), sum(x in _NEG for x in w)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def collect_news_sentiment(universe: pd.DataFrame, out_csv: str | Path, session=None) -> dict:
    """Optional: headline sentiment from Google News RSS (lexicon-based, crude). Fail-soft.

    Headlines are stamped with their publication date; the model only sees a trailing average.
    """
    import requests
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    sess = session or requests.Session()
    rows, failed = [], []
    for tkr, r in universe.iterrows():
        q = f"{r['name']} Kenya stock NSE".replace(" ", "+")
        try:
            resp = sess.get(f"https://news.google.com/rss/search?q={q}&hl=en-KE&gl=KE&ceid=KE:en",
                            headers={"User-Agent": "nse-ai-research/0.1"}, timeout=20)
            resp.raise_for_status()
            for it in ET.fromstring(resp.content).iter("item"):
                title = it.findtext("title") or ""
                dt = parsedate_to_datetime(it.findtext("pubDate")).date()
                rows.append({"date": dt, "ticker": tkr, "score": lexicon_score(title)})
        except Exception as exc:
            failed.append(f"{tkr}:{type(exc).__name__}")
    if rows:
        new = pd.DataFrame(rows)
        p = Path(out_csv)
        if p.exists():
            new = pd.concat([pd.read_csv(p), new]).drop_duplicates()
        new.to_csv(p, index=False)
    return {"headlines": len(rows), "failed": failed}


def load_sentiment(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["ticker"] = df["ticker"].str.upper()
    return df


def sentiment_daily(sent: pd.DataFrame, price: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    piv = sent.pivot_table(index="date", columns="ticker", values="score", aggfunc="mean")
    piv = piv.reindex(piv.index.union(price.index)).reindex(columns=price.columns)
    # headline on day d is usable from d+1 close onward
    return piv.rolling(window, min_periods=1).mean().shift(1).reindex(price.index)


# --------------------------------------------------------------------------------------
# synthetic NSE-like market (demo/tests)
# --------------------------------------------------------------------------------------
def make_synthetic_nse(n_stocks: int = 40, years: float = 12, seed: int = 1, signal: bool = True,
                       signal_strength: float = 0.10, end: str = "2026-10-30") -> MarketData:
    """Simulated market with NSE-style frictions: market regimes, sector factors, thin trading
    (stale prices / zero-volume days), wide idiosyncratic risk.

    signal=True plants a *slowly drifting per-stock alpha* (so momentum-type features carry real
    information); signal=False is a pure placebo where nothing is predictable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=int(years * 252))
    T, N = len(dates), n_stocks
    # market regimes
    state = np.zeros(T, int)
    for t in range(1, T):
        state[t] = state[t - 1] if rng.random() > 0.01 else rng.integers(0, 3)
    mu_m = np.array([0.0007, -0.0008, 0.0001])[state]
    sd_m = np.array([0.007, 0.013, 0.008])[state]
    mkt = mu_m + sd_m * rng.standard_normal(T)
    n_sec = 6
    sec = rng.integers(0, n_sec, N)
    sec_ret = 0.006 * rng.standard_normal((T, n_sec))
    beta = rng.uniform(0.5, 1.4, N)
    idio = rng.uniform(0.011, 0.028, N)
    alpha = np.zeros((T, N))
    if signal:
        a = rng.standard_normal(N) * signal_strength / 252
        for t in range(1, T):
            a = 0.994 * a + np.sqrt(1 - 0.994 ** 2) * signal_strength / 252 * rng.standard_normal(N)
            alpha[t] = a
    latent = beta * mkt[:, None] + sec_ret[:, sec] + idio * rng.standard_normal((T, N)) + alpha
    latent = np.clip(latent, -0.25, 0.25)
    # thin trading: probability of trading each day
    p_trade = np.clip(rng.beta(2.2, 1.0, N) * 1.05, 0.25, 1.0)
    p_trade[: N // 3] = np.clip(p_trade[: N // 3] + 0.3, 0, 1)
    traded = rng.random((T, N)) < p_trade
    lat_px = 20 * np.exp(np.cumsum(latent, axis=0)) * rng.uniform(0.5, 6, N)
    obs = np.where(traded, lat_px, np.nan)
    obs = pd.DataFrame(obs, index=dates).ffill().to_numpy()
    base_vol = np.exp(rng.normal(11.5, 1.1, N))             # shares per trade day
    vol = np.where(traded, base_vol * np.exp(0.6 * rng.standard_normal((T, N))), 0.0)
    # listing/delisting staggering
    first = np.where(rng.random(N) < 0.25, rng.integers(0, int(T * 0.35), N), 0)
    last = np.where(rng.random(N) < 0.08, rng.integers(int(T * 0.6), T - 100, N), T)
    close = pd.DataFrame(obs, index=dates, columns=[f"S{i:02d}" for i in range(N)])
    volume = pd.DataFrame(vol, index=dates, columns=close.columns)
    for j in range(N):
        close.iloc[: first[j], j] = np.nan
        close.iloc[last[j]:, j] = np.nan
        volume.iloc[: first[j], j] = 0
        volume.iloc[last[j]:, j] = 0
    fx = pd.Series(130 * np.exp(np.cumsum(0.0002 + 0.003 * rng.standard_normal(T))), index=dates)
    macro = pd.DataFrame({"usdkes": fx})
    sec_names = ["Banking", "Manufacturing", "Energy", "Insurance", "Commercial", "Agricultural"]
    meta = pd.DataFrame({"name": [f"Synthetic {c}" for c in close.columns],
                         "sector": [sec_names[s] for s in sec]}, index=close.columns)
    return MarketData(close=close, volume=volume, meta=meta, macro=macro, is_synthetic=True,
                      notes=[f"SYNTHETIC market (signal={'planted' if signal else 'none/placebo'}) -- "
                             "not real NSE data"])
