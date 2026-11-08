"""Feature engineering + the monthly decision panel.

Rule: a value stamped at decision date t may only use information available at the close
of t.  Targets (period returns, forward returns) are kept in separate arrays and only
read by the evaluator.  ``tests/test_no_lookahead.py`` enforces this by corrupting the
future and checking that nothing at or before t changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import MarketData, fundamentals_daily, sentiment_daily

BASE_FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252_21",         # trend / momentum
    "rev_5",                                              # short-term reversal
    "vol_63", "dvol_63", "idio_vol_126", "beta_252",      # risk
    "dist_52w_high", "trend_200", "ma_50_200", "rsi_14",  # technical
    "rel_mom_126",                                        # strength vs NSE universe
    "ln_value_63", "zero_frac_63", "ln_amihud_63",        # liquidity (very important on NSE)
    "fx_beta_252",                                        # USD/KES sensitivity
    "ey", "bp", "dy", "roe", "eps_g", "rev_g",            # fundamentals (point-in-time, optional)
    "sent_30",                                            # news sentiment (optional)
]
ML_FEATURE = "ml_pred"                                    # walk-forward ML forecast, filled during learning
ALL_FEATURES = BASE_FEATURES + [ML_FEATURE]

FEATURE_DOC = {
    "mom_21": "1-month momentum", "mom_63": "3-month momentum", "mom_126": "6-month momentum",
    "mom_252_21": "12-1 month momentum", "rev_5": "1-week move (reversal)",
    "vol_63": "3-month volatility", "dvol_63": "downside volatility", "idio_vol_126": "idiosyncratic risk",
    "beta_252": "beta to NSE universe", "dist_52w_high": "distance from 52-week high",
    "trend_200": "price vs 200-day average", "ma_50_200": "50/200-day trend", "rsi_14": "RSI(14)",
    "rel_mom_126": "6-month strength vs market", "ln_value_63": "traded value (liquidity)",
    "zero_frac_63": "no-trade days (illiquidity)", "ln_amihud_63": "price impact (Amihud)",
    "fx_beta_252": "USD/KES sensitivity", "ey": "earnings yield", "bp": "book-to-price",
    "dy": "dividend yield", "roe": "return on equity", "eps_g": "EPS growth", "rev_g": "revenue growth",
    "sent_30": "news sentiment", "ml_pred": "machine-learning forecast",
}


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list
    rb: np.ndarray               # row index (in dates) of each decision date (month-ends + latest day)
    ex: np.ndarray               # execution row: next session close
    rb_dates: pd.DatetimeIndex
    pr: np.ndarray               # R x N realised period return exec_r -> exec_{r+1}   (TARGET)
    cash: np.ndarray             # R  cash return over the same period
    cash_rate: np.ndarray        # R  annual cash rate known at decision date
    elig: np.ndarray             # R x N investable at decision date
    cube: np.ndarray             # F x R x N cross-sectionally rank-normalised features in [-1,1]
    feat_names: list
    macro: np.ndarray            # R x M scalar market-level state at decision date
    macro_names: list
    mkt_trend: np.ndarray        # R  6-month EW-universe return at decision date
    inv_vol: np.ndarray          # R x N
    bench_w: np.ndarray          # R x N equal weight of investable names
    fwd3: np.ndarray             # R x N realised 3-period forward return (TARGET)
    vol_ann: np.ndarray          # R x N
    adv_value: np.ndarray        # R x N median daily traded value (KES) 63d
    last_price: np.ndarray       # N latest price
    meta: pd.DataFrame = None
    notes: list = field(default_factory=list)
    ml_idx: int = -1
    px_df: pd.DataFrame = None

    @property
    def R(self): return len(self.rb)

    @property
    def N(self): return len(self.tickers)

    @property
    def F(self): return len(self.feat_names)

    def known_end(self, r0: int) -> int:
        """Exclusive end row of period returns that are realised when deciding at row r0."""
        live = self.ex[r0] <= self.rb[r0]       # latest-day row: its own exec does not exist yet
        return r0 if live else r0 - 1


def _rank_norm(frame: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    z = (frame.where(mask).rank(axis=1, pct=True) - 0.5) * 2.0
    return z.fillna(0.0).to_numpy(dtype=np.float32)


def compute_daily_features(md: MarketData, cfg: dict) -> tuple[dict, dict]:
    """Return (feature frames T x N, aux frames)."""
    m = cfg["market"]
    px_raw = md.close.sort_index()
    px = px_raw.ffill(limit=int(cfg["data"].get("max_stale_days", 10)))
    has_vol = not md.volume.isna().all().all()
    vol = md.volume.reindex(px.index).fillna(0.0) if has_vol else pd.DataFrame(1.0, index=px.index, columns=px.columns)
    ret = px / px.shift(1) - 1.0
    ret = ret.clip(-0.5, 0.5)
    mkt = ret.mean(axis=1, skipna=True).fillna(0.0)
    mkt_idx = (1 + mkt).cumprod()

    F: dict[str, pd.DataFrame] = {}
    F["mom_21"] = px / px.shift(21) - 1
    F["mom_63"] = px / px.shift(63) - 1
    F["mom_126"] = px / px.shift(126) - 1
    F["mom_252_21"] = px.shift(21) / px.shift(252) - 1
    F["rev_5"] = px / px.shift(5) - 1
    F["vol_63"] = ret.rolling(63, min_periods=40).std() * np.sqrt(252)
    F["dvol_63"] = np.sqrt((ret.clip(upper=0) ** 2).rolling(63, min_periods=40).mean()) * np.sqrt(252)
    var_m = mkt.rolling(252, min_periods=126).var()
    cov = ret.rolling(252, min_periods=126).cov(mkt)
    beta = cov.div(var_m, axis=0)
    F["beta_252"] = beta
    var_i = ret.rolling(126, min_periods=80).var()
    var_m126 = mkt.rolling(126, min_periods=80).var()
    F["idio_vol_126"] = np.sqrt((var_i - (beta ** 2).mul(var_m126, axis=0)).clip(lower=0)) * np.sqrt(252)
    F["dist_52w_high"] = px / px.rolling(252, min_periods=120).max() - 1
    ma200 = px.rolling(200, min_periods=120).mean()
    F["trend_200"] = px / ma200 - 1
    F["ma_50_200"] = px.rolling(50, min_periods=30).mean() / ma200 - 1
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    F["rsi_14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50) / 50
    F["rel_mom_126"] = F["mom_126"].sub(mkt_idx / mkt_idx.shift(126) - 1, axis=0)

    value = px * vol
    med_val = value.rolling(63, min_periods=30).median()
    zero_frac = (vol <= 0).astype(float).rolling(63, min_periods=30).mean()
    amihud = (ret.abs() / value.replace(0, np.nan)).rolling(63, min_periods=20).mean()
    F["ln_value_63"] = np.log1p(med_val)
    F["zero_frac_63"] = zero_frac
    F["ln_amihud_63"] = np.log(amihud * 1e6 + 1e-9)

    nan_frame = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    fx_series = md.macro["usdkes"] if (md.macro is not None and "usdkes" in md.macro) else None
    if fx_series is not None:
        fx_ret = (fx_series / fx_series.shift(1) - 1).reindex(px.index)
        F["fx_beta_252"] = ret.rolling(252, min_periods=126).cov(fx_ret).div(
            fx_ret.rolling(252, min_periods=126).var(), axis=0)
    else:
        F["fx_beta_252"] = nan_frame

    if md.fundamentals is not None:
        F.update(fundamentals_daily(md.fundamentals, px))
    for k in ("ey", "bp", "dy", "roe", "eps_g", "rev_g"):
        F.setdefault(k, nan_frame)
    F["sent_30"] = sentiment_daily(md.sentiment, px) if md.sentiment is not None else nan_frame

    hist = px_raw.notna().cumsum()
    traded_recent = (vol > 0).astype(float).rolling(10, min_periods=1).max() > 0 if has_vol else px.notna()
    elig = (px.notna() & (hist >= m["min_history_days"]) & traded_recent)
    if has_vol:
        elig &= (med_val >= m["min_median_value_kes"]) & (zero_frac <= m["max_zero_volume_frac"])
    aux = {"px": px, "ret": ret, "mkt_idx": mkt_idx, "elig": elig, "med_val": med_val,
           "vol63": F["vol_63"], "has_vol": has_vol}
    return F, aux


def build_panel(md: MarketData, cfg: dict) -> Panel:
    m = cfg["market"]
    F, aux = compute_daily_features(md, cfg)
    px, elig_d = aux["px"], aux["elig"]
    dates, tickers = px.index, list(px.columns)
    T, N = px.shape

    # decision dates: last session of each month + the latest session
    s = pd.Series(np.arange(T), index=dates)
    rb = s.groupby(dates.to_period("M")).last().to_numpy()
    if rb[-1] != T - 1:
        rb = np.append(rb, T - 1)
    ex = np.minimum(rb + 1, T - 1)
    R = len(rb)

    P = px.to_numpy()
    pr = np.full((R, N), np.nan)
    cash_rate = np.full(R, float(m["cash_annual_rate"]))
    if md.macro is not None and "tbill91" in md.macro:
        tb = md.macro["tbill91"].reindex(dates).ffill().to_numpy()[rb]
        cash_rate = np.where(np.isfinite(tb), tb / (100.0 if np.nanmedian(tb) > 1 else 1.0), cash_rate)
    cash = np.full(R, np.nan)
    for r in range(R - 1):
        a, b = ex[r], ex[r + 1]
        if b > a:
            pr[r] = P[b] / P[a] - 1.0
            cash[r] = (1 + cash_rate[r]) ** ((b - a) / 252.0) - 1.0
    cash[R - 1] = 0.0
    pr = np.where(np.isfinite(pr), np.clip(pr, -0.95, 3.0), np.nan)

    fwd3 = np.full((R, N), np.nan)
    for r in range(R - 3):
        blk = 1.0 + pr[r:r + 3]
        fwd3[r] = np.where(np.isfinite(blk).all(axis=0), blk.prod(axis=0) - 1.0, np.nan)

    elig = elig_d.to_numpy()[rb]
    cube = np.zeros((len(ALL_FEATURES), R, N), dtype=np.float32)
    for i, name in enumerate(BASE_FEATURES):
        fr = F[name].iloc[rb]
        cube[i] = _rank_norm(fr, pd.DataFrame(elig, index=fr.index, columns=fr.columns))

    mkt_idx = aux["mkt_idx"]
    trend = (mkt_idx / mkt_idx.shift(126) - 1).to_numpy()[rb]
    mvol = (aux["ret"].mean(axis=1).rolling(63, min_periods=40).std() * np.sqrt(252)).to_numpy()[rb]
    fx_mom = np.zeros(R)
    macro_names = ["mkt_trend_126", "mkt_vol_63", "fx_mom_63", "cash_rate"]
    if md.macro is not None and "usdkes" in md.macro:
        fx = md.macro["usdkes"].reindex(dates).ffill()
        fx_mom = (fx / fx.shift(63) - 1).to_numpy()[rb]
    macro = np.column_stack([trend, mvol, fx_mom, cash_rate])
    macro = np.nan_to_num(macro, nan=0.0)
    trend = np.nan_to_num(trend, nan=0.0)

    vola = aux["vol63"].to_numpy()[rb]
    inv_vol = np.where(np.isfinite(vola) & (vola > 0.02), 1.0 / np.maximum(vola, 0.02), 0.0)
    bench_w = elig / np.maximum(elig.sum(1, keepdims=True), 1)
    adv = aux["med_val"].to_numpy()[rb] if aux["has_vol"] else np.full((R, N), np.nan)
    notes = list(md.notes)
    if not aux["has_vol"]:
        notes.append("Volume missing: liquidity filters disabled -- results will be optimistic.")
    if md.fundamentals is None:
        notes.append("No fundamentals file: value/quality/growth features inactive.")
    if md.sentiment is None:
        notes.append("No sentiment file: news-sentiment feature inactive.")
    if md.macro is None or "usdkes" not in md.macro:
        notes.append("No USD/KES series: FX features inactive.")
    return Panel(dates=dates, tickers=tickers, rb=rb, ex=ex, rb_dates=dates[rb], pr=pr, cash=cash,
                 cash_rate=cash_rate, elig=elig, cube=cube, feat_names=list(ALL_FEATURES), macro=macro,
                 macro_names=macro_names, mkt_trend=trend, inv_vol=inv_vol, bench_w=bench_w, fwd3=fwd3,
                 vol_ann=vola, adv_value=adv, last_price=P[-1], meta=md.meta, notes=notes,
                 ml_idx=ALL_FEATURES.index(ML_FEATURE), px_df=px)
