"""Append-only forward ledger: every run logs its predictions; later runs score the matured ones.

This is the only *genuinely* out-of-sample evidence (nothing in it could have been tuned on the
outcome).  It is what should decide whether to trust the system with real money.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["as_of", "ticker", "action", "target_weight", "exp_3m_return", "exp_3m_excess_vs_market", "p_beat_tbill", "price"]


def log_predictions(df: pd.DataFrame, as_of: str, path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = df[[c for c in COLS if c != "as_of" and c in df.columns]].copy()
    new.insert(0, "as_of", as_of)
    if p.exists():
        old = pd.read_csv(p)
        if (old["as_of"] == as_of).any():
            return 0                                           # idempotent per date
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(p, index=False)
    return len(df)


def score_ledger(path: str | Path, px: pd.DataFrame, horizon_days: int = 63) -> dict:
    p = Path(path)
    if not p.exists():
        return {"resolved_dates": 0}
    led = pd.read_csv(p, parse_dates=["as_of"])
    out = []
    for d, g in led.groupby("as_of"):
        pos = px.index.searchsorted(d, side="right")           # execute next session
        if pos + horizon_days >= len(px):
            continue
        a, b = px.iloc[pos], px.iloc[pos + horizon_days]
        real = (b / a - 1)
        g = g.set_index("ticker")
        g["realized"] = real.reindex(g.index)
        mkt = real.dropna().mean()
        g["realized_excess"] = g["realized"] - mkt
        out.append(g.assign(as_of=d))
    if not out:
        return {"resolved_dates": 0, "note": "No prediction is old enough to be scored yet."}
    r = pd.concat(out).dropna(subset=["realized"])
    buys = r[r["action"].isin(["BUY / HOLD", "SMALL HOLD"])]
    ic = r.groupby("as_of").apply(lambda g: g["exp_3m_excess_vs_market"].corr(g["realized_excess"], method="spearman"),
                                  include_groups=False)
    return {"resolved_dates": int(r["as_of"].nunique()), "n": len(r),
            "buy_mean_excess": float(buys["realized_excess"].mean()) if len(buys) else np.nan,
            "buy_hit_rate_vs_market": float((buys["realized_excess"] > 0).mean()) if len(buys) else np.nan,
            "mean_rank_ic": float(ic.mean())}
