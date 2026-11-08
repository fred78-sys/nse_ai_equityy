"""Strategy genomes, the vectorised backtest engine and the fitness function.

A strategy ("genome") is a signed weighting of cross-sectional features plus portfolio
rules (how many stocks, how often to change, sizing, and a market-regime exposure switch).
Genomes are cheap to evaluate, which lets the evolution engine test thousands of them
against history.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .features import ALL_FEATURES, FEATURE_DOC, Panel

F_ALL = len(ALL_FEATURES)
ML_IDX = ALL_FEATURES.index("ml_pred")
TOP_N_CHOICES = [4, 5, 6, 8, 10, 12, 15]
HOLD_CHOICES = [1, 2, 3]
REGIME_THR = [np.nan, -0.10, -0.05, 0.0]       # nan = always invested
BEAR_EXPO = [0.0, 0.5]


@dataclass
class Genome:
    w: np.ndarray = field(default_factory=lambda: np.zeros(F_ALL))
    top_n: int = 8
    hold_k: int = 1
    weighting: str = "equal"                    # equal | invvol
    regime_thr: float = np.nan
    bear_exposure: float = 0.0

    def copy(self) -> "Genome":
        return Genome(self.w.copy(), self.top_n, self.hold_k, self.weighting, self.regime_thr, self.bear_exposure)

    @property
    def n_active(self) -> int:
        return int((np.abs(self.w) > 1e-9).sum())

    def key(self) -> str:
        t = (tuple(np.round(self.w / max(np.abs(self.w).max(), 1e-9), 1)), self.top_n, self.hold_k,
             self.weighting, None if np.isnan(self.regime_thr) else self.regime_thr, self.bear_exposure)
        return hashlib.md5(repr(t).encode()).hexdigest()[:10]

    def describe(self) -> str:
        idx = np.argsort(-np.abs(self.w))[:3]
        parts = [f"{'+' if self.w[i] > 0 else '-'}{ALL_FEATURES[i]}" for i in idx if abs(self.w[i]) > 1e-9]
        reg = "" if np.isnan(self.regime_thr) else f" | regime>{self.regime_thr:+.2f}->{self.bear_exposure:.0%}"
        return f"{' '.join(parts) or 'neutral'} | top{self.top_n} hold{self.hold_k}m {self.weighting}{reg}"

    def to_dict(self) -> dict:
        return {"w": {ALL_FEATURES[i]: float(v) for i, v in enumerate(self.w) if abs(v) > 1e-9},
                "top_n": self.top_n, "hold_k": self.hold_k, "weighting": self.weighting,
                "regime_thr": None if np.isnan(self.regime_thr) else float(self.regime_thr),
                "bear_exposure": self.bear_exposure}

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        w = np.zeros(F_ALL)
        for k, v in d["w"].items():
            if k in ALL_FEATURES:
                w[ALL_FEATURES.index(k)] = v
        return Genome(w, int(d["top_n"]), int(d["hold_k"]), d["weighting"],
                      np.nan if d["regime_thr"] is None else float(d["regime_thr"]), float(d["bear_exposure"]))


def _g(**kw) -> Genome:
    w = np.zeros(F_ALL)
    extra = {k: kw.pop(k) for k in list(kw) if k in ("top_n", "hold_k", "weighting", "regime_thr", "bear_exposure")}
    for k, v in kw.items():
        w[ALL_FEATURES.index(k)] = v
    return Genome(w, **extra)


# classic, hand-written ideas the system starts from (it will test them like everything else)
SEED_STRATEGIES = {
    "momentum_12_1": _g(mom_252_21=1.0, top_n=8),
    "momentum_6m": _g(mom_126=1.0, top_n=8),
    "trend_following": _g(trend_200=1.0, ma_50_200=0.7, top_n=8, regime_thr=-0.05, bear_exposure=0.0),
    "low_volatility": _g(vol_63=-1.0, dvol_63=-0.5, top_n=8, weighting="invvol"),
    "reversal_1w": _g(rev_5=-1.0, mom_21=-0.5, top_n=8),
    "breakout_52w": _g(dist_52w_high=1.0, mom_63=0.4, top_n=8),
    "liquid_quality_trend": _g(ln_value_63=0.6, zero_frac_63=-0.4, mom_126=0.5, top_n=10),
    "value_income": _g(dy=1.0, ey=0.8, bp=0.6, top_n=10, hold_k=3),
    "relative_strength": _g(rel_mom_126=1.0, beta_252=-0.3, top_n=8),
    "sentiment_momentum": _g(sent_30=1.0, mom_63=0.5, top_n=8),
    "ml_forecast": _g(ml_pred=1.0, top_n=8),
}


# --------------------------------------------------------------------------------------
# weights + returns
# --------------------------------------------------------------------------------------
def genome_weights(g: Genome, P: Panel, nrows: int | None = None, max_w: float = 0.25) -> np.ndarray:
    """Target portfolio weights (rows x N) at each decision date. Row r uses data <= decision date r."""
    nrows = P.R if nrows is None else nrows
    cube = P.cube[:, :nrows, :]
    S = np.tensordot(g.w.astype(np.float32), cube, axes=(0, 0)).astype(np.float64)
    el = P.elig[:nrows]
    S = np.where(el, S, -np.inf)
    N = S.shape[1]
    kk = max(1, min(g.top_n, N))
    fin = np.isfinite(S)
    spread = np.where(fin, S, -np.inf).max(1) - np.where(fin, S, np.inf).min(1)
    neutral = ~np.isfinite(spread) | (spread < 1e-9)          # no information -> index sleeve
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    ok = np.take_along_axis(fin, idx, 1)
    val = ok.astype(float)
    if g.weighting == "invvol":
        val = val * np.take_along_axis(P.inv_vol[:nrows], idx, 1)
    ssum = val.sum(1, keepdims=True)
    bad = (ssum[:, 0] <= 0)
    val = np.divide(val, np.where(ssum > 0, ssum, 1.0))
    W = np.zeros((nrows, N))
    np.put_along_axis(W, idx, val, 1)
    W = np.minimum(W, max_w)
    nb = neutral | bad
    W[nb] = P.bench_w[:nrows][nb]
    if g.hold_k > 1:
        base = (np.arange(nrows) // g.hold_k) * g.hold_k
        W = W[base]
    if not np.isnan(g.regime_thr):
        expo = np.where(P.mkt_trend[:nrows] > g.regime_thr, 1.0, g.bear_exposure)
        W = W * expo[:, None]
    return W


def portfolio_returns(W: np.ndarray, P: Panel, cost_bps: float) -> np.ndarray:
    """Net return per period after turnover costs. Weight drift is ignored (conservative on turnover)."""
    n = W.shape[0]
    pr = np.nan_to_num(P.pr[:n], nan=0.0)
    gross = (W * pr).sum(1) + (1.0 - W.sum(1)) * P.cash[:n]
    turn = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(1)
    net = gross - turn * (cost_bps / 1e4)
    net[~np.isfinite(P.pr[:n]).any(1)] = np.nan
    return net


def turnover(W: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(1)


# --------------------------------------------------------------------------------------
# metrics + fitness
# --------------------------------------------------------------------------------------
def ann_stats(r: np.ndarray, ppy: int = 12) -> dict:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(n=len(r), cagr=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, total=np.nan)
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(eq) - 1
    sd = r.std(ddof=1)
    return dict(n=len(r), cagr=eq[-1] ** (ppy / len(r)) - 1, vol=sd * np.sqrt(ppy),
                sharpe=(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan, mdd=dd.min(), total=eq[-1] - 1)


def window_metrics(net, bench, cash) -> dict:
    ok = np.isfinite(net) & np.isfinite(bench)
    net, bench, cash = net[ok], bench[ok], cash[ok]
    n = len(net)
    if n < 6:
        return dict(n=n, sr=np.nan, ir=np.nan, mean_ex=np.nan, mdd=np.nan, frac=np.nan, years=n / 12)
    exc, exb = net - cash, net - bench
    sr = exc.mean() / (exc.std(ddof=1) + 1e-9) * np.sqrt(12)
    ir = exb.mean() / (exb.std(ddof=1) + 1e-9) * np.sqrt(12)
    eq = np.cumprod(1 + net)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    chunks = np.array_split(exb, 3)
    frac = float(np.mean([c.mean() > 0 for c in chunks if len(c)]))
    return dict(n=n, sr=sr, ir=ir, mean_ex=exb.mean(), std_ex=exb.std(ddof=1), mdd=mdd, frac=frac, years=n / 12)


def fitness(m: dict, g: Genome) -> float:
    """Risk-adjusted, benchmark-relative, consistency- and complexity-penalised score."""
    if not np.isfinite(m["sr"]) or m["n"] < 24:
        return -9.0
    return float(0.5 * m["sr"] + 0.5 * m["ir"] + 0.6 * (m["frac"] - 0.5)
                 - 0.04 * g.n_active - 2.0 * max(0.0, abs(m["mdd"]) - 0.35))


def passes_hurdle(m: dict, z: float) -> bool:
    """Multiple-testing guard: in-sample IR must clear z standard errors (~ z/sqrt(years))."""
    return bool(np.isfinite(m["ir"]) and m["n"] >= 24 and m["mean_ex"] > 0 and m["ir"] > z / np.sqrt(m["years"]))


def explain(g_list, weights, P: Panel, row: int, top: int = 3) -> list[list[tuple[str, float]]]:
    """Per-stock top feature contributions of the weighted ensemble at one decision row."""
    N = P.N
    contrib = np.zeros((F_ALL, N))
    for g, a in zip(g_list, weights):
        nrm = max(np.abs(g.w).sum(), 1e-9)
        contrib += a * (g.w / nrm)[:, None] * P.cube[:, row, :]
    out = []
    for i in range(N):
        idx = np.argsort(-contrib[:, i])[:top]
        out.append([(FEATURE_DOC[ALL_FEATURES[j]], float(contrib[j, i])) for j in idx if contrib[j, i] > 1e-6])
    return out
