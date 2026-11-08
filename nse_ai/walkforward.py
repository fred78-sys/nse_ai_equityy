"""The self-learning loop: honest walk-forward replay of history.

At every re-learn date t the system
  1. fits the ML forecaster on labels that had *matured* before t,
  2. scores every strategy it already owns on data realised since that strategy was born,
     retiring the losers and promoting the winners,
  3. breeds new strategies (genetic search) on the trailing training window only,
  4. blends the survivors (weights from posterior evidence) into one portfolio,
  5. trades that portfolio on data it has NOT seen yet until the next re-learn date.

The stitched-together returns of step 5 are the only performance numbers that mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .evolution import Registry, evolve, pick_diverse, update_and_retire
from .features import ALL_FEATURES, Panel
from .strategies import (ML_IDX, SEED_STRATEGIES, Genome, ann_stats, genome_weights, portfolio_returns,
                         turnover)

F_BASE = [i for i in range(len(ALL_FEATURES)) if i != ML_IDX]


# --------------------------------------------------------------------------------------
# ML forecaster (walk-forward, matured labels only)
# --------------------------------------------------------------------------------------
def _xy(P: Panel, rows, with_y: bool):
    X, Y, idx = [], [], []
    for r in rows:
        el = P.elig[r]
        if with_y:
            y = P.fwd3[r]
            el = el & np.isfinite(y)
        if el.sum() < 6:
            continue
        x = np.hstack([P.cube[F_BASE, r, :][:, el].T, np.tile(P.macro[r], (el.sum(), 1))])
        X.append(x)
        idx.append((r, np.flatnonzero(el)))
        if with_y:
            yy = y[el] - np.mean(y[el])
            rk = (pd.Series(yy).rank(pct=True).to_numpy() - 0.5) * 2
            Y.append(rk)
    if not X:
        return np.empty((0, len(F_BASE) + P.macro.shape[1])), np.empty(0), []
    return np.vstack(X), (np.concatenate(Y) if with_y else None), idx


def ml_step(P: Panel, r0: int, r1: int, seed: int, min_rows: int = 600) -> bool:
    """Fit on labels matured before r0, write forecasts for decision rows [r0, r1) into the cube."""
    kend = P.known_end(r0)
    train_rows = range(0, max(kend - 2, 0))      # fwd3[r] needs pr[r..r+2] realised -> r <= kend-3
    X, Y, _ = _xy(P, train_rows, True)
    if len(Y) < min_rows:
        return False
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=60,
                                          l2_regularization=3.0, random_state=seed)
    model.fit(X, Y)
    for r in range(r0, r1):
        Xp, _, idx = _xy(P, [r], False)
        if not idx:
            continue
        pred = model.predict(Xp)
        z = (pd.Series(pred).rank(pct=True).to_numpy() - 0.5) * 2
        P.cube[ML_IDX, r, :] = 0.0
        P.cube[ML_IDX, r, idx[0][1]] = z
    return True


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def strategy_z(g: Genome, P: Panel, rows: slice) -> np.ndarray:
    S = np.tensordot(g.w.astype(np.float32), P.cube[:, rows, :], axes=(0, 0)).astype(float)
    el = P.elig[rows]
    df = pd.DataFrame(np.where(el, S, np.nan))
    return ((df.rank(axis=1, pct=True) - 0.5) * 2).fillna(0.0).to_numpy()


def block_bootstrap_p(x: np.ndarray, B: int = 3000, block: int = 3, seed: int = 0) -> float:
    """One-sided p-value for mean(x) > 0 using a stationary-style block bootstrap."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan
    rng = np.random.default_rng(seed)
    obs, xc = x.mean(), x - x.mean()
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n] % n
    return float((xc[idx].mean(1) >= obs).mean())


@dataclass
class WFResult:
    P: Panel
    reg: Registry
    r_start: int
    W_oos: np.ndarray
    comp: np.ndarray                  # composite z-score of the ensemble at each OOS decision row
    net: np.ndarray                   # OOS net return per period (NaN before r_start)
    bench_net: np.ndarray
    baselines: dict
    final: dict                       # state at the latest row: recs, weights, conviction
    curve: list
    metrics: dict
    ml_ok: bool


def run_walk_forward(P: Panel, cfg: dict, log=print) -> WFResult:
    L, mk = cfg["learning"], cfg["market"]
    cost = mk["one_way_cost_bps"]
    rng = np.random.default_rng(L["seed"])
    R, N = P.R, P.N
    bench_net = portfolio_returns(P.bench_w, P, cost)
    enough = np.flatnonzero(P.elig.sum(1) >= 8)
    if len(enough) == 0:
        raise RuntimeError("Fewer than 8 investable stocks at every date -- relax liquidity filters or add data.")
    r_first = int(enough[0])
    r_start = r_first + L["min_train_months"] + 1
    if r_start >= R - 2:
        raise RuntimeError(f"Not enough history: need > {L['min_train_months']} months of investable data.")
    epochs = sorted(set(list(range(r_start, R - 1, L["learn_every_months"])) + [R - 1]))
    reg = Registry()
    W_oos = np.zeros((R, N))
    W_shadow = P.bench_w.copy()          # strategy mix at FULL conviction (evidence for self-doubt, not traded)
    comp = np.zeros((R, N))
    curve, ml_ok_any = [], False
    final = {}
    for e, r0 in enumerate(epochs):
        r1 = epochs[e + 1] if e + 1 < len(epochs) else R
        date = P.rb_dates[r0]
        ml_ok = bool(L.get("use_ml", True)) and ml_step(P, r0, r1, L["seed"])
        ml_ok_any |= ml_ok
        update_and_retire(reg, P, bench_net, r0, cfg)
        kend = P.known_end(r0)
        lo = max(r_first, kend - L["train_months"])
        scored = evolve(P, bench_net, lo, kend, cfg, rng, seeds=[r.genome for r in reg.alive()],
                        exclude=reg.keys())
        winners = pick_diverse(scored, L["new_candidates_per_epoch"], L["hurdle_z"])
        for s in winners:
            reg.add(s, r0, date, cfg)
        if not winners:
            reg.log(date, "search", "-", f"no candidate cleared the hurdle (best train fit {scored[0].fit:.2f})")
        stats = update_and_retire(reg, P, bench_net, r0, cfg)
        alive = [reg.recs[i] for i in stats]
        irs = np.array([max(stats[r.id], 0.0) for r in alive])
        if len(alive) and irs.sum() > 0:
            a = irs / irs.sum()
            conv = float(np.clip(np.sort(irs)[::-1][:3].mean() / L["full_conviction_ir"], 0, 1))
        else:
            a, conv = np.zeros(len(alive)), 0.0
        # self-doubt: scale conviction by the learner's OWN realised out-of-sample record so far
        meta = 1.0
        n_oos = P.known_end(r0) - r_start
        if n_oos >= 12:
            so_far = portfolio_returns(W_shadow[:P.known_end(r0)], P, cost)[r_start:]
            exo = (so_far - bench_net[r_start:P.known_end(r0)])
            exo = exo[np.isfinite(exo)]
            ir_oos = exo.mean() / (exo.std(ddof=1) + 1e-12) * np.sqrt(12)
            meta = float(np.clip(0.5 + ir_oos, 0.15, 1.0))
        conv *= meta
        rows = slice(r0, r1)
        Wc = (1 - conv) * P.bench_w[rows].copy()
        Zc = np.zeros((r1 - r0, N))
        Wsh = (P.bench_w[rows].copy() if a.sum() <= 0 else np.zeros((r1 - r0, N)))
        for rec, ai in zip(alive, a):
            if ai <= 0:
                continue
            Ws = genome_weights(rec.genome, P, r1, mk["max_position_weight"])[rows]
            Wc += conv * ai * Ws
            Wsh += ai * Ws
            Zc += conv * ai * strategy_z(rec.genome, P, rows)
        W_shadow[rows] = Wsh
        band, min_w = float(L.get("rebalance_band", 0.0)), float(mk.get("min_position_weight", 0.0))
        if band > 0 or min_w > 0:                                   # no-trade band: ignore tiny, costly adjustments
            prev = W_oos[r0 - 1] if r0 > r_start else np.zeros(N)
            for i in range(Wc.shape[0]):
                w_ = np.where((np.abs(Wc[i] - prev) < band) & (Wc[i] > 0) & (prev > 0), prev, Wc[i])
                tot = w_.sum()
                w_ = np.where(w_ < min_w, 0.0, w_)             # no dust positions
                if w_.sum() > 0:
                    w_ = w_ * (tot / w_.sum())
                if w_.sum() > 1:
                    w_ = w_ / w_.sum()
                Wc[i], prev = w_, w_
        W_oos[rows], comp[rows] = Wc, Zc
        curve.append({"date": str(date)[:10], "alive": len(alive), "retired_total": sum(r.status == "retired" for r in reg.recs.values()),
                      "conviction": conv, "self_doubt": meta, "best_post_ir": float(max(r.post_ir for r in alive)) if alive else np.nan,
                      "new": len(winners)})
        if e % 4 == 0 or e == len(epochs) - 1:
            log(f"  epoch {e + 1}/{len(epochs)} {str(date)[:10]}: alive={len(alive)} conviction={conv:.0%} "
                f"new={len(winners)} retired_total={curve[-1]['retired_total']}")
        final = {"row": r0, "alive": alive, "a": a, "conviction": conv, "stats": stats}

    shadow_net = np.where(np.arange(R) >= r_start, portfolio_returns(W_shadow, P, cost), np.nan)
    full_net = portfolio_returns(W_oos, P, cost)
    net = np.where(np.arange(R) >= r_start, full_net, np.nan)
    baselines = {}
    for name, g in SEED_STRATEGIES.items():
        if g.w[ML_IDX] != 0:
            continue
        baselines[name] = np.where(np.arange(R) >= r_start, portfolio_returns(genome_weights(g, P, R, mk["max_position_weight"]), P, cost), np.nan)
    metrics = oos_metrics(P, net, bench_net, W_oos, r_start, baselines, cost)
    metrics["shadow"] = ann_stats(shadow_net[r_start:][np.isfinite(shadow_net[r_start:]) & np.isfinite(bench_net[r_start:])])
    return WFResult(P, reg, r_start, W_oos, comp, net, bench_net, baselines, final, curve, metrics, ml_ok_any)


def oos_metrics(P, net, bench_net, W, r_start, baselines, cost) -> dict:
    sl = slice(r_start, P.R)
    ok = np.isfinite(net[sl]) & np.isfinite(bench_net[sl])
    n, b, c = net[sl][ok], bench_net[sl][ok], P.cash[sl][ok]
    ex = n - b
    st = lambda x: ann_stats(x)
    m = {"ensemble": st(n), "benchmark": st(b), "cash": st(c), "months": int(ok.sum()),
         "start": str(P.rb_dates[r_start])[:10], "end": str(P.rb_dates[np.flatnonzero(ok)[-1] + r_start])[:10] if ok.any() else "",
         "excess_ann": float(ex.mean() * 12) if len(ex) else np.nan,
         "ir": float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(12)) if len(ex) > 2 else np.nan,
         "p_value": block_bootstrap_p(ex), "hit_rate": float((ex > 0).mean()) if len(ex) else np.nan,
         "avg_turnover": float(turnover(W[sl])[ok].mean()) if ok.any() else np.nan,
         "baselines": {}}
    for k, v in baselines.items():
        x = v[sl][ok]
        m["baselines"][k] = {**st(x), "excess_ann": float((x - b).mean() * 12)}
    m["tier"] = evidence_tier(m)
    return m


def evidence_tier(m: dict) -> str:
    p, ir = m.get("p_value", np.nan), m.get("ir", np.nan)
    if not np.isfinite(p) or m.get("months", 0) < 24:
        return "INSUFFICIENT HISTORY"
    if p < 0.05 and ir > 0.5:
        return "MODERATE EVIDENCE"
    if p < 0.20 and ir > 0.2:
        return "WEAK EVIDENCE"
    return "NO DEMONSTRATED EDGE"
