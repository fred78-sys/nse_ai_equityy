"""Generative strategy search (genetic algorithm) + persistent strategy registry.

*Generate*  random strategies, hand-written seeds, and mutations / cross-overs of winners.
*Test*      every candidate is replayed over the training window of history (with costs).
*Keep/drop* winners are kept and bred; losers die.  Survivors must then prove themselves on
            data that arrives AFTER they were born (live track record) or they are retired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import ALL_FEATURES, Panel
from .strategies import (BEAR_EXPO, F_ALL, HOLD_CHOICES, ML_IDX, REGIME_THR, SEED_STRATEGIES, TOP_N_CHOICES,
                         Genome, fitness, genome_weights, passes_hurdle, portfolio_returns, window_metrics)


# --------------------------------------------------------------------------------------
# genetic operators
# --------------------------------------------------------------------------------------
def random_genome(rng: np.random.Generator, use_ml: bool = True) -> Genome:
    pool = [i for i in range(F_ALL) if use_ml or i != ML_IDX]
    w = np.zeros(F_ALL)
    for i in rng.choice(pool, size=rng.integers(1, 5), replace=False):
        w[i] = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
    return Genome(w, int(rng.choice(TOP_N_CHOICES)), int(rng.choice(HOLD_CHOICES, p=[.5, .3, .2])),
                  str(rng.choice(["equal", "invvol"])), float(rng.choice(REGIME_THR)), float(rng.choice(BEAR_EXPO)))


def mutate(g: Genome, rng: np.random.Generator, use_ml: bool = True) -> Genome:
    c = g.copy()
    pool = [i for i in range(F_ALL) if use_ml or i != ML_IDX]
    for _ in range(rng.integers(1, 3)):
        op = rng.choice(["perturb", "toggle", "flip", "n", "hold", "wt", "regime"], p=[.28, .22, .1, .12, .08, .08, .12])
        if op == "perturb":
            nz = np.flatnonzero(np.abs(c.w) > 1e-9)
            if len(nz):
                j = rng.choice(nz)
                c.w[j] = np.clip(c.w[j] + rng.normal(0, 0.3), -1, 1)
        elif op == "toggle":
            j = int(rng.choice(pool))
            c.w[j] = 0.0 if abs(c.w[j]) > 1e-9 else rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
        elif op == "flip":
            nz = np.flatnonzero(np.abs(c.w) > 1e-9)
            if len(nz):
                j = rng.choice(nz)
                c.w[j] = -c.w[j]
        elif op == "n":
            c.top_n = int(rng.choice(TOP_N_CHOICES))
        elif op == "hold":
            c.hold_k = int(rng.choice(HOLD_CHOICES))
        elif op == "wt":
            c.weighting = "invvol" if c.weighting == "equal" else "equal"
        else:
            c.regime_thr, c.bear_exposure = float(rng.choice(REGIME_THR)), float(rng.choice(BEAR_EXPO))
    if c.n_active == 0:
        c.w[int(rng.choice(pool))] = 1.0
    return c


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    m = rng.random(F_ALL) < 0.5
    c = Genome(np.where(m, a.w, b.w))
    pick = (lambda x, y: x if rng.random() < 0.5 else y)
    c.top_n, c.hold_k = pick(a.top_n, b.top_n), pick(a.hold_k, b.hold_k)
    c.weighting = pick(a.weighting, b.weighting)
    c.regime_thr, c.bear_exposure = (a.regime_thr, a.bear_exposure) if rng.random() < .5 else (b.regime_thr, b.bear_exposure)
    if c.n_active == 0:
        return a.copy()
    return c


# --------------------------------------------------------------------------------------
# evolution on a training window
# --------------------------------------------------------------------------------------
@dataclass
class Scored:
    g: Genome
    fit: float
    m: dict
    net: np.ndarray            # net returns on the window
    origin: str = "evolved"
    parents: tuple = ()
    trials: int = 1               # how many candidates were tried in this search (multiple-testing)


def evolve(P: Panel, bench_net: np.ndarray, lo: int, hi: int, cfg: dict, rng: np.random.Generator,
           seeds: list[Genome] | None = None, exclude: set | None = None) -> list[Scored]:
    """Search for strategies that would have beaten the benchmark on rows [lo, hi)."""
    L, mk = cfg["learning"], cfg["market"]
    cost, maxw = mk["one_way_cost_bps"], mk["max_position_weight"]
    use_ml = bool(L.get("use_ml", True)) and bool(np.abs(P.cube[ML_IDX, lo:hi]).sum() > 0)
    cache: dict[str, Scored] = {}

    def score(g: Genome, origin: str, parents=()) -> Scored | None:
        k = g.key()
        if k in cache:
            return cache[k]
        W = genome_weights(g, P, hi, maxw)
        net = portfolio_returns(W, P, cost)[lo:hi]
        m = window_metrics(net, bench_net[lo:hi], P.cash[lo:hi])
        s = Scored(g, fitness(m, g), m, net, origin, parents)
        cache[k] = s
        return s

    pop: list[Scored] = []
    for name, g in SEED_STRATEGIES.items():
        if g.w[ML_IDX] != 0 and not use_ml:
            continue
        pop.append(score(g.copy(), f"seed:{name}"))
    for g in (seeds or []):
        pop.append(score(g.copy(), "carried"))
    while len(pop) < L["population"]:
        pop.append(score(random_genome(rng, use_ml), "random"))

    for _ in range(L["generations"]):
        pop = sorted(pop, key=lambda s: -s.fit)
        elite = pop[: L["elite"]]
        nxt = list(elite)
        while len(nxt) < L["population"]:
            r = rng.random()
            if r < 0.45:
                a, b = (elite[i] for i in rng.choice(len(elite), 2, replace=False)) if len(elite) > 1 else (elite[0], elite[0])
                child = crossover(a.g, b.g, rng)
                nxt.append(score(mutate(child, rng, use_ml) if rng.random() < .5 else child, "evolved"))
            elif r < 0.85:
                a = elite[int(rng.integers(0, len(elite)))]
                nxt.append(score(mutate(a.g, rng, use_ml), "evolved"))
            else:
                nxt.append(score(random_genome(rng, use_ml), "random"))
        pop = nxt
    allv = sorted(cache.values(), key=lambda s: -s.fit)
    for s in allv:
        s.trials = len(cache)
    if exclude:
        allv = [s for s in allv if s.g.key() not in exclude]
    return allv


def pick_diverse(scored: list[Scored], k: int, z: float, max_corr: float = 0.92) -> list[Scored]:
    """Top-k winners that clear the hurdle and are not near-duplicates of each other."""
    chosen: list[Scored] = []
    for s in scored:
        if not passes_hurdle(s.m, z):
            continue
        ok = np.isfinite(s.net)
        dup = False
        for c in chosen:
            both = ok & np.isfinite(c.net)
            if both.sum() > 12 and np.corrcoef(s.net[both], c.net[both])[0, 1] > max_corr:
                dup = True
                break
        if not dup:
            chosen.append(s)
        if len(chosen) >= k:
            break
    return chosen


# --------------------------------------------------------------------------------------
# registry (the system's memory of what worked and what was dropped)
# --------------------------------------------------------------------------------------
@dataclass
class Record:
    id: str
    genome: Genome
    born_row: int
    born_date: str
    origin: str
    parents: list
    prior_mu: float                 # monthly benchmark-relative return believed (shrunk in-sample)
    prior_sigma: float
    train_ir: float
    status: str = "probation"       # probation | active | retired
    retired_date: str | None = None
    retired_reason: str | None = None
    live_n: int = 0
    live_excess_cum: float = 0.0
    post_ir: float = 0.0

    def name(self) -> str:
        return self.genome.describe()


@dataclass
class Registry:
    recs: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    counter: int = 0

    def log(self, date, event, rid, detail=""):
        self.events.append({"date": str(date)[:10], "event": event, "id": rid, "detail": detail})

    def add(self, s: Scored, row: int, date, cfg) -> Record:
        L = cfg["learning"]
        self.counter += 1
        rid = f"S{self.counter:04d}"
        sig = max(s.m.get("std_ex", 0.02), 0.01)
        # Deflate the in-sample IR by what the best of `trials` random strategies would show by luck
        # (selection bias), then shrink again for decay/regime change. Only the remainder is believed.
        null_ir = L.get("deflation_k", 0.5) * np.sqrt(2 * np.log(max(s.trials, 2))) / np.sqrt(s.m["years"])
        ir_def = max(0.0, float(s.m["ir"]) - null_ir)
        prior_mu = L["prior_shrink"] * ir_def * sig / np.sqrt(12)
        rec = Record(rid, s.g.copy(), row, str(date)[:10], s.origin, list(s.parents),
                     prior_mu, sig, float(s.m["ir"]))
        self.recs[rid] = rec
        self.log(date, "born", rid, f"{s.origin} | train IR {s.m['ir']:.2f} (deflated {ir_def:.2f}) | {rec.name()}")
        return rec

    def alive(self) -> list[Record]:
        return [r for r in self.recs.values() if r.status != "retired"]

    def keys(self) -> set:
        return {r.genome.key() for r in self.recs.values()}

    def retire(self, rec: Record, date, reason: str):
        rec.status, rec.retired_date, rec.retired_reason = "retired", str(date)[:10], reason
        self.log(date, "retired", rec.id, f"{reason} | {rec.name()}")

    def to_json(self) -> dict:
        return {"counter": self.counter, "events": self.events,
                "recs": {k: {**{a: b for a, b in v.__dict__.items() if a != "genome"}, "genome": v.genome.to_dict()}
                         for k, v in self.recs.items()}}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1, default=float))


def posterior(rec: Record, ex_live: np.ndarray, n0: int) -> tuple[float, float, float]:
    """Blend shrunk in-sample belief with live evidence -> (mu, sigma, annualised IR)."""
    n = len(ex_live)
    mu = (n0 * rec.prior_mu + ex_live.sum()) / (n0 + n)
    var_live = ex_live.var(ddof=1) if n > 2 else rec.prior_sigma ** 2
    sig = np.sqrt((n0 * rec.prior_sigma ** 2 + n * var_live) / (n0 + n))
    sig = max(sig, 0.01)
    return mu, sig, mu / sig * np.sqrt(12)


def update_and_retire(reg: Registry, P: Panel, bench_net: np.ndarray, r0: int, cfg: dict) -> dict:
    """Score every living strategy on realised data since its birth; retire the losers.

    Returns {id: annualised posterior IR lower bound (drives capital)} for survivors.
    """
    L, mk = cfg["learning"], cfg["market"]
    kend = P.known_end(r0)
    date = P.rb_dates[r0]
    stats = {}
    for rec in list(reg.alive()):
        W = genome_weights(rec.genome, P, max(kend, 1), mk["max_position_weight"])
        net = portfolio_returns(W, P, mk["one_way_cost_bps"])
        lo = rec.born_row
        ex = (net[lo:kend] - bench_net[lo:kend]) if kend > lo else np.array([])
        ex = ex[np.isfinite(ex)]
        mu, sig, ir = posterior(rec, ex, L["prior_months"])
        rec.live_n, rec.live_excess_cum, rec.post_ir = len(ex), float(ex.sum()), float(ir)
        # capital follows the LOWER confidence bound of the evidence, not the point estimate
        stats[rec.id] = float(ir - L.get("conviction_se_k", 0.5) * np.sqrt(12.0 / (L["prior_months"] + len(ex))))
        if len(ex) >= L["min_live_months_to_judge"]:
            t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)) + 1e-12)
            streak = 0
            for v in ex[::-1]:
                if v < 0:
                    streak += 1
                else:
                    break
            reason = None
            if streak >= L["retire_loss_streak"]:
                reason = f"{streak} consecutive months behind benchmark"
            elif ex.sum() < -L["retire_cum_excess"]:
                reason = f"live excess {ex.sum():+.1%} vs benchmark"
            elif t < L["retire_tstat"]:
                reason = f"live t-stat {t:.2f}"
            elif len(ex) >= 12 and ir <= 0:
                reason = f"posterior IR {ir:.2f} <= 0 after {len(ex)}m"
            if reason:
                reg.retire(rec, date, reason)
                stats.pop(rec.id, None)
                continue
            if rec.status == "probation" and ex.mean() > 0:
                rec.status = "active"
                reg.log(date, "promoted", rec.id, f"live excess {ex.sum():+.1%} over {len(ex)}m")
    # capacity: keep the best, drop the rest
    alive = sorted(reg.alive(), key=lambda r: -r.post_ir)
    for rec in alive[L["max_registry_active"]:]:
        reg.retire(rec, date, f"outranked (post IR {rec.post_ir:.2f})")
        stats.pop(rec.id, None)
    return stats
