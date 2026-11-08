import copy
import numpy as np

from nse_ai.evolution import Registry, Scored, evolve, pick_diverse, posterior, update_and_retire
from nse_ai.strategies import SEED_STRATEGIES, portfolio_returns, window_metrics, fitness
from nse_ai.walkforward import run_walk_forward, block_bootstrap_p


def test_evolution_beats_random_on_train_window(panel, cfg):
    bench = portfolio_returns(panel.bench_w, panel, cfg["market"]["one_way_cost_bps"])
    rng = np.random.default_rng(1)
    lo, hi = 20, 100
    scored = evolve(panel, bench, lo, hi, cfg, rng)
    assert len(scored) > 30
    assert scored[0].fit > np.median([s.fit for s in scored])
    assert len({s.g.key() for s in scored}) == len(scored)           # no duplicates evaluated


def test_losing_strategy_is_retired(panel, cfg):
    """A strategy that is the exact opposite of the planted signal must be dropped on live evidence."""
    bench = portfolio_returns(panel.bench_w, panel, cfg["market"]["one_way_cost_bps"])
    reg = Registry()
    bad = SEED_STRATEGIES["momentum_12_1"].copy()
    bad.w = -bad.w
    bad.top_n = 5
    r_born = 50
    s = Scored(bad, 1.0, {"mean_ex": 0.02, "std_ex": 0.03, "ir": 2.0, "years": 4}, np.zeros(10), "test")
    s.trials = 1
    rec = reg.add(s, r_born, panel.rb_dates[r_born], cfg)
    for r0 in range(r_born + 2, r_born + 40, 3):
        update_and_retire(reg, panel, bench, r0, cfg)
        if rec.status == "retired":
            break
    assert rec.status == "retired", "a persistently losing strategy should have been dropped"
    assert rec.retired_reason


def test_winning_strategy_is_promoted(panel, cfg):
    bench = portfolio_returns(panel.bench_w, panel, cfg["market"]["one_way_cost_bps"])
    reg = Registry()
    good = SEED_STRATEGIES["momentum_12_1"].copy()
    good.top_n = 5
    s = Scored(good, 1.0, {"mean_ex": 0.02, "std_ex": 0.03, "ir": 2.0, "years": 4}, np.zeros(10), "test")
    s.trials = 1
    rec = reg.add(s, 50, panel.rb_dates[50], cfg)
    update_and_retire(reg, panel, bench, 50 + 12, cfg)
    assert rec.status in ("active", "probation")
    assert rec.live_n >= 8


def test_posterior_shrinks_toward_evidence():
    from nse_ai.evolution import Record
    from nse_ai.strategies import Genome
    r = Record("x", Genome(), 0, "d", "t", [], prior_mu=0.01, prior_sigma=0.03, train_ir=2)
    mu0, _, ir0 = posterior(r, np.array([]), 12)
    mu1, _, ir1 = posterior(r, np.full(36, -0.02), 12)
    assert mu0 == 0.01 and mu1 < 0 and ir1 < ir0


def test_bootstrap_p_value():
    rng = np.random.default_rng(0)
    assert block_bootstrap_p(rng.normal(0.02, 0.02, 80)) < 0.01
    assert block_bootstrap_p(rng.normal(0.0, 0.02, 80), seed=1) > 0.05


def test_walk_forward_runs_and_reports(panel, cfg):
    res = run_walk_forward(copy.deepcopy(panel), cfg, log=lambda *a: None)
    assert res.metrics["months"] > 24
    assert res.metrics["tier"] in {"MODERATE EVIDENCE", "WEAK EVIDENCE", "NO DEMONSTRATED EDGE", "INSUFFICIENT HISTORY"}
    assert (res.W_oos[res.r_start:].sum(1) <= 1 + 1e-9).all()
    assert any(e["event"] == "retired" for e in res.reg.events)      # losers really are dropped
    assert any(e["event"] == "born" for e in res.reg.events)
    assert len(res.reg.alive()) <= cfg["learning"]["max_registry_active"]
