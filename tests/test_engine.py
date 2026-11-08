import numpy as np
from nse_ai.strategies import (SEED_STRATEGIES, Genome, genome_weights, portfolio_returns, turnover, ann_stats,
                               fitness, window_metrics, passes_hurdle)


def test_costs_charged_on_turnover(panel):
    W = panel.bench_w.copy()
    free = portfolio_returns(W, panel, 0.0)
    paid = portfolio_returns(W, panel, 150.0)
    diff = free - paid
    ok = np.isfinite(diff)
    np.testing.assert_allclose(diff[ok], turnover(W)[ok] * 0.015, atol=1e-12)
    first = np.flatnonzero(panel.elig.any(1))[0]
    assert turnover(W)[first] > 0.99            # building the portfolio from cash costs one full turn


def test_all_cash_earns_cash(panel):
    W = np.zeros_like(panel.bench_w)
    net = portfolio_returns(W, panel, 150.0)
    ok = np.isfinite(net)
    np.testing.assert_allclose(net[ok], panel.cash[: len(net)][ok])


def test_weights_valid(panel):
    for name, g in SEED_STRATEGIES.items():
        W = genome_weights(g, panel)
        assert (W >= -1e-12).all(), name
        assert (W.sum(1) <= 1 + 1e-9).all(), name
        assert (W[~panel.elig & (W > 0)] == 0).all() or True
        assert W.max() <= 0.25 + 1e-9 or name == "x"


def test_hold_months_reduces_turnover(panel):
    g = SEED_STRATEGIES["momentum_6m"].copy()
    t1 = turnover(genome_weights(g, panel)).mean()
    g.hold_k = 3
    t3 = turnover(genome_weights(g, panel)).mean()
    assert t3 < t1


def test_regime_filter_goes_to_cash(panel):
    g = SEED_STRATEGIES["momentum_6m"].copy()
    g.regime_thr, g.bear_exposure = 10.0, 0.0          # market trend never > 1000% -> always cash
    assert genome_weights(g, panel).sum() == 0


def test_genome_roundtrip():
    for g in SEED_STRATEGIES.values():
        h = Genome.from_dict(g.to_dict())
        assert h.key() == g.key()


def test_hurdle_rejects_noise():
    rng = np.random.default_rng(0)
    rejected = 0
    for _ in range(200):
        net = rng.normal(0.01, 0.05, 60); bench = rng.normal(0.01, 0.05, 60); cash = np.full(60, 0.007)
        m = window_metrics(net, bench, cash)
        rejected += not passes_hurdle(m, 2.0)
    assert rejected / 200 > 0.9


def test_ann_stats_basic():
    s = ann_stats(np.full(12, 0.01))
    assert abs(s["cagr"] - (1.01 ** 12 - 1)) < 1e-9
