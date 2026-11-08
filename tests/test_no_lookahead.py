"""The most important tests: nothing at or before date t may depend on data after t."""
import copy
import numpy as np
import pandas as pd

from nse_ai.features import build_panel
from nse_ai.walkforward import run_walk_forward


def _corrupt_after(md, cut):
    m = copy.deepcopy(md)
    rng = np.random.default_rng(0)
    after = m.close.index > cut
    noise = np.exp(rng.normal(0, 0.5, size=m.close.loc[after].shape))
    m.close.loc[after] = m.close.loc[after].to_numpy() * noise
    m.volume.loc[after] = m.volume.loc[after].to_numpy() * 3
    m.macro.loc[after] = m.macro.loc[after].to_numpy() * 2
    return m


def test_features_unchanged_by_future(md, cfg, panel):
    r0 = len(panel.rb) - 30
    cut = panel.rb_dates[r0]
    p2 = build_panel(_corrupt_after(md, cut), cfg)
    assert (p2.rb_dates[: r0 + 1] == panel.rb_dates[: r0 + 1]).all()
    for name in ("cube", "elig", "inv_vol", "bench_w", "mkt_trend", "macro", "cash_rate"):
        a, b = getattr(panel, name), getattr(p2, name)
        sl = (slice(None), slice(0, r0 + 1)) if name == "cube" else slice(0, r0 + 1)
        np.testing.assert_allclose(np.nan_to_num(a[sl].astype(float)), np.nan_to_num(b[sl].astype(float)),
                                   err_msg=f"{name} leaks future information")


def test_period_return_is_next_close_to_next_close(panel):
    # decision at close of rb[r]; trade at ex[r]=rb[r]+1; earn exec_r -> exec_{r+1}
    r = 20
    a, b = panel.ex[r], panel.ex[r + 1]
    j = int(np.flatnonzero(panel.elig[r])[0])
    expect = panel.px_df.iloc[b, j] / panel.px_df.iloc[a, j] - 1
    assert np.isclose(np.clip(expect, -.95, 3), panel.pr[r, j])
    assert panel.ex[r] == panel.rb[r] + 1          # never earns the move that produced the signal


def test_whole_learner_is_causal(md, cfg):
    """Replay the learner on history truncated at D: all decisions made before D must be identical."""
    full = build_panel(md, cfg)
    D = full.rb_dates[-14]
    m2 = copy.deepcopy(md)
    keep = m2.close.index <= D
    m2.close, m2.volume, m2.macro = m2.close[keep], m2.volume[keep], m2.macro[keep]
    short = build_panel(m2, cfg)
    assert short.rb_dates[-1] == D
    a = run_walk_forward(full, cfg, log=lambda *x: None)
    b = run_walk_forward(short, cfg, log=lambda *x: None)
    last_epoch_short = b.final["row"]
    assert last_epoch_short == short.R - 1
    n = last_epoch_short                          # rows strictly before the final live epoch
    np.testing.assert_allclose(a.W_oos[a.r_start:n], b.W_oos[a.r_start:n], atol=1e-9,
                               err_msg="learner decisions changed when future data was removed -> look-ahead")
    ev_a = [e for e in a.reg.events if e["date"] < str(D)[:10]]
    ev_b = [e for e in b.reg.events if e["date"] < str(D)[:10]]
    assert ev_a == ev_b
