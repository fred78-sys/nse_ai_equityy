import numpy as np
import pandas as pd

from nse_ai import data as D
from nse_ai.forecast import fama_macbeth, simulate_returns
from nse_ai.ledger import log_predictions, score_ledger


def test_fundamentals_visible_only_after_publication():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    px = pd.DataFrame({"AAA": 10.0}, index=idx)
    f = pd.DataFrame({"ticker": ["AAA", "AAA"], "period_end": pd.to_datetime(["2023-12-31", "2024-06-30"]),
                      "publish_date": pd.to_datetime(["2024-04-15", "2024-09-20"]),
                      "eps": [1.0, 2.0], "dps": [0.5, 0.5], "bvps": [10.0, 10.0], "revenue": [100.0, 150.0]})
    out = D.fundamentals_daily(f, px)
    assert out["ey"].loc["2024-04-10", "AAA"] != out["ey"].loc["2024-04-10", "AAA"]      # NaN before publish
    assert out["ey"].loc["2024-04-16", "AAA"] == 0.1
    assert out["ey"].loc["2024-09-19", "AAA"] == 0.1                                        # still old number
    assert out["ey"].loc["2024-09-23", "AAA"] == 0.2
    assert np.isclose(out["eps_g"].loc["2024-09-23", "AAA"], 1.0)


def test_macro_publication_lag():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    p = pd.DataFrame({"date": pd.to_datetime(["2024-01-31"]), "series": ["inflation"], "value": [6.0]})
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "m.csv"); p.to_csv(path, index=False)
    m = D.load_macro([path], idx)
    assert np.isnan(m.loc["2024-02-05", "inflation"])         # published ~20 days after the month it describes
    assert m.loc["2024-02-26", "inflation"] == 6.0


def test_csv_loader(tmp_path):
    dates = pd.bdate_range("2022-01-01", periods=300)
    pd.DataFrame({"Date": dates, "Adj Close": np.linspace(10, 20, 300), "Volume": 1000}).to_csv(tmp_path / "SCOM.csv", index=False)
    md = D.load_csv_store(tmp_path)
    assert md.tickers == ["SCOM"] and md.close.shape[0] == 300


def test_synthetic_has_nse_frictions(md):
    zero_share = (md.volume == 0).to_numpy().mean()
    assert 0.1 < zero_share < 0.7                               # thin trading is modelled
    assert md.is_synthetic


def test_lexicon_sentiment():
    assert D.lexicon_score("Profit surge as dividend rises") > 0
    assert D.lexicon_score("Probe into fraud, shares fall") < 0


def test_fama_macbeth_detects_signal_and_noise():
    rng = np.random.default_rng(0)
    R, N = 60, 30
    x = rng.uniform(-1, 1, (R, N)); el = np.ones((R, N), bool)
    y = 0.05 * x + 0.05 * rng.standard_normal((R, N))
    assert fama_macbeth(x, y, el, range(R))["t"] > 3
    assert abs(fama_macbeth(x, 0.05 * rng.standard_normal((R, N)), el, range(R))["t"]) < 3


def test_simulation_matches_target_mean_and_is_fat_tailed():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(3, 800) * 0.01)
    sim = simulate_returns(r, 0.05, 63, 4000, 5, rng)
    assert abs(sim.mean() - 0.05) < 0.01
    assert sim.std() > 0.03


def test_ledger_roundtrip(tmp_path):
    idx = pd.bdate_range("2025-01-01", periods=200)
    px = pd.DataFrame({"A": np.linspace(10, 20, 200), "B": np.linspace(10, 10, 200)}, index=idx)
    df = pd.DataFrame({"ticker": ["A", "B"], "action": ["BUY / HOLD", "NEUTRAL"], "target_weight": [.5, 0],
                       "exp_3m_return": [.1, 0], "exp_3m_excess_vs_market": [.05, -.05], "p_beat_tbill": [.7, .3], "price": [10, 10]})
    p = tmp_path / "led.csv"
    assert log_predictions(df, "2025-01-10", p) == 2
    assert log_predictions(df, "2025-01-10", p) == 0               # idempotent
    s = score_ledger(p, px, 63)
    assert s["resolved_dates"] == 1 and s["buy_mean_excess"] > 0


def test_import_long_and_investing_formats(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=120)
    long = pd.concat([pd.DataFrame({"Date": dates, "Code": t, "Price": np.linspace(10, 12, 120), "Volume": "1.5M"})
                      for t in ("SCOM", "KCB")])
    long.to_csv(tmp_path / "all.csv", index=False)
    inv = pd.DataFrame({"Date": dates.strftime("%b %d, %Y"), "Price": np.linspace(5, 6, 120), "Vol.": "350K"})
    inv.to_csv(tmp_path / "EQTY.csv", index=False)
    out = tmp_path / "store"
    rep = D.import_prices(tmp_path / "all.csv", out)
    rep2 = D.import_prices(tmp_path / "EQTY.csv", out)
    assert rep["imported"] == ["KCB", "SCOM"] and rep2["imported"] == ["EQTY"]
    md = D.load_csv_store(out)
    assert set(md.tickers) == {"SCOM", "KCB", "EQTY"}
    assert md.volume["SCOM"].iloc[0] == 1.5e6 and md.volume["EQTY"].iloc[0] == 3.5e5


def test_import_mendeley_style_yearly_files_are_merged(tmp_path):
    cols = ["Date", "Stock Code", "Stock Name", "12month LowPrice", "12month HighPrice", "Day LowPrice", "Day HighPrice",
            "Day FinalPrice", "Previous traded price", "Change in price value", "Change in price%", "Volume traded", "Adjusted price"]
    def year(y, start):
        d = pd.bdate_range(f"{y}-01-02", periods=130)
        rows = []
        for code in ("SCOM", "EQTY"):
            for i, dt in enumerate(d):
                rows.append([dt.strftime("%d-%b-%Y"), code, code, 1, 2, 1, 2, start + i * .1, 0, 0, "0%",
                             "-" if i % 10 == 0 else "2,500", start + i * .1 + 1])
        pd.DataFrame(rows, columns=cols).to_csv(tmp_path / f"NSE_data_all_stocks_{y}.csv", index=False)
    year(2013, 10); year(2014, 40)
    out = tmp_path / "store"
    rep = D.import_prices(tmp_path, out)
    assert rep["imported"] == ["EQTY", "SCOM"]
    assert any("Adjusted price" in n for n in rep["notes"])
    md = D.load_csv_store(out)
    assert len(md.close) == 260                                  # both years joined, not overwritten
    assert md.close["SCOM"].iloc[0] == 10.0                      # default = Day FinalPrice
    assert md.volume["SCOM"].iloc[1] == 2500 and md.volume["SCOM"].iloc[0] == 0
    D.import_prices(tmp_path, out, price_col="adjusted_price")
    assert D.load_csv_store(out).close["SCOM"].iloc[0] == 11.0
