# NSE Adaptive Research System

A generative, self-learning stock research system for the Nairobi Securities Exchange. It builds on the
idea of the attached `ai_equity_research` prototype (price features -> ranking -> walk-forward ML ->
loss-streak ledger) and extends it into a closed learning loop.

> Research software, not investment advice. No model can guarantee returns. The system is designed to
> say **"no demonstrated edge"** and fall back to an index-like portfolio when it cannot prove one.

## What it does

1. **Collects** NSE prices (Yahoo `.NR`, or your own CSVs), USD/KES, optional point-in-time fundamentals,
   macro (T-bill, CBK rate, inflation, with publication lags) and news sentiment.
2. **Generates** thousands of candidate strategies (genetic search: seeds, random, mutation, crossover) over
   26 features: momentum, reversal, volatility, beta, 52-week high, trend, RSI, liquidity/zero-trade days,
   Amihud impact, FX sensitivity, value, dividend yield, quality, growth, sentiment and a walk-forward ML forecast.
3. **Replays history** for each candidate with realistic NSE costs (150 bps one-way default), liquidity filters,
   next-close execution, and keeps the winners / kills the losers.
4. **Adapts**: survivors must prove themselves on data after their birth. Strategies are retired for a loss
   streak, negative live t-stat, cumulative underperformance or lost posterior edge, and promoted otherwise.
   Capital follows the *lower confidence bound* of evidence, scaled by a "self-doubt" factor from the learner's own record.
5. **Forecasts** each stock: calibrated 3-month expected return (shrunk by out-of-sample significance), plus a
   generative Monte-Carlo distribution (P10/P50/P90, P(beat T-bill), worst-5% loss).
6. **Recommends** BUY/HOLD/AVOID with sizes in KES, liquidity warnings, and the reasons (top drivers).
7. **Logs** every prediction in an append-only forward ledger, later scored when matured: the only true live test.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                      # 27 tests incl. look-ahead tests
python -m nse_ai.cli demo                # simulated market, ~15 s, writes outputs/demo_signal/report.html
python -m nse_ai.cli demo --placebo      # market with NO signal: should find no edge
python -m nse_ai.cli study --n 6         # calibration study (signal vs placebo markets)

python -m nse_ai.cli collect             # try Yahoo (see 'Getting NSE data' below)
python -m nse_ai.cli run                 # learn, forecast, outputs/report.html + recommendations.csv
```

Run `collect` + `run` regularly (e.g. weekly). The forward ledger in `state/` accumulates and is scored automatically.

## Getting NSE data

Yahoo does **not** carry Nairobi stocks (every `.NR` symbol returns 404; the `.NS` hits are India's exchange). Use files instead:

1. **Free history 2007-2025:** the public "Kenya Nairobi Securities Exchange (NSE) All Stocks Prices" datasets on Mendeley Data
   (CC BY 4.0; search that title; separate sets for 2007-2012, 2013-2020, 2021, 2022, 2023-2024, 2025; use "Download All").
   Unzip every set into one folder, then:
   ```bash
   python -m nse_ai.cli import /path/to/that/folder
   ```
   Yearly files for the same stock are joined automatically. Default price = `Day FinalPrice`; the import prints how often
   `Adjusted price` differs, and `--price-col adjusted_price` switches to it if that looks like the better series.
2. **Recent months (the datasets end at 2025):** add newer daily data from the NSE website's data services, your broker,
   or a scraper such as the open-source `NSE-Stock-Price-Crawler` on GitHub; put the files in a folder and `import` again
   (existing history is kept and merged).
3. Then `python -m nse_ai.cli run`. **Check the "store now covers ... -> date" line: recommendations are only as fresh as
   the last price date. Do not act on stale data.**

Dividends: Day FinalPrice excludes dividends, so total return of high-yield NSE names is understated unless you use a
properly adjusted series.

## Your own data

Drop `data/csv/<TICKER>.csv` with columns `date,close,volume` (dividend-adjusted close preferred). Optional files
(see `config/settings.yaml`): `data/macro.csv` (`date,series,value`), `data/fundamentals.csv`
(`ticker,period_end,publish_date,eps,dps,bvps,revenue`; only visible after `publish_date`),
`data/sentiment.csv` (`date,ticker,score`). Edit `config/nse_universe.csv` to add/remove tickers
(the list is a starter; verify listings and Yahoo symbols).

## Validation (simulated markets only; real NSE results must be measured by you)

| Market | Edge found | Mean excess vs index |
|---|---|---|
| Planted signal (6 markets) | 50% | +2.8%/yr |
| Placebo, no signal (6 markets) | 17% (false "weak") | -0.9%/yr |

Look-ahead is tested two ways: corrupting future prices must not change earlier features, and replaying the
whole learner on truncated history must reproduce identical past decisions. The tests were mutation-checked
(an injected leak makes them fail).

## Known limits

* Yahoo does not appear to carry NSE symbols (confirmed by a user run); use `symbols` / `import`. The news collector is untested live.
* Free feeds omit delisted stocks (survivorship bias) and have gaps; thin NSE liquidity, price limits, T+3 and taxes are only approximated.
* The universe is small (~30-50 liquid names), so statistical power is modest; expect "weak" or "no" evidence often.
  That is the correct answer, not a bug. Use the forward ledger before risking real money.
