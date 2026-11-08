"""Command line:  python -m nse_ai.cli {collect|run|demo}"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import data as D
from .features import build_panel
from .forecast import recommend
from .ledger import log_predictions, score_ledger
from .report import write_outputs
from .walkforward import run_walk_forward

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str | None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "config" / "settings.yaml").read_text())


def attach_optional(md: D.MarketData, cfg: dict) -> D.MarketData:
    d = cfg["data"]
    paths = [ROOT / d["macro_csv"], ROOT / Path(d["csv_dir"]).parent / "macro_usdkes.csv"]
    px_idx = md.close.index
    if md.macro is None:
        md.macro = D.load_macro(paths, px_idx)
    f = D.load_fundamentals(ROOT / d["fundamentals_csv"])
    s = D.load_sentiment(ROOT / d["sentiment_csv"])
    md.fundamentals, md.sentiment = f, s
    return md


def cmd_collect(cfg, args):
    uni = D.load_universe(ROOT / cfg["universe_file"])
    rep = D.collect_yahoo(uni, ROOT / cfg["data"]["csv_dir"], cfg["data"]["start"], cfg["data"].get("fx_symbol"))
    print(f"downloaded {len(rep['ok'])}/{len(uni)} tickers; fx: {rep.get('fx')}")
    for t, why in list(rep["failed"].items())[:5]:
        print(f"  skipped {t}: {why[:90]}")
    if len(rep["failed"]) > 5:
        print(f"  ... and {len(rep['failed']) - 5} more")
    if len(rep["ok"]) < len(uni) // 2:
        print("\nYahoo does not appear to serve these NSE symbols. Next steps:\n"
              "  1) python -m nse_ai.cli symbols      # ask Yahoo which symbols exist for each company\n"
              "  2) or download prices elsewhere and run: python -m nse_ai.cli import <file-or-folder>")
    if args.news:
        print("news:", D.collect_news_sentiment(uni, ROOT / cfg["data"]["sentiment_csv"]))
    if not rep["ok"]:
        print("Nothing downloaded. If you are offline/blocked, drop per-ticker CSVs (date,close,volume) into "
              f"{cfg['data']['csv_dir']}/ instead.")
        return 1


def _pipeline(cfg, md, out_dir, state_dir, synthetic):
    print(f"data: {md.close.shape[1]} tickers, {md.close.index[0].date()} -> {md.close.index[-1].date()}")
    P = build_panel(md, cfg)
    print(f"panel: {P.R} monthly decision dates, {int(P.elig[-1].sum())} investable names now")
    print("learning (walk-forward) ...")
    res = run_walk_forward(P, cfg)
    recs, info = recommend(res, cfg)
    ledger_path = Path(state_dir) / ("prediction_log_demo.csv" if synthetic else "prediction_log.csv")
    if not synthetic:
        log_predictions(recs, info["as_of"], ledger_path)
    led = score_ledger(ledger_path, P.px_df, cfg["forecast"]["horizon_days"]) if not synthetic else {"resolved_dates": 0}
    paths = write_outputs(res, recs, info, led, cfg, out_dir, synthetic)
    m = res.metrics
    print(f"\nOOS {m['start']} -> {m['end']}  ({m['months']} months)")
    print(f"  adaptive CAGR {m['ensemble']['cagr']:.1%}  benchmark {m['benchmark']['cagr']:.1%}  excess/yr {m['excess_ann']:+.1%}  IR {m['ir']:.2f}  p={m['p_value']:.3f}")
    print(f"  evidence: {m['tier']}")
    top = recs[recs["action"].isin(["BUY / HOLD", "SMALL HOLD"])].head(8)
    print("\ncurrent portfolio (top):")
    print(top[["ticker", "target_weight", "exp_3m_return", "p_beat_tbill", "confidence"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nreport: {paths['report']}")
    return 0


def cmd_symbols(cfg, args):
    uni = D.load_universe(ROOT / cfg["universe_file"])
    if args.limit:
        uni = uni.head(args.limit)
    df = D.find_yahoo_symbols(uni)
    out = ROOT / "outputs" / "yahoo_symbol_candidates.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    hits = df.dropna(subset=["symbol"])
    # "NSE" alone is India's exchange (.NS) -- only accept Nairobi / Kenya / .NR
    ke = hits[hits["exchange"].astype(str).str.contains("Nairobi|Kenya", case=False, na=False)
              | hits["symbol"].astype(str).str.endswith(".NR")]
    print(f"{len(hits)} candidate symbols found ({len(ke)} on a Nairobi/Kenya exchange). Saved to {out}")
    print(ke.drop_duplicates("symbol").head(30).to_string(index=False) if len(ke) else
          "None of them are listed on a Nairobi exchange: Yahoo probably does not carry NSE prices. Use `import`.")


def cmd_import(cfg, args):
    rep = D.import_prices(args.path, ROOT / cfg["data"]["csv_dir"], args.ticker, args.price_col)
    print(f"imported {len(rep['imported'])} tickers into {cfg['data']['csv_dir']}: {', '.join(rep['imported'])}")
    if rep["imported"]:
        md = D.load_csv_store(ROOT / cfg["data"]["csv_dir"])
        print(f"store now covers {md.close.index[0].date()} -> {md.close.index[-1].date()} "
              f"({md.close.shape[1]} tickers). Latest data date matters: recommendations are only as fresh as this.")
    for n in rep["notes"]:
        print("  note:", n)


def cmd_run(cfg, args):
    uni = D.load_universe(ROOT / cfg["universe_file"])
    md = D.load_csv_store(ROOT / cfg["data"]["csv_dir"], uni)
    md = attach_optional(md, cfg)
    return _pipeline(cfg, md, ROOT / "outputs", ROOT / "state", False)


def cmd_demo(cfg, args):
    md = D.make_synthetic_nse(n_stocks=args.stocks, years=args.years, seed=args.seed, signal=not args.placebo,
                              signal_strength=args.strength)
    out = ROOT / "outputs" / ("demo_placebo" if args.placebo else "demo_signal")
    return _pipeline(cfg, md, out, ROOT / "state", True)


def cmd_study(cfg, args):
    """Calibration study: how often does the system find an edge where one exists, and where none does?"""
    import numpy as np
    rows = []
    for kind in ("signal", "placebo"):
        for seed in range(1, args.n + 1):
            md = D.make_synthetic_nse(n_stocks=40, years=12, seed=seed, signal=(kind == "signal"), signal_strength=args.strength)
            res = run_walk_forward(build_panel(md, cfg), cfg, log=lambda *a: None)
            m = res.metrics
            rows.append({"market": kind, "seed": seed, "excess_per_yr": m["excess_ann"], "ir": m["ir"], "p": m["p_value"], "verdict": m["tier"]})
            print(f"{kind:8s} seed {seed}: excess {m['excess_ann']:+6.1%}  IR {m['ir']:+.2f}  p={m['p_value']:.3f}  {m['tier']}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "validation_study.csv", index=False)
    for kind, g in df.groupby("market"):
        found = g["verdict"].isin(["MODERATE EVIDENCE", "WEAK EVIDENCE"]).mean()
        print(f"{kind}: mean excess {g['excess_per_yr'].mean():+.1%}/yr, 'edge found' in {found:.0%} of markets")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nse_ai")
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="download NSE prices (Yahoo .NR) into data/csv")
    c.add_argument("--news", action="store_true", help="also collect headline sentiment")
    sy = sub.add_parser("symbols", help="discover which Yahoo symbols exist for the NSE companies")
    sy.add_argument("--limit", type=int, default=0)
    im = sub.add_parser("import", help="import price CSV(s) you downloaded elsewhere into data/csv")
    im.add_argument("path")
    im.add_argument("--ticker", help="ticker for a single-stock file (default: file name)")
    im.add_argument("--price-col", help="force a price column, e.g. adjusted_price (default: Day FinalPrice / close)")
    sub.add_parser("run", help="learn from data/csv, forecast, write report")
    d = sub.add_parser("demo", help="run on a SIMULATED NSE-like market")
    d.add_argument("--placebo", action="store_true", help="no predictable signal (should find no edge)")
    d.add_argument("--seed", type=int, default=2)
    d.add_argument("--stocks", type=int, default=40)
    d.add_argument("--years", type=float, default=12)
    d.add_argument("--strength", type=float, default=0.35)
    st = sub.add_parser("study", help="validation: planted-signal vs placebo markets")
    st.add_argument("--n", type=int, default=6)
    st.add_argument("--strength", type=float, default=0.35)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    return {"collect": cmd_collect, "run": cmd_run, "demo": cmd_demo, "study": cmd_study, "symbols": cmd_symbols, "import": cmd_import}[args.cmd](cfg, args) or 0


if __name__ == "__main__":
    sys.exit(main())
