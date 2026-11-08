"""HTML report (charts embedded) + CSV/JSON outputs."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .walkforward import WFResult

INK, MUTED, BLUE, ORANGE, GREEN, RED = "#1d2433", "#6b7385", "#2d5bff", "#e07b00", "#1a8f5a", "#c23b3b"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=.25)
    ax.tick_params(colors=MUTED)


def chart_equity(res: WFResult) -> str:
    P, s = res.P, res.r_start
    d = P.rb_dates[s:]
    ok = np.isfinite(res.net[s:]) & np.isfinite(res.bench_net[s:])
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    eq = lambda x: np.cumprod(1 + np.nan_to_num(x[s:][ok]))
    ax.plot(d[ok], eq(res.net), color=BLUE, lw=2.2, label="Adaptive system (out-of-sample)")
    ax.plot(d[ok], eq(res.bench_net), color=ORANGE, lw=1.8, label="Equal-weight NSE benchmark")
    ax.plot(d[ok], eq(P.cash), color=MUTED, lw=1.2, ls="--", label="T-bill cash")
    ax.set_ylabel("Growth of 1 KES"); ax.legend(frameon=False, fontsize=8); _style(ax)
    return _png(fig)


def chart_excess(res: WFResult) -> str:
    P, s = res.P, res.r_start
    ok = np.isfinite(res.net[s:]) & np.isfinite(res.bench_net[s:])
    ex = (res.net[s:] - res.bench_net[s:])[ok]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.bar(P.rb_dates[s:][ok], ex * 100, width=20, color=np.where(ex >= 0, GREEN, RED))
    ax.set_ylabel("Monthly excess %"); ax.axhline(0, color=INK, lw=.8); _style(ax)
    return _png(fig)


def chart_learning(res: WFResult) -> str:
    c = pd.DataFrame(res.curve)
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    x = pd.to_datetime(c["date"])
    ax.plot(x, c["conviction"] * 100, color=BLUE, lw=2, label="Capital in strategies (%)")
    ax.plot(x, c["self_doubt"] * 100, color=GREEN, lw=1.4, label="Self-doubt factor (%)")
    ax2 = ax.twinx()
    ax2.plot(x, c["retired_total"], color=RED, lw=1.4, label="Strategies retired (cumulative)")
    ax2.set_ylabel("retired"); ax.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3); _style(ax)
    return _png(fig)


def _pct(x, d=1): return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"
def _num(x, d=2): return "–" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


def _table(df: pd.DataFrame, cls="") -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    wi = list(df.columns).index("Why") if "Why" in df.columns else -1
    b = "".join("<tr>" + "".join(f'<td class="why">{v}</td>' if j == wi else f"<td>{v}</td>" for j, v in enumerate(r)) + "</tr>"
                for r in df.itertuples(index=False))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def build_html(res: WFResult, recs: pd.DataFrame, info: dict, ledger: dict, cfg: dict, synthetic: bool) -> str:
    m, P = res.metrics, res.P
    e, b = m["ensemble"], m["benchmark"]
    tier = m["tier"]
    tier_col = {"MODERATE EVIDENCE": GREEN, "WEAK EVIDENCE": ORANGE}.get(tier, RED)
    banner = ('<div class="warn">SYNTHETIC DEMO DATA – every number below comes from a simulated market and says nothing about the real NSE.</div>'
              if synthetic else "")
    kpi = lambda label, val, sub="": f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{val}</div><div class="ks">{sub}</div></div>'
    kpis = "".join([
        kpi("Out-of-sample CAGR", _pct(e["cagr"]), f"benchmark {_pct(b['cagr'])}"),
        kpi("Excess vs benchmark", _pct(m["excess_ann"]), "per year, after costs"),
        kpi("Information ratio", _num(m["ir"]), f"p-value {_num(m['p_value'], 3)}"),
        kpi("Max drawdown", _pct(e["mdd"], 0), f"benchmark {_pct(b['mdd'], 0)}"),
        kpi("Months beating benchmark", _pct(m["hit_rate"], 0), f"{m['months']} months tested"),
    ])
    r = recs.copy()
    show = r[r["action"].isin(["BUY / HOLD", "SMALL HOLD"])].copy()
    def fmt(df):
        o = pd.DataFrame({
            "Ticker": df["ticker"], "Name": df["name"], "Action": df["action"],
            "Weight": df["target_weight"].map(lambda v: _pct(v)),
            "KES": df["suggested_kes"].map(lambda v: f"{v:,.0f}"),
            "Price": df["price"].map(lambda v: _num(v)),
            "Exp 3m": df["exp_3m_return"].map(lambda v: _pct(v)),
            "P10 / P50 / P90": [f"{_pct(a, 0)} / {_pct(b_, 0)} / {_pct(c, 0)}" for a, b_, c in zip(df["p10"], df["p50"], df["p90"])],
            "P(beat T-bill)": df["p_beat_tbill"].map(lambda v: _pct(v, 0)),
            "Worst-5% avg": df["cvar5"].map(lambda v: _pct(v, 0)),
            "Why": df["drivers"].fillna(""),
            "Liquidity": df["liquidity_flag"],
        })
        return o
    avoid = r[r["action"] == "AVOID"].head(10)
    reg = res.reg
    alive = sorted(reg.alive(), key=lambda x: -x.post_ir)
    board = pd.DataFrame([{
        "ID": x.id, "Status": x.status, "Strategy": html.escape(x.name()), "Born": x.born_date,
        "Live months": x.live_n, "Live excess": _pct(x.live_excess_cum), "Posterior IR": _num(x.post_ir),
    } for x in alive])
    dead = sorted([x for x in reg.recs.values() if x.status == "retired"], key=lambda x: x.retired_date or "", reverse=True)[:12]
    gy = pd.DataFrame([{"ID": x.id, "Strategy": html.escape(x.name()), "Retired": x.retired_date, "Why dropped": html.escape(x.retired_reason or "")} for x in dead])
    base = pd.DataFrame([{"Fixed strategy (no learning)": k, "CAGR": _pct(v["cagr"]), "Excess/yr": _pct(v["excess_ann"]), "Max DD": _pct(v["mdd"], 0)}
                         for k, v in sorted(m["baselines"].items(), key=lambda kv: -kv[1]["excess_ann"])])
    cal = info["calibration"]
    calt = pd.DataFrame([{"Signal": k, "Mean rank-IC": _num(v["ic"], 3), "IC t-stat": _num(v["ic_t"]), "Slope t-stat": _num(v["t"]),
                          "Used (0-100%)": _pct(info["shrink"][k], 0)} for k, v in cal.items()])
    led = ("<p>No matured forward predictions yet – this run starts the ledger.</p>" if not ledger.get("resolved_dates")
           else f"<p>{ledger['resolved_dates']} dates resolved: BUY picks beat the market by {_pct(ledger['buy_mean_excess'])} on average "
                f"(hit rate {_pct(ledger['buy_hit_rate_vs_market'], 0)}, rank-IC {_num(ledger['mean_rank_ic'], 3)}).</p>")
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in P.notes)
    css = f"""
    :root{{--ink:{INK};--muted:{MUTED};--line:#e3e6ee;--bg:#fff;--soft:#f6f7fb}}
    @media(prefers-color-scheme:dark){{:root{{--ink:#e8ebf3;--muted:#98a0b3;--line:#2a3042;--bg:#12151d;--soft:#1a1e2a}}}}
    body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
    main{{max-width:1040px;margin:0 auto;padding:28px 16px 60px}}
    h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 8px;padding-top:10px;border-top:1px solid var(--line)}}
    .sub{{color:var(--muted);margin:0 0 16px}} img{{max-width:100%;border-radius:8px;background:#fff}}
    .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}}
    .kpi{{background:var(--soft);border-radius:10px;padding:12px 14px}} .kl{{color:var(--muted);font-size:12px}}
    .kv{{font-size:24px;font-weight:650}} .ks{{color:var(--muted);font-size:12px}}
    .tier{{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:650;font-size:13px;color:#fff;background:{tier_col}}}
    .warn{{background:#fff3cd;color:#664d03;border:1px solid #ffe69c;padding:10px 14px;border-radius:8px;margin:12px 0}}
    .tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
    th,td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{color:var(--muted);font-weight:600}}
    td{{vertical-align:top}} td.why{{white-space:normal;min-width:260px}} small,.note{{color:var(--muted);font-size:13px}} li{{margin:3px 0}}
    """
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE Adaptive Research Report</title><style>{css}</style></head><body><main>
<h1>NSE adaptive research report</h1>
<p class="sub">As of {info['as_of']} · decision horizon ≈ {info['horizon_days']} trading days · all returns net of assumed {cfg['market']['one_way_cost_bps']:.0f} bps one-way costs</p>
{banner}
<p>Evidence the system can beat the market: <span class="tier">{tier}</span></p>
<p class="note">Strategies currently get {_pct(info['strategy_conviction'], 0)} of the capital; the rest sits in an equal-weight NSE index sleeve
(plus {_pct(info['cash_weight'], 0)} in T-bills). Capital shifts toward strategies only as their out-of-sample record earns it.</p>
<div class="kpis">{kpis}</div>
<h2>1 · Would it have won? Walk-forward test</h2>
<p class="note">{m['start']} → {m['end']}. Each month the system only used data available at that time: it generated and tested strategies on the preceding {cfg['learning']['train_months']} months, kept winners, dropped losers, then traded the result on unseen data.</p>
<img src="{chart_equity(res)}" alt="Equity curves"><img src="{chart_excess(res)}" alt="Monthly excess return">
<h2>2 · Recommended portfolio now</h2>
<p class="note">Confidence: <b>{info['conviction_label']}</b>. Probabilities come from 4,000 simulated 3-month paths per stock, centred on a calibrated (significance-shrunk) expected return – not a promise. Position sizes are suggestions for KES {cfg['forecast']['capital_kes']:,.0f}.</p>
{_table(fmt(show)) if len(show) else '<p>No positions – portfolio is entirely in T-bills/index sleeve.</p>'}
<h3>Names the model would avoid</h3>
{_table(fmt(avoid)[["Ticker","Name","Exp 3m","P(beat T-bill)","Worst-5% avg","Liquidity"]]) if len(avoid) else '<p class="note">None flagged.</p>'}
<h2>3 · Self-learning: what survived and what was dropped</h2>
<img src="{chart_learning(res)}" alt="Learning curve">
<h3>Living strategies</h3>{_table(board) if len(board) else '<p>None.</p>'}
<h3>Recently retired</h3>{_table(gy) if len(gy) else '<p>None yet.</p>'}
<h3>Do the adaptive layers add value over fixed textbook strategies?</h3>{_table(base)}
<h2>4 · Is the forecast calibrated?</h2>{_table(calt)}
<p class="note">IC = rank correlation between the signal and the next 3-month relative return, measured only on past out-of-sample months. An IC near 0 means the signal is not
distinguishable from noise, so its expected-return contribution is automatically shrunk to ~0.</p>
<h2>5 · Forward ledger (true live test)</h2>{led}
<h2>6 · Data & caveats</h2><ul>{notes}</ul>
<ul class="note"><li>Research tool, not investment advice. No method can guarantee returns; thin NSE liquidity, price limits, T+3 settlement and 15% capital-gains/withholding taxes are only roughly captured.</li>
<li>Delisted/suspended stocks are missing from most free data feeds (survivorship bias flatters backtests). Treat results as an upper bound.</li>
<li>Costs are an assumption ({cfg['market']['one_way_cost_bps']:.0f} bps one-way); confirm with your broker. Dividends only count if your prices are dividend-adjusted.</li></ul>
</main></body></html>"""


def write_outputs(res: WFResult, recs: pd.DataFrame, info: dict, ledger: dict, cfg: dict, out_dir: str | Path,
                  synthetic: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out / "recommendations.csv", index=False)
    (out / "report.html").write_text(build_html(res, recs, info, ledger, cfg, synthetic), encoding="utf-8")
    pd.DataFrame(res.reg.events).to_csv(out / "strategy_events.csv", index=False)
    res.reg.save(out / "strategy_registry.json")
    P, s = res.P, res.r_start
    pd.DataFrame({"date": P.rb_dates[s:], "adaptive": res.net[s:], "benchmark": res.bench_net[s:], "cash": P.cash[s:]}).to_csv(
        out / "oos_monthly_returns.csv", index=False)
    summary = {"as_of": info["as_of"], "tier": res.metrics["tier"], "metrics": json.loads(json.dumps(res.metrics, default=float)),
               "calibration": info["calibration"], "ledger": ledger, "synthetic": synthetic}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return {"report": str(out / "report.html"), "recommendations": str(out / "recommendations.csv")}
