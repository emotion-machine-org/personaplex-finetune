"""Compare two batch_eval runs with statistical tests and control-chart visuals.

Takes two results.json files (from batch_eval.py) and produces:
  - Paired Wilcoxon signed-rank tests per metric (with Holm-Bonferroni correction)
  - Cliff's delta effect sizes
  - Bootstrap 95% CIs on the mean difference
  - Control chart: run A as baseline, run B plotted against ±2σ/±3σ limits
  - Per-scenario breakdown

Usage:
    python pipeline/compare_runs.py \
        --run-a runs/compare_base/results.json \
        --run-b runs/compare_lora/results.json \
        --output runs/comparison_report

    Labels default to directory names; override with --label-a / --label-b.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_per_dialogue_metrics(run: dict) -> dict[str, dict]:
    """Build {dialogue_id: {metric: value}} from a run's results.

    dialogue_id is (scenario_id, seed) so we can pair across runs.
    """
    dialogues = run["dialogues"]
    reviews_by_id = {r["id"]: r for r in run.get("reviews", [])}
    silence_by_id = {
        s["id"]: s for s in run.get("silence", {}).get("per_dialogue", [])
    }

    records = {}
    for d in dialogues:
        if d["status"] != "ok":
            continue
        key = (d["scenario_id"], d["seed"])
        rec = {}

        # Silence metrics
        si = silence_by_id.get(d["id"], {})
        if "silence_pct" in si:
            rec["silence_pct"] = si["silence_pct"]
        if "longest_silence_s" in si:
            rec["longest_silence_s"] = si["longest_silence_s"]

        # LLM review scores
        rev = reviews_by_id.get(d["id"], {})
        for m in ("coherence", "naturalness", "effectiveness", "grounding"):
            v = rev.get(m)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                rec[m] = v

        records[key] = rec

    return records


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples. Returns (statistic, p-value)."""
    from scipy.stats import wilcoxon
    diff = b - a
    # wilcoxon requires non-zero differences
    nonzero = diff[diff != 0]
    if len(nonzero) < 5:
        return float("nan"), float("nan")
    stat, p = wilcoxon(nonzero, alternative="two-sided")
    return float(stat), float(p)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta effect size (non-parametric).

    Returns (delta, magnitude) where magnitude is one of:
    negligible, small, medium, large.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan"), "n/a"
    # Vectorized pairwise comparison
    more = np.sum(np.subtract.outer(b, a) > 0)
    less = np.sum(np.subtract.outer(b, a) < 0)
    delta = (more - less) / (n_a * n_b)

    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"

    return float(delta), mag


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap CI on mean(b) - mean(a). Returns (mean_diff, ci_low, ci_high)."""
    rng = np.random.default_rng(42)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(b[idx]) - np.mean(a[idx])
    alpha = 1 - ci
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return float(np.mean(b) - np.mean(a)), lo, hi


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


# ---------------------------------------------------------------------------
# Control chart helpers
# ---------------------------------------------------------------------------

def control_limits(values: np.ndarray) -> dict:
    """Compute X-bar control chart limits from baseline values."""
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    return {
        "mean": mu,
        "sigma": sigma,
        "ucl_2s": mu + 2 * sigma,
        "lcl_2s": mu - 2 * sigma,
        "ucl_3s": mu + 3 * sigma,
        "lcl_3s": mu - 3 * sigma,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(run_a: dict, run_b: dict, label_a: str, label_b: str) -> dict:
    records_a = extract_per_dialogue_metrics(run_a)
    records_b = extract_per_dialogue_metrics(run_b)

    # Find paired keys (scenario_id, seed) present in both runs
    paired_keys = sorted(set(records_a.keys()) & set(records_b.keys()))
    if not paired_keys:
        print("ERROR: No paired dialogues found (no matching scenario+seed combinations)")
        sys.exit(1)

    # Determine metrics present in both
    all_metrics = set()
    for k in paired_keys:
        all_metrics |= set(records_a[k].keys()) & set(records_b[k].keys())
    metrics = sorted(all_metrics)

    print(f"Paired dialogues: {len(paired_keys)}")
    print(f"Metrics: {metrics}")

    # Build paired arrays per metric
    arrays_a = defaultdict(list)
    arrays_b = defaultdict(list)
    for k in paired_keys:
        for m in metrics:
            if m in records_a[k] and m in records_b[k]:
                arrays_a[m].append(records_a[k][m])
                arrays_b[m].append(records_b[k][m])

    # Run tests
    results = {}
    raw_pvals = []
    metric_order = []

    for m in metrics:
        a = np.array(arrays_a[m], dtype=float)
        b = np.array(arrays_b[m], dtype=float)

        stat, pval = paired_wilcoxon(a, b)
        delta, mag = cliffs_delta(a, b)
        mean_diff, ci_lo, ci_hi = bootstrap_ci(a, b)
        ctrl = control_limits(a)

        # Count how many B values fall outside A's control limits
        b_outside_2s = int(np.sum((b > ctrl["ucl_2s"]) | (b < ctrl["lcl_2s"])))
        b_outside_3s = int(np.sum((b > ctrl["ucl_3s"]) | (b < ctrl["lcl_3s"])))

        results[m] = {
            "n_paired": len(a),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "std_a": float(np.std(a, ddof=1)),
            "std_b": float(np.std(b, ddof=1)),
            "wilcoxon_stat": stat,
            "wilcoxon_p": pval,
            "cliffs_delta": delta,
            "cliffs_magnitude": mag,
            "bootstrap_mean_diff": mean_diff,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "control_chart": ctrl,
            "b_outside_2sigma": b_outside_2s,
            "b_outside_3sigma": b_outside_3s,
        }
        raw_pvals.append(pval)
        metric_order.append(m)

    # Holm-Bonferroni correction
    adjusted = holm_bonferroni(raw_pvals)
    for m, adj_p in zip(metric_order, adjusted):
        results[m]["wilcoxon_p_adjusted"] = adj_p

    # Per-scenario breakdown
    scenarios = sorted(set(k[0] for k in paired_keys))
    per_scenario = {}
    for sc in scenarios:
        sc_keys = [k for k in paired_keys if k[0] == sc]
        sc_results = {}
        for m in metrics:
            a_vals = [records_a[k][m] for k in sc_keys if m in records_a[k] and m in records_b[k]]
            b_vals = [records_b[k][m] for k in sc_keys if m in records_a[k] and m in records_b[k]]
            if a_vals:
                sc_results[m] = {
                    "mean_a": float(np.mean(a_vals)),
                    "mean_b": float(np.mean(b_vals)),
                    "diff": float(np.mean(b_vals) - np.mean(a_vals)),
                }
        per_scenario[sc] = sc_results

    return {
        "labels": {"a": label_a, "b": label_b},
        "n_paired": len(paired_keys),
        "metrics": results,
        "per_scenario": per_scenario,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_report(report: dict):
    la = report["labels"]["a"]
    lb = report["labels"]["b"]

    print()
    print("=" * 72)
    print(f"  COMPARISON: {la}  vs  {lb}")
    print(f"  Paired dialogues: {report['n_paired']}")
    print("=" * 72)

    # Header
    print(f"\n{'Metric':<22} {'Mean A':>8} {'Mean B':>8} {'Diff':>8} "
          f"{'CI 95%':>16} {'Cliff δ':>8} {'Mag':>7} "
          f"{'p-val':>8} {'p-adj':>8} {'Sig':>4}")
    print("-" * 115)

    for m, data in sorted(report["metrics"].items()):
        ci = data["bootstrap_ci_95"]
        sig = ""
        p_adj = data["wilcoxon_p_adjusted"]
        if not math.isnan(p_adj):
            if p_adj < 0.01:
                sig = "**"
            elif p_adj < 0.05:
                sig = "*"

        p_raw = data["wilcoxon_p"]
        p_raw_s = f"{p_raw:.4f}" if not math.isnan(p_raw) else "N/A"
        p_adj_s = f"{p_adj:.4f}" if not math.isnan(p_adj) else "N/A"
        delta_s = f"{data['cliffs_delta']:+.3f}" if not math.isnan(data['cliffs_delta']) else "N/A"

        print(f"{m:<22} {data['mean_a']:>8.2f} {data['mean_b']:>8.2f} "
              f"{data['bootstrap_mean_diff']:>+8.2f} "
              f"[{ci[0]:>+6.2f}, {ci[1]:>+6.2f}] "
              f"{delta_s:>8} {data['cliffs_magnitude']:>7} "
              f"{p_raw_s:>8} {p_adj_s:>8} {sig:>4}")

    # Control chart summary
    print(f"\n{'CONTROL CHART (A = baseline)':}")
    print(f"{'Metric':<22} {'A mean':>8} {'A σ':>8} {'B outside 2σ':>14} {'B outside 3σ':>14}")
    print("-" * 72)
    for m, data in sorted(report["metrics"].items()):
        ctrl = data["control_chart"]
        print(f"{m:<22} {ctrl['mean']:>8.2f} {ctrl['sigma']:>8.2f} "
              f"{data['b_outside_2sigma']:>14} {data['b_outside_3sigma']:>14}")

    # Grounding rate (if provenance available)
    if "grounding" in report:
        gr = report["grounding"]
        print(f"\n{'GROUNDING RATE (factual claims traceable to provenance)':}")
        for label_key, label_name in [("a", la), ("b", lb)]:
            g = gr[label_key]
            if g["n_dialogues"] > 0:
                print(f"  {label_name}: {g['grounding_rate']:.1%} "
                      f"({g['total_grounded']}/{g['total_claims']} claims, "
                      f"{g['n_dialogues']} dialogues)")
            else:
                print(f"  {label_name}: N/A (no provenance data)")

    # Per-scenario
    print(f"\n{'PER-SCENARIO MEAN DIFFERENCES (B - A)':}")
    scenarios = sorted(report["per_scenario"].keys())
    metrics = sorted(report["metrics"].keys())
    header = f"{'Scenario':<14}" + "".join(f"{m:>18}" for m in metrics)
    print(header)
    print("-" * (14 + 18 * len(metrics)))
    for sc in scenarios:
        row = f"{sc:<14}"
        for m in metrics:
            d = report["per_scenario"][sc].get(m, {})
            diff = d.get("diff")
            if diff is not None:
                row += f"{diff:>+18.2f}"
            else:
                row += f"{'N/A':>18}"
        print(row)

    print()


def generate_html(report: dict, run_a: dict, run_b: dict) -> str:
    """Generate self-contained HTML dashboard with embedded Chart.js."""
    la = report["labels"]["a"]
    lb = report["labels"]["b"]
    metrics = sorted(report["metrics"].keys())

    # Build per-dialogue paired data for scatter/strip charts
    records_a = extract_per_dialogue_metrics(run_a)
    records_b = extract_per_dialogue_metrics(run_b)
    paired_keys = sorted(set(records_a.keys()) & set(records_b.keys()))

    # Prepare chart data per metric
    chart_data = {}
    for m in metrics:
        vals_a = []
        vals_b = []
        labels = []
        for k in paired_keys:
            if m in records_a[k] and m in records_b[k]:
                vals_a.append(records_a[k][m])
                vals_b.append(records_b[k][m])
                labels.append(f"{k[0]}/seed{k[1]}")
        chart_data[m] = {"a": vals_a, "b": vals_b, "labels": labels}

    # Build summary table rows
    table_rows = ""
    for m in metrics:
        d = report["metrics"][m]
        ci = d["bootstrap_ci_95"]
        p_adj = d["wilcoxon_p_adjusted"]
        sig_class = ""
        sig_text = ""
        if not math.isnan(p_adj):
            if p_adj < 0.01:
                sig_class = "sig-high"
                sig_text = "**"
            elif p_adj < 0.05:
                sig_class = "sig-med"
                sig_text = "*"

        p_raw_s = f"{d['wilcoxon_p']:.4f}" if not math.isnan(d['wilcoxon_p']) else "N/A"
        p_adj_s = f"{p_adj:.4f}" if not math.isnan(p_adj) else "N/A"
        delta_s = f"{d['cliffs_delta']:+.3f}" if not math.isnan(d['cliffs_delta']) else "N/A"
        diff_s = f"{d['bootstrap_mean_diff']:+.2f}"
        ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"

        table_rows += f"""<tr class="{sig_class}">
            <td>{m}</td><td>{d['mean_a']:.2f}</td><td>{d['mean_b']:.2f}</td>
            <td>{diff_s}</td><td>{ci_s}</td><td>{delta_s}</td>
            <td>{d['cliffs_magnitude']}</td><td>{p_raw_s}</td><td>{p_adj_s}</td>
            <td>{sig_text}</td></tr>"""

    # Control chart table
    ctrl_rows = ""
    for m in metrics:
        d = report["metrics"][m]
        ctrl = d["control_chart"]
        ctrl_rows += f"""<tr>
            <td>{m}</td><td>{ctrl['mean']:.2f}</td><td>{ctrl['sigma']:.2f}</td>
            <td>{d['b_outside_2sigma']}</td><td>{d['b_outside_3sigma']}</td></tr>"""

    # Per-scenario table
    scenarios = sorted(report["per_scenario"].keys())
    scenario_header = "".join(f"<th>{m}</th>" for m in metrics)
    scenario_rows = ""
    for sc in scenarios:
        cells = ""
        for m in metrics:
            diff = report["per_scenario"][sc].get(m, {}).get("diff")
            if diff is not None:
                color = "#c0392b" if diff < -0.3 else "#27ae60" if diff > 0.3 else "#7f8c8d"
                cells += f'<td style="color:{color}">{diff:+.2f}</td>'
            else:
                cells += "<td>N/A</td>"
        scenario_rows += f"<tr><td>{sc}</td>{cells}</tr>"

    chart_data_json = json.dumps(chart_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Run Comparison: {la} vs {lb}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; font-size: 1.6em; }}
  h2 {{ color: #8b949e; margin: 28px 0 12px; font-size: 1.1em; text-transform: uppercase;
        letter-spacing: 0.05em; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.95em; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.85em; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 8px 10px;
       font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.04em; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #161b22; }}
  .sig-high {{ background: #1a0f0f !important; }}
  .sig-high td {{ color: #f85149; }}
  .sig-med {{ background: #1a1a0f !important; }}
  .sig-med td {{ color: #d29922; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; margin: 20px 0; }}
  .chart-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; }}
  .chart-card h3 {{ color: #c9d1d9; font-size: 0.95em; margin-bottom: 10px; }}
  canvas {{ max-height: 260px; }}
  .legend {{ display: flex; gap: 20px; margin: 16px 0; font-size: 0.9em; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<h1>Run Comparison Dashboard</h1>
<p class="subtitle">
  <strong>{la}</strong> (A) &nbsp;vs&nbsp; <strong>{lb}</strong> (B)
  &mdash; {report['n_paired']} paired dialogues
</p>

<div class="legend">
  <span><span class="dot" style="background:#58a6ff"></span> {la}</span>
  <span><span class="dot" style="background:#f97583"></span> {lb}</span>
</div>

<h2>Statistical Summary</h2>
<table>
<tr><th>Metric</th><th>Mean A</th><th>Mean B</th><th>Diff</th><th>95% CI</th>
    <th>Cliff's &delta;</th><th>Magnitude</th><th>p-value</th><th>p-adj (Holm)</th><th>Sig</th></tr>
{table_rows}
</table>

<h2>Paired Comparison Charts</h2>
<div class="charts" id="charts"></div>

<h2>Control Chart Summary (A = baseline)</h2>
<table>
<tr><th>Metric</th><th>A Mean</th><th>A &sigma;</th><th>B outside &plusmn;2&sigma;</th><th>B outside &plusmn;3&sigma;</th></tr>
{ctrl_rows}
</table>

<h2>Control Charts</h2>
<div class="charts" id="control-charts"></div>

<h2>Per-Scenario Mean Differences (B &minus; A)</h2>
<table>
<tr><th>Scenario</th>{scenario_header}</tr>
{scenario_rows}
</table>

<script>
const chartData = {chart_data_json};
const metrics = {json.dumps(metrics)};
const report = {json.dumps(report["metrics"])};
const labelA = {json.dumps(la)};
const labelB = {json.dumps(lb)};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

// Paired comparison: dumbbell / dot plot per dialogue
const chartsDiv = document.getElementById('charts');
metrics.forEach(m => {{
  const card = document.createElement('div');
  card.className = 'chart-card';
  const d = report[m];
  const sig = d.wilcoxon_p_adjusted < 0.05 ? ' *' : '';
  card.innerHTML = `<h3>${{m}}${{sig}} (diff: ${{d.bootstrap_mean_diff >= 0 ? '+' : ''}}${{d.bootstrap_mean_diff.toFixed(2)}})</h3><canvas id="pair-${{m}}"></canvas>`;
  chartsDiv.appendChild(card);

  const cd = chartData[m];
  const indices = cd.a.map((_, i) => i);

  new Chart(document.getElementById(`pair-${{m}}`), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{ label: labelA, data: indices.map((i, idx) => ({{ x: cd.a[idx], y: idx }})),
           backgroundColor: '#58a6ff', pointRadius: 5 }},
        {{ label: labelB, data: indices.map((i, idx) => ({{ x: cd.b[idx], y: idx }})),
           backgroundColor: '#f97583', pointRadius: 5 }},
      ]
    }},
    options: {{
      indexAxis: 'y',
      scales: {{
        y: {{ display: false }},
        x: {{ title: {{ display: true, text: m }} }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(2)}} (${{cd.labels[ctx.parsed.y]}})`
          }}
        }}
      }}
    }}
  }});
}});

// Control charts
const ctrlDiv = document.getElementById('control-charts');
metrics.forEach(m => {{
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = `<h3>${{m}} &mdash; Control Chart</h3><canvas id="ctrl-${{m}}"></canvas>`;
  ctrlDiv.appendChild(card);

  const d = report[m];
  const ctrl = d.control_chart;
  const cd = chartData[m];
  const n = cd.b.length;
  const indices = Array.from({{length: n}}, (_, i) => i);

  const annotations = {{
    mean: {{ type: 'line', yMin: ctrl.mean, yMax: ctrl.mean,
             borderColor: '#58a6ff', borderWidth: 1, borderDash: [4,4],
             label: {{ display: true, content: 'A mean', position: 'start', color: '#58a6ff', font: {{size: 10}} }} }},
    ucl2: {{ type: 'line', yMin: ctrl.ucl_2s, yMax: ctrl.ucl_2s,
             borderColor: '#d29922', borderWidth: 1, borderDash: [3,3],
             label: {{ display: true, content: '+2\u03c3', position: 'start', color: '#d29922', font: {{size: 9}} }} }},
    lcl2: {{ type: 'line', yMin: ctrl.lcl_2s, yMax: ctrl.lcl_2s,
             borderColor: '#d29922', borderWidth: 1, borderDash: [3,3],
             label: {{ display: true, content: '-2\u03c3', position: 'start', color: '#d29922', font: {{size: 9}} }} }},
    ucl3: {{ type: 'line', yMin: ctrl.ucl_3s, yMax: ctrl.ucl_3s,
             borderColor: '#f85149', borderWidth: 1, borderDash: [2,2],
             label: {{ display: true, content: '+3\u03c3', position: 'start', color: '#f85149', font: {{size: 9}} }} }},
    lcl3: {{ type: 'line', yMin: ctrl.lcl_3s, yMax: ctrl.lcl_3s,
             borderColor: '#f85149', borderWidth: 1, borderDash: [2,2],
             label: {{ display: true, content: '-3\u03c3', position: 'start', color: '#f85149', font: {{size: 9}} }} }},
  }};

  // Color points: green if inside 2s, yellow if between 2s-3s, red if outside 3s
  const colors = cd.b.map(v => {{
    if (v > ctrl.ucl_3s || v < ctrl.lcl_3s) return '#f85149';
    if (v > ctrl.ucl_2s || v < ctrl.lcl_2s) return '#d29922';
    return '#3fb950';
  }});

  new Chart(document.getElementById(`ctrl-${{m}}`), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: labelB,
        data: indices.map(i => ({{x: i, y: cd.b[i]}})),
        backgroundColor: colors,
        pointRadius: 6,
      }}]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Dialogue #' }}, ticks: {{ stepSize: 1 }} }},
        y: {{ title: {{ display: true, text: m }} }}
      }},
      plugins: {{
        legend: {{ display: false }},
        annotation: {{ annotations }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{cd.labels[ctx.parsed.x]}}: ${{ctx.parsed.y.toFixed(2)}}`
          }}
        }}
      }}
    }},
    plugins: [{{ id: 'annotation' }}]
  }});
}});
</script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
</body>
</html>"""


def compute_grounding_rate(run: dict) -> dict:
    """Extract factual claims from broker transcripts and check against provenance.

    For each dialogue that has provenance (system_prompt + context_injections),
    extracts numbers, dates, names, and policy references from the broker
    transcript, then checks each against the provided reference material.

    Returns {dialogue_id: {n_claims, n_grounded, grounding_rate}}.
    """
    import re

    # Pattern for factual claims: dollar amounts, percentages, dates, policy/form numbers
    CLAIM_PATTERNS = [
        (r'\$[\d,]+(?:\.\d{2})?', 'dollar'),         # $13,100 or $4,800.00
        (r'\d+(?:\.\d+)?%', 'percent'),                # 12.3% or 3%
        (r'\d{1,2}/\d{1,2}/\d{2,4}', 'date'),         # 3/15/2026
        (r'(?:policy|form|endorsement)\s*#?\s*\w+', 'reference'),  # policy 4471
    ]

    results = {}
    for d in run.get("dialogues", []):
        if d["status"] != "ok":
            continue
        system_prompt = d.get("system_prompt", "")
        injections = d.get("context_injections", [])
        if not system_prompt:
            continue

        # Build reference text from all provenance sources
        reference = system_prompt.lower()
        for inj in injections:
            if isinstance(inj, dict):
                reference += " " + inj.get("text", "").lower()

        # Read broker transcript
        try:
            with open(d["text_a_path"]) as f:
                tokens = json.load(f)
            transcript = "".join(
                t for t in tokens if t not in ("EPAD", "BOS", "EOS", "PAD", "<CTX>")
            ).strip().lower()
        except Exception:
            continue

        # Extract claims from transcript
        claims = []
        for pattern, claim_type in CLAIM_PATTERNS:
            for match in re.finditer(pattern, transcript, re.IGNORECASE):
                claims.append({"text": match.group(), "type": claim_type})

        if not claims:
            continue

        # Check each claim against reference material
        n_grounded = 0
        for claim in claims:
            # Normalize: strip $ and commas for numeric comparison
            claim_text = claim["text"].replace(",", "").replace("$", "")
            if claim_text in reference.replace(",", "").replace("$", ""):
                n_grounded += 1

        results[d["id"]] = {
            "n_claims": len(claims),
            "n_grounded": n_grounded,
            "grounding_rate": n_grounded / len(claims) if claims else 1.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two batch_eval runs")
    parser.add_argument("--run-a", required=True, help="Path to run A results.json")
    parser.add_argument("--run-b", required=True, help="Path to run B results.json")
    parser.add_argument("--label-a", default=None, help="Label for run A")
    parser.add_argument("--label-b", default=None, help="Label for run B")
    parser.add_argument("--output", default=None, help="Output directory for report JSON + HTML")

    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)

    label_a = args.label_a or Path(args.run_a).parent.name
    label_b = args.label_b or Path(args.run_b).parent.name

    report = compare(run_a, run_b, label_a, label_b)

    # Compute grounding rates if provenance is available
    gr_a = compute_grounding_rate(run_a)
    gr_b = compute_grounding_rate(run_b)
    if gr_a or gr_b:
        def _agg_gr(gr):
            if not gr:
                return {"grounding_rate": float("nan"), "n_dialogues": 0}
            rates = [v["grounding_rate"] for v in gr.values()]
            return {
                "grounding_rate": sum(rates) / len(rates),
                "n_dialogues": len(rates),
                "total_claims": sum(v["n_claims"] for v in gr.values()),
                "total_grounded": sum(v["n_grounded"] for v in gr.values()),
            }
        report["grounding"] = {"a": _agg_gr(gr_a), "b": _agg_gr(gr_b)}

    print_report(report)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / "comparison.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {report_path}")

        html_path = out_dir / "dashboard.html"
        html = generate_html(report, run_a, run_b)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"Dashboard saved to {html_path}")


if __name__ == "__main__":
    main()
