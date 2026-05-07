"""
compare_runs.py
───────────────
Parse one or more training-log text files and generate a self-contained
HTML comparison report (charts + tables).

USAGE
─────
1. Put each run's log in a separate .txt file.
   The first line of each file should be a label for that run, e.g.:
       === No augmentation ===
   (anything between === and === is used as the run name)
   If no such line is found the filename is used as the label.

2. Edit INPUT_FILES at the bottom of this script to list your files.

3. Run:
       python compare_runs.py

4. Open  comparison_report.html  in your browser.
"""

import re, json, math, os, sys
from pathlib import Path

# ── colours (one per run) ────────────────────────────────────────────────────
PALETTE = ["#0D4DBB", "#E07A00", "#0FA077", "#5A5A5A", "#8C2DAA", "#D53333"]

# ── subtypes that are "reliable" enough to show in subtype charts ─────────────
MIN_N = 5

# ═════════════════════════════════════════════════════════════════════════════
# PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_log(text: str, filename: str) -> dict:
    """Extract everything we care about from one log dump."""
    d = {}

    # ── run label ────────────────────────────────────────────────────────────
    m = re.search(r"===\s*(.+?)\s*===", text)
    d["label"] = m.group(1).strip() if m else Path(filename).stem

    # ── split sizes ──────────────────────────────────────────────────────────
    m = re.search(r"Split sizes:\s*train=(\d+),\s*val=(\d+),\s*test=(\d+)", text)
    if m:
        d["train_n"], d["val_n"], d["test_n"] = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # ── label distribution test ───────────────────────────────────────────────
    m = re.search(r"Label distribution test=\{0:\s*(\d+),\s*1:\s*(\d+)\}", text)
    if m:
        d["test_neg"], d["test_pos"] = int(m.group(1)), int(m.group(2))

    # ── overall test metrics ─────────────────────────────────────────────────
    m = re.search(
        r"test_loss=([\d.]+)\s+test_accuracy=([\d.]+)\s+auc=([\d.]+)\s+"
        r"aupr=([\d.]+)\s+precision=([\d.]+)\s+recall=([\d.]+)\s+f1=([\d.]+)",
        text,
    )
    if m:
        keys = ["test_loss","test_accuracy","auc","aupr","precision","recall","f1"]
        d["overall"] = {k: float(v) for k, v in zip(keys, m.groups())}

    # ── early stopping epoch ─────────────────────────────────────────────────
    m = re.search(r"epoch=(\d+).*?\(STOP\)", text)
    d["stop_epoch"] = int(m.group(1)) if m else None

    # ── per-subtype table ─────────────────────────────────────────────────────
    subtypes = {}
    # find the table block
    table_m = re.search(r"Per-cas_subtype test metrics:(.*)", text, re.DOTALL)
    if table_m:
        for row in re.finditer(
            r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.nan]+)\s+([\d.nan]+)\s+"
            r"([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)",
            table_m.group(1), re.MULTILINE
        ):
            st = row.group(1)
            def fv(s):
                try: return float(s)
                except: return float("nan")
            subtypes[st] = {
                "n":         int(row.group(2)),
                "accuracy":  fv(row.group(3)),
                "auc":       fv(row.group(4)),
                "aupr":      fv(row.group(5)),
                "precision": fv(row.group(6)),
                "recall":    fv(row.group(7)),
                "f1":        fv(row.group(8)),
            }
    d["subtypes"] = subtypes
    return d


# ═════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def safe(v):
    """Format float; blank for nan."""
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return f"{v:.4f}"

def pct(v):
    return f"{v*100:.1f}%"

def _js_array(values):
    def jv(x):
        if isinstance(x, float) and math.isnan(x):
            return "null"
        return str(round(x, 4))
    return "[" + ", ".join(jv(x) for x in values) + "]"


def build_html(runs: list[dict]) -> str:
    n_runs = len(runs)
    colors = PALETTE[:n_runs]

    # ── shared reliable subtypes ──────────────────────────────────────────────
    all_sts = {}
    for run in runs:
        for st, vals in run["subtypes"].items():
            all_sts[st] = max(all_sts.get(st, 0), vals["n"])
    reliable = sorted([st for st, n in all_sts.items() if n >= MIN_N])

    # ── legend HTML ──────────────────────────────────────────────────────────
    legend_items = "".join(
        f'<span><span class="dot" style="background:{c}"></span>{r["label"]}</span>'
        for r, c in zip(runs, colors)
    )

    # ── overall metrics table ─────────────────────────────────────────────────
    metrics = ["test_loss","test_accuracy","auc","aupr","precision","recall","f1"]
    metric_labels = ["Loss ↓","Accuracy","AUC","AUPR","Precision","Recall","F1"]
    higher_better = [False, True, True, True, True, True, True]

    def best_idx(metric, hb):
        vals = [r["overall"].get(metric, float("nan")) for r in runs]
        valid = [(v, i) for i, v in enumerate(vals) if not math.isnan(v)]
        if not valid: return -1
        return max(valid, key=lambda x: x[0])[1] if hb else min(valid, key=lambda x: x[0])[1]

    overall_rows = ""
    for m_key, m_lbl, hb in zip(metrics, metric_labels, higher_better):
        bi = best_idx(m_key, hb)
        cells = ""
        for i, (run, c) in enumerate(zip(runs, colors)):
            v = run["overall"].get(m_key, float("nan"))
            bold = "font-weight:700;" if i == bi else ""
            bg = f"background:{c}22;" if i == bi else ""
            disp = pct(v) if m_key == "test_accuracy" else safe(v)
            cells += f'<td style="color:{c};{bold}{bg}">{disp}</td>'
        overall_rows += f"<tr><td>{m_lbl}</td>{cells}</tr>"

    run_headers = "".join(f'<th style="color:{c}">{r["label"]}</th>' for r, c in zip(runs, colors))

    # ── split info table ──────────────────────────────────────────────────────
    split_rows = ""
    for key, lbl in [("train_n","Train size"),("val_n","Val size"),("test_n","Test size"),("stop_epoch","Stop epoch")]:
        cells = "".join(f"<td>{run.get(key,'—')}</td>" for run in runs)
        split_rows += f"<tr><td>{lbl}</td>{cells}</tr>"

    # ── subtype table (F1 + accuracy) ─────────────────────────────────────────
    subtype_rows = ""
    for st in reliable:
        f1_vals  = [run["subtypes"].get(st, {}).get("f1",  float("nan")) for run in runs]
        acc_vals = [run["subtypes"].get(st, {}).get("accuracy", float("nan")) for run in runs]
        n_val    = max(run["subtypes"].get(st, {}).get("n", 0) for run in runs)

        def best_cell(vals, row_vals):
            valid = [(v,i) for i,v in enumerate(vals) if not math.isnan(v)]
            if not valid: return -1
            return max(valid, key=lambda x: x[0])[1]

        bi_f1  = best_cell(f1_vals, f1_vals)
        bi_acc = best_cell(acc_vals, acc_vals)

        f1_cells = ""
        for i, (v, c) in enumerate(zip(f1_vals, colors)):
            bold = "font-weight:700;" if i == bi_f1 else ""
            bg   = f"background:{c}22;" if i == bi_f1 else ""
            f1_cells += f'<td style="color:{c};{bold}{bg}">{safe(v)}</td>'

        acc_cells = ""
        for i, (v, c) in enumerate(zip(acc_vals, colors)):
            bold = "font-weight:700;" if i == bi_acc else ""
            bg   = f"background:{c}22;" if i == bi_acc else ""
            acc_cells += f'<td style="color:{c};{bold}{bg}">{safe(v)}</td>'

        subtype_rows += f"<tr><td><strong>{st}</strong></td><td>{n_val}</td>{f1_cells}{acc_cells}</tr>"

    f1_acc_headers = (
        "".join(f'<th style="color:{c}">F1</th>' for c in colors) +
        "".join(f'<th style="color:{c}">Acc</th>' for c in colors)
    )

    # ── Chart.js datasets ─────────────────────────────────────────────────────
    def radar_ds(run, color, metric):
        vals = [run["subtypes"].get(st, {}).get(metric, float("nan")) for st in reliable]
        return (
            f'{{"label":"{run["label"]}",'
            f'"data":{_js_array(vals)},'
            f'"borderColor":"{color}",'
            f'"backgroundColor":"transparent",'
            f'"fill":false,"borderWidth":2,"pointRadius":3,'
            f'"pointBackgroundColor":"{color}"}}'
        )

    def bar_ds(run, color, metric):
        vals = [run["subtypes"].get(st, {}).get(metric, 0) for st in reliable]
        return (
            f'{{"label":"{run["label"]}",'
            f'"data":{_js_array(vals)},'
            f'"backgroundColor":"{color}"}}'
        )

    radar_labels_js = json.dumps(reliable)
    f1_radar_ds  = ",".join(radar_ds(r, c, "f1")       for r, c in zip(runs, colors))
    acc_radar_ds = ",".join(radar_ds(r, c, "accuracy")  for r, c in zip(runs, colors))
    f1_bar_ds    = ",".join(bar_ds(r,  c, "f1")         for r, c in zip(runs, colors))
    acc_bar_ds   = ",".join(bar_ds(r,  c, "accuracy")   for r, c in zip(runs, colors))

    overall_radar_labels = json.dumps(["AUC","AUPR","F1","Recall","Precision"])
    def overall_radar_ds(run, color):
        o = run.get("overall", {})
        vals = [o.get(k, 0) for k in ["auc","aupr","f1","recall","precision"]]
        return (
            f'{{"label":"{run["label"]}",'
            f'"data":{_js_array(vals)},'
            f'"borderColor":"{color}",'
            f'"backgroundColor":"transparent",'
            f'"fill":false,"borderWidth":2,"pointRadius":3,'
            f'"pointBackgroundColor":"{color}"}}'
        )
    overall_radar_ds_str = ",".join(overall_radar_ds(r, c) for r, c in zip(runs, colors))

    # ── assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Run Comparison Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; background: #f4f5f6; color: #1a1a18;
         margin: 0; padding: 2rem 1.5rem; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1rem; font-weight: 500; color: #444; margin: 2rem 0 0.6rem; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 1.25rem; font-size: 13px; color: #666; }}
  .legend span {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; border: 1px solid #cfcfcf; }}
  table {{ width: auto; max-width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; margin-bottom: 0.5rem; border: 1px solid #ddd; table-layout: auto; }}
  thead th {{ text-align: left; padding: 6px 6px; color: #444; font-weight: 600; border-bottom: 2px solid #cfcfcf; background: #f6f6f6; font-size:13px; white-space:nowrap; }}
  th {{ text-align: left; padding: 6px 6px; color: #444; font-weight: 600; font-size:13px; white-space:nowrap; }}
  td {{ padding: 4px 6px; border-top: 1px solid #e6e6e6; background: #fff; font-size:14px; white-space:nowrap; }}
  tbody tr:nth-child(even) td {{ background: #fbfbfa; }}
  tr:last-child td {{ border-bottom: none; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px #0001; border: 1px solid #e9e9e9; }}
  .card-title {{ font-size: 12px; font-weight: 500; color: #666; margin-bottom: 6px; text-align: center; }}
  .chart-wrap {{ position: relative; width: 100%; height: 200px; max-width:100%; }}
  .chart-wrap-lg {{ position: relative; width: 100%; height: 260px; max-width:100%; }}
  @media (max-width: 680px) {{
    .grid2, .grid3 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<h1>Run Comparison Report</h1>
<div class="legend">{legend_items}</div>

<h2>Overall metrics</h2>
<div class="card" style="margin-bottom:1rem">
<table>
  <thead><tr><th>Metric</th>{run_headers}</tr></thead>
  <tbody>{overall_rows}</tbody>
</table>
</div>

<h2>Overall profile (radar)</h2>
<div class="card" style="margin-bottom:1rem">
  <div class="chart-wrap" style="max-width:460px;margin:auto">
    <canvas id="overallRadar"></canvas>
  </div>
</div>

<h2>Split &amp; training info</h2>
<div class="card" style="margin-bottom:1rem">
<table>
  <thead><tr><th>Info</th>{run_headers}</tr></thead>
  <tbody>{split_rows}</tbody>
</table>
</div>

<h2>Per-subtype metrics (n ≥ {MIN_N})</h2>
<div class="card" style="margin-bottom:1rem">
<table>
  <thead><tr><th>Subtype</th><th>n</th>{f1_acc_headers}</tr></thead>
  <tbody>{subtype_rows}</tbody>
</table>
</div>

<h2>Subtype F1 — radars &amp; bar chart</h2>
<div class="grid2" style="margin-bottom:1rem">
  <div class="card">
    <div class="card-title">F1 per subtype (radar)</div>
    <div class="chart-wrap"><canvas id="f1Radar"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Accuracy per subtype (radar)</div>
    <div class="chart-wrap"><canvas id="accRadar"></canvas></div>
  </div>
</div>
<div class="card" style="margin-bottom:1rem">
  <div class="card-title">F1 per subtype (bar)</div>
  <div class="chart-wrap-lg"><canvas id="f1Bar"></canvas></div>
</div>
<div class="card" style="margin-bottom:2rem">
  <div class="card-title">Accuracy per subtype (bar)</div>
  <div class="chart-wrap-lg"><canvas id="accBar"></canvas></div>
</div>

<script>
const radarLabels = {radar_labels_js};
const overallLabels = {overall_radar_labels};

const radarOpts = {{
  responsive:true, maintainAspectRatio:false,
  scales:{{ r:{{ min:0, max:1,
    ticks:{{ stepSize:0.25, font:{{size:9}}, color:'#888', backdropColor:'transparent' }},
    grid:{{ color:'rgba(136,135,128,0.2)' }},
    angleLines:{{ color:'rgba(136,135,128,0.2)' }},
    pointLabels:{{ font:{{size:10}}, color:'#666' }}
  }}}},
  plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.r != null ? ctx.parsed.r.toFixed(3) : 'n/a'}}` }} }} }}
}};

const overallRadarOpts = {{
  responsive:true, maintainAspectRatio:false,
  scales:{{ r:{{ min:0.4, max:1.0,
    ticks:{{ stepSize:0.1, font:{{size:9}}, color:'#888', backdropColor:'transparent' }},
    grid:{{ color:'rgba(136,135,128,0.2)' }},
    angleLines:{{ color:'rgba(136,135,128,0.2)' }},
    pointLabels:{{ font:{{size:11}}, color:'#666' }}
  }}}},
  plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{size:11}}, boxWidth:10 }} }}, tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.r != null ? ctx.parsed.r.toFixed(3) : 'n/a'}}` }} }} }}
}};

const barOpts = (title) => ({{
  responsive:true, maintainAspectRatio:false,
  scales:{{
    y:{{ min:0, max:1, ticks:{{ font:{{size:11}}, callback: v => v.toFixed(1) }}, title:{{display:true, text:title, font:{{size:11}}}} }},
    x:{{ ticks:{{ font:{{size:11}} }} }}
  }},
  plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{size:11}}, boxWidth:10 }} }}, tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y != null ? ctx.parsed.y.toFixed(3) : 'n/a'}}` }} }} }}
}});

new Chart(document.getElementById('overallRadar'), {{
  type:'radar',
  data:{{ labels:overallLabels, datasets:[{overall_radar_ds_str}] }},
  options: overallRadarOpts
}});

new Chart(document.getElementById('f1Radar'), {{
  type:'radar',
  data:{{ labels:radarLabels, datasets:[{f1_radar_ds}] }},
  options: radarOpts
}});

new Chart(document.getElementById('accRadar'), {{
  type:'radar',
  data:{{ labels:radarLabels, datasets:[{acc_radar_ds}] }},
  options: radarOpts
}});

new Chart(document.getElementById('f1Bar'), {{
  type:'bar',
  data:{{ labels:radarLabels, datasets:[{f1_bar_ds}] }},
  options: barOpts('F1')
}});

new Chart(document.getElementById('accBar'), {{
  type:'bar',
  data:{{ labels:radarLabels, datasets:[{acc_bar_ds}] }},
  options: barOpts('Accuracy')
}});
</script>
</body>
</html>"""
    return html


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  —  edit INPUT_FILES below
# ═════════════════════════════════════════════════════════════════════════════

INPUT_FILES = [
    "seed-42.txt",   # ← replace with your actual filenames
    "seed-63.txt",
    # "run3.txt",  # add / remove as needed (up to 6)
]

OUTPUT_FILE = "comparison_report.html"


def main():
    runs = []
    for path in INPUT_FILES:
        if not os.path.exists(path):
            print(f"⚠  File not found: {path}  — skipping")
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        run = parse_log(text, path)
        runs.append(run)
        print(f"✓  Parsed '{run['label']}' from {path}  "
              f"({len(run['subtypes'])} subtypes, "
              f"overall F1={run.get('overall',{}).get('f1','?')})")

    if not runs:
        print("No valid input files found. Edit INPUT_FILES in the script.")
        sys.exit(1)

    html = build_html(runs)
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"\n✅  Report written → {OUTPUT_FILE}")
    print("    Open it in any browser.")


if __name__ == "__main__":
    main()
