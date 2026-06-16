#!/usr/bin/env python3
"""
make_summary_report.py

Generate a self-contained HTML cross-site summary of epi_analysis figures
plus the cross-site comparison tables produced by cross_site_vasopressin_analysis.py.

Automatically discovers all output/epi_analysis_{SITE}/ directories.
To add a new site: run epi_analysis.py --site <NEWSITE>, then re-run this script.

Usage:
    uv run python code/make_summary_report.py
    uv run python code/make_summary_report.py --embed   # base64-embed all images (portable)

Output: output/cross_site_summary.html
"""

import argparse
import base64
import importlib.util
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_ROOT = BASE_DIR / "output"
CROSS_SITE_PLOTS = BASE_DIR / "cross-site output" / "plots"
OUT_FILE = OUTPUT_ROOT / "cross_site_summary.html"

ap = argparse.ArgumentParser()
ap.add_argument("--embed", action="store_true",
                help="Embed images as base64 (single portable file; larger)")
args = ap.parse_args()


# ── Load site palette from config ──────────────────────────────────────────────
def _load_logo_palette() -> dict[str, str]:
    cfg_path = BASE_DIR / "config" / "config.py"
    if not cfg_path.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location("clif_cfg", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {k.upper(): v for k, v in getattr(mod, "LOGO_PALETTE", {}).items()}
    except Exception:
        return {}

LOGO_PALETTE = _load_logo_palette()
_DEFAULT_HDR_COLOR = "#2c5f8a"


# ── Figure manifest ────────────────────────────────────────────────────────────
# Each tuple: (paper_label, question, description, [(suffix, sub_caption), ...])
# suffix is the part after "{site_lower}_" in the filename, without .png
FIGURES = [
    (
        "Figure 1",
        "Do patients who receive vasopressin have worse survival than those who don't?",
        (
            "Kaplan–Meier survival curves stratified by vasopressin use and pre-vasopressin "
            "norepinephrine-equivalent (NEE) dose. Unadjusted survival is lower in the "
            "ever-vasopressin group, reflecting higher severity of illness. The cumulative "
            "incidence plot shows how quickly patients in each NEE bin progressed to vasopressin "
            "initiation. KM curves stratified by pre-vaso max NEE illustrate a dose-response "
            "relationship between vasopressor requirements and mortality."
        ),
        [
            ("analysis1_5_km_evervaso",
             "1a — KM survival: ever- vs never-vasopressin"),
            ("analysis0_time_to_vaso_by_nee",
             "1b — Cumulative incidence of vasopressin initiation by pre-vaso NEE bin"),
            ("analysis1_km_nee_dose",
             "1c — KM survival by pre-vaso max NEE dose bin"),
        ],
    ),
    (
        "Figure 2",
        "Does the probability of being on vasopressin rise with NEE dose, and at what dose does it plateau?",
        (
            "The proportion of patient-hours with vasopressin active rises steeply with NEE dose "
            "and plateaus around 0.25–0.3 μg/kg/min. Wilson 95% confidence intervals are shown "
            "for each 0.1 μg/kg/min bin. The stacked-bar view distinguishes patient-hours where "
            "vasopressin was never started (grey), is currently active (blue), and has already "
            "been weaned (red), revealing that at higher NEE doses a substantial proportion of "
            "hours represent patients who had vasopressin successfully weaned. "
            "The stratified panel shows how initiation probability varies by care unit. "
            "The drug-mix panel (stacked to 100%) shows which component vasopressors are active "
            "in the 24h window before vasopressin is started."
        ),
        [
            ("analysis2_B",
             "2a — Proportion of patient-hours on vasopressin by NEE dose (Wilson 95% CI)"),
            ("analysis2_C",
             "2b — Patient-hours by NEE dose and vasopressin state (stacked bar)"),
            ("analysis2B_stratified",
             "2c — Proportion on vasopressin by NEE, stratified by care unit (Wilson 95% CI)"),
            ("analysis2D_nee_components_prevaso",
             "2d — Vasopressor drug mix in 24h before vasopressin initiation (stacked to 100%)"),
        ],
    ),
    (
        "Figure 3",
        "When is vasopressin initiated?",
        (
            "Vasopressin initiation timing relative to trajectory start (ICU admission / "
            "norepinephrine start). Most patients who receive vasopressin do so within the first "
            "24–48 hours. Despite this, clinicians frequently wait many hours after crossing "
            "NEE > 0.25 μg/kg/min, lactate > 2 mmol/L, or MAP < 65 mmHg thresholds before "
            "adding vasopressin. The time-of-day panel shows LOWESS-smoothed physiologic values "
            "at the moment of initiation across clock hours, probing for diurnal variation in "
            "initiation practice."
        ),
        [
            ("analysis4a_time_to_vaso",
             "3a — Distribution of time to vasopressin initiation (hours from trajectory start)"),
            ("analysis4d_wait_time",
             "3b — Hours above clinical thresholds before vasopressin (waiting time)"),
            ("analysis3_TOD",
             "3c — Physiologic features at vasopressin initiation by time of day"),
        ],
    ),
    (
        "Figure 4",
        "Who gets vasopressin — early (low NEE) vs late (high NEE)?",
        (
            "Among vasopressin initiators, patients in the bottom quartile of NEE at initiation "
            "(Q1, early/low-dose) differ systematically from those in the top quartile (Q4, "
            "late/high-dose). Q1 patients tend to have higher SOFA, lower MAP, and more organ "
            "support at the moment of initiation, suggesting two distinct initiation phenotypes: "
            "prophylactic early use at lower vasopressor loads vs rescue use after maximal "
            "norepinephrine dose. Brackets show Bonferroni-corrected Mann-Whitney Q1 vs Q4 "
            "comparisons; the IQR ribbon plot shows medians across all NEE bins continuously."
        ),
        [
            ("analysis5_A",
             "4a — Patient features by NEE quartile at vasopressin initiation (violin + stats)"),
            ("analysis5_B",
             "4b — Median + IQR ribbon by NEE dose at initiation (continuous)"),
            ("analysis5_D_rate_of_change",
             "4c — Rate of change in SOFA / lactate / MAP in final 6h before vasopressin (by NEE quartile)"),
            ("analysis5_E_tod_vs_sofa",
             "4d — Time of day of vasopressin initiation vs SOFA at initiation (LOWESS)"),
        ],
    ),
]

# Cross-site comparison outputs (single-image panels, not per-site)
CROSS_SITE_FIGURES = [
    ("Cohort flowchart",                    "consort_flowchart.png"),
    ("Baseline characteristics comparison", "baseline_comparison_combined.png"),
    ("Features at vasopressin initiation",  "initiation_features_combined.png"),
]


# ── Discover sites ─────────────────────────────────────────────────────────────
# Sites are discovered from upload_to_box_*/epi_analysis/ directories so that
# the coordinating site only needs the shared upload_to_box_<SITE> folders —
# no PHI epi_analysis_<SITE>/ directories required.
sites = []
for d in sorted(OUTPUT_ROOT.glob("upload_to_box_*")):
    m = re.match(r"upload_to_box_(.+)$", d.name)
    if m and (d / "epi_analysis").is_dir():
        sites.append((m.group(1), d / "epi_analysis"))

if not sites:
    sys.exit(f"No upload_to_box_*/epi_analysis/ directories found under {OUTPUT_ROOT}\n"
             f"Run epi_analysis.py first, or ensure shared upload_to_box_<SITE> folders "
             f"are placed in {OUTPUT_ROOT}")

print(f"Found sites: {[s for s, _ in sites]}")


# ── HTML helpers ──────────────────────────────────────────────────────────────
def img_tag(path: Path, alt: str, embed: bool, relative_to: Path | None = None) -> str:
    if not path.exists():
        return (
            f'<div class="missing">&#9888; Not found<br>'
            f'<span class="fname">{path.name}</span></div>'
        )
    if embed:
        data = base64.b64encode(path.read_bytes()).decode()
        src = f"data:image/png;base64,{data}"
    else:
        base = relative_to or OUTPUT_ROOT
        try:
            src = str(path.relative_to(base))
        except ValueError:
            src = str(path)
        src = src.replace("\\", "/")
    return f'<img src="{src}" alt="{alt}" loading="lazy">'


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def site_hdr_color(site_name: str) -> str:
    return LOGO_PALETTE.get(site_name.upper(), _DEFAULT_HDR_COLOR)


# ── Build HTML ─────────────────────────────────────────────────────────────────
site_labels = [s for s, _ in sites]
n_sites = len(sites)

col_width = max(300, min(600, 1100 // n_sites))

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    background: #f7f7f7;
    color: #222;
    padding: 24px;
    max-width: {max(900, col_width * n_sites + 240)}px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.6em; margin-bottom: 6px; }}
h2 {{ font-size: 1.25em; font-weight: bold; margin: 40px 0 12px; color: #1a1a1a;
     border-bottom: 2px solid #ccc; padding-bottom: 6px; }}
.subtitle {{ color: #555; margin-bottom: 28px; font-size: 0.9em; }}
.paper-title {{ font-size: 1.25em; font-weight: bold; margin: 36px 0 4px; color: #1a1a1a; }}
.fig-block {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 20px;
    margin: 18px 0 28px;
}}
.fig-question {{
    font-size: 1.05em;
    font-weight: 600;
    color: #1a3a5c;
    margin-bottom: 8px;
}}
.fig-desc {{
    color: #444;
    line-height: 1.55;
    margin-bottom: 18px;
    max-width: 900px;
}}
.panel-grid {{
    display: grid;
    grid-template-columns: 180px repeat({n_sites}, 1fr);
    gap: 0;
    border: 1px solid #ccc;
    border-radius: 4px;
    overflow: hidden;
}}
.hdr-blank {{ background: #eef; padding: 10px 8px; }}
.hdr-site {{
    color: white;
    font-weight: bold;
    font-size: 0.95em;
    padding: 10px 8px;
    text-align: center;
}}
.sub-label {{
    background: #f0f4f8;
    font-size: 0.82em;
    color: #334;
    padding: 10px 10px;
    display: flex;
    align-items: center;
    border-top: 1px solid #dde;
    font-weight: 500;
    line-height: 1.4;
}}
.panel-cell {{
    border-top: 1px solid #dde;
    border-left: 1px solid #dde;
    padding: 6px;
    text-align: center;
    background: #fafafa;
}}
.panel-cell img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}}
.missing {{
    color: #a33;
    font-size: 0.8em;
    padding: 20px 10px;
    background: #fff3f3;
    border: 1px dashed #e88;
    border-radius: 3px;
    text-align: center;
}}
.fname {{ font-family: monospace; font-size: 0.85em; color: #666; }}
.site-count {{ font-size: 0.78em; font-weight: normal; }}
.cross-block {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 20px;
    margin: 14px 0 22px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}}
.cross-block .cross-label {{
    font-size: 0.9em;
    font-weight: 600;
    color: #444;
    margin-bottom: 10px;
}}
.cross-block img {{
    max-width: 100%;
    height: auto;
}}
"""

rows_html_parts = []

# ── Per-site panels ────────────────────────────────────────────────────────────
rows_html_parts.append(
    '<div class="paper-title">Paper 1 &mdash; Characterizing Vasopressin Initiation in Septic Shock</div>'
)

for fig_label, question, description, subplots in FIGURES:
    header_cells = ['<div class="hdr-blank"></div>']
    for site_name, site_dir in sites:
        color = site_hdr_color(site_name)
        header_cells.append(
            f'<div class="hdr-site" style="background:{color};">{esc(site_name)}</div>'
        )

    subplot_rows = []
    for suffix, sub_caption in subplots:
        label_cell = f'<div class="sub-label">{esc(sub_caption)}</div>'
        img_cells = []
        for site_name, site_dir in sites:
            site_lower = site_name.lower()
            img_path = site_dir / f"{site_lower}_{suffix}.png"
            tag = img_tag(img_path, f"{site_name} {sub_caption}", args.embed)
            img_cells.append(f'<div class="panel-cell">{tag}</div>')
        subplot_rows.append(label_cell + "".join(img_cells))

    grid_inner = "".join(header_cells) + "".join(subplot_rows)
    grid_html = f'<div class="panel-grid">{grid_inner}</div>'

    fig_html = (
        f'<div class="fig-block">'
        f'<div class="fig-question"><strong>{esc(fig_label)}</strong> &mdash; {esc(question)}</div>'
        f'<div class="fig-desc">{esc(description)}</div>'
        f'{grid_html}'
        f'</div>'
    )
    rows_html_parts.append(fig_html)

# ── Cross-site comparison section ─────────────────────────────────────────────
if CROSS_SITE_PLOTS.exists():
    rows_html_parts.append('<h2>Cross-Site Comparison (cross_site_vasopressin_analysis.py)</h2>')
    for label, fname in CROSS_SITE_FIGURES:
        img_path = CROSS_SITE_PLOTS / fname
        tag = img_tag(img_path, label, args.embed, relative_to=OUTPUT_ROOT.parent)
        rows_html_parts.append(
            f'<div class="cross-block">'
            f'<div class="cross-label">{esc(label)}</div>'
            f'{tag}'
            f'</div>'
        )

site_list_str = ", ".join(site_labels)
body_html = "\n".join(rows_html_parts)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Site Vasopressin Analysis Summary</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>Cross-Site Vasopressin Analysis Summary</h1>
<div class="subtitle">
  Sites: {esc(site_list_str)} &nbsp;&bull;&nbsp;
  Generated: <span id="ts"></span>
  {'&nbsp;&bull;&nbsp;<em>Images embedded (portable)</em>' if args.embed else '&nbsp;&bull;&nbsp;<em>Images linked (keep output/ folder)</em>'}
</div>
<script>document.getElementById('ts').textContent = new Date().toLocaleString();</script>

{body_html}

</body>
</html>
"""

OUT_FILE.write_text(html, encoding="utf-8")
print(f"\nWrote: {OUT_FILE}")
print(f"  Sites included: {site_list_str}")
print(f"  Figures: {len(FIGURES)}")
print(f"  Sub-panels: {sum(len(sp) for *_, sp in FIGURES)}")
cross_avail = sum(1 for _, f in CROSS_SITE_FIGURES if (CROSS_SITE_PLOTS / f).exists())
print(f"  Cross-site panels: {cross_avail}/{len(CROSS_SITE_FIGURES)}")
if not args.embed:
    print("  Note: images are linked, not embedded — keep output/ folder alongside the HTML.")
    print("  Use --embed for a single portable file.")
