#!/usr/bin/env python3
"""
make_threshold_summary_report.py

Generate a self-contained HTML cross-site summary for the threshold decision-rule
analyses, addressing three questions:

  (1) Which features predict clinician vasopressin initiation?
  (2) Can a simple decision tree (combination of threshold rules) predict initiation?
  (3) Does following a threshold rule (or a decision tree) associate with better
      hospital mortality compared with discordant clinical practice?

Data sources (per site in output/upload_to_box_<SITE>/threshold/):
  site_threshold_sweep.py outputs:
    threshold_comparison_table.csv  -- step-level kappa, AUROC, sens, spec per feature
    patient_level_table.csv         -- patient-level kappa, AUROC per feature
    plots/threshold_sweep.png       -- kappa vs threshold curves for each feature
    plots/decision_tree_fidelity.png -- depth 1-7 agreement + per-feature AUROC

  site_threshold_outcome.py outputs:
    threshold_outcome_table.csv           -- adj + IPTW mortality ORs per feature rule
    threshold_concordance_summary.csv     -- concordance rates + crude hourly hazard

Cross-site aggregate figures (from cross_site_vasopressin_analysis.py):
  cross-site output/plots/feature_ranking_heatmap.png
  cross-site output/plots/kappa_initiation_step.png
  cross-site output/plots/kappa_initiation_patient.png
  cross-site output/plots/auroc_step.png
  cross-site output/plots/auroc_patient.png
  cross-site output/plots/sens_spec_table_step.png

Output: output/threshold_summary_report.html

Usage:
    uv run python code/make_threshold_summary_report.py
    uv run python code/make_threshold_summary_report.py --embed
"""

import argparse
import base64
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent
OUTPUT_ROOT      = BASE_DIR / "output"
CROSS_SITE_DIR   = BASE_DIR / "cross-site output"
CROSS_SITE_PLOTS = CROSS_SITE_DIR / "plots"
OUT_FILE         = OUTPUT_ROOT / "threshold_summary_report.html"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--embed", action="store_true",
                help="Embed images as base64 (single portable file; larger)")
args = ap.parse_args()
EMBED = args.embed


# ── CSV reader ─────────────────────────────────────────────────────────────────
def _read_csv(path: Path):
    """Return list-of-dicts or None."""
    if not path.exists():
        return None
    try:
        import polars as pl
        return pl.read_csv(path, null_values=["", "NA", "N/A", "nan"]).to_dicts()
    except ImportError:
        pass
    try:
        import csv as _csv
        with open(path, encoding="utf-8") as f:
            return list(_csv.DictReader(f))
    except Exception:
        return None


def _safe_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("None", "nan", "N/A", "NA") or s.startswith("<"):
        return None
    try:
        f = float(s.replace(",", ""))
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, dp=3) -> str:
    f = _safe_float(v)
    return f"{{:.{dp}f}}".format(f) if f is not None else "—"


def _fmt_or(or_v, lo, hi, pval) -> str:
    """Format OR (95% CI) p= string."""
    o = _safe_float(or_v)
    l = _safe_float(lo)
    h = _safe_float(hi)
    p = _safe_float(pval)
    if o is None:
        return "—"
    ci = f" ({l:.2f}–{h:.2f})" if l is not None and h is not None else ""
    ps = ""
    if p is not None:
        ps = " p<0.001" if p < 0.001 else f" p={p:.3f}"
    return f"{o:.2f}{ci}{ps}"


# ── Site palette ───────────────────────────────────────────────────────────────
def _load_logo_palette():
    cfg = BASE_DIR / "config" / "config.py"
    if not cfg.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location("clif_cfg", cfg)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {k.upper(): v for k, v in getattr(mod, "LOGO_PALETTE", {}).items()}
    except Exception:
        return {}


LOGO_PALETTE = _load_logo_palette()
CLIF_PALETTE = ["#ffd328", "#ffb00e", "#fe8211", "#f04122",
                "#cb0824", "#ad1c54", "#6b1461", "#39114b"]

FEAT_LABELS = {
    "time_hour":      "Time (h)",
    "norepinephrine": "NE (µg/kg/min)",
    "nee":            "NEE (µg/kg/min)",
    "mbp":            "MAP (mmHg)",
    "sofa":           "SOFA",
    "lactate":        "Lactate (mmol/L)",
    "urine_output":   "Urine output (mL/h)",
    "creatinine":     "Creatinine (mg/dL)",
    "bun":            "BUN (mg/dL)",
    "ventil":         "Ventilation",
    "rrt":            "RRT",
    "steroid":        "Corticosteroid",
    "fluids":         "Fluids (mL/hr)",
}


# ── Site discovery ─────────────────────────────────────────────────────────────
def _detect_sites():
    SENTINELS = ["threshold_comparison_table.csv", "patient_level_table.csv",
                 "threshold_outcome_table.csv"]
    found = []
    for d in sorted(OUTPUT_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("upload_to_box_"):
            continue
        site = d.name[len("upload_to_box_"):]
        thr  = d / "threshold"
        if any((thr / s).exists() for s in SENTINELS):
            found.append(site)
    mimic  = [s for s in found if s.upper() == "MIMIC"]
    others = [s for s in found if s.upper() != "MIMIC"]
    return mimic + sorted(others)


SITE_NAMES = _detect_sites()
if not SITE_NAMES:
    sys.exit(
        f"No threshold data found in output/upload_to_box_*/threshold/.\n"
        "Run site_threshold_sweep.py and/or site_threshold_outcome.py first."
    )
print(f"Found {len(SITE_NAMES)} site(s): {SITE_NAMES}")

_non_special = [s for s in SITE_NAMES
                if s.upper() not in LOGO_PALETTE and s.upper() != "MIMIC"]


def _site_color(site: str) -> str:
    upper = site.upper()
    if upper in LOGO_PALETTE:
        return LOGO_PALETTE[upper]
    if upper == "MIMIC":
        return "#b0b0b0"
    idx = _non_special.index(site) if site in _non_special else 0
    return CLIF_PALETTE[idx % len(CLIF_PALETTE)]


def _darken(hex_color: str, f: float = 0.65) -> str:
    h = hex_color.lstrip("#")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


def _site_label(site: str) -> str:
    return "MIMIC-IV" if site.upper() == "MIMIC" else f"CLIF ({site.upper()})"


def _site_hdr(site: str) -> str:
    return _darken(_site_color(site))


def _thr_dir(site: str) -> Path:
    return OUTPUT_ROOT / f"upload_to_box_{site}" / "threshold"


# ── HTML helpers ───────────────────────────────────────────────────────────────
def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return (f'<div class="missing">&#9888; Not found<br>'
                f'<span class="fname">{path.name}</span></div>')
    if EMBED:
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{esc(alt)}" loading="lazy">'
    try:
        src = str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/")
    except ValueError:
        try:
            src = str(path.relative_to(BASE_DIR)).replace("\\", "/")
            src = "../" + src
        except ValueError:
            src = str(path).replace("\\", "/")
    return f'<img src="{src}" alt="{esc(alt)}" loading="lazy">'


def fig_as_tag(fig: plt.Figure, alt: str, fname: str) -> str:
    """Save figure to cross-site plots dir and return img tag."""
    CROSS_SITE_PLOTS.mkdir(parents=True, exist_ok=True)
    path = CROSS_SITE_PLOTS / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return img_tag(path, alt)


def html_table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    """Render a list-of-dicts as an HTML table."""
    if not rows:
        return '<p class="missing">No data available.</p>'
    if headers is None:
        headers = cols
    th  = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = []
    for i, r in enumerate(rows):
        cls = ' class="alt"' if i % 2 == 0 else ""
        tds = "".join(f'<td>{esc(str(r.get(c) if r.get(c) is not None else "—"))}</td>'
                      for c in cols)
        trs.append(f"<tr{cls}>{tds}</tr>")
    return (f'<table class="data-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table>')


def per_site_panel(plot_fn) -> str:
    """Build a CSS-grid panel with one column per site. plot_fn(site) -> HTML str."""
    n = len(SITE_NAMES)
    hdrs = "".join(
        f'<div class="hdr-site" style="background:{_site_hdr(s)};">'
        f'{esc(_site_label(s))}</div>'
        for s in SITE_NAMES
    )
    cells = "".join(
        f'<div class="panel-cell">{plot_fn(s)}</div>'
        for s in SITE_NAMES
    )
    return (f'<div class="site-grid" style="grid-template-columns: repeat({n}, 1fr);">'
            f'{hdrs}{cells}</div>')


def site_tbl_block(site: str, inner_html: str) -> str:
    return (f'<div class="site-tbl-block">'
            f'<div class="site-tbl-hdr" style="background:{_site_hdr(site)};">'
            f'{esc(_site_label(site))}</div>'
            f'{inner_html}'
            f'</div>')


# ── Cross-site figures ─────────────────────────────────────────────────────────
def cross_fig_block(fname: str, label: str, desc: str) -> str:
    path = CROSS_SITE_PLOTS / fname
    return (f'<div class="fig-block">'
            f'<div class="fig-label">{esc(label)}</div>'
            f'<div class="fig-desc">{esc(desc)}</div>'
            f'<div class="cross-content">{img_tag(path, label)}</div>'
            f'</div>')


# ── Forest plot ────────────────────────────────────────────────────────────────
def make_forest_plot() -> str:
    """Generate cross-site forest plot of threshold-mortality ORs."""
    site_data: dict[str, dict] = {}
    for site in SITE_NAMES:
        rows = _read_csv(_thr_dir(site) / "threshold_outcome_table.csv")
        if rows:
            site_data[site] = {r["feature"]: r for r in rows if r.get("feature")}

    if not site_data:
        return '<div class="missing">&#9888; No threshold_outcome_table.csv found for any site.</div>'

    present = [s for s in SITE_NAMES if s in site_data]
    n_sites = len(present)

    # Features union, excluding time_hour (not clinical threshold)
    feat_union = list(dict.fromkeys(
        f for s in present for f in site_data[s] if f != "time_hour"
    ))
    if not feat_union:
        return '<div class="missing">&#9888; No usable features in outcome tables.</div>'

    # Sort ascending by mean adj OR (most protective at top of plot = highest y)
    def _mean_adj(feat):
        vals = [_safe_float(site_data[s][feat].get("adj_or"))
                for s in present if feat in site_data[s]]
        vals = [v for v in vals if v is not None]
        return np.mean(vals) if vals else 1.0

    feats   = sorted(feat_union, key=_mean_adj)  # ascending: lowest OR at bottom
    n_feats = len(feats)

    row_spacing = n_sites * 0.5 + 0.9
    fig_h = max(5.0, n_feats * row_spacing * 0.6 + 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, fig_h), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    estimators = [
        ("Covariate-adjusted OR\n(conditional; clustered SE)",
         "adj_or",  "adj_or_lo",  "adj_or_hi",  "adj_pval"),
        ("IPTW-MSM OR\n(marginal; clustered SE)",
         "iptw_or", "iptw_or_lo", "iptw_or_hi", "iptw_pval"),
    ]

    ytick_pos    = []
    ytick_labels = []
    for fi, feat in enumerate(feats):
        base_y = fi * row_spacing
        ytick_pos.append(base_y + (n_sites - 1) * 0.25)
        ytick_labels.append(FEAT_LABELS.get(feat, feat))

    for ax, (title, or_col, lo_col, hi_col, pval_col) in zip(axes, estimators):
        for fi, feat in enumerate(feats):
            base_y = fi * row_spacing
            for si, site in enumerate(present):
                if feat not in site_data[site]:
                    continue
                row  = site_data[site][feat]
                ypos = base_y + si * 0.5
                col  = _site_color(site)

                or_v = _safe_float(row.get(or_col))
                lo_v = _safe_float(row.get(lo_col))
                hi_v = _safe_float(row.get(hi_col))
                pval = _safe_float(row.get(pval_col))

                if or_v is None:
                    ax.scatter([1.0], [ypos], color="#cccccc", s=12, marker="x", zorder=3)
                    continue

                lo_p = max(lo_v if lo_v is not None else or_v * 0.5, 0.05)
                hi_p = min(hi_v if hi_v is not None else or_v * 2.0, 30.0)
                sig  = pval is not None and pval < 0.05

                ax.plot([lo_p, hi_p], [ypos, ypos], color=col, lw=1.5, alpha=0.85, zorder=3)
                ax.scatter(
                    [or_v], [ypos],
                    color=col if sig else "none",
                    edgecolors=col,
                    s=60 if sig else 35,
                    linewidths=1.5,
                    marker="D" if sig else "o",
                    zorder=5,
                    label=_site_label(site) if fi == 0 else "",
                )

        ax.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=8.5)
        ax.set_xlabel("Odds Ratio (log scale)", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.set_ylim(-0.5, n_feats * row_spacing - 0.3)
        ax.set_xlim(0.05, 30)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", alpha=0.22, which="major", ls=":")
        ax.axvspan(0.05, 1.0, alpha=0.04, color="green",  zorder=0)
        ax.axvspan(1.0,  30,  alpha=0.04, color="salmon", zorder=0)

    axes[0].legend(fontsize=8, title="Site", title_fontsize=8,
                   loc="lower left", framealpha=0.85)

    fig.suptitle(
        "Hospital mortality OR for concordant vs discordant vasopressin care\n"
        "Concordance = clinician action matches the κ-initiation threshold at each patient-hour\n"
        "◆ filled = p<0.05  ○ open = p≥0.05  |  green shading = OR<1 (potentially protective)",
        fontsize=9.5, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig_as_tag(fig, "Cross-site forest plot: threshold-mortality association",
                      "threshold_outcome_forest.png")


# ── Section 1: Feature predictiveness ─────────────────────────────────────────
def build_section1() -> str:
    parts = []

    parts.append(
        '<div class="section-intro">'
        "<p>For each clinical feature, Cohen's κ was computed between a threshold-based "
        "rule (feature ≥ or ≤ threshold) and the observed clinician vasopressin action. "
        "The <strong>κ-initiation</strong> threshold maximises agreement on "
        "<em>vaso-naive at-risk steps</em> — it predicts when a clinician starts "
        "vasopressin, not when it is maintained. "
        "The <strong>κ-allsteps</strong> threshold is optimised across all timesteps "
        "(including vasopressin maintenance). "
        "AUROC and sensitivity/specificity complement κ by assessing discriminability "
        "across all thresholds.</p>"
        "</div>"
    )

    # Cross-site aggregate plots (from cross_site_vasopressin_analysis.py)
    parts.append(cross_fig_block(
        "feature_ranking_heatmap.png",
        "Cross-site feature ranking: Cohen's κ heatmap",
        "Heatmap of Cohen's κ by feature and site. Left panel = κ-initiation (vaso-naive "
        "training steps); right = κ-allsteps (all timesteps); far right = patient-level "
        "κ-initiation (max feature per stay → ever-vasopressin). Features sorted by "
        "mean κ across sites. Orange/green = moderate-good agreement; red = poor.",
    ))
    parts.append(cross_fig_block(
        "kappa_initiation_step.png",
        "Cross-site κ-initiation: step-level",
        "Bar chart of κ-initiation per feature and site at the step level. "
        "κ-initiation is computed on vaso-naive at-risk steps: which threshold best "
        "predicts when clinicians decide to start vasopressin?",
    ))
    parts.append(cross_fig_block(
        "auroc_step.png",
        "Cross-site AUROC: step-level",
        "AUROC for each feature at the step level. Measures how well a continuous "
        "threshold on the feature discriminates timesteps with vs without clinician "
        "vasopressin action.",
    ))
    parts.append(cross_fig_block(
        "sens_spec_table_step.png",
        "Cross-site sensitivity and specificity: step-level",
        "Sensitivity and specificity at the κ-initiation threshold for each feature "
        "and site. Threshold selected on training at-risk (vaso-naive) steps only.",
    ))

    # Per-site threshold sweep plots
    parts.append(
        '<div class="fig-block">'
        '<div class="fig-label">Per-site: Threshold sweep curves (kappa vs threshold value)</div>'
        '<div class="fig-desc">For each continuous feature, Cohen’s κ is shown as a '
        "function of the threshold value in both directions (≥ threshold = blue solid, "
        "≤ threshold = red dashed). The black dotted line marks the κ-initiation "
        "threshold selected on training at-risk steps. AUROC, κ, and agreement at the "
        "optimal threshold are annotated per panel.</div>"
        + per_site_panel(lambda s: img_tag(
            _thr_dir(s) / "plots" / "threshold_sweep.png",
            f"{_site_label(s)} threshold sweep",
        ))
        + "</div>"
    )

    # Per-site step-level comparison tables
    tbl_blocks = []
    for site in SITE_NAMES:
        rows = _read_csv(_thr_dir(site) / "threshold_comparison_table.csv")
        if rows:
            disp = [{
                "Feature":       FEAT_LABELS.get(r.get("feature", ""), r.get("feature", "")),
                "Threshold":     str(r.get("threshold") or "—"),
                "Agreement":     _fmt(r.get("agreement"), 3),
                "κ-init":   _fmt(r.get("kappa_initiation"), 3),
                "κ-all":    _fmt(r.get("kappa_allsteps"), 3),
                "AUROC":         _fmt(r.get("auroc"), 3),
                "Sens":          _fmt(r.get("sensitivity"), 3),
                "Spec":          _fmt(r.get("specificity"), 3),
            } for r in rows]
            inner = html_table(disp, list(disp[0].keys()))
        else:
            inner = '<div class="missing">threshold_comparison_table.csv not found</div>'
        tbl_blocks.append(site_tbl_block(site, inner))

    parts.append(
        '<div class="fig-block">'
        '<div class="fig-label">Per-site: Step-level feature performance</div>'
        '<div class="fig-desc">Step-level Cohen’s κ-initiation, κ-allsteps, '
        "AUROC, sensitivity, and specificity for each feature at its optimal threshold. "
        "Agreement = fraction of all steps where the rule matches clinician vasopressin action."
        "</div>"
        + "".join(tbl_blocks)
        + "</div>"
    )

    return "\n".join(parts)


# ── Section 2: Decision tree ───────────────────────────────────────────────────
def build_section2() -> str:
    parts = []

    parts.append(
        '<div class="section-intro">'
        "<p>A <strong>depth-1 decision tree</strong> is equivalent to a single threshold "
        "rule on one feature. Deeper trees combine multiple features into an interpretable "
        "rule set. The fidelity curve shows how well trees of increasing depth agree with "
        "clinician vasopressin actions on the full dataset.</p>"
        "<p>The right panel of each site’s figure shows per-feature AUROC at the step "
        "level, serving as an honest measure of discriminability (unlike tree feature "
        "importance, which is misleading for shallow trees).</p>"
        "<p>The patient-level analysis below asks the depth-1 question at the patient level: "
        "does the maximum feature value during the stay predict whether the patient ever "
        "received vasopressin?</p>"
        "</div>"
    )

    # Per-site decision tree fidelity plots
    parts.append(
        '<div class="fig-block">'
        '<div class="fig-label">Per-site: Decision tree fidelity (depth 1–7)</div>'
        '<div class="fig-desc">Left: agreement between decision-tree predictions and '
        "clinician vasopressin action as a function of tree depth. Depth=1 is a single-feature "
        "threshold rule; deeper trees combine multiple features. "
        "Right: AUROC per feature from the threshold sweep (discriminability at the step level)."
        "</div>"
        + per_site_panel(lambda s: img_tag(
            _thr_dir(s) / "plots" / "decision_tree_fidelity.png",
            f"{_site_label(s)} decision tree fidelity",
        ))
        + "</div>"
    )

    # Cross-site patient-level plots (from cross_site_vasopressin_analysis.py)
    parts.append(cross_fig_block(
        "kappa_initiation_patient.png",
        "Cross-site κ-initiation: patient-level (depth-1 equivalent)",
        "At the patient level, the κ-initiation threshold is applied to the maximum "
        "feature value per stay to predict whether the patient ever received vasopressin. "
        "This is the single-threshold (depth-1 tree) equivalent applied to each patient’s "
        "peak feature value.",
    ))
    parts.append(cross_fig_block(
        "auroc_patient.png",
        "Cross-site AUROC: patient-level",
        "AUROC for max(feature) per stay → ever-vasopressin classification. "
        "Measures the discriminative ability of each feature’s peak value for identifying "
        "vasopressin recipients.",
    ))

    # Per-site patient-level tables
    tbl_blocks = []
    for site in SITE_NAMES:
        rows = _read_csv(_thr_dir(site) / "patient_level_table.csv")
        if rows:
            disp = [{
                "Feature":         FEAT_LABELS.get(r.get("feature", ""), r.get("feature", "")),
                "κ-init (pat)": _fmt(r.get("patient_kappa_initiation"), 3),
                "AUROC (pat)":     _fmt(r.get("patient_auroc"), 3),
                "n vaso":          str(r.get("n_vaso_patients") or "—"),
                "n no-vaso":       str(r.get("n_novaso_patients") or "—"),
            } for r in rows]
            inner = html_table(disp, list(disp[0].keys()))
        else:
            inner = '<div class="missing">patient_level_table.csv not found</div>'
        tbl_blocks.append(site_tbl_block(site, inner))

    parts.append(
        '<div class="fig-block">'
        '<div class="fig-label">Per-site: Patient-level κ and AUROC</div>'
        '<div class="fig-desc">Patient-level performance of a single-feature rule: '
        "threshold applied to max(feature) per stay predicts ever-vasopressin. "
        "κ-initiation and AUROC are reported for each feature.</div>"
        + "".join(tbl_blocks)
        + "</div>"
    )

    return "\n".join(parts)


# ── Section 3: Mortality outcomes ──────────────────────────────────────────────
def build_section3() -> str:
    parts = []

    parts.append(
        '<div class="section-intro">'
        "<p>We ask whether patients managed <em>concordantly</em> with each threshold rule "
        "— i.e., where the clinician vasopressin action matches the feature threshold "
        "at each hour — have different hospital mortality than discordantly managed patients.</p>"
        "<p>Two estimators are reported:</p>"
        "<ul>"
        "<li><strong>Covariate-adjusted OR</strong> — pooled logistic regression with "
        "baseline + time-varying confounders and a restricted cubic spline for time, "
        "using clustered sandwich SE by patient (conditional OR).</li>"
        "<li><strong>IPTW-MSM OR</strong> — marginal structural model using "
        "stabilized inverse-probability weights with cumulative within-patient product "
        "(trimmed at 1st/99th percentile), to account for treatment selection bias "
        "(marginal OR).</li>"
        "</ul>"
        "<p>OR&nbsp;&lt;&nbsp;1 = concordant care associated with lower mortality. "
        "OR&nbsp;&gt;&nbsp;1 = concordant care associated with higher mortality or residual "
        "confounding (e.g., sicker patients are more likely to both receive vasopressin "
        "and have worse outcomes).</p>"
        "</div>"
    )

    # Forest plot
    forest_html = make_forest_plot()
    parts.append(
        '<div class="fig-block">'
        "<div class=\"fig-label\">Cross-site: Mortality OR per threshold rule (forest plot)</div>"
        "<div class=\"fig-desc\">Odds ratio for hospital mortality when clinical care is "
        "concordant with each feature-threshold rule. "
        "Left panel: covariate-adjusted conditional OR. "
        "Right panel: IPTW-MSM marginal OR. "
        "Error bars = 95% CI. "
        "◆ filled diamond = p&lt;0.05; ○ open circle = p≥0.05. "
        "Green shading = OR&lt;1 region; features near the bottom have the lowest adj OR "
        "(most potentially protective concordance).</div>"
        f'<div class="cross-content">{forest_html}</div>'
        "</div>"
    )

    # Per-site outcome tables
    tbl_blocks = []
    for site in SITE_NAMES:
        rows = _read_csv(_thr_dir(site) / "threshold_outcome_table.csv")
        if rows:
            disp = [{
                "Feature":          FEAT_LABELS.get(r.get("feature", ""), r.get("feature", "")),
                "Threshold":        f"{r.get('direction', '?')} {_fmt(r.get('threshold_value'), 3)}",
                "κ-init":      _fmt(r.get("kappa_initiation"), 3),
                "n":                str(r.get("n_patients") or "—"),
                "Events":           str(r.get("n_events") or "—"),
                "Conc. rate":       _fmt(r.get("concordance_rate"), 3),
                "adj OR (95% CI)":  _fmt_or(r.get("adj_or"),  r.get("adj_or_lo"),
                                            r.get("adj_or_hi"),  r.get("adj_pval")),
                "iptw OR (95% CI)": _fmt_or(r.get("iptw_or"), r.get("iptw_or_lo"),
                                            r.get("iptw_or_hi"), r.get("iptw_pval")),
            } for r in rows if r.get("feature") != "time_hour"]
            inner = html_table(disp, list(disp[0].keys()))
        else:
            inner = '<div class="missing">threshold_outcome_table.csv not found</div>'
        tbl_blocks.append(site_tbl_block(site, inner))

    parts.append(
        '<div class="fig-block">'
        "<div class=\"fig-label\">Per-site: Mortality ORs per feature rule</div>"
        "<div class=\"fig-desc\">Concordance rate = fraction of patient-hours where "
        "action_vaso matches the threshold rule. adj OR = covariate-adjusted conditional OR. "
        "iptw OR = IPTW-MSM marginal OR. n = patients; Events = hospital deaths.</div>"
        + "".join(tbl_blocks)
        + "</div>"
    )

    # Per-site concordance summary tables
    conc_blocks = []
    for site in SITE_NAMES:
        rows = _read_csv(_thr_dir(site) / "threshold_concordance_summary.csv")
        if rows:
            disp = [{
                "Feature":              FEAT_LABELS.get(r.get("feature", ""), r.get("feature", "")),
                "n":                    str(r.get("n_patients") or "—"),
                "Events":               str(r.get("n_events") or "—"),
                "Conc. rate":           _fmt(r.get("concordance_rate"), 3),
                "Hazard (concordant)":  _fmt(r.get("hourly_hazard_concordant"), 4),
                "Hazard (discordant)":  _fmt(r.get("hourly_hazard_discordant"), 4),
            } for r in rows if r.get("feature") != "time_hour"]
            inner = html_table(disp, list(disp[0].keys()))
        else:
            inner = '<div class="missing">threshold_concordance_summary.csv not found</div>'
        conc_blocks.append(site_tbl_block(site, inner))

    parts.append(
        '<div class="fig-block">'
        "<div class=\"fig-label\">Per-site: Concordance rates and crude hourly hazard</div>"
        "<div class=\"fig-desc\">For each threshold rule, the concordance rate (fraction of "
        "patient-hours where action matches rule) and the raw hourly hazard of death in "
        "concordant vs discordant patient-hours. Low concordance may reflect rule "
        "over-specificity or infrequent crossing of the threshold.</div>"
        + "".join(conc_blocks)
        + "</div>"
    )

    return "\n".join(parts)


# ── CSS ────────────────────────────────────────────────────────────────────────
n_sites   = len(SITE_NAMES)
col_width = max(300, min(700, 1100 // n_sites))

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    background: #f7f7f7;
    color: #222;
    padding: 24px;
    max-width: {max(1000, col_width * n_sites + 180)}px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.55em; margin-bottom: 6px; }}
h2 {{
    font-size: 1.2em; font-weight: bold; margin: 50px 0 12px; color: #1a1a1a;
    border-bottom: 3px solid #2c5f8a; padding-bottom: 8px;
}}
.subtitle {{ color: #555; margin-bottom: 24px; font-size: 0.88em; }}
.toc {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 14px 20px; margin: 18px 0 30px; display: inline-block; min-width: 300px;
}}
.toc p {{ font-weight: bold; margin-bottom: 6px; }}
.toc ul {{ padding-left: 18px; }}
.toc li {{ margin: 4px 0; }}
.toc a {{ color: #2c5f8a; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.section-intro {{
    background: #f0f4f8; border-left: 4px solid #2c5f8a;
    padding: 13px 16px; margin: 12px 0 20px; border-radius: 0 6px 6px 0;
    line-height: 1.6; color: #333;
}}
.section-intro p {{ margin: 5px 0; }}
.section-intro ul {{ margin: 6px 0 6px 18px; }}
.section-intro li {{ margin: 3px 0; }}
.fig-block {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 18px 20px; margin: 16px 0 22px;
}}
.fig-label {{
    font-size: 1.0em; font-weight: bold; color: #1a3a5c; margin-bottom: 6px;
}}
.fig-desc {{
    color: #444; line-height: 1.55; margin-bottom: 14px; max-width: 960px;
    font-size: 0.9em;
}}
.cross-content {{ text-align: center; }}
.cross-content img {{ max-width: 100%; height: auto; }}
.site-grid {{
    display: grid; gap: 0;
    border: 1px solid #ccc; border-radius: 4px; overflow: hidden;
}}
.hdr-site {{
    color: white; font-weight: bold; font-size: 0.9em;
    padding: 8px 6px; text-align: center;
}}
.panel-cell {{
    border-top: 1px solid #dde; border-left: 1px solid #dde;
    padding: 5px; text-align: center; background: #fafafa;
}}
.panel-cell img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.data-table {{
    width: 100%; border-collapse: collapse; font-size: 0.82em; margin: 8px 0;
}}
.data-table th {{
    background: #2c5f8a; color: white; padding: 6px 8px;
    text-align: left; font-weight: bold; white-space: nowrap;
}}
.data-table td {{
    padding: 5px 8px; border-bottom: 1px solid #eee; text-align: left;
    white-space: nowrap;
}}
.data-table tr.alt {{ background: #f4f7fb; }}
.data-table tr:hover {{ background: #e8f0f8; }}
.site-tbl-block {{ margin-bottom: 22px; }}
.site-tbl-hdr {{
    color: white; font-weight: bold; padding: 7px 12px;
    border-radius: 4px 4px 0 0; font-size: 0.9em;
}}
.missing {{
    color: #a33; font-size: 0.85em; padding: 12px 10px;
    background: #fff3f3; border: 1px dashed #e88; border-radius: 3px;
    text-align: center;
}}
.fname {{ font-family: monospace; font-size: 0.85em; color: #666; }}
"""

# ── Assemble ───────────────────────────────────────────────────────────────────
print("\nBuilding Section 1: Feature predictiveness...")
s1 = build_section1()
print("\nBuilding Section 2: Decision tree...")
s2 = build_section2()
print("\nBuilding Section 3: Mortality outcomes...")
s3 = build_section3()

site_list_str = ", ".join(_site_label(s) for s in SITE_NAMES)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Threshold Decision-Rule Summary Report</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>Threshold Decision-Rule Analysis Summary</h1>
<div class="subtitle">
  Sites: {esc(site_list_str)}&nbsp;&bull;&nbsp;
  Generated: <span id="ts"></span>
  {"&nbsp;&bull;&nbsp;<em>Images embedded (portable)</em>" if EMBED
   else "&nbsp;&bull;&nbsp;<em>Images linked &mdash; keep output/ folder alongside the HTML</em>"}
</div>
<script>document.getElementById("ts").textContent = new Date().toLocaleString();</script>

<div class="toc">
  <p>Contents</p>
  <ul>
    <li><a href="#s1">1. Which features predict clinician vasopressin initiation?</a></li>
    <li><a href="#s2">2. Can a simple decision tree predict vasopressin initiation?</a></li>
    <li><a href="#s3">3. Does threshold-concordant care associate with better mortality?</a></li>
  </ul>
</div>

<h2 id="s1">1. Which features predict clinician vasopressin initiation?</h2>
{s1}

<h2 id="s2">2. Can a simple decision tree predict vasopressin initiation?</h2>
{s2}

<h2 id="s3">3. Does threshold-concordant care associate with better mortality?</h2>
{s3}

</body>
</html>
"""

OUT_FILE.write_text(html, encoding="utf-8")
print(f"\nWrote: {OUT_FILE}")
print(f"  Sites: {site_list_str}")
if not EMBED:
    print("  Note: images are linked — keep the output/ folder alongside the HTML.")
    print("  Use --embed for a single portable file.")
