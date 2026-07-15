#!/usr/bin/env python3
"""
04_site_variation_analysis.py  [PER-SITE — run at each participating institution]

Hospital/Site Variation Analysis for vasopressin initiation.

Terminology: "site" = CLIF site / larger data pool (SITE_NAME); "hospital" =
an individual community or academic hospital within a site's data pool
(hospital_id in clif_adt.parquet — a site may bundle several).

Shared logistic spec (see _build_design_matrix), fit at every level below:
  vaso_on ~ intercept + age_c
          + p_f_ratio_c + device_category (Room Air ref)   [respiratory]
          + creatinine_c + rrt                              [renal]
          + platelet_c                                      [coagulation]
          + bilirubin_c                                      [liver]
          + gcs_c                                            [neuro]
          + rcs(NEE,4) + rcs(time,4)
  SOFA components replace the aggregate SOFA score so NEE's mechanical
  overlap with the cardiovascular SOFA axis is excluded; cardiovascular
  itself is intentionally omitted (captured by NEE on the RHS instead).

Approach 1 — Fixed-effects / intercept comparison:
  Fit the shared spec per-site (independent refit, own knots/scaling/device
  categories). Compare site-specific intercepts (baseline propensity
  variation). Compute P(vasopressor) for a reference patient at each site.

Approach 2 — Time/dose-varying probability of vasopressor initiation:
  Using per-site model coefficients, all components except the one being
  varied are held at fixed reference values (REF_AGE, REF_PF, REF_DEVICE, ...):
    Plot A: P(vaso_on) vs. time
    Plot B: P(vaso_on) vs. NEE dose

Mixed-effects pooling (runs when ≥2 sites' data are available locally):
  DL random-effects meta-analysis of site intercepts → τ², ICC, MOR
  Pooled logistic GLM with site dummies (fixed effects) for comparison
  Optional GLMM with site random intercept via BinomialBayesMixedGLM

Approach 3 — Ward/ICU-type + hospital variance, 4-level decomposition
(this site only):
  Same fixed-effects-intercept + DL pooling technique as the site-level mixed
  effects above (fit_grouped_variance), applied twice within this site:
    - ICU/ward type (from cohort.location_type, canonicalized) -> tau2_ward
    - hospital_id (community/academic hospital within this site) -> tau2_hospital
  Combined with the site-level tau2 (mixed_effects.dl_tau2) and the fixed
  patient-level logistic residual variance (pi^2/3), this gives a 4-level
  variance decomposition: patient / ward (ICU type) / hospital / site.

Outputs:
  output/upload_to_box_<SITE>/<cohort>/
    site_variation_packet_<cohort>_<SITE>.json
    coefficient_table_<cohort>_<SITE>.csv
    ward_intercepts_<cohort>_<SITE>.csv / hospital_intercepts_<cohort>_<SITE>.csv
  output/cross_site_variation/<cohort>/
    approach1_intercept_comparison_<cohort>.png
    approach2_time_varying_<cohort>.png
    approach2_nee_varying_<cohort>.png
    mixed_effects_forest_<cohort>.png
    variance_decomposition_<cohort>_<SITE>.png

Usage:
  uv run python code/04_site_variation_analysis.py
  uv run python code/04_site_variation_analysis.py --cohort rhee
  uv run python code/04_site_variation_analysis.py --cohort both  # default
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--site",   default=None)
_ap.add_argument("--cohort", default="both", choices=["sepsis3", "rhee", "both"])
_args, _ = _ap.parse_known_args()


def _load_site_config():
    import importlib.util as _ilu
    cfg_path = BASE_DIR / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit("ERROR: config/config.py not found.")

SITE_NAME   = _args.site if _args.site else getattr(_cfg, "SITE_NAME", "UCMC")
OUTPUT_ROOT = Path(getattr(_cfg, "OUTPUT_ROOT", "."))
COHORT_ARG  = _args.cohort
LOGO_PAL    = getattr(_cfg, "LOGO_PALETTE", {})

PATIENT_LEVEL_DIR = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"

_COHORT_LABELS = {"sepsis3": "Sepsis-3 (CMS)", "rhee": "Rhee/CDC ASE"}

# Reference patient for Approach 2 plots. SOFA components replace the
# aggregate SOFA score (see fit_site_logistic) so NEE's mechanical overlap
# with the cardiovascular SOFA axis is excluded; each reference value below
# is a normal/mild-severity default for that organ axis.
REF_AGE       = 60.0
REF_NEE       = 0.3    # mcg/kg/min
REF_PF        = 300.0  # p/f ratio (mild/no ARDS)
REF_DEVICE    = "Room Air"
REF_CREATININE = 0.8   # mg/dL
REF_RRT       = 0
REF_PLATELET  = 200.0  # x10^3/uL
REF_BILIRUBIN = 0.7    # mg/dL
REF_GCS       = 15.0

# Continuous SOFA-component covariates (z-scored like age); device_category
# is dummy-coded separately with REF_DEVICE as the reference level.
_CONT_COLS = ["age", "p_f_ratio", "creatinine", "platelet", "bilirubin", "gcs"]
_DEVICE_REF = REF_DEVICE

_DEFAULT_COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2"]

# Minimum patients an ICU type / hospital needs before it gets its own
# fixed-effects fit in the Approach 3 local variance decomposition.
MIN_ICU_N_PATIENTS = 30
MIN_HOSPITAL_N_PATIENTS = 30

# fit_site_logistic fits ~15 parameters (6 continuous SOFA-component covariates
# + rrt + device dummies + 3 NEE-spline + 3 time-spline terms) with no
# separation guard. A group with few vaso_on=1 hours (or a rare device
# category) can hit quasi-complete separation, inflating the intercept SE
# into the thousands — MIN_ICU_N_PATIENTS/MIN_HOSPITAL_N_PATIENTS only gate on
# patient count, which doesn't catch this since event rate varies by group.
# Group-level fits with an intercept SE at or above this threshold are kept in
# the packet (for transparency) but excluded from DL pooling — same cutoff
# already used in 06_cross_site_variation_analysis.py's intercept plots to
# flag unusable points.
MAX_STABLE_INTERCEPT_SE = 10.0

# ICU-type canonicalization (same mapping used in 03_epi_analysis.py /
# 06_cross_site_variation_analysis.py, kept in sync for consistent labels).
_ICU_CANON = {
    "medical_icu": "MICU", "micu": "MICU",
    "cardiac_icu": "CICU", "cicu": "CICU", "coronary_icu": "CICU",
    "surgical_icu": "SICU", "sicu": "SICU",
    "mixed_neuro_icu": "Neuro ICU", "neuro_icu": "Neuro ICU", "neuro_sicu": "Neuro ICU",
    "mixed_cardiothoracic_icu": "CT ICU", "cardiothoracic_icu": "CT ICU",
    "cardiothoracic_surgical_icu": "CT ICU",
    "burn_icu": "Burn ICU",
    "general_icu": "Mixed/General ICU", "mixed_icu": "Mixed/General ICU",
    "medical intensive care unit (micu)": "MICU",
    "cardiac vascular intensive care unit (cvicu)": "CICU",
    "coronary care unit (ccu)": "CICU",
    "surgical intensive care unit (sicu)": "SICU",
    "trauma sicu (tsicu)": "SICU",
    "medical/surgical intensive care unit (micu/sicu)": "Mixed/General ICU",
    "neuro surgical intensive care unit (neuro sicu)": "Neuro ICU",
    "intensive care unit (icu)": "Mixed/General ICU",
}


def _pick_location_col(cohort: pd.DataFrame) -> str | None:
    """First of location_type/location_name/location_category with >1 non-null value."""
    return next(
        (c for c in ["location_type", "location_name", "location_category"]
         if c in cohort.columns and cohort[c].notna().sum() > 0 and cohort[c].nunique() > 1),
        None,
    )


def _canon_icu_series(raw: pd.Series) -> pd.Series:
    """Canonical ICU-type label per row; NaN for missing/unrecorded location
    (dropped from the ward-level analysis, not lumped into "Other ICU")."""
    low = raw.astype(str).str.lower().str.strip()
    is_missing = raw.isna() | low.isin(["none", "nan", ""])
    canon = low.map(_ICU_CANON).fillna("Other ICU")
    canon[is_missing] = np.nan
    return canon


def _dl_pool(alphas: np.ndarray, alpha_var: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling of K independent (alpha_k, se_k)
    intercept estimates -> between-group variance tau2, ICC, and MOR (median
    odds ratio) on the patient-level logistic residual variance pi^2/3.
    """
    K = len(alphas)
    if K < 2:
        return {}
    w = 1.0 / alpha_var; ws = w.sum()
    theta_fe = float((w * alphas).sum() / ws)
    Q = float((w * (alphas - theta_fe) ** 2).sum())
    C = float(ws - (w ** 2).sum() / ws)
    tau2 = max(0.0, (Q - (K - 1)) / C) if C > 0 else 0.0

    w_re = 1.0 / (alpha_var + tau2); ws_re = w_re.sum()
    theta_re = float((w_re * alphas).sum() / ws_re)
    se_re = float(np.sqrt(1.0 / ws_re))
    icc = tau2 / (tau2 + np.pi ** 2 / 3)
    mor = float(np.exp(np.sqrt(2 * tau2) * 0.6745)) if tau2 > 0 else 1.0

    return {"tau2": tau2, "icc": icc, "mor": mor, "pooled_alpha": theta_re, "se": se_re, "k": K, "q": Q}


# ── RCS helpers ───────────────────────────────────────────────────────────────
def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell RCS basis: K-1 columns (linear term + K-2 nonlinear terms)."""
    K    = len(knots)
    kK   = knots[-1]; kKm1 = knots[-2]; denom = kK - kKm1
    cols = [x.copy()]
    for j in range(K - 2):
        kj  = knots[j]
        col = (
            np.maximum(x - kj,   0) ** 3
            - np.maximum(x - kKm1, 0) ** 3 * (kK - kj)   / denom
            + np.maximum(x - kK,   0) ** 3 * (kKm1 - kj) / denom
        )
        cols.append(col)
    return np.column_stack(cols)


def _rcs_knots(x: np.ndarray, df: int = 4) -> np.ndarray:
    quantiles = {4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95], 3: [10, 50, 90]}.get(df, [5, 35, 65, 95])
    return np.percentile(x[np.isfinite(x)], quantiles)


# ── Data prep ─────────────────────────────────────────────────────────────────
# Labs/GCS/p_f_ratio are recorded sparsely (not every hour), unlike
# device_category/fio2_set which are already forward-filled at extraction
# time (persistent ventilator/device settings). Forward-fill them here per
# stay_id so they're usable as hourly time-varying covariates instead of
# dropping nearly every hour with no fresh measurement.
_LOCF_COLS = ["p_f_ratio", "creatinine", "platelet", "bilirubin", "gcs"]


def _prep_person_hours(cohort: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    cols = (["stay_id", "time_hour", "vaso_dose", "nee", "rrt", "device_category"]
            + _LOCF_COLS)
    ph = features[cols].copy()
    ph["vaso_on"] = (ph["vaso_dose"] > 0).astype(int)
    ph = ph.sort_values(["stay_id", "time_hour"])
    ph[_LOCF_COLS] = ph.groupby("stay_id")[_LOCF_COLS].ffill()
    ph = ph.merge(cohort[["stay_id", "age"]], on="stay_id", how="left")
    ph = ph.dropna(subset=["nee", "age", "rrt", "device_category"] + _LOCF_COLS).copy()
    return ph


# ── Shared design matrix (per-site refits and the pooled cross-site GLM) ─────
def _build_design_matrix(
    df: pd.DataFrame,
    nee_knots: np.ndarray = None,
    time_knots: np.ndarray = None,
    device_categories: list = None,
    means: dict = None,
    sds: dict = None,
):
    """vaso_on ~ intercept + age_c + p_f_ratio_c + device_category + creatinine_c
    + rrt + platelet_c + bilirubin_c + gcs_c + rcs(NEE,4) + rcs(time,4).

    SOFA components replace the aggregate SOFA score so NEE's mechanical
    overlap with the cardiovascular SOFA axis is excluded; cardiovascular
    itself is intentionally omitted (captured by NEE on the RHS instead).

    knots/means/sds/device_categories can be passed in (pooled cross-site fit
    reusing one shared scaling) or left None to derive from `df` (per-group
    refits, each on their own scale).
    """
    if nee_knots is None:
        nee_knots = _rcs_knots(df["nee"].values, df=4)
    if time_knots is None:
        time_knots = _rcs_knots(df["time_hour"].values, df=4)
    nee_b  = _rcs_basis(df["nee"].values, nee_knots)
    time_b = _rcs_basis(df["time_hour"].values, time_knots)

    if means is None or sds is None:
        means, sds = {}, {}
        for c in _CONT_COLS:
            means[c] = float(df[c].mean())
            sds[c]   = float(df[c].std()) or 1.0
    centered = {c: (df[c].values - means[c]) / sds[c] for c in _CONT_COLS}

    if device_categories is None:
        device_categories = sorted(set(df["device_category"].unique()) - {_DEVICE_REF})
    device_cols = {d: (df["device_category"].values == d).astype(np.float64)
                   for d in device_categories}

    nee_cols  = [f"rcs_nee_{i}"  for i in range(nee_b.shape[1])]
    time_cols = [f"rcs_time_{i}" for i in range(time_b.shape[1])]
    col_names = (["intercept"] + [f"{c}_c" for c in _CONT_COLS] + ["rrt"]
                 + [f"device_{d}" for d in device_categories] + nee_cols + time_cols)

    X = np.column_stack(
        [np.ones(len(df))]
        + [centered[c] for c in _CONT_COLS]
        + [df["rrt"].values.astype(np.float64)]
        + [device_cols[d] for d in device_categories]
        + [nee_b, time_b]
    ).astype(np.float64)

    return X, col_names, means, sds, nee_knots, time_knots, device_categories


# ── Approach 1 & 2: Per-site logistic GLM ─────────────────────────────────────
def fit_site_logistic(ph: pd.DataFrame, cohort_label: str, site: str) -> dict:
    """Fit the shared logistic spec (see _build_design_matrix) independently
    on this group's person-hours (its own knots/scaling/device categories)."""
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as fam

    df = ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt", "device_category"]
            + _CONT_COLS].dropna().copy()

    X, col_names, means, sds, nee_knots, time_knots, device_categories = (
        _build_design_matrix(df)
    )
    Y = df["vaso_on"].values.astype(np.float64)

    fit = GLM(Y, X, family=fam.Binomial()).fit(maxiter=200)

    coef_out = {
        name: {"beta": float(fit.params[i]), "se": float(fit.bse[i]),
               "or":   float(np.exp(fit.params[i]))}
        for i, name in enumerate(col_names)
    }

    result = {
        "site":        site,
        "cohort":      cohort_label,
        "n_patients":  int(df["stay_id"].nunique()),
        "n_ph_rows":   int(len(df)),
        "n_vaso_on":   int(Y.sum()),
        "means": means, "sds": sds,
        "device_categories": device_categories,
        "nee_knots":   nee_knots.tolist(),
        "time_knots":  time_knots.tolist(),
        "coefficients": coef_out,
    }
    result["ref_patient_p_t0"] = float(predict_p(result, REF_AGE, REF_NEE, 5.0))
    return result


def predict_p(
    model: dict, ages, nees, times,
    p_f_ratio=REF_PF, device_category=REF_DEVICE, creatinine=REF_CREATININE,
    rrt=REF_RRT, platelet=REF_PLATELET, bilirubin=REF_BILIRUBIN, gcs=REF_GCS,
) -> np.ndarray:
    """Vectorized P(vaso_on) prediction from a fitted model dict. Component
    axes not being varied by the caller stay at their reference values."""
    arrs = {
        "age": ages, "p_f_ratio": p_f_ratio, "creatinine": creatinine,
        "platelet": platelet, "bilirubin": bilirubin, "gcs": gcs,
        "nee": nees, "time": times, "rrt": rrt,
    }
    arrs = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in arrs.items()}
    n = max(len(v) for v in arrs.values())
    arrs = {k: np.broadcast_to(v, (n,)).copy() for k, v in arrs.items()}
    device_arr = np.broadcast_to(np.asarray(device_category, dtype=object), (n,))

    means, sds, coef = model["means"], model["sds"], model["coefficients"]
    nee_b  = _rcs_basis(arrs["nee"],  np.array(model["nee_knots"]))
    time_b = _rcs_basis(arrs["time"], np.array(model["time_knots"]))

    lp = np.full(n, coef["intercept"]["beta"])
    for c in _CONT_COLS:
        centered = (arrs[c] - means[c]) / sds[c]
        lp += coef[f"{c}_c"]["beta"] * centered
    lp += coef["rrt"]["beta"] * arrs["rrt"]
    for d in model.get("device_categories", []):
        col = f"device_{d}"
        if col in coef:
            lp += coef[col]["beta"] * (device_arr == d).astype(float)
    for i in range(nee_b.shape[1]):
        lp += coef[f"rcs_nee_{i}"]["beta"]  * nee_b[:, i]
    for i in range(time_b.shape[1]):
        lp += coef[f"rcs_time_{i}"]["beta"] * time_b[:, i]

    return 1.0 / (1.0 + np.exp(-lp))


# ── Plot: Approach 1 — reference-patient P and intercept forest ───────────────
def plot_approach1(site_models: dict, out_dir: Path, cohort_label: str, pal: dict):
    sites    = list(site_models.keys())
    colors   = [pal.get(s, "#888888") for s in sites]
    ref_p    = [site_models[s]["ref_patient_p_t0"] for s in sites]
    alphas   = [site_models[s]["coefficients"]["intercept"]["beta"] for s in sites]
    alpha_se = [site_models[s]["coefficients"]["intercept"]["se"]   for s in sites]
    ns       = [site_models[s]["n_patients"] for s in sites]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(sites) * 0.9 + 2)))

    # Panel A: P(vaso) bar for reference patient
    ax = axes[0]
    y_pos = np.arange(len(sites))
    bars = ax.barh(y_pos, ref_p, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s}\n(n={n:,})" for s, n in zip(sites, ns)], fontsize=9)
    ax.set_xlabel(
        f"P(vasopressin | age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, "
        f"NEE={REF_NEE} mcg/kg/min, t=0)", fontsize=9
    )
    ax.set_title("Reference-patient P(vasopressin)", fontsize=10, fontweight="bold")
    for bar, p in zip(bars, ref_p):
        ax.text(p + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, max(ref_p) * 1.4)

    # Panel B: Forest plot of logit intercepts ± 95% CI
    ax2 = axes[1]
    for yi, (s, alpha, se, col) in enumerate(zip(sites, alphas, alpha_se, colors)):
        ax2.errorbar(alpha, yi, xerr=1.96 * se, fmt="D", color=col,
                     capsize=5, markersize=8, linewidth=2, zorder=3)
        ax2.text(alpha + 1.96 * se + 0.02, yi,
                 f"α={alpha:+.3f}", va="center", fontsize=8, color=col)
    ax2.axvline(0, color="lightgrey", linestyle="--", linewidth=1)
    ax2.set_yticks(np.arange(len(sites))); ax2.set_yticklabels(sites)
    ax2.set_xlabel("Logit intercept ± 95% CI\n(components/age at means, NEE=0, t=0)", fontsize=9)

    sd_alpha = float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0
    ax2.set_title(
        f"Site intercepts (baseline propensity)\nSD = {sd_alpha:.3f} log-odds",
        fontsize=10, fontweight="bold",
    )

    fig.suptitle(
        f"Approach 1 — Fixed-effects intercept comparison\n"
        f"Cohort: {_COHORT_LABELS.get(cohort_label, cohort_label)}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"approach1_intercept_comparison_{cohort_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: approach1_intercept_comparison_{cohort_label}.png")


# ── Plot: Approach 2A — P(vaso) vs time ──────────────────────────────────────
def plot_approach2_time(site_models: dict, out_dir: Path, cohort_label: str, pal: dict):
    """P(vaso_on) over time for reference patient at each site."""
    if not site_models:
        return

    t_max     = min(max(max(m["time_knots"]) for m in site_models.values()), 120)
    time_grid = np.linspace(0, t_max, 300)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, m in site_models.items():
        p_t = predict_p(m, REF_AGE, REF_NEE, time_grid)
        ax.plot(time_grid, p_t, color=pal.get(s, "#888888"), linewidth=2.5, label=s)

    ax.set_xlabel("Hours from NE start (t = 0)", fontsize=11)
    ax.set_ylabel("P(vasopressin on | hour)", fontsize=11)
    ax.set_title(
        f"Approach 2A — Time-varying P(vasopressin)\n"
        f"Reference patient: age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, NEE={REF_NEE} mcg/kg/min  "
        f"[{_COHORT_LABELS.get(cohort_label, cohort_label)}]",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_xlim(0, t_max); ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"approach2_time_varying_{cohort_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: approach2_time_varying_{cohort_label}.png")


# ── Plot: Approach 2B — P(vaso) vs NEE, reference patient ────────────────────
def plot_approach2_nee(
    site_models: dict,
    ph_by_site: dict,
    out_dir: Path,
    cohort_label: str,
    pal: dict,
    ref_time: float = 0.0,
):
    """P(vaso_on) vs. NEE dose for the fixed reference patient (all SOFA
    components held at reference values, matching Approach 2A) — simplified
    from the old aggregate-SOFA version, which fit an empirical SOFA~NEE
    curve; with SOFA decomposed there's no single severity axis to feature."""
    if not site_models:
        return

    sites   = list(site_models.keys())
    all_nee = np.concatenate([
        ph_by_site[s]["nee"].dropna().values for s in sites if s in ph_by_site
    ])
    nee_max  = float(np.percentile(all_nee[np.isfinite(all_nee)], 95))
    nee_grid = np.linspace(0.0, min(nee_max, 2.0), 300)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, m in site_models.items():
        p_nee = predict_p(m, REF_AGE, nee_grid, ref_time)
        ax.plot(nee_grid, p_nee, color=pal.get(s, "#888888"), linewidth=2.5, label=s)

    ax.set_xlabel("NEE dose (mcg/kg/min)", fontsize=11)
    ax.set_ylabel("P(vasopressin on)", fontsize=11)
    ax.set_title(
        f"Approach 2B — P(vasopressin) vs. NEE dose\n"
        f"Reference patient: age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, t={ref_time:.0f} h  "
        f"[{_COHORT_LABELS.get(cohort_label, cohort_label)}]",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_xlim(0, nee_grid[-1]); ax.set_ylim(0, 1)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"approach2_nee_varying_{cohort_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: approach2_nee_varying_{cohort_label}.png")


# ── Mixed-effects pooling ─────────────────────────────────────────────────────
def fit_and_plot_mixed_effects(
    site_models: dict,
    ph_by_site: dict,
    out_dir: Path,
    cohort_label: str,
    pal: dict,
) -> dict:
    """DL random-effects meta-analysis of site intercepts + pooled GLM + optional GLMM.

    Returns dict with tau2, ICC, MOR, and GLMM results if available.
    """
    sites = [s for s in site_models if s in ph_by_site]
    if len(sites) < 2:
        print("  Skipping mixed effects: <2 sites with data.")
        return {}

    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as fam

    # ── DL random-effects meta-analysis of site intercepts ─────────────────
    alphas    = np.array([site_models[s]["coefficients"]["intercept"]["beta"] for s in sites])
    alpha_var = np.array([site_models[s]["coefficients"]["intercept"]["se"]   for s in sites]) ** 2

    dl        = _dl_pool(alphas, alpha_var)
    tau2_dl, icc_dl, mor_dl = dl["tau2"], dl["icc"], dl["mor"]
    theta_re, se_re         = dl["pooled_alpha"], dl["se"]

    print(f"  DL  τ²={tau2_dl:.4f}  ICC={icc_dl:.3f}  MOR={mor_dl:.3f}")
    print(f"  DL pooled logit intercept = {theta_re:.4f} ± {se_re:.4f} (SE)  "
          f"[P={1/(1+np.exp(-theta_re)):.1%}]")

    # ── Pooled GLM with site dummies ────────────────────────────────────────
    frames = []
    for s in sites:
        ph = ph_by_site[s][["stay_id", "time_hour", "vaso_on", "nee", "rrt",
                             "device_category"] + _CONT_COLS].copy()
        ph["site"] = s
        frames.append(ph)
    all_ph = pd.concat(frames, ignore_index=True).dropna(
        subset=["nee", "rrt", "device_category", "vaso_on"] + _CONT_COLS
    )

    X_base, base_col_names, _means, _sds, _nee_kp, _time_kp, _dev_cats = (
        _build_design_matrix(all_ph)
    )
    n_base = len(base_col_names)

    site_cats = sorted(sites)
    REF_SITE  = "UCMC" if "UCMC" in site_cats else site_cats[0]
    non_ref_cats = [s for s in site_cats if s != REF_SITE]
    site_dum  = pd.get_dummies(all_ph["site"], prefix="s", drop_first=False).astype(np.float64)
    site_ref  = f"s_{REF_SITE}"
    if site_ref in site_dum.columns:
        site_dum = site_dum.drop(columns=[site_ref])
    site_dum  = site_dum[[f"s_{s}" for s in non_ref_cats]]

    X_pool = np.column_stack([X_base, site_dum.values]).astype(np.float64)
    Y_pool = all_ph["vaso_on"].values.astype(np.float64)

    pooled_intercepts: dict = {}
    try:
        fit_pool = GLM(Y_pool, X_pool, family=fam.Binomial()).fit(maxiter=200)
        pooled_intercepts[REF_SITE] = {
            "alpha": float(fit_pool.params[0]),
            "se":    float(fit_pool.bse[0]),
        }
        for ji, s in enumerate(non_ref_cats):
            pi    = n_base + ji
            alpha = float(fit_pool.params[0] + fit_pool.params[pi])
            se    = float(np.sqrt(fit_pool.bse[0]**2 + fit_pool.bse[pi]**2))
            pooled_intercepts[s] = {"alpha": alpha, "se": se}
        print(f"  Pooled GLM (site dummies) converged OK.")
    except Exception as _ep:
        print(f"  WARNING pooled GLM failed: {_ep}")

    # ── Optional GLMM with site random intercept ────────────────────────────
    glmm_result: dict = {}
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as BMGLM
        site_idx = np.array([site_cats.index(s) for s in all_ph["site"]])
        Z        = (site_idx[:, None] == np.arange(len(site_cats))[None, :]).astype(np.float64)
        id_arr   = np.zeros(len(site_cats), dtype=int)
        X_glmm   = X_pool[:, :n_base]
        # Try progressively stronger priors; with few sites the log-variance can
        # drift to huge values under a weak prior (MAP finds a degenerate mode).
        s2u = float("nan")
        for _vcp_p in [2, 4, 8, 16]:
            _fit = BMGLM(Y_pool, X_glmm, Z, id_arr, vcp_p=_vcp_p, fe_p=2).fit_map()
            _s2u = float(np.exp(_fit.vcp_mean[0])) if len(_fit.vcp_mean) > 0 else float("nan")
            if np.isfinite(_s2u) and _s2u < 5:
                s2u = _s2u
                break
        if not np.isfinite(s2u) or s2u >= 5:
            raise RuntimeError(
                f"GLMM non-convergent (σ²={s2u:.2f}) with {len(sites)} sites — "
                "variance estimate is unreliable; use DL result"
            )
        icc_glmm = s2u / (s2u + np.pi**2 / 3)
        mor_glmm = float(np.exp(np.sqrt(2 * s2u) * 0.6745))
        glmm_result = {"sigma2_site": s2u, "icc": icc_glmm, "mor": mor_glmm}
        print(f"  GLMM  σ²_site={s2u:.4f}  ICC={icc_glmm:.3f}  MOR={mor_glmm:.3f}")
    except Exception as _eg:
        print(f"  GLMM skipped ({type(_eg).__name__}: {_eg})")

    # ── Forest plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, len(sites) * 1.3 + 2.5)))
    y_pos = np.arange(len(sites))

    for yi, s in enumerate(sites):
        col = pal.get(s, "#888888")
        # Per-site independent GLM
        a_ind = site_models[s]["coefficients"]["intercept"]["beta"]
        se_ind = site_models[s]["coefficients"]["intercept"]["se"]
        ax.errorbar(a_ind, yi - 0.15, xerr=1.96 * se_ind, fmt="D", color=col,
                    capsize=4, markersize=8, linewidth=2, zorder=3)
        # Pooled GLM (shrinkage via common covariate model)
        if s in pooled_intercepts:
            a_pool = pooled_intercepts[s]["alpha"]
            se_pool = pooled_intercepts[s]["se"]
            ax.errorbar(a_pool, yi + 0.15, xerr=1.96 * se_pool, fmt="s", color=col,
                        capsize=4, markersize=6, linewidth=1.5, alpha=0.75, zorder=3)

    # DL random-effects pooled line + CI band
    ax.axvline(theta_re, color="#222222", linewidth=2, alpha=0.9,
               label=f"DL pooled: {theta_re:.3f} (P={1/(1+np.exp(-theta_re)):.1%})")
    ax.axvspan(theta_re - 1.96 * se_re, theta_re + 1.96 * se_re,
               alpha=0.12, color="grey")
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)

    ax.set_yticks(y_pos); ax.set_yticklabels(sites, fontsize=10)
    ax.set_xlabel("Logit intercept ± 95% CI  (components/age at means, NEE=0, t=0)", fontsize=10)

    glmm_str = ""
    if glmm_result:
        glmm_str = (f"  |  GLMM σ²={glmm_result['sigma2_site']:.3f}, "
                    f"ICC={glmm_result['icc']:.3f}, MOR={glmm_result['mor']:.3f}")
    ax.set_title(
        f"Mixed-effects pooling  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]\n"
        f"DL τ²={tau2_dl:.4f},  ICC={icc_dl:.3f},  MOR={mor_dl:.3f}{glmm_str}",
        fontsize=10, fontweight="bold",
    )

    legend_els = [
        mlines.Line2D([], [], marker="D", color="grey", linestyle="none",
                      markersize=8, label="Per-site GLM"),
        mlines.Line2D([], [], marker="s", color="grey", linestyle="none",
                      markersize=6, alpha=0.75, label="Pooled GLM (site dummies)"),
        mlines.Line2D([], [], color="#222222", linewidth=2,
                      label="DL random-effects pooled ± 95% CI"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"mixed_effects_forest_{cohort_label}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mixed_effects_forest_{cohort_label}.png")

    return {
        "dl_tau2": tau2_dl, "dl_icc": icc_dl, "dl_mor": mor_dl,
        "dl_pooled_alpha": theta_re, "dl_se": se_re,
        **({"glmm": glmm_result} if glmm_result else {}),
    }


# ── Approach 3: ward/ICU-type + hospital variance (this site only) ───────────
def fit_grouped_variance(
    cohort: pd.DataFrame, ph: pd.DataFrame, cohort_label: str, site: str,
    group_col: str, min_n: int, label: str, canon_fn=None,
) -> dict:
    """Same fixed-effects-intercept + DL pooling technique as the site-level
    mixed effects, applied within `site` to whatever local grouping factor is
    passed in (ICU/ward type or hospital) — shared machinery for both levels
    of the local variance hierarchy.

    Fits the shared logistic spec (see _build_design_matrix) independently
    within each group with >= min_n patients, then DL-pools those intercepts
    -> tau2/ICC/MOR for that grouping factor. Purely local to `site` — no
    cross-site data needed, so this is safe to compute (and share as an
    aggregate number) at any single site.
    """
    if group_col not in cohort.columns:
        print(f"  No usable {label} column ('{group_col}') found — skipping {label}-level variance.")
        return {}

    group_map = cohort[["stay_id", group_col]].copy()
    if canon_fn is not None:
        group_map[group_col] = canon_fn(group_map[group_col])
    group_map = group_map.dropna(subset=[group_col])[["stay_id", group_col]]

    ph_grp = ph.merge(group_map, on="stay_id", how="inner")
    if ph_grp.empty:
        print(f"  No person-hours with a known {label} — skipping {label}-level variance.")
        return {}

    group_models: dict = {}
    for grp, sub in ph_grp.groupby(group_col):
        n_pat = sub["stay_id"].nunique()
        if n_pat < min_n:
            print(f"    Skipping {label} '{grp}' (n={n_pat} < {min_n} patients)")
            continue
        try:
            group_models[grp] = fit_site_logistic(sub, cohort_label, f"{site}:{grp}")
            m = group_models[grp]
            se = m["coefficients"]["intercept"]["se"]
            m["stable"] = bool(se < MAX_STABLE_INTERCEPT_SE)
            flag = "" if m["stable"] else "  [UNSTABLE — likely separation; excluded from DL pooling]"
            print(f"    [{grp}] n={m['n_patients']:,}  vaso_on={m['n_vaso_on']:,}  "
                  f"α={m['coefficients']['intercept']['beta']:+.4f} (SE={se:.3f}){flag}")
        except Exception as _e:
            print(f"    WARNING: model fit failed for {label} '{grp}': {_e}")

    stable_models = {g: m for g, m in group_models.items() if m.get("stable", True)}
    if len(stable_models) < 2:
        print(f"  <2 stable {label}s with >= {min_n} patients — skipping {label}-level variance.")
        return {"group_column": group_col, "group_models": group_models}

    alphas    = np.array([m["coefficients"]["intercept"]["beta"] for m in stable_models.values()])
    alpha_var = np.array([m["coefficients"]["intercept"]["se"]   for m in stable_models.values()]) ** 2
    dl = _dl_pool(alphas, alpha_var)

    n_excluded = len(group_models) - len(stable_models)
    if n_excluded:
        print(f"  {label.capitalize()}-level DL: excluded {n_excluded} unstable group(s) from pooling")
    print(f"  {label.capitalize()}-level DL: τ²={dl['tau2']:.4f}  ICC={dl['icc']:.3f}  "
          f"MOR={dl['mor']:.3f}  (k={dl['k']} {label}s)")

    return {
        "group_column": group_col,
        "group_models": group_models,
        "tau2": dl["tau2"], "icc": dl["icc"], "mor": dl["mor"],
        "pooled_alpha": dl["pooled_alpha"], "se": dl["se"], "k": dl["k"],
    }


def fit_ward_level_variance(cohort: pd.DataFrame, ph: pd.DataFrame, cohort_label: str, site: str) -> dict:
    """ICU/ward-type variance via fit_grouped_variance (dynamic location
    column pick + ICU-type canonicalization)."""
    loc_col = _pick_location_col(cohort)
    if loc_col is None:
        print("  No usable ICU/ward location column found — skipping ward-level variance.")
        return {}
    return fit_grouped_variance(
        cohort, ph, cohort_label, site,
        group_col=loc_col, min_n=MIN_ICU_N_PATIENTS, label="ward", canon_fn=_canon_icu_series,
    )


def fit_hospital_level_variance(cohort: pd.DataFrame, ph: pd.DataFrame, cohort_label: str, site: str) -> dict:
    """Hospital variance via fit_grouped_variance — hospital_id identifies
    the individual community/academic hospital within this CLIF site's data
    pool (single-hospital sites/pulls just yield <2 groups and are skipped)."""
    return fit_grouped_variance(
        cohort, ph, cohort_label, site,
        group_col="hospital_id", min_n=MIN_HOSPITAL_N_PATIENTS, label="hospital",
    )


def compute_variance_decomposition(
    ward_result: dict, hospital_result: dict, site_mixed_effects: dict
) -> dict:
    """4-level variance decomposition: patient (fixed logistic residual,
    pi^2/3) / ward-ICU-type (tau2_ward, within this site) / hospital
    (tau2_hospital, within this site's own community/academic hospitals) /
    site (tau2_dl, this site's contribution to the cross-site DL pooling —
    "site" here is the CLIF-site / larger data pool, not an individual
    hospital).
    """
    tau2_ward = ward_result.get("tau2")
    tau2_hosp = hospital_result.get("tau2")
    tau2_site = site_mixed_effects.get("dl_tau2")
    if tau2_ward is None or tau2_site is None:
        return {}
    # Single/no-hospital sites (e.g. UCMC, MIMIC) can't estimate a hospital-level
    # component — report it as 0% rather than dropping the whole decomposition.
    if tau2_hosp is None:
        tau2_hosp = 0.0

    pi2_3 = float(np.pi ** 2 / 3)
    total = pi2_3 + tau2_ward + tau2_hosp + tau2_site
    return {
        "patient_variance": pi2_3, "ward_variance": tau2_ward,
        "hospital_variance": tau2_hosp, "site_variance": tau2_site,
        "total_variance": total,
        "pct_patient":  pi2_3     / total * 100,
        "pct_ward":     tau2_ward / total * 100,
        "pct_hospital": tau2_hosp / total * 100,
        "pct_site":     tau2_site / total * 100,
    }


def plot_variance_decomposition(decomp: dict, out_dir: Path, cohort_label: str, site: str) -> None:
    if not decomp:
        return
    labels = ["Patient", "Ward / ICU type", "Hospital", "Site (CLIF data pool)"]
    pcts   = [decomp["pct_patient"], decomp["pct_ward"], decomp["pct_hospital"], decomp["pct_site"]]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    left = 0.0
    bars = []
    for label, pct, col in zip(labels, pcts, colors):
        b = ax.barh(0, pct, left=left, color=col, edgecolor="white", height=0.6,
                    label=f"{label}  ({pct:.1f}%)")
        bars.append(b)
        if pct >= 12:
            ax.text(left + pct / 2, 0, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_ylim(-1.1, 0.6)
    ax.set_xlabel("% of total variance", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=8, frameon=False)
    ax.set_title(
        f"Variance decomposition — {site}  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]\n"
        f"Patient-specific factors accounted for {decomp['pct_patient']:.1f}% of variation; "
        f"{decomp['pct_ward']:.1f}%, {decomp['pct_hospital']:.1f}%, and {decomp['pct_site']:.1f}% "
        f"were attributed to ward/ICU type, hospital, and CLIF site respectively.",
        fontsize=9, fontweight="bold", wrap=True,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"variance_decomposition_{cohort_label}_{site}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_group_intercepts(
    group_result: dict, out_dir: Path, cohort_label: str, site: str, label: str
) -> None:
    """Forest plot of each ward/ICU-type or hospital's fixed-effects intercept
    ± 95% CI (the DL-pooling input for this site's ward-/hospital-level
    variance) — the group-level counterpart of plot_approach1's site-level
    intercept comparison."""
    group_models = group_result.get("group_models", {})
    if len(group_models) < 2:
        return

    groups = list(group_models.keys())
    alphas   = [group_models[g]["coefficients"]["intercept"]["beta"] for g in groups]
    alpha_se = [group_models[g]["coefficients"]["intercept"]["se"]   for g in groups]
    ns       = [group_models[g]["n_patients"] for g in groups]

    order = np.argsort(alphas)
    groups, alphas, alpha_se, ns = (
        [groups[i] for i in order], [alphas[i] for i in order],
        [alpha_se[i] for i in order], [ns[i] for i in order],
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(groups) * 0.5 + 1.5)))
    y_pos = np.arange(len(groups))
    for yi, (alpha, se) in enumerate(zip(alphas, alpha_se)):
        ax.errorbar(alpha, yi, xerr=1.96 * se, fmt="D", color="#4e79a7",
                    capsize=5, markersize=7, linewidth=2, zorder=3)
    ax.axvline(0, color="lightgrey", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{g}  (n={n:,})" for g, n in zip(groups, ns)], fontsize=9)
    ax.set_xlabel("Logit intercept ± 95% CI", fontsize=9)

    pooled = group_result.get("tau2")
    subtitle = f"τ²={pooled:.3f}" if pooled is not None else "τ² not estimable"
    ax.set_title(
        f"{label.capitalize()}-level intercept comparison — {site}  "
        f"[{_COHORT_LABELS.get(cohort_label, cohort_label)}]  ({subtitle})",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_intercepts_{cohort_label}_{site}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Interpretable coefficient / group-intercept tables ───────────────────────
def export_coefficient_table(model: dict) -> pd.DataFrame:
    """One row per fitted-model term: beta, SE, OR, 95% CI, p-value."""
    from scipy.stats import norm

    rows = []
    for term, c in model["coefficients"].items():
        beta, se = c["beta"], c["se"]
        z = beta / se if se else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append({
            "term": term, "beta": beta, "se": se, "or": c["or"],
            "or_ci_lo95": float(np.exp(beta - 1.96 * se)),
            "or_ci_hi95": float(np.exp(beta + 1.96 * se)),
            "p_value": p,
        })
    return pd.DataFrame(rows)


def export_group_intercept_table(group_result: dict) -> pd.DataFrame:
    """One row per hospital / ICU-ward-type: fixed-effects intercept (the
    input to DL pooling), SE, n_patients, n_vaso_on — persists what
    fit_grouped_variance already prints to console."""
    rows = []
    for grp, m in group_result.get("group_models", {}).items():
        c = m["coefficients"]["intercept"]
        rows.append({
            "group": grp, "alpha": c["beta"], "se": c["se"],
            "n_patients": m["n_patients"], "n_vaso_on": m["n_vaso_on"],
        })
    return pd.DataFrame(rows)


# ── Site discovery & main logic ───────────────────────────────────────────────
def _discover_sites(output_root: Path) -> list:
    return sorted(
        p.name.replace("patient_level_data_", "")
        for p in (output_root / "output").glob("patient_level_data_*")
        if p.is_dir()
    )


def run_for_cohort(cohort_label: str):
    cohort_path = PATIENT_LEVEL_DIR / f"cohort_{cohort_label}.parquet"
    feat_path   = PATIENT_LEVEL_DIR / "features.parquet"

    if not cohort_path.exists():
        print(f"  Cohort file not found: {cohort_path}"); return
    if not feat_path.exists():
        print(f"  Features file not found: {feat_path}"); return

    per_site_out = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / cohort_label
    per_site_out.mkdir(parents=True, exist_ok=True)
    cross_out = OUTPUT_ROOT / "output" / "cross_site_variation" / cohort_label
    cross_out.mkdir(parents=True, exist_ok=True)

    all_sites = _discover_sites(OUTPUT_ROOT)
    print(f"  Available sites: {all_sites}")

    # ── Fit per-site models for all sites with data ────────────────────────
    site_models:   dict = {}
    ph_by_site:    dict = {}
    cohort_by_site: dict = {}

    for site in all_sites:
        site_coh_p  = OUTPUT_ROOT / "output" / f"patient_level_data_{site}" / f"cohort_{cohort_label}.parquet"
        site_feat_p = OUTPUT_ROOT / "output" / f"patient_level_data_{site}" / "features.parquet"
        if not site_coh_p.exists() or not site_feat_p.exists():
            continue
        print(f"\n  [{site}] Fitting logistic GLM...")
        try:
            site_cohort   = pd.read_parquet(site_coh_p)
            site_features = pd.read_parquet(site_feat_p)
            site_ids      = set(site_cohort["stay_id"])
            site_features = site_features[site_features["stay_id"].isin(site_ids)].copy()
            ph = _prep_person_hours(site_cohort, site_features)
            site_models[site] = fit_site_logistic(ph, cohort_label, site)
            ph_by_site[site]  = ph
            cohort_by_site[site] = site_cohort
            m = site_models[site]
            print(f"    n={m['n_patients']:,}  vaso_on={m['n_vaso_on']:,}  "
                  f"α={m['coefficients']['intercept']['beta']:+.4f}  "
                  f"P_ref={m['ref_patient_p_t0']:.1%}")
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"    WARNING: model fit failed for {site}: {_e}")

    if not site_models:
        print("  No models fitted — skipping plots."); return

    # ── Palette ───────────────────────────────────────────────────────────
    pal = {**LOGO_PAL}
    for i, s in enumerate(site for site in site_models if site not in pal):
        pal[s] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

    # ── Approach 1: intercept comparison ──────────────────────────────────
    print("\n--- Approach 1: Intercept comparison ---")
    if len(site_models) >= 2:
        plot_approach1(site_models, cross_out, cohort_label, pal)
    else:
        print("  Skipping (need ≥2 sites for comparison).")

    # ── Approach 2A: time-varying ─────────────────────────────────────────
    print("\n--- Approach 2A: P(vaso) vs time ---")
    plot_approach2_time(site_models, cross_out, cohort_label, pal)

    # ── Approach 2B: NEE-varying ──────────────────────────────────────────
    print("\n--- Approach 2B: P(vaso) vs NEE ---")
    plot_approach2_nee(site_models, ph_by_site, cross_out, cohort_label, pal)

    # ── Mixed effects pooling ─────────────────────────────────────────────
    print("\n--- Mixed-effects pooling ---")
    me_result: dict = {}
    if len(site_models) >= 2:
        me_result = fit_and_plot_mixed_effects(
            site_models, ph_by_site, cross_out, cohort_label, pal
        ) or {}
    else:
        print("  Skipping: only 1 site available.")

    # ── Approach 3: ward/ICU-type + hospital variance + 4-level decomposition ──
    print("\n--- Approach 3: Ward/ICU-type + hospital variance + 4-level decomposition ---")
    ward_results:     dict = {}
    hospital_results: dict = {}
    variance_decomp:  dict = {}
    if me_result.get("dl_tau2") is not None:
        for site in site_models:
            print(f"\n  [{site}] ward/ICU type:")
            ward_results[site] = fit_ward_level_variance(
                cohort_by_site[site], ph_by_site[site], cohort_label, site
            )
            print(f"  [{site}] hospital:")
            hospital_results[site] = fit_hospital_level_variance(
                cohort_by_site[site], ph_by_site[site], cohort_label, site
            )
            decomp = compute_variance_decomposition(
                ward_results[site], hospital_results[site], me_result
            )
            if decomp:
                variance_decomp[site] = decomp
                print(f"    Patient-specific factors accounted for {decomp['pct_patient']:.1f}% of "
                      f"variation; {decomp['pct_ward']:.1f}%, {decomp['pct_hospital']:.1f}%, and "
                      f"{decomp['pct_site']:.1f}% were attributed to ward/ICU type, hospital, and "
                      f"CLIF site respectively.")
                plot_variance_decomposition(decomp, cross_out, cohort_label, site)
            plot_group_intercepts(ward_results[site], cross_out, cohort_label, site, "ward")
            plot_group_intercepts(hospital_results[site], cross_out, cohort_label, site, "hospital")
    else:
        print("  Skipping: need >=2 sites pooled for site-level τ² (mixed effects).")

    # ── Save variation packets ────────────────────────────────────────────
    # One packet per site, each containing all site models + mixed effects.
    # This lets the HTML report read any one packet and get all site data.
    packet = {
        "site":             SITE_NAME,
        "cohort":           cohort_label,
        "per_site_models":  site_models,          # all sites
        "per_site_model":   site_models.get(SITE_NAME, {}),  # back-compat
        "mixed_effects":    me_result,
        "ward_level":             ward_results,      # per site: group_models + tau2/icc/mor (ICU type)
        "hospital_level":         hospital_results,   # per site: group_models + tau2/icc/mor (hospital_id)
        "variance_decomposition": variance_decomp,   # per site: patient/ward/hospital/site % of total variance
    }

    # Write into each site's upload directory so the report can find it
    # regardless of which site's config was active when the script ran.
    for s in site_models:
        s_out = OUTPUT_ROOT / "output" / f"upload_to_box_{s}" / cohort_label
        s_out.mkdir(parents=True, exist_ok=True)
        p = s_out / f"site_variation_packet_{cohort_label}_{s}.json"
        pkt_s = dict(packet, site=s, per_site_model=site_models[s])
        with open(p, "w", encoding="utf-8") as fout:
            json.dump(pkt_s, fout, indent=2)

        export_coefficient_table(site_models[s]).to_csv(
            s_out / f"coefficient_table_{cohort_label}_{s}.csv", index=False
        )
        if ward_results.get(s, {}).get("group_models"):
            export_group_intercept_table(ward_results[s]).to_csv(
                s_out / f"ward_intercepts_{cohort_label}_{s}.csv", index=False
            )
        if hospital_results.get(s, {}).get("group_models"):
            export_group_intercept_table(hospital_results[s]).to_csv(
                s_out / f"hospital_intercepts_{cohort_label}_{s}.csv", index=False
            )

    pkt_path = per_site_out / f"site_variation_packet_{cohort_label}_{SITE_NAME}.json"
    print(f"\nSaved packet:  {pkt_path}  (+ copies for {list(site_models.keys())})")
    print(f"Cross-site plots: {cross_out}")


def main():
    cohorts = ["sepsis3", "rhee"] if COHORT_ARG == "both" else [COHORT_ARG]
    for c in cohorts:
        print("\n" + "=" * 60)
        print(f"COHORT: {c.upper()}  |  SITE: {SITE_NAME}")
        print("=" * 60)
        try:
            run_for_cohort(c)
        except Exception as _e:
            import traceback
            print(f"ERROR running {c}: {_e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
