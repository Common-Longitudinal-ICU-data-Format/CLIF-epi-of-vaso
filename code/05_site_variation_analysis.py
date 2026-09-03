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

Approach 3 — Ward/ICU-type + hospital variance, 4-level variance attribution
(this site only):
  ONE expanded logistic GLM per site using site-wide centering (globally
  comparable intercepts) with ward/ICU-type and hospital fixed-effect dummies.
  Effective intercepts computed via delta method from the full covariance matrix.
  DL-pools effective intercepts -> tau2_ward / tau2_hosp for this site.
  Combined with the cross-site tau2 (mixed_effects.dl_tau2) from Approach 1 and
  the logistic latent residual (pi^2/3 ≈ 3.29), Nakagawa & Schielzeth R²
  decomposition gives: covariate groups + ward + hospital + site + unexplained = 100%.

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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Global ICU/hospital-type color map ────────────────────────────────────────
# Fixed tab10 hex values so the same type string always gets the same color
# regardless of which site's data is being plotted (avoids sorted-index shifts).
# tab10: 0=#1f77b4, 1=#ff7f0e, 2=#2ca02c, 3=#d62728, 4=#9467bd,
#        5=#8c564b, 6=#e377c2, 7=#7f7f7f, 8=#bcbd22, 9=#17becf
_GROUP_TYPE_COLOR_MAP: dict = {
    # ICU types
    "MICU":              "#1f77b4",
    "CICU":              "#ff7f0e",
    "SICU":              "#2ca02c",
    "CT ICU":            "#d62728",
    "Neuro ICU":         "#9467bd",
    "Burn ICU":          "#8c564b",
    "Mixed/General ICU": "#e377c2",
    "ICU (unspecified)": "#7f7f7f",
    "Other ICU":         "#bcbd22",
    "Ward":              "#17becf",
    "Step-down / IMC":   "#b07aa1",
    "ED / Emergency":    "#e15759",
    "OR / Procedural":   "#76b7b2",
    "Other":             "#aaaaaa",
    # Hospital types
    "academic":          "#2c5f8a",
    "community":         "#e07b39",
    "Academic":          "#2c5f8a",
    "Community":         "#e07b39",
}

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--site",   default=None)
_ap.add_argument("--cohort", default="both", choices=["sepsis3", "rhee", "rhee_clifpy", "both"])
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

_COHORT_LABELS = {"sepsis3": "Sepsis-3 (CMS)", "rhee": "Rhee/CDC ASE", "rhee_clifpy": "Rhee/CDC ASE (clifpy)"}

# Reference patient for Approach 2 plots. SOFA components replace the
# aggregate SOFA score (see fit_site_logistic) so NEE's mechanical overlap
# with the cardiovascular SOFA axis is excluded; each reference value below
# is a normal/mild-severity default for that organ axis.
REF_AGE       = 60.0
REF_NEE       = 0.3    # mcg/kg/min
REF_TIME      = 5.0    # hours from NE start used as the fixed time in dose-varying plots

# Fixed centering for DLMM design matrix — pre-specified so all sites share
# the same column interpretation without a coordination round.
# Approximate clinical references: SOFA-normal / mild-threshold values.
_DLMM_REFS: dict = {
    "age":                    65.0,   # years
    "p_f_ratio":             200.0,   # mmHg (ARDS threshold)
    "creatinine":              1.0,   # mg/dL
    "platelet":              150.0,   # x10³/µL
    "bilirubin":               1.0,   # mg/dL
    "gcs":                    15.0,   # GCS total score (normal)
    "rrt":                     0.0,   # binary; reference = not on RRT
    "invasive_vent":           0.0,   # binary; reference = not invasively ventilated
    "nee":                     0.15,  # mcg/kg/min — approximate typical dose at vaso initiation
    "time_to_vaso_h_log1p":    1.79,  # ≈ log1p(5 h) — approximate median delay to initiation
    # Binary comorbidity / procedure covariates — all centered at 0 (no comorbidity = reference)
    "comorbid_cirrhosis":       0.0,
    "comorbid_liver_nocirrh":   0.0,
    "comorbid_cad":             0.0,
    "comorbid_aortic_stenosis": 0.0,
    "comorbid_carotid_stenosis":0.0,
}
_DLMM_AGE_REF = _DLMM_REFS["age"]  # back-compat alias
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

# Binary baseline covariates used as fixed effects in the GLM / GLMM / DLMM.
# All have reference value 0 (no comorbidity / no procedure).
# NOTE: comorbid_liver_disease is deliberately excluded here — it is collinear with
# comorbid_cirrhosis.  comorbid_liver_nocirrh (non-cirrhotic liver disease) is used
# instead to form a mutually exclusive pair.
# Must stay in sync with _BIN_COLS_MODELS in 08_cross_site_variation_analysis.py.
_BIN_COLS_MODELS = [
    "comorbid_cirrhosis",
    "comorbid_liver_nocirrh",
    "comorbid_cad",
    "comorbid_aortic_stenosis",
    "comorbid_carotid_stenosis",
]
# Drop a binary covariate from a site's model if fewer than this fraction of
# that site's patients carry it — avoids separation / inflated SEs.
_MIN_BIN_PREVALENCE = 0.01

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

# Fallback labels for location_category values (when location_type is absent).
_LOC_CAT_CANON = {
    "icu":           "ICU (unspecified)",
    "ward":          "Ward",
    "ed":            "ED / Emergency",
    "emergency":     "ED / Emergency",
    "or":            "OR / Procedural",
    "operating_room": "OR / Procedural",
    "procedure_room": "OR / Procedural",
    "procedural":    "OR / Procedural",
    "surgery":       "OR / Procedural",
    "pacu":          "PACU",
    "stepdown":      "Step-down / IMC",
    "step_down":     "Step-down / IMC",
    "intermediate":  "Step-down / IMC",
    "imc":           "Step-down / IMC",
    "other":         "Other",
}


def _effective_location(
    cohort: pd.DataFrame,
    type_col: str = "location_type",
    cat_col: str  = "location_category",
) -> pd.Series:
    """Per-row effective location: location_type when non-null/non-blank,
    else location_category as fallback."""
    if type_col not in cohort.columns:
        return cohort.get(cat_col, pd.Series(dtype=object, index=cohort.index))
    raw = cohort[type_col].copy().astype(object)
    null_mask = raw.isna() | raw.astype(str).str.lower().str.strip().isin(["none", "nan", ""])
    if cat_col in cohort.columns:
        raw[null_mask] = cohort.loc[null_mask, cat_col]
    return raw


def _canon_icu_series(raw: pd.Series) -> pd.Series:
    """Canonical location label per row.

    ICU subtypes (from location_type) → abbreviated names via _ICU_CANON.
    Broad location categories (from location_category fallback) → readable
    labels via _LOC_CAT_CANON.  Genuinely missing → NaN (dropped from
    ward-level analysis rather than lumped into "Other").
    """
    low = raw.astype(str).str.lower().str.strip()
    is_missing = raw.isna() | low.isin(["none", "nan", ""])
    canon = low.map(_ICU_CANON)
    # For values not in _ICU_CANON, try the broader category map; fall back "Other ICU"
    unmapped = canon.isna() & ~is_missing
    canon[unmapped] = low[unmapped].map(_LOC_CAT_CANON).fillna("Other ICU")
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

    # Merge static baseline features (age + binary comorbidities)
    _static = ["stay_id", "age"] + [c for c in _BIN_COLS_MODELS if c in cohort.columns]
    ph = ph.merge(cohort[_static], on="stay_id", how="left")
    # Binary cols default to 0 when absent (e.g. procedures file missing at a site)
    for c in _BIN_COLS_MODELS:
        if c in ph.columns:
            ph[c] = ph[c].fillna(0).astype(int)

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
    present_bin_cols: list = None,
    candidate_bin_cols: list = None,  # if None, uses _BIN_COLS_MODELS
    extra_cont_cols: list = None,     # additional continuous covariates to standardise & include
):
    """vaso_on ~ intercept + age_c + p_f_ratio_c + device_category + creatinine_c
    + rrt + platelet_c + bilirubin_c + gcs_c
    + [binary comorbidities: cirrhosis, liver_nocirrh, cad, aortic_stenosis, carotid_stenosis]
    + rcs(NEE,4) + rcs(time,4).

    SOFA components replace the aggregate SOFA score so NEE's mechanical
    overlap with the cardiovascular SOFA axis is excluded; cardiovascular
    itself is intentionally omitted (captured by NEE on the RHS instead).

    Binary comorbidity covariates are not z-scored (reference = 0); any binary
    covariate whose patient-level prevalence is below _MIN_BIN_PREVALENCE is
    dropped from that site's model to avoid quasi-separation.

    extra_cont_cols: list of additional continuous column names already present in df;
    they are standardised (mean 0, sd 1) and appended after the SOFA components.

    knots/means/sds/device_categories/present_bin_cols can be passed in
    (pooled/refit reusing one shared structure) or left None to derive from `df`.
    """
    _extra = list(extra_cont_cols) if extra_cont_cols else []
    _all_cont = _CONT_COLS + _extra

    if nee_knots is None:
        nee_knots = _rcs_knots(df["nee"].values, df=4)
    if time_knots is None:
        time_knots = _rcs_knots(df["time_hour"].values, df=4)
    nee_b  = _rcs_basis(df["nee"].values, nee_knots)
    time_b = _rcs_basis(df["time_hour"].values, time_knots)

    if means is None or sds is None:
        means, sds = {}, {}
        for c in _all_cont:
            means[c] = float(df[c].mean())
            sds[c]   = float(df[c].std()) or 1.0
    else:
        # Compute means/sds for any extra cols not already stored
        for c in _extra:
            if c not in means:
                means[c] = float(df[c].mean())
                sds[c]   = float(df[c].std()) or 1.0
    centered = {c: (df[c].values - means[c]) / sds[c] for c in _all_cont}

    if device_categories is None:
        device_categories = sorted(set(df["device_category"].unique()) - {_DEVICE_REF})
    device_cols = {d: (df["device_category"].values == d).astype(np.float64)
                   for d in device_categories}

    # Binary comorbidity covariates — prevalence-checked, not z-scored
    if present_bin_cols is None:
        _cand_bin = candidate_bin_cols if candidate_bin_cols is not None else _BIN_COLS_MODELS
        present_bin_cols = []
        for c in _cand_bin:
            if c not in df.columns:
                continue
            # Prevalence at patient level (binary cols are constant within a stay_id)
            if "stay_id" in df.columns:
                prev = (df.groupby("stay_id")[c].max() == 1).mean()
            else:
                prev = float((df[c] == 1).mean())
            if prev >= _MIN_BIN_PREVALENCE:
                present_bin_cols.append(c)
            else:
                print(f"    [{c}]: prevalence {prev:.1%} < {_MIN_BIN_PREVALENCE:.0%}"
                      f" — excluded from model")

    nee_cols  = [f"rcs_nee_{i}"  for i in range(nee_b.shape[1])]
    time_cols = [f"rcs_time_{i}" for i in range(time_b.shape[1])]
    col_names = (["intercept"] + [f"{c}_c" for c in _all_cont] + ["rrt"]
                 + present_bin_cols
                 + [f"device_{d}" for d in device_categories] + nee_cols + time_cols)

    X = np.column_stack(
        [np.ones(len(df))]
        + [centered[c] for c in _all_cont]
        + [df["rrt"].values.astype(np.float64)]
        + [df[c].values.astype(np.float64) for c in present_bin_cols]
        + [device_cols[d] for d in device_categories]
        + [nee_b, time_b]
    ).astype(np.float64)

    return X, col_names, means, sds, nee_knots, time_knots, device_categories, present_bin_cols


# ── Approach 1 & 2: Per-site logistic GLM ─────────────────────────────────────
def fit_site_logistic(
    ph: pd.DataFrame,
    cohort_label: str,
    site: str,
    bin_cols: list = None,
    extra_cont_cols: list = None,
) -> dict:
    """Fit the shared logistic spec (see _build_design_matrix) independently
    on this group's person-hours (its own knots/scaling/device categories).

    bin_cols:        list of binary covariate names to include; defaults to _BIN_COLS_MODELS.
    extra_cont_cols: list of additional continuous covariate names (already in ph);
                     they are standardised and included alongside the base _CONT_COLS.
    """
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as fam

    _cand  = bin_cols if bin_cols is not None else _BIN_COLS_MODELS
    _extra = list(extra_cont_cols) if extra_cont_cols else []
    _bin_present = [c for c in _cand if c in ph.columns]
    _extra_present = [c for c in _extra if c in ph.columns]
    df = ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt", "device_category"]
            + _CONT_COLS + _extra_present + _bin_present].dropna().copy()

    X, col_names, means, sds, nee_knots, time_knots, device_categories, present_bin_cols = (
        _build_design_matrix(df, candidate_bin_cols=_cand, extra_cont_cols=_extra_present)
    )
    Y = df["vaso_on"].values.astype(np.float64)

    fit = GLM(Y, X, family=fam.Binomial()).fit(maxiter=200)

    coef_out = {
        name: {"beta": float(fit.params[i]), "se": float(fit.bse[i]),
               "or":   float(np.exp(fit.params[i]))}
        for i, name in enumerate(col_names)
    }

    result = {
        "site":              site,
        "cohort":            cohort_label,
        "n_patients":        int(df["stay_id"].nunique()),
        "n_ph_rows":         int(len(df)),
        "n_vaso_on":         int(Y.sum()),
        "means":             means,
        "sds":               sds,
        "device_categories": device_categories,
        "present_bin_cols":  present_bin_cols,
        "extra_cont_cols":   _extra_present,
        "nee_knots":         nee_knots.tolist(),
        "time_knots":        time_knots.tolist(),
        "coefficients":      coef_out,
    }
    result["ref_patient_p_t5"] = float(predict_p(result, REF_AGE, REF_NEE, REF_TIME).item())
    return result


def predict_p(
    model: dict, ages, nees, times,
    p_f_ratio=REF_PF, device_category=REF_DEVICE, creatinine=REF_CREATININE,
    rrt=REF_RRT, platelet=REF_PLATELET, bilirubin=REF_BILIRUBIN, gcs=REF_GCS,
    bin_vals: dict = None,
) -> np.ndarray:
    """Vectorized P(vaso_on) prediction from a fitted model dict. Component
    axes not being varied by the caller stay at their reference values.

    bin_vals: optional dict mapping binary covariate name → scalar (0 or 1).
              Defaults to 0 for each binary covariate (reference patient:
              no comorbidities).
    """
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
    # Binary comorbidity terms (reference = 0; caller may pass specific values)
    _bv = bin_vals or {}
    for c in model.get("present_bin_cols", []):
        if c in coef:
            lp += coef[c]["beta"] * float(_bv.get(c, 0.0))
    for d in model.get("device_categories", []):
        col = f"device_{d}"
        if col in coef:
            lp += coef[col]["beta"] * (device_arr == d).astype(float)
    for i in range(nee_b.shape[1]):
        lp += coef[f"rcs_nee_{i}"]["beta"]  * nee_b[:, i]
    for i in range(time_b.shape[1]):
        lp += coef[f"rcs_time_{i}"]["beta"] * time_b[:, i]

    return 1.0 / (1.0 + np.exp(-lp))


def predict_p_structural(
    model: dict, ages, nees, times,
    ward_type: str = None,
    hospital_id: str = None,
    p_f_ratio=REF_PF, device_category=REF_DEVICE,
    creatinine=REF_CREATININE, rrt=REF_RRT,
    platelet=REF_PLATELET, bilirubin=REF_BILIRUBIN, gcs=REF_GCS,
) -> np.ndarray:
    """P(vaso_on) from the structural (expanded) model.

    Identical to predict_p but also activates the appropriate ward / hospital
    dummy.  ward_type=None → reference ward (all ward dummies = 0).
    hospital_id=None → reference hospital.  Coefficients not present in the
    model dict (e.g. ward dummies for a single-ward site) are silently skipped.
    """
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
        lp += coef[f"{c}_c"]["beta"] * (arrs[c] - means[c]) / sds[c]
    lp += coef["rrt"]["beta"] * arrs["rrt"]
    for d in model.get("device_categories", []):
        col = f"device_{d}"
        if col in coef:
            lp += coef[col]["beta"] * (device_arr == d).astype(float)
    for i in range(nee_b.shape[1]):
        lp += coef[f"rcs_nee_{i}"]["beta"] * nee_b[:, i]
    for i in range(time_b.shape[1]):
        lp += coef[f"rcs_time_{i}"]["beta"] * time_b[:, i]
    if ward_type is not None:
        wd_col = f"ward_{ward_type}"
        if wd_col in coef:
            lp += coef[wd_col]["beta"]
    if hospital_id is not None:
        hd_col = f"hosp_{hospital_id}"
        if hd_col in coef:
            lp += coef[hd_col]["beta"]

    return 1.0 / (1.0 + np.exp(-lp))


# ── Plot: Approach 1 — reference-patient P and intercept forest ───────────────
def plot_approach1(site_models: dict, out_dir: Path, cohort_label: str, pal: dict):
    sites    = list(site_models.keys())
    colors   = [pal.get(s, "#888888") for s in sites]
    ref_p    = [site_models[s]["ref_patient_p_t5"] for s in sites]
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
        f"NEE={REF_NEE} mcg/kg/min, t={REF_TIME:.0f} h)", fontsize=9
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
    ax2.set_xlabel(f"Logit intercept ± 95% CI\n(components/age at means, NEE=0, t={REF_TIME:.0f} h)", fontsize=9)

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
    time_grid = np.linspace(5, t_max, 300)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s, m in site_models.items():
        p_t = predict_p(m, REF_AGE, REF_NEE, time_grid)
        ax.plot(time_grid, p_t, color=pal.get(s, "#888888"), linewidth=2.5, label=s)

    ax.set_xlabel("Hours from NE start", fontsize=11)
    ax.set_ylabel("P(vasopressin on | hour)", fontsize=11)
    ax.set_title(
        f"Approach 2A — Time-varying P(vasopressin)\n"
        f"Reference patient: age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, NEE={REF_NEE} mcg/kg/min  "
        f"[{_COHORT_LABELS.get(cohort_label, cohort_label)}]",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_xlim(5, t_max); ax.set_ylim(0, 1)
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
    ref_time: float = REF_TIME,
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
        _bin_cols_s = [c for c in _BIN_COLS_MODELS if c in ph_by_site[s].columns]
        ph = ph_by_site[s][["stay_id", "time_hour", "vaso_on", "nee", "rrt",
                             "device_category"] + _CONT_COLS + _bin_cols_s].copy()
        ph["site"] = s
        frames.append(ph)
    all_ph = pd.concat(frames, ignore_index=True).dropna(
        subset=["nee", "rrt", "device_category", "vaso_on"] + _CONT_COLS
    )

    X_base, base_col_names, _means, _sds, _nee_kp, _time_kp, _dev_cats, _pbc = (
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
    ax.set_xlabel(f"Logit intercept ± 95% CI  (components/age at means, NEE=0, t={REF_TIME:.0f} h)", fontsize=10)

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


# ── Approach 3: expanded model with ward/ICU-type + hospital fixed effects ──────
def _build_cohort_structural_df(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per-stay structural lookup: canonical _ward and hospital_id.

    Used by fit_site_logistic_with_structure to join structural grouping
    columns onto person-hours by stay_id.
    """
    c = cohort.copy()
    c["_ward"] = _canon_icu_series(_effective_location(c))
    cols = ["stay_id", "_ward"]
    if "hospital_id" in c.columns:
        cols.append("hospital_id")
    return c[cols].drop_duplicates("stay_id")


def fit_site_logistic_with_structure(
    ph: pd.DataFrame,
    cohort: pd.DataFrame,
    cohort_label: str,
    site: str,
) -> dict:
    """Expanded logistic GLM: shared spec + ward/ICU-type dummies + hospital
    dummies, fit on this site's person-hours (Approach 3).

    Uses site-wide (global) means/sds for centering so per-ward effective
    intercepts are all evaluated at the same reference patient — making them
    directly comparable without the reference-shift bias of per-group refits.

    Effective intercept for ward j:
        α_j = β_intercept + β_{ward_j}
        SE_j = √(Var(β₀) + Var(β_j) + 2·Cov(β₀, β_j))

    Reference level: highest-N ward / hospital (most stable baseline).
    Groups below MIN_ICU_N_PATIENTS / MIN_HOSPITAL_N_PATIENTS are excluded
    from the dummy set; their person-hours remain in the fit and are absorbed
    into the reference intercept.

    Also DL-pools the effective intercepts to produce tau2_ward / tau2_hosp —
    equivalent to the old separate-fit DL approach but statistically sounder.
    """
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as fam

    struct_df = _build_cohort_structural_df(cohort)

    df = ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt", "device_category"]
            + _CONT_COLS].dropna().copy()

    # Site-wide means/sds — consistent centering across all groups
    means_g, sds_g = {}, {}
    for c_col in _CONT_COLS:
        means_g[c_col] = float(df[c_col].mean())
        sds_g[c_col]   = float(df[c_col].std()) or 1.0

    X_base, col_names_base, _, _, nee_knots, time_knots, device_categories, present_bin_cols_s = (
        _build_design_matrix(df, means=means_g, sds=sds_g)
    )

    # Merge structural labels (one assignment per stay_id)
    df = df.merge(struct_df, on="stay_id", how="left")

    # ── Ward dummies ───────────────────────────────────────────────────────
    ward_pat_n: dict = {}
    if "_ward" in df.columns:
        ward_pat_n = (
            df.dropna(subset=["_ward"])
            .groupby("_ward")["stay_id"].nunique()
            .sort_values(ascending=False)
            .to_dict()
        )
    ward_cats   = [w for w, n in ward_pat_n.items() if n >= MIN_ICU_N_PATIENTS]
    ward_ref    = ward_cats[0] if ward_cats else None
    ward_levels = [w for w in ward_cats if w != ward_ref]

    # ── Hospital dummies ───────────────────────────────────────────────────
    hosp_pat_n: dict = {}
    if "hospital_id" in df.columns:
        hosp_pat_n = (
            df.dropna(subset=["hospital_id"])
            .groupby("hospital_id")["stay_id"].nunique()
            .sort_values(ascending=False)
            .to_dict()
        )
    hosp_cats   = [h for h, n in hosp_pat_n.items() if n >= MIN_HOSPITAL_N_PATIENTS]
    hosp_ref    = hosp_cats[0] if len(hosp_cats) >= 2 else None
    hosp_levels = [h for h in hosp_cats if h != hosp_ref] if hosp_ref else []

    # ── Build expanded X ───────────────────────────────────────────────────
    ward_col_names = [f"ward_{w}" for w in ward_levels]
    hosp_col_names = [f"hosp_{h}" for h in hosp_levels]
    col_names      = col_names_base + ward_col_names + hosp_col_names

    parts = [X_base]
    if ward_levels:
        wv = df["_ward"].values
        parts.append(np.column_stack(
            [(wv == w).astype(np.float64) for w in ward_levels]
        ))
    if hosp_levels:
        hv = df["hospital_id"].values
        parts.append(np.column_stack(
            [(hv == h).astype(np.float64) for h in hosp_levels]
        ))
    X_full = np.column_stack(parts) if len(parts) > 1 else X_base
    Y      = df["vaso_on"].values.astype(np.float64)

    fit = GLM(Y, X_full, family=fam.Binomial()).fit(maxiter=200)
    cov = fit.cov_params()

    coef_out = {
        name: {"beta": float(fit.params[i]), "se": float(fit.bse[i]),
               "or":   float(np.exp(fit.params[i]))}
        for i, name in enumerate(col_names)
    }

    i_int    = col_names.index("intercept")
    ward_npt = df.groupby("_ward")["stay_id"].nunique().to_dict() if "_ward" in df.columns else {}
    hosp_npt = df.groupby("hospital_id")["stay_id"].nunique().to_dict() if "hospital_id" in df.columns else {}

    # ── Effective intercepts per ward ──────────────────────────────────────
    ward_ei: dict = {}
    if ward_ref:
        ward_ei[ward_ref] = {
            "alpha": float(fit.params[i_int]),
            "se":    float(fit.bse[i_int]),
            "n_patients": int(ward_npt.get(ward_ref, 0)),
            "is_reference": True, "stable": True,
        }
        print(f"    ward [{ward_ref}] n={ward_npt.get(ward_ref,0):,}  "
              f"α={fit.params[i_int]:+.4f} (SE={fit.bse[i_int]:.3f})  [reference]")
    for w in ward_levels:
        i_w    = col_names.index(f"ward_{w}")
        alpha  = float(fit.params[i_int] + fit.params[i_w])
        se     = float(np.sqrt(max(0.0, cov[i_int, i_int] + cov[i_w, i_w] + 2 * cov[i_int, i_w])))
        stable = bool(se < MAX_STABLE_INTERCEPT_SE)
        flag   = "" if stable else "  [UNSTABLE — likely separation]"
        print(f"    ward [{w}] n={ward_npt.get(w,0):,}  α={alpha:+.4f} (SE={se:.3f}){flag}")
        ward_ei[w] = {"alpha": alpha, "se": se,
                      "n_patients": int(ward_npt.get(w, 0)),
                      "is_reference": False, "stable": stable}

    # ── Effective intercepts per hospital ──────────────────────────────────
    hosp_ei: dict = {}
    if hosp_ref is not None:
        hosp_ei[str(hosp_ref)] = {
            "alpha": float(fit.params[i_int]),
            "se":    float(fit.bse[i_int]),
            "n_patients": int(hosp_npt.get(hosp_ref, 0)),
            "is_reference": True, "stable": True,
        }
    for h in hosp_levels:
        i_h   = col_names.index(f"hosp_{h}")
        alpha = float(fit.params[i_int] + fit.params[i_h])
        se    = float(np.sqrt(max(0.0, cov[i_int, i_int] + cov[i_h, i_h] + 2 * cov[i_int, i_h])))
        hosp_ei[str(h)] = {"alpha": alpha, "se": se,
                           "n_patients": int(hosp_npt.get(h, 0)),
                           "is_reference": False, "stable": bool(se < MAX_STABLE_INTERCEPT_SE)}

    result: dict = {
        "site": site, "cohort": cohort_label,
        "n_patients":  int(df["stay_id"].nunique()),
        "n_ph_rows":   int(len(df)),
        "n_vaso_on":   int(Y.sum()),
        "means": means_g, "sds": sds_g,
        "device_categories": device_categories,
        "nee_knots":   nee_knots.tolist(),
        "time_knots":  time_knots.tolist(),
        "ward_levels": ward_levels,
        "ward_ref":    ward_ref,
        "hosp_levels": [str(h) for h in hosp_levels],
        "hosp_ref":    str(hosp_ref) if hosp_ref is not None else None,
        "coefficients": coef_out,
        "ward_effective_intercepts":     ward_ei,
        "hospital_effective_intercepts": hosp_ei,
    }

    # DL-pool stable effective intercepts → tau2_ward / tau2_hosp
    stable_w = {g: v for g, v in ward_ei.items() if v.get("stable", True) and v["se"] > 0}
    if len(stable_w) >= 2:
        dl_w = _dl_pool(
            np.array([v["alpha"] for v in stable_w.values()]),
            np.array([v["se"]    for v in stable_w.values()]) ** 2,
        )
        result.update({"tau2_ward": dl_w["tau2"], "icc_ward": dl_w["icc"], "mor_ward": dl_w["mor"]})
        print(f"  Ward-level DL: τ²={dl_w['tau2']:.4f}  ICC={dl_w['icc']:.3f}  "
              f"MOR={dl_w['mor']:.3f}  (k={dl_w['k']} wards)")
    else:
        result.update({"tau2_ward": None, "icc_ward": None, "mor_ward": None})
        print("  Ward-level DL: <2 stable wards — τ² not estimable")

    stable_h = {g: v for g, v in hosp_ei.items() if v.get("stable", True) and v["se"] > 0}
    if len(stable_h) >= 2:
        dl_h = _dl_pool(
            np.array([v["alpha"] for v in stable_h.values()]),
            np.array([v["se"]    for v in stable_h.values()]) ** 2,
        )
        result.update({"tau2_hosp": dl_h["tau2"], "icc_hosp": dl_h["icc"], "mor_hosp": dl_h["mor"]})
        print(f"  Hospital-level DL: τ²={dl_h['tau2']:.4f}  ICC={dl_h['icc']:.3f}  "
              f"MOR={dl_h['mor']:.3f}  (k={dl_h['k']} hospitals)")
    else:
        result.update({"tau2_hosp": None, "icc_hosp": None, "mor_hosp": None})

    return result


def plot_variance_decomposition(decomp: dict, out_dir: Path, cohort_label: str, site: str) -> None:
    if not decomp:
        return
    labels = ["Patient", "Ward / ICU type", "Hospital", "Site"]
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
        f"were attributed to ward/ICU type, hospital, and site respectively.",
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


# ── Covariate-level variance decomposition (Nakagawa & Schielzeth 2013) ───────

# Maps group label -> function: col_name -> bool (True = belongs to this group)
_COVAR_GROUPS: dict = {
    "Age":             lambda c: c == "age_c",
    "P/F ratio":       lambda c: c == "p_f_ratio_c",
    "Creatinine":      lambda c: c == "creatinine_c",
    "Platelets":       lambda c: c == "platelet_c",
    "Bilirubin":       lambda c: c == "bilirubin_c",
    "GCS":             lambda c: c == "gcs_c",
    "RRT":             lambda c: c == "rrt",
    "Device category": lambda c: c.startswith("device_"),
    "NEE dose":        lambda c: c.startswith("rcs_nee_"),
    "Time on NE":      lambda c: c.startswith("rcs_time_"),
}

_COVAR_COLORS: dict = {
    "Age":             "#a6cee3",
    "P/F ratio":       "#1f78b4",
    "Creatinine":      "#b2df8a",
    "Platelets":       "#33a02c",
    "Bilirubin":       "#fb9a99",
    "GCS":             "#e31a1c",
    "RRT":             "#fdbf6f",
    "Device category": "#ff7f00",
    "NEE dose":        "#cab2d6",
    "Time on NE":      "#6a3d9a",
    "Ward / ICU type": "#f28e2b",
    "Hospital":        "#e15759",
    "Site":            "#59a14f",
    "Unexplained":     "#cccccc",
}


def compute_covariate_variance_explained(
    ph_by_site: dict,
    site_models: dict,
    structural_models: dict,
    me_result: dict,
) -> dict:
    """Variance of each covariate's linear predictor contribution, per site.

    For predictor group g:  σ²_g = Var(β_g ᵀ X_g) over all patient-hours.

    Total latent variance = Σ σ²_g + σ²_ward + σ²_hosp + σ²_site + π²/3
    where:
      • σ²_g  = Var(β_g X_g) from the Approach 1 model (shared logistic spec)
      • σ²_ward = τ²_ward from DL pooling of Approach 3 effective intercepts
      • σ²_hosp = τ²_hosp from DL pooling of Approach 3 effective intercepts
      • σ²_site = dl_tau2 from cross-site mixed-effects pooling (Approach 1)
      • π²/3  ≈ 3.29 = logistic latent-variable residual (fixed mathematical constant)

    This gives a single decomposition across all 4 levels summing to 100%.

    R²_marginal    = Σ σ²_g  / σ²_total   (covariate fixed effects only)
    R²_conditional = (Σ σ²_g + σ²_ward + σ²_hosp + σ²_site) / σ²_total

    Reference: Nakagawa & Schielzeth (2013) Methods Ecol Evol 4:133-142.
    """
    pi2_3 = float(np.pi ** 2 / 3)
    results: dict = {}

    for site in site_models:
        if site not in ph_by_site:
            continue
        model = site_models[site]
        ph = ph_by_site[site].copy()
        _pbc = model.get("present_bin_cols", [])
        _bin_in_ph = [c for c in _pbc if c in ph.columns]
        ph = (ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt",
                   "device_category"] + _CONT_COLS + _bin_in_ph].dropna().copy())
        if ph.empty:
            continue

        try:
            X, col_names, _, _, _, _, _, _ = _build_design_matrix(
                ph,
                nee_knots=np.array(model["nee_knots"]),
                time_knots=np.array(model["time_knots"]),
                means=model["means"],
                sds=model["sds"],
                device_categories=model["device_categories"],
                present_bin_cols=_pbc,
            )
        except Exception as exc:
            print(f"  [{site}] covariate variance skipped — {exc}")
            continue

        beta_vec = np.array([model["coefficients"][c]["beta"] for c in col_names])

        group_sigma2: dict = {}
        for grp, match_fn in _COVAR_GROUPS.items():
            idxs = [i for i, c in enumerate(col_names) if match_fn(c)]
            lp = X[:, idxs] @ beta_vec[idxs] if idxs else np.zeros(len(ph))
            group_sigma2[grp] = float(np.var(lp))

        sigma2_fixed = float(sum(group_sigma2.values()))

        # Structural variance from Approach 3 DL pooling of effective intercepts
        struct_s    = structural_models.get(site, {})
        sigma2_ward = float(struct_s.get("tau2_ward") or 0.0)
        sigma2_hosp = float(struct_s.get("tau2_hosp") or 0.0)
        # Cross-site variance from Approach 1 DL pooling
        sigma2_site = float(me_result.get("dl_tau2") or 0.0)

        sigma2_rand  = sigma2_ward + sigma2_hosp + sigma2_site
        sigma2_total = sigma2_fixed + sigma2_rand + pi2_3

        def _pct(v: float) -> float:
            return v / sigma2_total * 100 if sigma2_total > 0 else 0.0

        results[site] = {
            "n_patients":    model["n_patients"],
            "group_sigma2":  group_sigma2,
            "sigma2_fixed":  sigma2_fixed,
            "sigma2_ward":   sigma2_ward,
            "sigma2_hosp":   sigma2_hosp,
            "sigma2_site":   sigma2_site,
            "sigma2_resid":  pi2_3,
            "sigma2_total":  sigma2_total,
            "r2_marginal":   sigma2_fixed / sigma2_total if sigma2_total > 0 else float("nan"),
            "r2_conditional": (sigma2_fixed + sigma2_rand) / sigma2_total if sigma2_total > 0 else float("nan"),
            "pct_groups":    {g: _pct(v) for g, v in group_sigma2.items()},
            "pct_ward":      _pct(sigma2_ward),
            "pct_hosp":      _pct(sigma2_hosp),
            "pct_site":      _pct(sigma2_site),
            "pct_resid":     _pct(pi2_3),
        }
        print(f"  [{site}] R²_marginal={results[site]['r2_marginal']:.3f}  "
              f"R²_conditional={results[site]['r2_conditional']:.3f}")

    return results


def plot_covariate_variance_breakdown(
    covar_result: dict,
    out_dir: Path,
    cohort_label: str,
    site: str,
) -> None:
    """Stacked horizontal bar: each covariate group + random levels + residual."""
    if not covar_result or site not in covar_result:
        return
    res = covar_result[site]

    # Covariates ordered descending by % contribution
    grp_items = sorted(res["pct_groups"].items(), key=lambda x: -x[1])
    labels = ([g for g, _ in grp_items]
              + ["Ward / ICU type", "Hospital", "Site", "Unexplained"])
    pcts   = ([p for _, p in grp_items]
              + [res["pct_ward"], res["pct_hosp"], res["pct_site"], res["pct_resid"]])
    colors = [_COVAR_COLORS.get(lbl, "#aaaaaa") for lbl in labels]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    left = 0.0
    for label, pct, col in zip(labels, pcts, colors):
        if pct < 0.05:
            left += pct
            continue
        ax.barh(0, pct, left=left, color=col, edgecolor="white", height=0.55,
                label=f"{label}  {pct:.1f}%")
        if pct >= 5:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% of total latent-scale variance", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=4, fontsize=8, frameon=False)
    r2m = res["r2_marginal"]
    r2c = res["r2_conditional"]
    ax.set_title(
        f"Covariate variance explained — {site}  "
        f"[{_COHORT_LABELS.get(cohort_label, cohort_label)}]\n"
        f"R²_marginal = {r2m:.3f}  (fixed effects only)  |  "
        f"R²_conditional = {r2c:.3f}  (fixed + random effects)",
        fontsize=9, fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"covariate_variance_{cohort_label}_{site}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def export_covariate_variance_csv(
    covar_result: dict,
    out_dir: Path,
    cohort_label: str,
    site: str,
) -> None:
    if not covar_result or site not in covar_result:
        return
    res = covar_result[site]
    rows = [
        {"component": g, "type": "covariate",
         "sigma2": round(s2, 6), "pct_total": round(res["pct_groups"][g], 3)}
        for g, s2 in res["group_sigma2"].items()
    ] + [
        {"component": "Ward / ICU type", "type": "random_effect",
         "sigma2": round(res["sigma2_ward"], 6), "pct_total": round(res["pct_ward"], 3)},
        {"component": "Hospital", "type": "random_effect",
         "sigma2": round(res["sigma2_hosp"], 6), "pct_total": round(res["pct_hosp"], 3)},
        {"component": "Site", "type": "random_effect",
         "sigma2": round(res["sigma2_site"], 6), "pct_total": round(res["pct_site"], 3)},
        {"component": "Unexplained (residual)", "type": "residual",
         "sigma2": round(res["sigma2_resid"], 6), "pct_total": round(res["pct_resid"], 3)},
    ]
    df_out = pd.DataFrame(rows)
    df_out["r2_marginal"] = round(res["r2_marginal"], 4)
    df_out["r2_conditional"] = round(res["r2_conditional"], 4)
    df_out["site"] = site
    df_out["cohort"] = cohort_label
    out = out_dir / f"covariate_variance_{cohort_label}_{site}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"  Saved: {out.name}")


def plot_group_intercepts(
    structural_model: dict, out_dir: Path, cohort_label: str, site: str, label: str
) -> None:
    """Forest plot of each ward/ICU-type or hospital's effective logit intercept
    ± 95% CI — computed via delta method from the single expanded structural model.

    structural_model: result of fit_site_logistic_with_structure for this site.
    label: "ward"     → reads ward_effective_intercepts
           "hospital" → reads hospital_effective_intercepts
    """
    ei_key = "ward_effective_intercepts" if label == "ward" else "hospital_effective_intercepts"
    ei = structural_model.get(ei_key, {})
    if len(ei) < 2:
        return

    groups   = list(ei.keys())
    alphas   = [ei[g]["alpha"]      for g in groups]
    alpha_se = [ei[g]["se"]         for g in groups]
    ns       = [ei[g]["n_patients"] for g in groups]

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
    ax.set_xlabel("Effective logit intercept ± 95% CI\n(ref patient at site-wide mean covariates)", fontsize=9)

    tau2_key = "tau2_ward" if label == "ward" else "tau2_hosp"
    tau2     = structural_model.get(tau2_key)
    subtitle = f"τ²={tau2:.3f}" if tau2 is not None else "τ² not estimable"
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


# ── Approach 2A/2B broken down by ward/ICU-type or hospital (per site) ────────
def plot_approach2_time_by_group(
    structural_model: dict,
    out_dir: Path,
    cohort_label: str,
    site: str,
    label: str,
) -> None:
    """P(vaso_on) vs time for the reference patient, one curve per ward/ICU-type or hospital.

    Uses predict_p_structural so all curves share the same base model — the
    only difference between curves is the ward or hospital dummy being activated.
    structural_model: result of fit_site_logistic_with_structure for this site.
    label: "ward" or "hospital".
    """
    ei_key = "ward_effective_intercepts" if label == "ward" else "hospital_effective_intercepts"
    ei = structural_model.get(ei_key, {})
    if len(ei) < 2:
        return

    t_max = min(max(structural_model.get("time_knots", [120])), 120)
    time_grid = np.linspace(5, t_max, 300)

    keys = sorted(ei.keys())
    cmap = plt.cm.tab10
    color_map = {
        k: _GROUP_TYPE_COLOR_MAP.get(k, cmap(i / max(len(keys) - 1, 1)))
        for i, k in enumerate(keys)
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for grp in keys:
        ward_arg = grp if label == "ward"     else None
        hosp_arg = grp if label == "hospital" else None
        p_t = predict_p_structural(
            structural_model, REF_AGE, REF_NEE, time_grid,
            ward_type=ward_arg, hospital_id=hosp_arg,
        )
        ax.plot(time_grid, p_t, color=color_map[grp], linewidth=2.0,
                label=f"{grp}  (n={ei[grp]['n_patients']:,})")

    ax.set_xlabel("Hours from NE start", fontsize=11)
    ax.set_ylabel("P(vasopressin on | hour)", fontsize=11)
    ax.set_title(
        f"Approach 2A — Time-varying P(vasopressin) by {label}\n"
        f"Ref patient: age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, NEE={REF_NEE} mcg/kg/min  "
        f"|  {site}  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_xlim(5, t_max)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"approach2_time_by_{label}_{cohort_label}_{site}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_approach2_nee_by_group(
    structural_model: dict,
    site_ph: "pd.DataFrame",
    out_dir: Path,
    cohort_label: str,
    site: str,
    label: str,
    ref_time: float = REF_TIME,
) -> None:
    """P(vaso_on) vs NEE dose for the reference patient, one curve per ward/ICU-type or hospital.

    Uses predict_p_structural so curves share the same base model.
    """
    ei_key = "ward_effective_intercepts" if label == "ward" else "hospital_effective_intercepts"
    ei = structural_model.get(ei_key, {})
    if len(ei) < 2:
        return

    all_nee = site_ph["nee"].dropna().values if site_ph is not None else np.array([])
    finite  = all_nee[np.isfinite(all_nee)]
    nee_max = float(np.percentile(finite, 95)) if len(finite) else 2.0
    nee_grid = np.linspace(0.0, min(nee_max, 2.0), 300)

    keys = sorted(ei.keys())
    cmap = plt.cm.tab10
    color_map = {
        k: _GROUP_TYPE_COLOR_MAP.get(k, cmap(i / max(len(keys) - 1, 1)))
        for i, k in enumerate(keys)
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for grp in keys:
        ward_arg = grp if label == "ward"     else None
        hosp_arg = grp if label == "hospital" else None
        p_nee = predict_p_structural(
            structural_model, REF_AGE, nee_grid, ref_time,
            ward_type=ward_arg, hospital_id=hosp_arg,
        )
        ax.plot(nee_grid, p_nee, color=color_map[grp], linewidth=2.0,
                label=f"{grp}  (n={ei[grp]['n_patients']:,})")

    ax.set_xlabel("NEE dose (mcg/kg/min)", fontsize=11)
    ax.set_ylabel("P(vasopressin on)", fontsize=11)
    ax.set_title(
        f"Approach 2B — P(vasopressin) vs NEE by {label}\n"
        f"Ref patient: age={REF_AGE}, p/f={REF_PF}, {REF_DEVICE}, t={ref_time:.0f} h  "
        f"|  {site}  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_xlim(0, nee_grid[-1])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"approach2_nee_by_{label}_{cohort_label}_{site}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# Minimum patients per cell for outcome heatmaps (privacy floor + stability).
# Must be defined before _draw_outcome_heatmap uses it as a default argument.
_MIN_CELL_HM = 11

# ── Location transition heatmap ───────────────────────────────────────────────
def _draw_transition_heatmap(
    counts: "pd.DataFrame",
    title: str,
    out_path: Path,
    loc_order: "list | None" = None,
    min_cell: int = _MIN_CELL_HM,
) -> None:
    """Transition matrix heatmap: rows = start location, columns = end location.

    Color = count / row_total (row-normalised fraction, 0–1).
    Blank (white) cells have zero actual transitions.
    Cells with count < min_cell show 'n<K' instead of the exact count.
    """
    # Square matrix over the canonical location universe
    if loc_order is not None:
        all_locs = loc_order
    else:
        all_locs = sorted(set(counts.index) | set(counts.columns))
    counts = counts.reindex(index=all_locs, columns=all_locs, fill_value=0)

    # Same order for rows AND columns so the diagonal = "stayed in same location"
    if loc_order is not None:
        order = loc_order
    else:
        freq = counts.sum(axis=1) + counts.sum(axis=0)
        order = freq.sort_values(ascending=False).index.tolist()
    counts = counts.loc[order, order]

    N = int(counts.values.sum())
    if N == 0:
        return

    row_sums = counts.sum(axis=1).values.astype(float)   # shape (nrows,)

    # fraction[i,j] = count[i,j] / row_sum[i]  (0 = nobody, 1 = everyone in row went here)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            row_sums[:, None] > 0,
            counts.values / row_sums[:, None],
            np.nan,
        )
    # White out cells with zero count (frac could be 0 from division or NaN from row_sum=0)
    frac = np.where(counts.values > 0, frac, np.nan)

    nrows, ncols = counts.shape
    cell_px = 0.9
    fig, ax = plt.subplots(figsize=(max(4, ncols * cell_px + 2.0),
                                    max(3, nrows * cell_px + 1.5)))
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("white")
    im = ax.imshow(frac, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(nrows):
        for j in range(ncols):
            cnt = int(counts.values[i, j])
            if cnt == 0:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=12, color="#cccccc")
            elif cnt < min_cell:
                ax.text(j, i, f"n<{min_cell}", ha="center", va="center",
                        fontsize=9, color="#aaaaaa")
            else:
                pct = frac[i, j] * 100
                text_color = "white" if frac[i, j] > 0.55 else "black"
                ax.text(
                    j, i,
                    f"{cnt:,}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=12, color=text_color,
                )

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(counts.columns.tolist(), rotation=40, ha="right", fontsize=12)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(counts.index.tolist(), fontsize=12)
    ax.set_xlabel("Location at trajectory end  (death / ICU discharge / 120 h)", fontsize=7)
    ax.set_ylabel("Location at NE start  (t = 0)", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("Fraction of row patients\n(0 = none, 1 = all)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(title, fontsize=7, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _cohort_to_transition_counts(cohort: "pd.DataFrame") -> "pd.DataFrame | None":
    """Build a (loc_start × loc_end) count matrix for one cohort dataframe."""
    if ("location_category_end" not in cohort.columns
            and "location_type_end" not in cohort.columns):
        return None
    df = cohort.copy()
    df["loc_start"] = _canon_icu_series(
        _effective_location(df, "location_type", "location_category")
    )
    df["loc_end"] = _canon_icu_series(
        _effective_location(df, "location_type_end", "location_category_end")
    ).fillna("Unknown / Missing")
    df = df.dropna(subset=["loc_start"])  # keep rows with unknown end-location
    if df.empty:
        return None
    return df.groupby(["loc_start", "loc_end"]).size().unstack(fill_value=0)


def plot_location_transition_heatmap(
    cohort: "pd.DataFrame", out_dir: Path, cohort_label: str, site: str
) -> None:
    counts = _cohort_to_transition_counts(cohort)
    if counts is None:
        print(f"  [{site}] End-location columns absent or no data — skipping heatmap.")
        return
    N = int(counts.values.sum())
    _draw_transition_heatmap(
        counts,
        title=(
            f"Location transition — {site}  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]  "
            f"N = {N:,} patients\n"
            f"Cell: count (% of row)   |   Color: fraction of row patients ending here"
        ),
        out_path=out_dir / f"location_transition_{cohort_label}_{site}.png",
    )


def plot_pooled_location_transition_heatmap(
    cohort_by_site: dict, out_dir: Path, cohort_label: str
) -> None:
    """Pool all sites' transition counts and draw one combined heatmap."""
    pooled: "pd.DataFrame | None" = None
    for site, cohort in cohort_by_site.items():
        c = _cohort_to_transition_counts(cohort)
        if c is None:
            continue
        if pooled is None:
            pooled = c
        else:
            pooled = pooled.add(c, fill_value=0).fillna(0)
    if pooled is None or pooled.values.sum() == 0:
        print(f"  No transition data found across any site for {cohort_label} — skipping pooled heatmap.")
        return
    N = int(pooled.values.sum())
    _draw_transition_heatmap(
        pooled,
        title=(
            f"Location transition — ALL SITES POOLED  [{_COHORT_LABELS.get(cohort_label, cohort_label)}]  "
            f"N = {N:,} patients\n"
            f"Cell: count (% of row)   |   Color: fraction of row patients ending here"
        ),
        out_path=out_dir / f"location_transition_{cohort_label}_pooled.png",
    )


# ── Outcome-annotated location transition heatmaps ───────────────────────────


def _compute_cell_outcomes(
    cohort: "pd.DataFrame",
    features: "pd.DataFrame | None",
) -> "pd.DataFrame | None":
    """Build patient-level outcome table for outcome heatmaps.

    Returns DataFrame with: loc_start, loc_end, _ever_vaso, hospital_death,
    _time_to_vaso_h, _ne_at_init, _nee_at_init.  Returns None when
    end-location columns are absent from the cohort.
    """
    if ("location_category_end" not in cohort.columns
            and "location_type_end" not in cohort.columns):
        return None
    df = cohort.copy()
    df["loc_start"] = _canon_icu_series(
        _effective_location(df, "location_type", "location_category")
    )
    df["loc_end"] = _canon_icu_series(
        _effective_location(df, "location_type_end", "location_category_end")
    ).fillna("Unknown / Missing")
    df = df.dropna(subset=["loc_start"])
    if df.empty:
        return None

    # ever_vaso and first_vaso_hour come from features (action_vaso column).
    # The cohort's first_vaso_time is only populated for the prior-vaso group
    # (vaso_before_traj == 1) and cannot be used here.
    df["_ever_vaso"] = 0
    df["_time_to_vaso_h"] = np.nan
    df["_ne_at_init"] = np.nan
    df["_nee_at_init"] = np.nan

    if features is not None and "action_vaso" in features.columns:
        _vaso_f = features[(features["action_vaso"] == 1) & (features["time_hour"] >= 0)]
        if not _vaso_f.empty:
            _first_vhr = (
                _vaso_f.groupby("stay_id")["time_hour"]
                .min()
                .reset_index()
                .rename(columns={"time_hour": "_first_vhr"})
            )
            df = df.merge(_first_vhr, on="stay_id", how="left")
            df["_ever_vaso"] = df["_first_vhr"].notna().astype(int)
            df["_time_to_vaso_h"] = df["_first_vhr"].astype(float)
            df.drop(columns=["_first_vhr"], inplace=True)

            if "norepinephrine" in features.columns and "nee" in features.columns:
                _vt = df[df["_ever_vaso"] == 1][["stay_id", "_time_to_vaso_h"]].dropna()
                if not _vt.empty:
                    _vt = _vt.copy()
                    _vt["_init_hr"] = _vt["_time_to_vaso_h"].astype(int)
                    _feat = features[["stay_id", "time_hour", "norepinephrine", "nee"]].copy()
                    _feat = _feat.merge(_vt[["stay_id", "_init_hr"]], on="stay_id", how="inner")
                    _feat = _feat[_feat["time_hour"] == _feat["_init_hr"]]
                    _feat = _feat.groupby("stay_id")[["norepinephrine", "nee"]].first().reset_index()
                    _feat.rename(columns={"norepinephrine": "_ne_f", "nee": "_nee_f"}, inplace=True)
                    df = df.merge(_feat[["stay_id", "_ne_f", "_nee_f"]], on="stay_id", how="left")
                    df["_ne_at_init"] = df["_ne_f"]
                    df["_nee_at_init"] = df["_nee_f"]
                    df.drop(columns=["_ne_f", "_nee_f"], inplace=True)

    need = ["loc_start", "loc_end", "_ever_vaso", "hospital_death",
            "_time_to_vaso_h", "_ne_at_init", "_nee_at_init"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    return df[need].copy()


def _draw_outcome_heatmap(
    value_mat: "pd.DataFrame",
    counts_mat: "pd.DataFrame",
    title: str,
    cmap_name: str,
    cbar_label: str,
    out_path: Path,
    vmin=None,
    vmax=None,
    vcenter=None,
    q25_mat: "pd.DataFrame | None" = None,
    q75_mat: "pd.DataFrame | None" = None,
    pct: bool = False,
    min_cell: int = _MIN_CELL_HM,
    loc_order: "list | None" = None,
) -> None:
    """Single outcome heatmap over the (loc_start × loc_end) grid."""
    # When loc_order is supplied it IS the canonical universe; use it for reindex
    # so that locations with no outcome data still appear as empty rows/columns.
    if loc_order is not None:
        all_locs = loc_order
    else:
        all_locs = sorted(set(value_mat.index) | set(value_mat.columns))
    value_mat = value_mat.reindex(index=all_locs, columns=all_locs)
    counts_mat = counts_mat.reindex(index=all_locs, columns=all_locs, fill_value=0)

    order = loc_order if loc_order is not None else (
        (counts_mat.sum(axis=1) + counts_mat.sum(axis=0))
        .sort_values(ascending=False).index.tolist()
    )
    value_mat = value_mat.loc[order, order]
    counts_mat = counts_mat.loc[order, order]
    if q25_mat is not None:
        q25_mat = q25_mat.reindex(index=all_locs, columns=all_locs).loc[order, order]
        q75_mat = q75_mat.reindex(index=all_locs, columns=all_locs).loc[order, order]

    vals = value_mat.values.astype(float).copy()
    cnts = counts_mat.values.astype(int)
    vals[cnts == 0] = np.nan

    valid_vals = vals[(~np.isnan(vals)) & (cnts >= min_cell)]
    if len(valid_vals) == 0:
        print(f"  Skipping {out_path.name} — no cells with n≥{min_cell}")
        return
    _vmin = float(vmin) if vmin is not None else float(np.nanmin(valid_vals))
    _vmax = float(vmax) if vmax is not None else float(np.nanmax(valid_vals))
    if _vmin == _vmax:
        _vmax = _vmin + 1e-6

    nrows, ncols = vals.shape
    cell_px = 0.9
    fig, ax = plt.subplots(figsize=(max(4, ncols * cell_px + 2.0),
                                    max(3, nrows * cell_px + 1.5)))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    disp = vals.copy()
    disp[cnts < min_cell] = np.nan

    _norm = None
    if vcenter is not None:
        from matplotlib.colors import TwoSlopeNorm
        _vc = float(np.clip(vcenter, _vmin + 1e-6, _vmax - 1e-6))
        _norm = TwoSlopeNorm(vmin=_vmin, vcenter=_vc, vmax=_vmax)
        im = ax.imshow(disp, cmap=cmap, norm=_norm, aspect="auto")
    else:
        im = ax.imshow(disp, cmap=cmap, vmin=_vmin, vmax=_vmax, aspect="auto")

    for i in range(nrows):
        for j in range(ncols):
            cnt = int(cnts[i, j])
            v = float(vals[i, j]) if not np.isnan(vals[i, j]) else np.nan
            if cnt == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#cccccc")
            elif cnt < min_cell:
                ax.text(j, i, f"n={cnt}", ha="center", va="center", fontsize=7, color="#aaaaaa")
            elif np.isnan(v):
                ax.text(j, i, f"n={cnt}", ha="center", va="center", fontsize=7, color="#aaaaaa")
            else:
                if _norm is not None:
                    nv = float(_norm(v))
                    tc = "white" if (nv < 0.3 or nv > 0.7) else "black"
                else:
                    nv = (v - _vmin) / (_vmax - _vmin)
                    tc = "white" if nv > 0.6 else "black"
                if pct:
                    cell_text = f"{v * 100:.0f}%\n(n={cnt})"
                elif q25_mat is not None:
                    _q25v = float(q25_mat.values[i, j])
                    _q75v = float(q75_mat.values[i, j])
                    iqr_str = (f"({_q25v:.1f}–{_q75v:.1f})"
                               if (not np.isnan(_q25v) and not np.isnan(_q75v)) else "")
                    cell_text = f"{v:.1f}\n{iqr_str}\nn={cnt}"
                else:
                    cell_text = f"{v:.2f}\n(n={cnt})"
                ax.text(j, i, cell_text, ha="center", va="center", fontsize=8, color=tc)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(value_mat.columns.tolist(), rotation=40, ha="right", fontsize=12)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(value_mat.index.tolist(), fontsize=12)
    ax.set_xlabel("Location at trajectory end  (death / ICU discharge / 120 h)", fontsize=7)
    ax.set_ylabel("Location at NE start  (t = 0)", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(cbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(title, fontsize=7, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_location_transition_outcome_heatmaps(
    cohort: "pd.DataFrame",
    features: "pd.DataFrame | None",
    out_dir: Path,
    cohort_label: str,
    site: str,
) -> None:
    """Generate 5 outcome-annotated transition heatmaps for one site."""
    cell_df = _compute_cell_outcomes(cohort, features)
    if cell_df is None or cell_df.empty:
        print(f"  [{site}] Outcome heatmaps skipped — no location/outcome data.")
        return

    counts = cell_df.groupby(["loc_start", "loc_end"]).size().unstack(fill_value=0)
    cohort_lbl = _COHORT_LABELS.get(cohort_label, cohort_label)

    # Canonical location order shared by all 6 heatmaps so rows/columns align
    _all_locs = sorted(set(counts.index) | set(counts.columns))
    _counts_sq = counts.reindex(index=_all_locs, columns=_all_locs, fill_value=0)
    _freq = _counts_sq.sum(axis=1) + _counts_sq.sum(axis=0)
    loc_order = _freq.sort_values(ascending=False).index.tolist()

    # Regenerate the count heatmap with this ordering so it matches the outcome heatmaps
    N = int(counts.values.sum())
    _draw_transition_heatmap(
        counts,
        title=(
            f"Location transition — {site}  [{cohort_lbl}]  N = {N:,} patients\n"
            f"Cell: count (% of row)   |   Color: fraction of row patients ending here"
        ),
        out_path=out_dir / f"location_transition_{cohort_label}_{site}.png",
        loc_order=loc_order,
    )

    # 1. Vasopressin use rate
    n_vaso = cell_df.groupby(["loc_start", "loc_end"])["_ever_vaso"].sum().unstack(fill_value=0)
    vaso_rate = (n_vaso.astype(float) / counts.replace(0, np.nan)).clip(0.0, 1.0)
    _draw_outcome_heatmap(
        vaso_rate, counts,
        title=(f"Vasopressin use rate — {site}  [{cohort_lbl}]\n"
               f"Cell: % ever-vaso (n total)  |  Color: fraction receiving vasopressin"),
        cmap_name="Greens", cbar_label="Fraction ever receiving vasopressin",
        out_path=out_dir / f"transition_outcome_vaso_rate_{cohort_label}_{site}.png",
        vmin=0.0, vmax=1.0, pct=True, loc_order=loc_order,
    )

    # 2. In-hospital mortality rate (coolwarm centred at median mortality)
    if cell_df["hospital_death"].notna().any():
        n_deaths = (
            cell_df.groupby(["loc_start", "loc_end"])["hospital_death"]
            .sum().unstack(fill_value=0)
        )
        mort_rate = (n_deaths.astype(float) / counts.replace(0, np.nan)).clip(0.0, 1.0)
        _valid_m = mort_rate.values[(counts.values >= _MIN_CELL_HM) & ~np.isnan(mort_rate.values)]
        _mort_vcenter = float(np.nanmedian(_valid_m)) if len(_valid_m) > 0 else 0.5
        _draw_outcome_heatmap(
            mort_rate, counts,
            title=(f"In-hospital mortality — {site}  [{cohort_lbl}]\n"
                   f"Cell: % died (n total)  |  Color: coolwarm centred at median"),
            cmap_name="coolwarm", cbar_label="In-hospital mortality fraction",
            out_path=out_dir / f"transition_outcome_mortality_{cohort_label}_{site}.png",
            vmin=0.0, vmax=1.0, vcenter=_mort_vcenter, pct=True, loc_order=loc_order,
        )

    # Metrics 3-5 use only ever-vaso patients; n_vaso_mat drives suppression
    vaso_df = cell_df[cell_df["_ever_vaso"] == 1].copy()
    n_vaso_mat = vaso_df.groupby(["loc_start", "loc_end"]).size().unstack(fill_value=0)

    def _med_iqr(col):
        if col not in vaso_df.columns or vaso_df[col].notna().sum() < _MIN_CELL_HM:
            return None, None, None
        grp = vaso_df.groupby(["loc_start", "loc_end"])[col]
        med = grp.median().unstack(fill_value=np.nan)
        q25 = grp.quantile(0.25).unstack(fill_value=np.nan)
        q75 = grp.quantile(0.75).unstack(fill_value=np.nan)
        return med, q25, q75

    # 3. Time to vasopressin (hours from NE start)
    med, q25, q75 = _med_iqr("_time_to_vaso_h")
    if med is not None:
        _draw_outcome_heatmap(
            med, n_vaso_mat,
            title=(f"Median time to vasopressin — {site}  [{cohort_lbl}]\n"
                   f"Cell: median h (IQR)  n = ever-vaso patients  |  Color: median hours"),
            cmap_name="Blues", cbar_label="Median h from NE start to vasopressin initiation",
            out_path=out_dir / f"transition_outcome_time_to_vaso_{cohort_label}_{site}.png",
            vmin=0.0, q25_mat=q25, q75_mat=q75, loc_order=loc_order,
        )

    # 4. NE dose at vasopressin initiation
    med, q25, q75 = _med_iqr("_ne_at_init")
    if med is not None:
        _draw_outcome_heatmap(
            med, n_vaso_mat,
            title=(f"Median NE dose at vasopressin initiation — {site}  [{cohort_lbl}]\n"
                   f"Cell: median μg/kg/min (IQR)  n = ever-vaso patients  |  Color: median dose"),
            cmap_name="Purples", cbar_label="Median NE dose at vaso initiation (μg/kg/min)",
            out_path=out_dir / f"transition_outcome_ne_dose_{cohort_label}_{site}.png",
            vmin=0.0, q25_mat=q25, q75_mat=q75, loc_order=loc_order,
        )

    # 5. NEE-equivalent dose at vasopressin initiation
    med, q25, q75 = _med_iqr("_nee_at_init")
    if med is not None:
        _draw_outcome_heatmap(
            med, n_vaso_mat,
            title=(f"Median NEE at vasopressin initiation — {site}  [{cohort_lbl}]\n"
                   f"Cell: median μg/kg/min (IQR)  n = ever-vaso patients  |  Color: median dose"),
            cmap_name="Oranges", cbar_label="Median NEE-equivalent dose at vaso initiation (μg/kg/min)",
            out_path=out_dir / f"transition_outcome_nee_dose_{cohort_label}_{site}.png",
            vmin=0.0, q25_mat=q25, q75_mat=q75, loc_order=loc_order,
        )


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


def export_group_intercept_table(structural_model: dict, label: str = "ward") -> pd.DataFrame:
    """One row per ward/ICU-type or hospital: effective logit intercept (delta-method SE),
    n_patients — from fit_site_logistic_with_structure's effective intercepts.

    label: "ward"     → reads ward_effective_intercepts
           "hospital" → reads hospital_effective_intercepts
    """
    ei_key = "ward_effective_intercepts" if label == "ward" else "hospital_effective_intercepts"
    rows = []
    for grp, v in structural_model.get(ei_key, {}).items():
        rows.append({
            "group": grp, "alpha": v["alpha"], "se": v["se"],
            "n_patients": v["n_patients"],
            "is_reference": v.get("is_reference", False),
            "stable": v.get("stable", True),
        })
    return pd.DataFrame(rows)


# ── Site discovery & main logic ───────────────────────────────────────────────
def compute_dlmm_stats(cohort_by_site: dict, features_by_site: dict, bin_cols: list = None) -> dict:
    """Compute per-outcome DLMM aggregate statistics for all sites.

    bin_cols: list of binary covariate names to include; defaults to _BIN_COLS_MODELS.
    Extra columns (e.g. OD criterion flags) must already be present in cohort_by_site[site].

    Each site's statistics are mathematically sufficient for the coordinating
    script (08) to fit a Distributed Linear Mixed Model (Luo et al., Nat
    Commun 2022) without accessing patient-level data. One round of
    communication, no iteration.

    Covariates centered at _DLMM_REFS — pre-specified so all sites share the
    same column interpretation without a coordination round.

    Shared base design matrix (both outcomes):
      [1, age-65, p_f_ratio-200, creatinine-1, platelet-150,
       bilirubin-1, gcs-15, rrt, device_non_invasive, device_invasive_imv]

    device_category is encoded via 3 canonical buckets (Room Air = reference):
      device_non_invasive : 1 if NIV / CPAP / BiPAP / high-flow O2
      device_invasive_imv : 1 if invasive mechanical ventilation / intubation
    IMV check takes priority (avoids ambiguity with "non-invasive ventilation").

    Outcome-specific extra covariate:
      time_to_vaso_h_log1p outcome: + nee (NEE dose at vaso initiation, -0.15)
      nee_at_init outcome:          + time_to_vaso_h_log1p (log1p hours, -1.79)

    Outcomes (vasopressin-user population only):
      time_to_vaso_h_log1p : log(1 + hours from NE start to first vasopressin)
      nee_at_init           : NEE-equivalent dose (mcg/kg/min) at initiation

    Random effect: Z_i = column of 1s (random site intercept).
    ZtZ = n (scalar), ZtX = col sums of X, Zty = sum(y) — all already in sx/sy/n;
    also stored under explicit "ztz"/"ztx"/"zty" keys for clarity in script 08.

    Returned dict: {site: {outcome_key: {n, XtX, Xty, yty, sx, sy, ztz, ztx, zty, ...}}}
    """
    _BIN_COLS = bin_cols if bin_cols is not None else _BIN_COLS_MODELS
    _FEAT_COVS = ["p_f_ratio", "creatinine", "platelet", "bilirubin", "gcs", "rrt"]
    # Canonical device buckets — portable across sites with different vocabularies.
    # Room Air = reference (no dummy). Check IMV first (takes priority over NIV matches).
    _IMV_KWDS    = [r"\binvasive\b", "imv", "intubat", "mechanic", "ett", "endotrach"]
    _NONIMV_KWDS = [r"non.?inv", r"niv\b", "cpap", "bipap", r"high.?flow", "high_flow"]

    results: dict = {}

    for site, cohort in cohort_by_site.items():
        features = features_by_site.get(site)
        if features is None or cohort is None:
            continue

        if "action_vaso" not in features.columns:
            print(f"  [{site}] DLMM: action_vaso column missing — skipping")
            continue

        vaso_rows = features[(features["action_vaso"] == 1) & (features["time_hour"] >= 0)]
        if vaso_rows.empty:
            print(f"  [{site}] DLMM: no vasopressin events — skipping")
            continue

        first_vhr = (
            vaso_rows.groupby("stay_id")["time_hour"]
            .min()
            .reset_index()
            .rename(columns={"time_hour": "time_to_vaso_h"})
        )

        # Start with age + timing; binary comorbidities merged in below
        _cohort_base_cols = ["stay_id", "age"] + [c for c in _BIN_COLS
                                                   if c in cohort.columns]
        pat = (
            cohort[_cohort_base_cols]
            .merge(first_vhr, on="stay_id", how="inner")
            .dropna(subset=["age", "time_to_vaso_h"])
        )
        # Fill any missing binary cols (site may not have procedures/diagnosis table)
        for _bc in _BIN_COLS:
            if _bc in pat.columns:
                pat[_bc] = pat[_bc].fillna(0.0)

        # Grab NEE + device_category + severity covariates at the initiation hour,
        # plus raw NE dose, MAP, lactate, BUN for the feature-bin scatter plots.
        _BIN_EXTRAS = ["norepinephrine", "mbp", "lactate", "bun"]
        grab_cols = [c for c in ["nee", "device_category"] + _BIN_EXTRAS + _FEAT_COVS
                     if c in features.columns]
        if grab_cols:
            vt = pat[["stay_id", "time_to_vaso_h"]].copy()
            vt["_init_hr"] = vt["time_to_vaso_h"].astype(int)
            feat_at = (
                features[["stay_id", "time_hour"] + grab_cols]
                .merge(vt[["stay_id", "_init_hr"]], on="stay_id", how="inner")
            )
            feat_at = feat_at[feat_at["time_hour"] == feat_at["_init_hr"]]
            feat_at = feat_at.groupby("stay_id")[grab_cols].first().reset_index()
            pat = pat.merge(feat_at, on="stay_id", how="left")

        # Canonical device buckets (Room Air = reference, no dummy).
        # IMV check takes priority: "non-invasive ventilation" → non-invasive, not IMV.
        import re as _re
        if "device_category" in pat.columns:
            dcl = pat["device_category"].fillna("").astype(str).str.lower()
            pat["device_invasive_imv"]  = dcl.apply(
                lambda x: float(any(_re.search(kw, x) for kw in _IMV_KWDS))
            )
            pat["device_non_invasive"] = dcl.apply(
                lambda x: float(
                    not any(_re.search(kw, x) for kw in _IMV_KWDS)
                    and any(_re.search(kw, x) for kw in _NONIMV_KWDS)
                )
            )
        else:
            pat["device_invasive_imv"]  = 0.0
            pat["device_non_invasive"] = 0.0

        # log1p(time_to_vaso_h) used as a predictor in the nee_at_init model
        pat["time_to_vaso_h_log1p"] = np.log1p(pat["time_to_vaso_h"])

        # ── Attach ward/hospital location for 4-level variance decomp ───────
        # Merge from cohort: location_type / location_category → canonical ward,
        # hospital_id / hospital_type (when present) for hospital-level ICC.
        _loc_cols = [c for c in ["location_type", "location_category",
                                  "hospital_id", "hospital_type"]
                     if c in cohort.columns]
        if _loc_cols:
            loc_df = cohort[["stay_id"] + _loc_cols].copy()
            loc_df["_ward"] = _canon_icu_series(_effective_location(loc_df))
            _hcols = [c for c in ["hospital_id", "hospital_type"] if c in loc_df.columns]
            pat = pat.merge(loc_df[["stay_id", "_ward"] + _hcols],
                            on="stay_id", how="left")

        # ── Attach SOFA for feature-bin scatter plots ─────────────────────
        # Use first available SOFA column from cohort (sofa > sofa_total).
        _sofa_col = next((c for c in ["sofa", "sofa_total", "sepsis_onset_sofa"]
                          if c in cohort.columns), None)
        if _sofa_col:
            pat = pat.merge(cohort[["stay_id", _sofa_col]].rename(columns={_sofa_col: "_sofa"}),
                            on="stay_id", how="left")

        if len(pat) < 5:
            continue

        # Binary comorbidity covariates — prevalence-screened at patient level
        _bin_dlmm = []
        for _bc in _BIN_COLS:
            if _bc not in pat.columns:
                continue
            _prev = float((pat[_bc] == 1).mean())
            if _prev >= _MIN_BIN_PREVALENCE:
                _bin_dlmm.append(_bc)
            else:
                print(f"  [{site}] DLMM: {_bc} prevalence {_prev:.1%} — excluded")

        # Shared base covariates: SOFA components + device dummies + binary comorbidities
        _BASE_COV = (["age"]
                     + [c for c in _FEAT_COVS if c in pat.columns]
                     + ["device_non_invasive", "device_invasive_imv"]
                     + _bin_dlmm)

        # Per-outcome covariate lists:
        #   time_to_vaso model: + NEE at vaso initiation (the dose "threshold" clinicians waited for)
        #   nee_at_init  model: + log1p(time) elapsed before initiation
        _COV_TIME = _BASE_COV + (["nee"] if "nee" in pat.columns else [])
        _COV_NEE  = _BASE_COV + ["time_to_vaso_h_log1p"]

        def _cov_names(cov_cols_: list) -> list:
            """Human-readable column names: plain name for 0-reference, name_minus_ref otherwise."""
            names = ["intercept"]
            for c in cov_cols_:
                ref = _DLMM_REFS.get(c, 0.0)
                if ref == 0.0:
                    names.append(c)
                elif ref == int(ref):
                    names.append(f"{c}_minus_{int(ref)}")
                else:
                    names.append(f"{c}_minus_{ref:.2f}")
            return names

        def _build_X(mask_: np.ndarray, cov_cols_: list) -> np.ndarray:
            cols = [np.ones(mask_.sum())]
            for c in cov_cols_:
                arr = pat[c].values.astype(float)
                cols.append(arr[mask_] - _DLMM_REFS.get(c, 0.0))
            return np.column_stack(cols)

        def _finite_mask(cov_cols_: list) -> np.ndarray:
            m = np.ones(len(pat), dtype=bool)
            for c in cov_cols_:
                m &= np.isfinite(pat[c].values.astype(float))
            return m

        def _ward_hosp_stats(y: np.ndarray, mask_: np.ndarray) -> dict:
            """Per-ward and per-hospital mean/SD/N for DL pooling in 08.

            Only groups with >= MIN_ICU_N_PATIENTS patients are included.
            Returns dict with keys 'ward_stats', 'hospital_stats',
            'hospital_type_stats' — each a list of {group, n, mean, sd, se}.
            """
            out: dict = {}
            _group_cols = [("_ward", "ward_stats")]
            for hcol in ["hospital_id", "hospital_type"]:
                if hcol in pat.columns:
                    _group_cols.append((hcol, f"{hcol}_stats"))
            for col, key in _group_cols:
                if col not in pat.columns:
                    continue
                groups = pat[col].values[mask_]
                grp_set = sorted(
                    {g for g in groups
                     if pd.notna(g) and str(g).lower() not in ["nan", "other", "other icu"]},
                    key=str,
                )
                rows = []
                for g in grp_set:
                    sel = groups == g
                    yn  = y[sel]
                    n   = int(sel.sum())
                    if n < MIN_ICU_N_PATIENTS:
                        continue
                    rows.append({
                        "group": str(g),
                        "n":     n,
                        "mean":  round(float(yn.mean()), 6),
                        "sd":    round(float(yn.std(ddof=1)), 6) if n > 1 else 0.0,
                        "se":    round(float(yn.std(ddof=1) / np.sqrt(n)), 6) if n > 1 else 0.0,
                    })
                if len(rows) >= 2:
                    out[key] = rows
            return out

        def _outcome_feature_bins(y: np.ndarray, mask_: np.ndarray) -> dict:
            """Per-bin mean/SD of outcome y, grouped by patient features.

            Used for the feature-bin scatter plot in 10 — each dot = one
            feature bin at one site, axes = (SD, mean) of the outcome.
            Only bins with >= 5 patients included.

            Features computed:
              sofa         — integer SOFA bins
              age_10bins   — age in 10 quantile groups
              ne_dose      — NE dose at initiation (5 quantile bins)
              nee_dose     — NEE-equivalent dose at initiation (5 quantile bins)
              map          — MAP at initiation (5 quantile bins)
              lactate      — lactate at initiation (5 quantile bins)
              creatinine   — creatinine at initiation (5 quantile bins)
              bun          — BUN at initiation (5 quantile bins)
              time_to_vaso — hours from NE start to vasopressin (5 quantile bins)
            """
            out: dict = {}

            def _bin_stats(feature_vals, y_, label_fn) -> list:
                rows = []
                for uval in sorted(set(feature_vals)):
                    if not np.isfinite(uval):
                        continue
                    sel = feature_vals == uval
                    yn  = y_[sel]
                    n   = int(sel.sum())
                    if n < 5:
                        continue
                    rows.append({
                        "bin_label": label_fn(uval),
                        "bin_value": float(uval),
                        "n":         n,
                        "mean":      round(float(yn.mean()), 4),
                        "sd":        round(float(yn.std(ddof=1)), 4) if n > 1 else 0.0,
                    })
                return rows

            def _qbins(col: str, key: str, n_bins: int = 5, fmt: str = ".2g") -> None:
                """Quantile-bin pat[col] → n_bins groups; store per-bin mean/SD of y."""
                if col not in pat.columns:
                    return
                arr = pat[col].values.astype(float)[mask_]
                fin = np.isfinite(arr)
                if fin.sum() < n_bins * 5:
                    return
                arr_ok = arr[fin]
                edges  = np.unique(np.percentile(arr_ok, np.linspace(0, 100, n_bins + 1)))
                if len(edges) < 2:
                    return
                quantized = np.full(len(arr), np.nan)
                centers, labels = [], []
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    last   = (i == len(edges) - 2)
                    sel    = (arr >= lo) & (arr <= hi if last else arr < hi)
                    mid    = round(float((lo + hi) / 2), 4)
                    quantized[sel] = mid
                    centers.append(mid)
                    labels.append(f"{lo:{fmt}}–{hi:{fmt}}")
                lbl_map = {c: l for c, l in zip(centers, labels)}
                fin2    = np.isfinite(quantized)
                rows    = _bin_stats(
                    quantized[fin2], y[fin2],
                    lambda v, m=lbl_map: m.get(v, str(v)),
                )
                if rows:
                    out[key] = rows

            # ── SOFA by integer value ────────────────────────────────────────
            if "_sofa" in pat.columns:
                sofa_arr = pat["_sofa"].values.astype(float)[mask_]
                sofa_arr = np.round(sofa_arr).astype(float)  # snap to integer
                rows = _bin_stats(sofa_arr, y, lambda v: str(int(v)))
                if rows:
                    out["sofa"] = rows

            # ── Age in 10 quantile bins ──────────────────────────────────────
            age_arr = pat["age"].values.astype(float)[mask_]
            age_ok  = age_arr[np.isfinite(age_arr)]
            if len(age_ok) >= 20:
                edges = np.percentile(age_ok, np.linspace(0, 100, 11))
                edges = np.unique(edges)
                bin_centers = []
                bin_labels  = []
                quantized   = np.full(len(age_arr), np.nan)
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    sel = (age_arr >= lo) & (age_arr <= hi if i == len(edges) - 2 else age_arr < hi)
                    mid = round((lo + hi) / 2, 1)
                    quantized[sel] = mid
                    bin_centers.append(mid)
                    bin_labels.append(f"{int(lo)}-{int(hi)}")
                # Override with edge labels for bin_stats label_fn
                _lbl_map = {c: l for c, l in zip(bin_centers, bin_labels)}
                rows = _bin_stats(
                    quantized[np.isfinite(quantized)],
                    y[np.isfinite(quantized)],
                    lambda v, m=_lbl_map: m.get(v, str(v)),
                )
                if rows:
                    out["age_10bins"] = rows

            # ── Clinical feature quantile bins (5 groups each) ───────────────
            _qbins("norepinephrine", "ne_dose",     fmt=".2g")
            _qbins("nee",            "nee_dose",    fmt=".2g")
            _qbins("mbp",            "map",         fmt=".0f")
            _qbins("lactate",        "lactate",     fmt=".1f")
            _qbins("creatinine",     "creatinine",  fmt=".1f")
            _qbins("bun",            "bun",         fmt=".0f")
            _qbins("time_to_vaso_h", "time_to_vaso", fmt=".0f")

            return out

        def _stats(y: np.ndarray, X: np.ndarray, mask_: np.ndarray,
                   transform: str, unit: str, cov_names_: list) -> dict:
            n_  = int(len(y))
            sx_ = X.sum(axis=0).tolist()
            sy_ = float(y.sum())
            result = {
                "n":                 n_,
                "XtX":               (X.T @ X).tolist(),
                "Xty":               (X.T @ y).tolist(),
                "yty":               float(y @ y),
                "sx":                sx_,
                "sy":                sy_,
                # ZtZ / ZtX / Zty — explicit DLMM labels for the random site intercept.
                # Z_i = column of 1s  →  Z'Z = n,  Z'X = col sums of X,  Z'y = sum(y).
                "ztz":               n_,
                "ztx":               sx_,
                "zty":               sy_,
                "covariate_names":   cov_names_,
                "outcome_transform": transform,
                "y_mean":            float(y.mean()),
                "y_sd":              float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                "unit":              unit,
            }
            result.update(_ward_hosp_stats(y, mask_))
            result["feature_bin_stats"] = _outcome_feature_bins(y, mask_)
            return result

        site_stats: dict = {}

        # ── Outcome 1: log(1 + hours to vasopressin) ────────────────────────
        names1 = _cov_names(_COV_TIME)
        mask1 = (
            _finite_mask(_COV_TIME)
            & np.isfinite(pat["time_to_vaso_h"].values)
            & (pat["time_to_vaso_h"].values >= 0)
        )
        if mask1.sum() >= 5:
            y1 = np.log1p(pat["time_to_vaso_h"].values[mask1])
            site_stats["time_to_vaso_h_log1p"] = _stats(
                y1, _build_X(mask1, _COV_TIME), mask1, "log1p(hours)", "log1p-hours", names1
            )

        # ── Outcome 2: NEE dose at vasopressin initiation ───────────────────
        nee_vals = pat["nee"].values if "nee" in pat.columns else np.full(len(pat), np.nan)
        names2 = _cov_names(_COV_NEE)
        mask2 = (
            _finite_mask(_COV_NEE)
            & np.isfinite(nee_vals)
            & (nee_vals >= 0)
        )
        if mask2.sum() >= 5:
            y2 = nee_vals[mask2]
            site_stats["nee_at_init"] = _stats(
                y2, _build_X(mask2, _COV_NEE), mask2, "raw (mcg/kg/min)", "mcg/kg/min", names2
            )

        if site_stats:
            results[site] = site_stats
            for ok, s in site_stats.items():
                p_val = len(s["covariate_names"])
                print(f"  [{site}] DLMM {ok}: n={s['n']} (p={p_val}), "
                      f"mean={s['y_mean']:.3f} ({s['outcome_transform']})")

    return results


# ── CVC stratification table ──────────────────────────────────────────────────

def compute_cvc_stratification(cohort_by_site: dict, features_by_site: dict) -> dict:
    """Descriptive table: time-to-vaso and NEE-at-initiation by CVC-before-NE status.

    Among vasopressin initiators in each cohort, compares patients who had a CVC
    placed *before* their first NE dose ('CVC before NE start') vs. those who did
    not.  The hypothesis is that pre-existing vascular access may facilitate
    earlier or lower-dose vasopressin initiation.

    Outcomes (vasopressin-user population only):
      time_to_vaso_h  : hours from NE start to first vasopressin
      nee_at_init     : NEE-equivalent dose (mcg/kg/min) at vasopressin initiation
      cvc_lead_h      : hours by which CVC preceded NE start (CVC=1 group only)

    Returns {site: pd.DataFrame} where each DataFrame has two rows (cvc_flag=1/0)
    with summary statistics and Mann-Whitney U p-values.
    """
    from scipy import stats as _scipy_stats

    def _med_iqr(s: pd.Series) -> tuple:
        s = s.dropna()
        if len(s) < 5:
            return np.nan, np.nan, np.nan
        return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))

    results: dict = {}

    for site, cohort in cohort_by_site.items():
        features = features_by_site.get(site)
        if features is None:
            continue
        if "cvc_before_ne_start" not in cohort.columns:
            print(f"  [{site}] CVC stratification: cvc_before_ne_start not in cohort — skipping")
            continue
        if "action_vaso" not in features.columns:
            print(f"  [{site}] CVC stratification: action_vaso not in features — skipping")
            continue

        # ── Vasopressin initiators: first vaso hour per patient ───────────────
        vaso_rows = features[(features["action_vaso"] == 1) & (features["time_hour"] >= 0)]
        if vaso_rows.empty:
            print(f"  [{site}] CVC stratification: no vasopressin events — skipping")
            continue

        first_vhr = (
            vaso_rows.groupby("stay_id")["time_hour"]
            .min()
            .reset_index()
            .rename(columns={"time_hour": "time_to_vaso_h"})
        )

        # NEE at the initiation hour (take the row where time_hour == floor(time_to_vaso_h))
        vt2 = first_vhr.copy()
        vt2["_init_hr"] = vt2["time_to_vaso_h"].astype(int)
        feat_init = (
            features[["stay_id", "time_hour", "nee"]]
            .merge(vt2[["stay_id", "_init_hr"]], on="stay_id", how="inner")
        )
        feat_init = feat_init[feat_init["time_hour"] == feat_init["_init_hr"]]
        nee_init = (
            feat_init[["stay_id", "nee"]]
            .drop_duplicates("stay_id")
            .rename(columns={"nee": "nee_at_init"})
        )

        # ── Merge with cohort for CVC flag and timestamps ─────────────────────
        cvc_cols = ["stay_id", "cvc_before_ne_start"]
        if "first_cvc_dttm" in cohort.columns and "first_norepi_time" in cohort.columns:
            cvc_cols += ["first_cvc_dttm", "first_norepi_time"]

        pat = (
            cohort[cvc_cols]
            .merge(first_vhr, on="stay_id", how="inner")
            .merge(nee_init,  on="stay_id", how="left")
        )

        # CVC lead time: hours by which CVC preceded NE start (positive = CVC first)
        if "first_cvc_dttm" in pat.columns and "first_norepi_time" in pat.columns:
            _fcvc = pd.to_datetime(pat["first_cvc_dttm"],    utc=True, errors="coerce")
            _fne  = pd.to_datetime(pat["first_norepi_time"], utc=True, errors="coerce")
            pat["cvc_lead_h"] = (_fne - _fcvc).dt.total_seconds() / 3600.0
        else:
            pat["cvc_lead_h"] = np.nan

        # ── Summary statistics by group ───────────────────────────────────────
        rows = []
        for cvc_flag, label in [(1, "CVC before NE start"), (0, "No CVC before NE start")]:
            grp = pat[pat["cvc_before_ne_start"] == cvc_flag]
            tv_med, tv_q25, tv_q75 = _med_iqr(grp["time_to_vaso_h"])
            ni_med, ni_q25, ni_q75 = _med_iqr(grp["nee_at_init"])
            cl_med, cl_q25, cl_q75 = (
                _med_iqr(grp["cvc_lead_h"]) if cvc_flag == 1 else (np.nan, np.nan, np.nan)
            )
            rows.append({
                "site":                  site,
                "cvc_group":             label,
                "cvc_flag":              cvc_flag,
                "n_vaso_initiators":     len(grp),
                "time_to_vaso_h_median": tv_med,
                "time_to_vaso_h_q25":    tv_q25,
                "time_to_vaso_h_q75":    tv_q75,
                "nee_at_init_median":    ni_med,
                "nee_at_init_q25":       ni_q25,
                "nee_at_init_q75":       ni_q75,
                # CVC lead time (hours CVC preceded NE; CVC=1 group only)
                "cvc_lead_h_median":     cl_med,
                "cvc_lead_h_q25":        cl_q25,
                "cvc_lead_h_q75":        cl_q75,
            })

        # ── Mann-Whitney U tests ──────────────────────────────────────────────
        g1 = pat[pat["cvc_before_ne_start"] == 1]
        g0 = pat[pat["cvc_before_ne_start"] == 0]
        tv1, tv0 = g1["time_to_vaso_h"].dropna(), g0["time_to_vaso_h"].dropna()
        ni1, ni0 = g1["nee_at_init"].dropna(),    g0["nee_at_init"].dropna()

        tv_p = (
            _scipy_stats.mannwhitneyu(tv1, tv0, alternative="two-sided").pvalue
            if len(tv1) >= 5 and len(tv0) >= 5 else np.nan
        )
        ni_p = (
            _scipy_stats.mannwhitneyu(ni1, ni0, alternative="two-sided").pvalue
            if len(ni1) >= 5 and len(ni0) >= 5 else np.nan
        )

        df_out = pd.DataFrame(rows)
        df_out["n_total_vaso_initiators"] = len(pat)
        df_out["time_to_vaso_p"] = tv_p    # same p-value repeated on both rows
        df_out["nee_at_init_p"]  = ni_p

        print(
            f"  [{site}] CVC strat: CVC-before n={len(g1)}, no-CVC n={len(g0)}, "
            f"time-to-vaso p={tv_p:.3g}, NEE-at-init p={ni_p:.3g}"
        )
        results[site] = df_out

    return results


# ── Model variant comparison (original / +comorbid / +Rhee-OD) ───────────────
# Binary OD criterion flags derivable without re-running 01:
#   od_lactate    : initial_lactate >= 2.0 mmol/L (Rhee/lactate arm — in cohort parquet)
#   od_coagulation: platelet < 100 K/µL at NE start (Rhee thrombocytopenia criterion)
#   od_liver      : bilirubin > 2.0 mg/dL at NE start (Rhee hyperbilirubinemia criterion)
# IMV and AKI/RRT criteria are already captured by device dummies and rrt/creatinine
# covariates in the base model; these three add threshold-specific binary signals.
_OD_COLS_RHEE = ["od_lactate", "od_coagulation", "od_liver"]


def compute_model_variants(
    cohort_label: str,
    cohort_by_site: dict,
    ph_by_site: dict,
    features_by_site: dict,
) -> dict:
    """Run three model variants and return comparison dict for storage in the packet.

    v1_original  — SOFA components + age + device + RRT + RCS(NEE,time); no comorbidities
    v2_comorbid  — v1 + 5 comorbidity binaries (current specification)
    v3_rhee_od   — v2 + 3 Rhee OD criterion binary flags; Rhee cohorts only

    For each variant, computes:
      - DL random-effects pooling of site intercepts → tau2, ICC, MOR
      - DLMM sufficient statistics for 08 to fit LMMs and compute DLMM ICCs

    Returns: {variant_name: {"bin_cols": [...], "dl": {...}, "dlmm_stats": {...}}}
    """
    # ── Variant definitions ──────────────────────────────────────────────────
    _variants: dict = {
        "v1_original": [],
        "v2_comorbid": list(_BIN_COLS_MODELS),
    }

    # ── Derive Rhee OD criterion flags for v3 (Rhee cohorts only) ────────────
    _od_cohort_by_site: dict = {}
    _od_ph_by_site: dict    = {}
    if cohort_label.startswith("rhee"):
        for _s in cohort_by_site:
            _coh = cohort_by_site[_s].copy()
            _ph  = ph_by_site.get(_s, pd.DataFrame()).copy()

            # od_lactate: initial_lactate >= 2.0 mmol/L (in cohort parquet)
            if "initial_lactate" in _coh.columns:
                _coh["od_lactate"] = (
                    _coh["initial_lactate"].fillna(0.0) >= 2.0
                ).astype(float)
            else:
                _coh["od_lactate"] = 0.0

            # od_coagulation: platelet < 100 — use first available value per stay in ph
            if not _ph.empty and "platelet" in _ph.columns:
                _first_plt = _ph.groupby("stay_id")["platelet"].first()
                _coh["od_coagulation"] = (
                    _coh["stay_id"].map(_first_plt < 100.0).fillna(False)
                ).astype(float)
            else:
                _coh["od_coagulation"] = 0.0

            # od_liver: bilirubin > 2.0 mg/dL — use first available value per stay
            if not _ph.empty and "bilirubin" in _ph.columns:
                _first_bili = _ph.groupby("stay_id")["bilirubin"].first()
                _coh["od_liver"] = (
                    _coh["stay_id"].map(_first_bili > 2.0).fillna(False)
                ).astype(float)
            else:
                _coh["od_liver"] = 0.0

            _od_cohort_by_site[_s] = _coh

            if not _ph.empty:
                _od_merge = _coh[["stay_id"] + _OD_COLS_RHEE].copy()
                for _c in _OD_COLS_RHEE:
                    if _c not in _ph.columns:
                        _ph = _ph.merge(_od_merge[["stay_id", _c]], on="stay_id", how="left")
                        _ph[_c] = _ph[_c].fillna(0.0)
                _od_ph_by_site[_s] = _ph

        _variants["v3_rhee_od"] = list(_BIN_COLS_MODELS) + _OD_COLS_RHEE

    # ── Run each variant ─────────────────────────────────────────────────────
    variant_results: dict = {}
    for _vname, _bin_cols in _variants.items():
        print(f"  [variant: {_vname}]  bin_cols={_bin_cols or '(none)'}")

        _coh_dict = _od_cohort_by_site if _vname == "v3_rhee_od" else cohort_by_site
        _ph_dict  = _od_ph_by_site     if _vname == "v3_rhee_od" else ph_by_site

        # Fit per-site GLMs with this covariate set
        _vsite_models: dict = {}
        for _s in _ph_dict:
            try:
                _vsite_models[_s] = fit_site_logistic(
                    _ph_dict[_s], cohort_label, _s, bin_cols=_bin_cols
                )
                _m = _vsite_models[_s]
                print(f"    [{_s}] p={len(_m['coefficients'])}  "
                      f"α={_m['coefficients']['intercept']['beta']:+.4f}")
            except Exception as _exc:
                print(f"    [{_s}] GLM failed: {_exc}")

        if not _vsite_models:
            print(f"    No models fitted — skipping {_vname}.")
            continue

        # DL random-effects pooling of site intercepts
        _slist     = sorted(_vsite_models)
        _alphas    = np.array([_vsite_models[_s]["coefficients"]["intercept"]["beta"]
                               for _s in _slist])
        _alpha_var = np.array([_vsite_models[_s]["coefficients"]["intercept"]["se"]
                               for _s in _slist]) ** 2
        _dl = _dl_pool(_alphas, _alpha_var) if len(_slist) >= 2 else {}
        if _dl:
            print(f"    DL  tau2={_dl['tau2']:.4f}  ICC={_dl['icc']:.3f}  MOR={_dl['mor']:.3f}")
        else:
            print(f"    DL  (only 1 site — no pooling)")

        # DLMM sufficient stats for this variant
        _dlmm = compute_dlmm_stats(_coh_dict, features_by_site, bin_cols=_bin_cols)

        variant_results[_vname] = {
            "bin_cols":    _bin_cols,
            "n_sites":     len(_slist),
            "dl":          _dl,
            "dlmm_stats":  _dlmm,
        }

    return variant_results


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
    site_models:    dict = {}
    ph_by_site:     dict = {}
    cohort_by_site: dict = {}
    features_by_site: dict = {}

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
            features_by_site[site] = site_features
            ph = _prep_person_hours(site_cohort, site_features)
            site_models[site] = fit_site_logistic(ph, cohort_label, site)
            ph_by_site[site]  = ph
            cohort_by_site[site] = site_cohort
            m = site_models[site]
            print(f"    n={m['n_patients']:,}  vaso_on={m['n_vaso_on']:,}  "
                  f"α={m['coefficients']['intercept']['beta']:+.4f}  "
                  f"P_ref={m['ref_patient_p_t5']:.1%}")
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

    # ── Approach 3: expanded model with ward/ICU-type + hospital fixed effects ──
    # Fits one expanded logistic GLM per site using site-wide centering so
    # effective intercepts are comparable across groups (no reference-shift bias).
    # DL-pools the effective intercepts to get tau2_ward / tau2_hosp.
    # Results feed directly into covariate_variance_explained via structural_models.
    print("\n--- Approach 3: Expanded structural model (ward/ICU-type + hospital dummies) ---")
    structural_models: dict = {}
    for site in site_models:
        print(f"\n  [{site}] fitting expanded model:")
        try:
            structural_models[site] = fit_site_logistic_with_structure(
                ph_by_site[site], cohort_by_site[site], cohort_label, site
            )
            sm = structural_models[site]
            n_wards = len(sm.get("ward_effective_intercepts", {}))
            n_hosps = len(sm.get("hospital_effective_intercepts", {}))
            print(f"    {n_wards} wards  ·  {n_hosps} hospitals  ·  "
                  f"τ²_ward={sm.get('tau2_ward')}  τ²_hosp={sm.get('tau2_hosp')}")
        except Exception as _e:
            import traceback
            print(f"    WARNING: structural model failed for {site}: {_e}")
            traceback.print_exc()
            structural_models[site] = {}
        sm = structural_models.get(site, {})
        plot_group_intercepts(sm, cross_out, cohort_label, site, "ward")
        plot_group_intercepts(sm, cross_out, cohort_label, site, "hospital")
        # Approach 2A/2B broken down by ward/ICU-type and hospital
        plot_approach2_time_by_group(sm, cross_out, cohort_label, site, "ward")
        plot_approach2_nee_by_group(sm, ph_by_site[site], cross_out, cohort_label, site, "ward")
        plot_approach2_time_by_group(sm, cross_out, cohort_label, site, "hospital")
        plot_approach2_nee_by_group(sm, ph_by_site[site], cross_out, cohort_label, site, "hospital")

    # ── Covariate variance explained (Nakagawa & Schielzeth R²) ──────────────
    print("\n--- Covariate variance explained (Nakagawa & Schielzeth R²) ---")
    covar_variance: dict = compute_covariate_variance_explained(
        ph_by_site, site_models, structural_models, me_result
    )
    for site in site_models:
        _cv_out = OUTPUT_ROOT / "output" / f"upload_to_box_{site}" / cohort_label
        plot_covariate_variance_breakdown(covar_variance, _cv_out, cohort_label, site)
        export_covariate_variance_csv(covar_variance, _cv_out, cohort_label, site)

    # ── Location transition heatmaps ───────────────────────────────────────
    # Saved to upload_to_box so they are included in the consolidated report
    # and shared with the coordinating centre.
    # ── DLMM aggregate statistics ─────────────────────────────────────────
    # Federated-safe summary (X'X, X'y, y'y) for continuous practice outcomes.
    # The coordinating script (08) will read these from the packet and run
    # the REML optimization without any patient data leaving this site.
    print("\n--- DLMM aggregate statistics (time-to-vaso, NEE at initiation) ---")
    dlmm_stats = compute_dlmm_stats(cohort_by_site, features_by_site)

    # ── Model variant comparison (v1 original / v2 comorbid / v3 Rhee-OD) ─────
    print("\n--- Model variant comparison (original / +comorbid / +Rhee-OD) ---")
    model_variants = compute_model_variants(
        cohort_label, cohort_by_site, ph_by_site, features_by_site
    )

    # ── CVC stratification: time-to-vaso and NEE at initiation ───────────────
    print("\n--- CVC stratification: time-to-vaso and NEE at initiation ---")
    cvc_strat = compute_cvc_stratification(cohort_by_site, features_by_site)
    for site, df_strat in cvc_strat.items():
        _cvc_out = OUTPUT_ROOT / "output" / f"upload_to_box_{site}" / cohort_label
        _cvc_out.mkdir(parents=True, exist_ok=True)
        _cvc_path = _cvc_out / f"cvc_stratification_{cohort_label}_{site}.csv"
        df_strat.to_csv(_cvc_path, index=False)
        print(f"  Saved: cvc_stratification_{cohort_label}_{site}.csv")

    print("\n--- Location transition heatmaps ---")
    for site in site_models:
        plot_location_transition_heatmap(
            cohort_by_site[site], per_site_out, cohort_label, site
        )
    print("  [pooled] all sites combined:")
    plot_pooled_location_transition_heatmap(cohort_by_site, per_site_out, cohort_label)

    print("\n--- Location transition outcome heatmaps ---")
    for site in site_models:
        plot_location_transition_outcome_heatmaps(
            cohort_by_site[site],
            features_by_site.get(site),
            per_site_out,
            cohort_label,
            site,
        )

    # ── Save variation packets ────────────────────────────────────────────
    # One packet per site, each containing all site models + mixed effects.
    # This lets the HTML report read any one packet and get all site data.
    packet = {
        "site":             SITE_NAME,
        "cohort":           cohort_label,
        "per_site_models":  site_models,          # all sites — Approach 1 shared logistic spec
        "per_site_model":   site_models.get(SITE_NAME, {}),  # back-compat
        "mixed_effects":    me_result,            # cross-site DL pooling (Approach 1)
        "structural_models": structural_models,   # per site: expanded model with ward/hospital dummies (Approach 3)
        "covariate_variance": covar_variance,     # per site: Nakagawa R² decomposition across all 4 levels
        "dlmm_stats":        dlmm_stats,          # per site: DLMM aggregate stats for continuous outcomes
        "model_variants":    model_variants,      # variant comparison: v1_original / v2_comorbid / v3_rhee_od
    }

    # Write into each site's upload directory.
    # Each packet contains only that site's per-site results (structural_models,
    # covariate_variance, DLMM) so that upload_to_box_<SITE> never carries
    # another site's data.  Cross-site fields (per_site_models, mixed_effects)
    # are kept so the HTML report can show the pooled comparison when all
    # packets are combined at the coordinating level.
    for s in site_models:
        s_out = OUTPUT_ROOT / "output" / f"upload_to_box_{s}" / cohort_label
        s_out.mkdir(parents=True, exist_ok=True)
        p = s_out / f"site_variation_packet_{cohort_label}_{s}.json"
        # Per-variant dlmm_stats: filter to this site only
        _mv_s = {}
        for _vk, _vd in model_variants.items():
            _mv_s[_vk] = {
                "bin_cols":   _vd.get("bin_cols", []),
                "n_sites":    _vd.get("n_sites", 0),
                "dl":         _vd.get("dl", {}),
                "dlmm_stats": {s: _vd.get("dlmm_stats", {}).get(s, {})},
            }
        pkt_s = dict(
            packet,
            site=s,
            per_site_model=site_models[s],
            # Filter every per-site dict to this site only
            structural_models={s: structural_models.get(s, {})},
            covariate_variance={s: covar_variance.get(s, {})},
            dlmm_stats={s: dlmm_stats.get(s, {})},
            model_variants=_mv_s,
        )
        with open(p, "w", encoding="utf-8") as fout:
            json.dump(pkt_s, fout, indent=2)

        export_coefficient_table(site_models[s]).to_csv(
            s_out / f"coefficient_table_{cohort_label}_{s}.csv", index=False
        )
        sm_s = structural_models.get(s, {})
        if sm_s.get("ward_effective_intercepts"):
            export_group_intercept_table(sm_s, "ward").to_csv(
                s_out / f"ward_intercepts_{cohort_label}_{s}.csv", index=False
            )
        if sm_s.get("hospital_effective_intercepts"):
            export_group_intercept_table(sm_s, "hospital").to_csv(
                s_out / f"hospital_intercepts_{cohort_label}_{s}.csv", index=False
            )

    pkt_path = per_site_out / f"site_variation_packet_{cohort_label}_{SITE_NAME}.json"
    print(f"\nSaved packet:  {pkt_path}  (+ copies for {list(site_models.keys())})")
    print(f"Cross-site plots: {cross_out}")


def main():
    cohorts = ["sepsis3", "rhee", "rhee_clifpy"] if COHORT_ARG == "both" else [COHORT_ARG]
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
