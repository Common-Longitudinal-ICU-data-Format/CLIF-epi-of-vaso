#!/usr/bin/env python3
"""
05_site_variation_analysis.py  [PER-SITE — run at each participating institution]

Site-specific model of vasopressin use. Everything here uses ONLY this site's patient
data — there is no cross-site pooling in this script (DerSimonian-Laird across sites,
Love plots, pooled coefficient tables etc. happen downstream in the coordinating scripts,
from the per-site packets this script writes).

Terminology: "site" = CLIF site / data pool (SITE_NAME). "ICU type" = canonical
location_type of the patient's ICU at NE start.

The ONE site-specific GLM (person-hours, complementary log-log link):

  vaso_start ~ intercept + age_c + p_f_ratio_c + device_category (Room Air ref)
          + creatinine_c + rrt + platelet_c + bilirubin_c + gcs_c + ICU type
          + rcs(NEE, 4) + rcs(time, 4)

  OUTCOME: the discrete-time hazard of INITIATING vasopressin. Rows are the hours from NE start up to and
  including the first vasopressin hour (hours before vasopressin starts); vaso_start = 1 only in that first
  vasopressin hour. Hours after vasopressin has started are not in the risk set.
  NEE is the PREVIOUS hour's NEE (0 at the first hour): the dose the clinician was looking at before acting.
  This also keeps the vasopressin contribution out of NEE, because the hour before the first vasopressin hour
  contains none. Other covariates are last-observed values at the current hour. `time` is the hour (from NE start) being modelled.

  LINK: complementary log-log, log(-log(1 - h)) = eta, not logit. Because the outcome is a discrete-time
  hazard, cloglog makes this the grouped-time analogue of a Cox proportional-hazards model: exp(beta) is a
  HAZARD ratio (not an odds ratio) and the coefficients do not depend on the width of the hourly interval.
  Everything derived on the latent scale uses the cloglog residual variance pi^2/6 (standard Gumbel), where
  a logit model would use pi^2/3, and the between-group spread is summarised as a median HAZARD ratio (MHR).

  Individual SOFA organ-system components (respiratory: P/F ratio + respiratory
  device category; renal: creatinine + RRT; coagulation: platelets; liver:
  bilirubin; neuro: GCS) replace the aggregate SOFA score, so NEE's mechanical
  overlap with the cardiovascular SOFA axis is excluded; cardiovascular itself is
  omitted (captured by NEE on the right-hand side). Continuous covariates are
  z-scored with this site's own means/SDs. ICU type is dummy-coded against the
  reference (MICU when present, else the largest ICU type); ICU types with fewer
  than MIN_ICU_N_PATIENTS patients are absorbed into the reference.

Derived from that single fit (no additional models):
  - ICU-type effective intercepts  alpha_j = b0 + b_icu_j  (delta-method SE), and their
    within-site DL tau^2 / ICC / MHR across ICU types
  - reference-patient P(vasopressin) vs. time and vs. NEE (per site, and by ICU type)
  - per-covariate-group variance of the linear predictor (Nakagawa & Schielzeth);
    the site-level component (sigma2_site) needs other sites and is left at 0 here

Other per-site outputs: location-transition heatmaps (overall and by outcome).

Outputs: output/upload_to_box_<SITE>/<cohort>/
    site_variation_packet_<cohort>_<SITE>.json
    coefficient_table_<cohort>_<SITE>.csv
    ward_intercepts_<cohort>_<SITE>.csv           (ICU-type effective intercepts)
    ward_intercepts / approach2_* / location_transition_* / transition_outcome_* PNGs

Usage:
  uv run python code/05_site_variation_analysis.py --site UCMC
  uv run python code/05_site_variation_analysis.py --cohort rhee
  uv run python code/05_site_variation_analysis.py --cohort both  # default: sepsis3, rhee, rhee_clifpy
"""

import argparse
import importlib.util
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

# ── GLM link: complementary log-log ──────────────────────────────────────────
# The outcome is a discrete-time hazard (did vasopressin start in THIS hour?), so the
# site GLM uses a complementary log-log link instead of logit:
#
#     cloglog(h) = log(-log(1 - h)) = eta        h = 1 - exp(-exp(eta))
#
# That makes the fit the grouped-time analogue of a Cox proportional-hazards model:
# exp(beta) is a HAZARD ratio, and the coefficients are invariant to the width of the
# hourly interval (a logit coefficient is not).  On the latent scale the cloglog
# residual is a standard Gumbel (extreme-value) draw with variance pi^2/6; a logit
# model would use pi^2/3.  Every ICC / variance-decomposition / median-ratio quantity
# below therefore uses RESID_VAR, and "MHR" is a median hazard ratio.
LINK_NAME = "cloglog"
RESID_VAR = float(np.pi ** 2 / 6)      # Gumbel latent variance (logit would be pi^2/3)


def new_family():
    """Binomial family with the complementary log-log link (one place to change it)."""
    from statsmodels.genmod import families as fam
    return fam.Binomial(link=fam.links.CLogLog())


# Divergence guard. A cloglog IRLS that walks into the exp-overflow region comes back with
# |beta| ~ 1e14-1e17 and a NaN log-likelihood; a legitimate but near-collinear spline term
# sits far below that (NU reaches 2.3e3 at its optimum). So OVERFLOW_BETA only has to
# separate those two regimes, and SEPARATION_BETA is a softer warning threshold.
OVERFLOW_BETA = 1e6
# Only terms outside the spline bases are checked for separation. The un-normalized Harrell
# basis in _rcs_basis() leaves the NEE spline coefficients around 1e3 at EVERY site (UCMC
# 1.3e3, RUSH 2.8e3, NU 2.3e3): its near-collinear columns carry large offsetting values
# that cancel to a well-behaved eta. That is a property of the basis rather than separation,
# so including those terms would make the warning fire everywhere and mean nothing.
SEPARATION_BETA = 20.0
SEPARATION_SKIP_PREFIXES = ("rcs_",)
# Relative-gradient tolerance: max |score_j| x se_j at the returned coefficients. Zero at a
# true optimum and scale-free, so it is comparable across fitting routes. Newton on the QR
# basis scores ~1e-13; IRLS stops at its own looser working precision (~1e-3) even when it
# reports success; a diverged IRLS scores 1e10+ and an optimizer stalled on its starting
# values scores 1e4. The threshold sits below IRLS's precision deliberately, so that the
# strictly-better Newton optimum is the one used rather than merely a nearby point.
REL_GRAD_TOL = 1e-4


def fisher_cov(X, beta):
    """Expected-information covariance of a binomial cloglog GLM: (X' W X)^-1, with
    W = diag( (dmu/deta)^2 / (mu (1 - mu)) ) and dmu/deta = exp(eta - exp(eta)).

    This is the covariance IRLS reports, and it reproduces statsmodels' IRLS standard
    errors to ~1e-8. It is computed explicitly because the generic Newton path returns an
    OBSERVED-information covariance instead: cloglog is not a canonical link, so the two
    differ by a few tenths of a percent in these samples, and using this one keeps the
    standard errors the same whichever route converged.
    """
    Xa = np.asarray(X, dtype=float)
    eta = np.clip(Xa @ np.asarray(beta, dtype=float), -500.0, 30.0)
    ex = np.exp(eta)
    mu = 1.0 - np.exp(-ex)
    dmu = np.exp(eta - ex)
    v = np.clip(mu * (1.0 - mu), 1e-12, None)
    w = dmu ** 2 / v
    return np.linalg.pinv(Xa.T @ (Xa * w[:, None]))


class _CloglogFit:
    """View over an accepted cloglog fit, in the original design parameterization.

    Two things are fixed up here. The fit may have been performed on a reparameterized
    design X @ T (eta = (X T) b = X (T b)), so the coefficients are mapped back as
    beta = T b. And the covariance is recomputed from the expected information at the
    optimum by fisher_cov(), the standard GLM convention, rather than taken from whichever
    optimizer happened to run. Everything else -- log-likelihood, AIC / BIC, deviance,
    residuals, fitted values -- depends only on eta, which the reparameterization leaves
    unchanged, so it is delegated to the underlying fit untouched.
    """

    def __init__(self, fit, beta, cov, names=None):
        self._fit = fit
        beta = np.asarray(beta, dtype=float)
        cov = np.asarray(cov, dtype=float)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        self._cov = pd.DataFrame(cov, index=names, columns=names) if names is not None else cov
        self.params = pd.Series(beta, index=names) if names is not None else beta
        self.bse = pd.Series(se, index=names) if names is not None else se

    def cov_params(self):
        return self._cov

    def __getattr__(self, name):      # llf / aic / bic / deviance / resid_* / fittedvalues / ...
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._fit, name)


def fit_cloglog(Y, X, maxiter: int = 200, col_names: list = None):
    """Fit the binomial cloglog GLM for design matrix X, robustly.

    cloglog is far less forgiving than logit: eta enters the likelihood through
    exp(exp(eta)), so an IRLS step that overshoots lands in the overflow region and the
    iteration never recovers -- the fit returns |beta| ~ 1e14 with a NaN log-likelihood.
    That happens on real site data (NU and MIMIC-all; UCMC / RUSH / MIMIC-anchoryears are
    fine under plain IRLS), and three things were ruled out as the cause:

      * not an ill-posed optimum -- at NU the fitted eta stays within [-10.7, 2.5], so no
        person-hour has a hazard anywhere near 1;
      * not mere ill-conditioning -- column-scaling the design (cond 3e9 -> 1e5) and full
        QR orthonormalization (cond = 1) both still diverge under IRLS;
      * not fixable by warm-starting IRLS from the logit coefficients.

    The cause is that statsmodels IRLS has no step-halving or line search. The fix is Newton
    on the QR-orthonormalized design (X = Q R; fit on Q, then map back with beta = R^-1 b, a
    pure reparameterization). At NU that reaches a genuine optimum from a cold start --
    relative gradient ~1e-12 and the highest log-likelihood of every route tried -- where
    Newton on the raw design scores 279 and lbfgs on the raw design stalls on its starting
    values. Both ingredients matter: IRLS still diverges in the QR basis, and Newton still
    fails in the raw one. This is the primary route for every site, so that one algorithm
    fits them all; plain IRLS, warm-started IRLS and lbfgs remain as fallbacks for a design
    where the QR is unavailable (rank deficiency) or Newton fails.

    Every candidate must be converged, finite, outside the overflow region and *stationary*
    by the relative-gradient test: lbfgs in particular reports success without having moved
    off its starting values, and fit.converged does not catch that.

    col_names labels the columns of X when X is a plain array; it is only used to keep the
    spline terms out of the separation warning.

    Standard errors always come from the expected information at the optimum (fisher_cov),
    the usual GLM convention, so they do not depend on which route converged.

    Returns a fit whose params / bse / cov_params() are in the original X parameterization.
    Prints the route used when it was not the first; if nothing is usable the last attempt
    is returned with a warning and the caller's `diagnostics.converged` stays False.
    """
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as fam

    names = list(X.columns) if hasattr(X, "columns") else None
    labels = names if names is not None else (list(col_names) if col_names is not None else None)
    Xa = np.asarray(X, dtype=float)

    # QR of the design, skipped if X is rank deficient (R would not be invertible).
    QR = None
    try:
        q_, r_ = np.linalg.qr(Xa)
        d_ = np.abs(np.diag(r_))
        if d_.size and d_.min() > 1e-10 * max(d_.max(), 1.0):
            QR = (q_, np.linalg.inv(r_))
    except np.linalg.LinAlgError:
        pass

    _warm: dict = {}

    def warm_params():
        """Logit coefficients on the raw design: cheap, stable, and a good start point."""
        if "raw" not in _warm:
            _warm["raw"] = np.asarray(
                GLM(Y, Xa, family=fam.Binomial()).fit(maxiter=maxiter).params, dtype=float)
        return _warm["raw"]

    def rel_gradient(model, fit):
        """max |score_j| x se_j in the space the fit was performed in; ~0 at an optimum."""
        try:
            g = np.asarray(model.score(np.asarray(fit.params, dtype=float)), dtype=float)
            se = np.asarray(fit.bse, dtype=float)
            return float(np.max(np.abs(g) * se))
        except Exception:
            return float("nan")

    # QR-Newton leads: it is the only route that reaches a stationary point at every site,
    # so trying the cheaper ones first would just pay for two discarded IRLS fits before
    # landing here anyway, and fitting every site with one algorithm is worth more than the
    # runtime. The rest remain as fallbacks for a design where the QR is unavailable
    # (rank deficiency) or Newton fails.
    routes = []
    if QR is not None:
        routes.append(("Newton on the QR-orthonormalized design", "qr_newton"))
    routes += [("IRLS", "irls"), ("IRLS, logit warm start", "irls_warm")]
    if QR is not None:
        routes.append(("lbfgs on the QR-orthonormalized design", "qr_lbfgs"))
    routes.append(("lbfgs, logit warm start", "lbfgs_warm"))

    last = None
    for i, (label, kind) in enumerate(routes):
        if kind == "qr_newton":
            design, T, kw = QR[0], QR[1], {"method": "newton", "maxiter": 5000}
        elif kind == "qr_lbfgs":
            design, T, kw = QR[0], QR[1], {"method": "lbfgs", "maxiter": 5000}
        elif kind == "irls":
            design, T, kw = Xa, None, {"maxiter": maxiter}
        elif kind == "irls_warm":
            design, T, kw = Xa, None, {"maxiter": maxiter * 2, "start_params": warm_params()}
        else:
            design, T, kw = Xa, None, {"method": "lbfgs", "maxiter": 5000,
                                       "start_params": warm_params()}

        model = GLM(Y, design, family=new_family())
        try:
            raw = model.fit(**kw)
        except Exception as exc:
            print(f"  cloglog fit [{label}] raised {type(exc).__name__}: {str(exc)[:90]}")
            continue

        rel = rel_gradient(model, raw)
        b = np.asarray(raw.params, dtype=float)
        beta = (np.asarray(T, dtype=float) @ b) if T is not None else b
        # statsmodels only sets `converged` on the IRLS path; the generic optimizers leave it
        # None, which must read as "not reported" rather than "failed" -- the relative-gradient
        # test below is the stronger criterion and decides those cases.
        conv = getattr(raw, "converged", None)
        ok = (conv is not False
              and np.all(np.isfinite(beta))
              and np.isfinite(getattr(raw, "llf", np.nan))
              and np.abs(beta).max() < OVERFLOW_BETA
              and np.isfinite(rel) and rel < REL_GRAD_TOL)
        if ok:
            fit = _CloglogFit(raw, beta, fisher_cov(Xa, beta), names)
            if i:
                print(f"  cloglog fit used fallback route [{label}] (rel. gradient {rel:.2g})")
            mag = np.abs(beta)
            if labels is not None and len(labels) == len(mag):
                keep = [j for j, nm in enumerate(labels)
                        if not str(nm).startswith(SEPARATION_SKIP_PREFIXES)]
                worst, who = max(((mag[j], labels[j]) for j in keep), default=(0.0, None))
            else:
                worst, who = float(mag.max()), None
            if worst >= SEPARATION_BETA:
                print(f"  NOTE: largest non-spline |beta| = {worst:.3g}"
                      f"{f' ({who})' if who else ''} - that covariate is probably separating, so "
                      "its own coefficient is not interpretable (the intercept and the other "
                      "terms still are).")
            return fit
        mx = np.abs(beta).max() if np.all(np.isfinite(beta)) else float("inf")
        print(f"  cloglog fit [{label}] not usable (max|beta|={mx:.3g}, "
              f"converged={conv}, rel. gradient={rel:.3g})")
        # Kept only so the caller gets a result object (with converged=False) if nothing works.
        last = _CloglogFit(raw, beta, fisher_cov(Xa, beta), names) if np.all(np.isfinite(beta)) else raw

    if last is None:
        raise RuntimeError("cloglog GLM could not be fit: every route raised (see messages above)")
    print("  WARNING: cloglog GLM did not converge by any route - this site's coefficients are "
          "NOT usable; check for a separating covariate before pooling this site.")
    return last


def inv_link(eta):
    """cloglog inverse link: h = 1 - exp(-exp(eta)).  Clipped so exp cannot overflow."""
    e = np.clip(np.asarray(eta, dtype=float), -500.0, 30.0)
    return 1.0 - np.exp(-np.exp(e))


def link(h):
    """cloglog link: eta = log(-log(1 - h))."""
    q = np.clip(np.asarray(h, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(-np.log(1.0 - q))


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
# If this site has no (or too few) Room Air hours — e.g. MIMIC records a device only while
# respiratory support is running — the reference device falls back to the first of these
# that has enough patients; the choice is stored in the model as "device_ref".
_DEVICE_REF_FALLBACK = ["Nasal Cannula", "Face Mask", "High Flow NC", "CPAP", "NIPPV", "IMV"]

# Binary baseline covariates used as fixed effects in the GLM.
# All have reference value 0 (no comorbidity / no procedure).
# NOTE: comorbid_liver_disease is deliberately excluded here — it is collinear with
# comorbid_cirrhosis.  comorbid_liver_nocirrh (non-cirrhotic liver disease) is used
# instead to form a mutually exclusive pair.
# Must stay in sync with _BIN_COLS_MODELS in 08_cross_site_variation_analysis.py.
_BIN_COLS_MODELS = []

# Original comorbidity list preserved here for compute_model_variants (v2_comorbid variant)
_COMORBID_COLS = [
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

# Reference ICU type for dummy encoding (most common medical ICU).
_ICU_TYPE_REF = "MICU"

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
    intercept estimates -> between-group variance tau2, ICC, and MHR (median
    hazard ratio) on the cloglog residual variance RESID_VAR = pi^2/6.
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
    icc = tau2 / (tau2 + RESID_VAR)
    mhr = float(np.exp(np.sqrt(2 * tau2) * 0.6745)) if tau2 > 0 else 1.0

    return {"tau2": tau2, "icc": icc, "mhr": mhr, "pooled_alpha": theta_re, "se": se_re, "k": K, "q": Q}


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
    """Harrell-style percentile knots. If percentile knots collide (NEE: >35% of patient-hours are
    NEE = 0 after pressors are weaned, so the 5th and 35th percentiles are both 0), the spline's
    nonlinear terms become perfectly collinear and the covariance matrix is singular. In that case
    the knots are placed on the distribution of the positive values."""
    quantiles = {4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95], 3: [10, 50, 90]}.get(df, [5, 35, 65, 95])
    x = x[np.isfinite(x)]
    knots = np.percentile(x, quantiles)
    if len(np.unique(knots)) < len(knots) and (x > 0).sum() > 0:
        knots = np.percentile(x[x > 0], quantiles)
    return knots


# ── Data prep ─────────────────────────────────────────────────────────────────
# Labs/GCS/p_f_ratio are recorded sparsely (not every hour), unlike
# device_category/fio2_set which are already forward-filled at extraction
# time (persistent ventilator/device settings). Forward-fill them here per
# stay_id so they're usable as hourly time-varying covariates instead of
# dropping nearly every hour with no fresh measurement.
_LOCF_COLS = ["p_f_ratio", "creatinine", "platelet", "bilirubin", "gcs"]


# NEE enters the model as the PREVIOUS hour's value (what the clinician could see; no vasopressin term yet).
# The other covariates (labs / GCS / P-F / device / RRT) stay at the current hour: they are last-observed-value
# carried forward, so they are not affected by the vasopressin start, and lagging them would drop ~40% of
# initiators whose first P/F or device record falls in the hour of initiation. Set to the full list to lag all.
_LAGGED_COLS = ["nee"]
OUTCOME_NAME = "vaso_start_hazard"


def _prep_person_hours(cohort: pd.DataFrame, features: pd.DataFrame, hazard: bool = True) -> pd.DataFrame:
    """Person-hours for the initiation-hazard GLM.

    1. vaso_on / vaso_start / at-risk flags are computed on the FULL hourly grid, before any row is lagged or
       dropped (a dropped hour must not make a later 'already on' hour look like the first one).
    2. NEE is replaced by its value in the previous hour (0 at the first hour).
    3. With hazard=True (default) only at-risk hours are kept (hours up to and including the first
       vasopressin hour) and the outcome column `vaso_on` is set to vaso_start.
    """
    cols = (["stay_id", "time_hour", "vaso_dose", "nee", "rrt", "device_category"]
            + _LOCF_COLS)
    ph = features[cols].copy()
    ph = ph.sort_values(["stay_id", "time_hour"])
    ph["vaso_on"] = (ph["vaso_dose"] > 0).astype(int)
    cum_before = ph.groupby("stay_id")["vaso_on"].cumsum() - ph["vaso_on"]
    ph["vaso_start"] = ((ph["vaso_on"] == 1) & (cum_before == 0)).astype(int)
    ph["_at_risk_start"] = (cum_before == 0).astype(int)

    ph[_LOCF_COLS] = ph.groupby("stay_id")[_LOCF_COLS].ffill()
    ph[_LAGGED_COLS] = ph.groupby("stay_id")[_LAGGED_COLS].shift(1)        # state in the previous hour
    # hour 0 = NE start: nothing was running in the hour before, so previous-hour NEE is 0
    ph["nee"] = ph["nee"].fillna(0) if "nee" in _LAGGED_COLS else ph["nee"]

    # Merge static baseline features (age + binary comorbidities)
    _static = ["stay_id", "age"] + [c for c in _BIN_COLS_MODELS if c in cohort.columns]
    ph = ph.merge(cohort[_static], on="stay_id", how="left")
    # Binary cols default to 0 when absent (e.g. procedures file missing at a site)
    for c in _BIN_COLS_MODELS:
        if c in ph.columns:
            ph[c] = ph[c].fillna(0).astype(int)

    # Merge ICU type (canonical, from location_type / location_category)
    if "icu_type" not in ph.columns:
        _loc_cols = [c for c in ["location_type", "location_category"]
                     if c in cohort.columns]
        if _loc_cols:
            _loc_df = cohort[["stay_id"] + _loc_cols].copy()
            _loc_df["icu_type"] = _canon_icu_series(
                _effective_location(_loc_df, *(_loc_cols + [""] if len(_loc_cols) < 2 else _loc_cols))
            ).fillna("Other ICU")
            ph = ph.merge(_loc_df[["stay_id", "icu_type"]], on="stay_id", how="left")
        ph["icu_type"] = ph.get("icu_type", pd.Series("Other ICU", index=ph.index)).fillna("Other ICU")

    # Merge hospital_type (static per patient; useful for pooled model)
    if "hospital_type" not in ph.columns and "hospital_type" in cohort.columns:
        ph = ph.merge(cohort[["stay_id", "hospital_type"]], on="stay_id", how="left")
        ph["hospital_type"] = ph["hospital_type"].fillna("unknown")

    ph = ph.dropna(subset=["nee", "age", "rrt", "device_category"] + _LOCF_COLS).copy()
    if hazard:
        ph = _filter_ph_for_outcome(ph, "start")
    return ph


# ── Shared design matrix (per-site refits and the pooled cross-site GLM) ─────
def _build_design_matrix(
    df: pd.DataFrame,
    nee_knots: np.ndarray = None,
    time_knots: np.ndarray = None,
    device_categories: list = None,
    icu_categories: list = None,      # ICU type dummies; None → derive from df["icu_type"]
    means: dict = None,
    sds: dict = None,
    present_bin_cols: list = None,
    candidate_bin_cols: list = None,  # if None, uses _BIN_COLS_MODELS
    extra_cont_cols: list = None,     # additional continuous covariates to standardise & include
):
    """vaso_start ~ intercept + age_c + p_f_ratio_c + device_category + icu_type + creatinine_c
    + rrt + platelet_c + bilirubin_c + gcs_c
    + [binary comorbidities: cirrhosis, liver_nocirrh, cad, aortic_stenosis, carotid_stenosis]
    + rcs(NEE,4) + rcs(time,4).

    icu_type dummies are included when df["icu_type"] is present; reference = _ICU_TYPE_REF ("MICU").

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

    # ICU type dummies (reference = _ICU_TYPE_REF, typically "MICU")
    _has_icu = "icu_type" in df.columns
    if _has_icu:
        if icu_categories is None:
            _all_icu = set(df["icu_type"].dropna().unique()) - {_ICU_TYPE_REF}
            icu_categories = sorted(_all_icu)
        icu_cols = {icu: (df["icu_type"].values == icu).astype(np.float64)
                    for icu in icu_categories}
    else:
        icu_categories = []
        icu_cols = {}

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
                 + [f"device_{d}" for d in device_categories]
                 + [f"icu_{icu}" for icu in icu_categories]
                 + nee_cols + time_cols)

    X = np.column_stack(
        [np.ones(len(df))]
        + [centered[c] for c in _all_cont]
        + [df["rrt"].values.astype(np.float64)]
        + [df[c].values.astype(np.float64) for c in present_bin_cols]
        + [device_cols[d] for d in device_categories]
        + [icu_cols[icu] for icu in icu_categories]
        + [nee_b, time_b]
    ).astype(np.float64)

    return X, col_names, means, sds, nee_knots, time_knots, device_categories, present_bin_cols, icu_categories


def stabilize_devices(df: pd.DataFrame) -> tuple:
    """Pick the reference device and merge device categories that would cause (quasi-)separation.

    A non-reference category with fewer than MIN_ICU_N_PATIENTS patients, or with no vasopressin
    hours (or only vasopressin hours), is relabelled "Other"; if "Other" itself is then unusable it
    is absorbed into the reference. Returns (df, device_ref, merged_categories, device_categories).
    """
    df = df.copy()
    n_pat = df.groupby("device_category")["stay_id"].nunique()
    ref = _DEVICE_REF if n_pat.get(_DEVICE_REF, 0) >= MIN_ICU_N_PATIENTS else next(
        (d for d in _DEVICE_REF_FALLBACK if n_pat.get(d, 0) >= MIN_ICU_N_PATIENTS), n_pat.idxmax())

    def _bad(cat_series):
        out = []
        g = df.groupby("device_category")["vaso_on"].agg(["sum", "size"])
        for d in g.index:
            if d == ref:
                continue
            if n_pat.get(d, 0) < MIN_ICU_N_PATIENTS or g.loc[d, "sum"] == 0 or g.loc[d, "sum"] == g.loc[d, "size"]:
                out.append(d)
        return out

    merged = _bad(df["device_category"])
    if merged:
        df["device_category"] = df["device_category"].where(~df["device_category"].isin(merged), "Other")
        g = df.groupby("device_category")["vaso_on"].agg(["sum", "size"])
        n_oth = df.loc[df["device_category"] == "Other", "stay_id"].nunique()
        if "Other" in g.index and ref != "Other" and (n_oth < MIN_ICU_N_PATIENTS or g.loc["Other", "sum"] in (0, g.loc["Other", "size"])):
            df["device_category"] = df["device_category"].where(df["device_category"] != "Other", ref)
    cats = sorted(set(df["device_category"].unique()) - {ref})
    return df, ref, merged, cats


# ── Outcome selector ──────────────────────────────────────────────────────────
def _filter_ph_for_outcome(ph: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Return person-hours appropriate for the given outcome model.

    outcome="on":    all person-hours; outcome = vaso_on (vasopressin being administered).
    outcome="start": only at-risk hours (before or at first vasopressin hour);
                     outcome = vaso_start (first hour of initiation, at most 1 per patient).

    In both cases the returned df has a column named "vaso_on" so downstream
    fitting functions (fit_site_logistic, fit_site_logistic_with_structure) work
    without modification.
    """
    if outcome == "start":
        ph = ph[ph["_at_risk_start"] == 1].copy()
        ph["vaso_on"] = ph["vaso_start"]
    return ph


# ── Approach 1 & 2: Per-site discrete-time hazard GLM (cloglog) ───────────────
_FIT_CACHE: dict = {}      # (cohort, site) -> statsmodels fit + design matrix, for run_site_diagnostics


def fit_site_logistic(
    ph: pd.DataFrame,
    cohort_label: str,
    site: str,
    bin_cols: list = None,
    extra_cont_cols: list = None,
) -> dict:
    """Fit the shared discrete-time hazard spec (see _build_design_matrix) with a
    complementary log-log link, independently on this group's person-hours (its own
    knots/scaling/device categories).

    bin_cols:        list of binary covariate names to include; defaults to _BIN_COLS_MODELS.
    extra_cont_cols: list of additional continuous covariate names (already in ph);
                     they are standardised and included alongside the base _CONT_COLS.
    """
    _cand  = bin_cols if bin_cols is not None else _BIN_COLS_MODELS
    _extra = list(extra_cont_cols) if extra_cont_cols else []
    _bin_present = [c for c in _cand if c in ph.columns]
    _extra_present = [c for c in _extra if c in ph.columns]
    _opt_cols = [c for c in ["icu_type", "hospital_type"] if c in ph.columns]
    df = ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt", "device_category"]
            + _CONT_COLS + _extra_present + _bin_present + _opt_cols].dropna(
                subset=["nee", "rrt", "device_category"] + _CONT_COLS).copy()

    # ICU-type dummies: only types with >= MIN_ICU_N_PATIENTS patients get a dummy
    # (smaller ones are absorbed into the reference); reference = MICU if present,
    # otherwise the largest ICU type.
    icu_ref, icu_cats_fit, icu_absorbed = None, None, []
    if "icu_type" in df.columns:
        _icu_n = df.groupby("icu_type")["stay_id"].nunique().sort_values(ascending=False)
        _eligible = [g for g, n in _icu_n.items() if n >= MIN_ICU_N_PATIENTS]
        icu_absorbed = [g for g in _icu_n.index if g not in _eligible]
        if _eligible:
            icu_ref = _ICU_TYPE_REF if _ICU_TYPE_REF in _eligible else _eligible[0]
        icu_cats_fit = [g for g in _eligible if g != icu_ref]

    df, device_ref, device_merged, _dev_cats = stabilize_devices(df)
    X, col_names, means, sds, nee_knots, time_knots, device_categories, present_bin_cols, icu_categories = (
        _build_design_matrix(df, candidate_bin_cols=_cand, extra_cont_cols=_extra_present,
                             icu_categories=icu_cats_fit, device_categories=_dev_cats)
    )
    Y = df["vaso_on"].values.astype(np.float64)

    fit = fit_cloglog(Y, X, maxiter=200, col_names=col_names)
    _FIT_CACHE[(cohort_label, site)] = {"fit": fit, "X": X, "Y": Y, "col_names": col_names, "df": df}

    # exp(beta) under cloglog is a hazard ratio (ratio of hourly initiation hazards),
    # not an odds ratio — the key is named "hr" so downstream readers cannot mislabel it.
    coef_out = {
        name: {"beta": float(fit.params[i]), "se": float(fit.bse[i]),
               "hr":   float(np.exp(fit.params[i]))}
        for i, name in enumerate(col_names)
    }

    # ── Inline diagnostics (added to result dict; no external dependencies) ──
    _diag: dict = {
        "converged":    bool(getattr(fit, "converged", True)),
        "n_iter":       int(getattr(fit, "nit", -1)),
        "aic":          float(fit.aic),
        "bic":          float(fit.bic),
        "outcome_rate": float(Y.mean()),
    }
    # NEE SMD (vaso_on=1 vs 0)
    try:
        _nee = df["nee"].values
        _va  = _nee[Y == 1]; _ct = _nee[Y == 0]
        _sd  = np.sqrt((_va.var() + _ct.var()) / 2)
        _diag["smd_nee"] = float((_va.mean() - _ct.mean()) / _sd) if _sd > 0 else 0.0
    except Exception:
        _diag["smd_nee"] = None
    # VIF for continuous covariates (using correlation matrix determinant ratio)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
        _Xdf = X[:, 1:]                                 # drop intercept col
        _vif_dict = {}
        for _j, _nm in enumerate(col_names[1:min(6, len(col_names))]):  # first 5 non-intercept
            _vif_dict[_nm] = float(_vif(_Xdf, _j))
        _diag["vif_top5"] = _vif_dict
    except Exception:
        _diag["vif_top5"] = {}
    # Residual summary (deviance residuals)
    try:
        _resid = fit.resid_deviance
        _diag["residuals"] = {
            "mean":   float(_resid.mean()),
            "sd":     float(_resid.std()),
            "q25":    float(np.percentile(_resid, 25)),
            "q75":    float(np.percentile(_resid, 75)),
        }
    except Exception:
        _diag["residuals"] = {}

    result = {
        "site":              site,
        "cohort":            cohort_label,
        "n_patients":        int(df["stay_id"].nunique()),
        "n_ph_rows":         int(len(df)),
        "n_vaso_on":         int(Y.sum()),
        "outcome":           OUTCOME_NAME,          # hazard of initiating vasopressin
        "link":              LINK_NAME,             # complementary log-log
        "resid_var":         RESID_VAR,             # pi^2/6, the cloglog latent residual variance
        "covariate_timing":  "nee_previous_hour",
        "means":             means,
        "sds":               sds,
        "device_categories": device_categories,
        "device_ref":        device_ref,
        "device_merged":     device_merged,
        "present_bin_cols":  present_bin_cols,
        "extra_cont_cols":   _extra_present,
        "nee_knots":         nee_knots.tolist(),
        "time_knots":        time_knots.tolist(),
        "coefficients":      coef_out,
        "col_names":         col_names,
        "cov":               np.asarray(fit.cov_params()).tolist(),   # for downstream pooling / Love plot
        "icu_ref":           icu_ref,
        "icu_categories":    icu_categories,
        "icu_absorbed":      icu_absorbed,
        "diagnostics":       _diag,
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

    return inv_link(lp)


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

    return inv_link(lp)


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
    ax.set_ylabel("P(vasopressin starts in the hour)", fontsize=11)
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

    ax.set_xlabel("NEE of the previous hour (mcg/kg/min)", fontsize=11)
    ax.set_ylabel("P(vasopressin starts in the hour)", fontsize=11)
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
    "ICU type":        lambda c: c.startswith("icu_"),
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


def build_icu_structure(model: dict, ph: pd.DataFrame) -> dict:
    """ICU-type effective intercepts derived from the ONE site GLM (no extra fit).

        alpha_j = b_intercept + b_icu_j,   SE_j = sqrt(Var(b0) + Var(b_j) + 2 Cov(b0, b_j))

    All intercepts refer to the same reference patient (this site's mean covariates).
    Returns a dict shaped like the old "structural model" (ward == ICU type) so the
    ward plots, predict_p_structural and downstream readers keep working. Hospital
    effects are not part of the model, so hospital_* fields are empty.
    """
    ref = model.get("icu_ref")
    if ref is None:
        return {}
    coef  = model["coefficients"]
    names = model["col_names"]
    cov   = np.asarray(model["cov"])
    i0    = names.index("intercept")
    npt   = ph.groupby("icu_type")["stay_id"].nunique().to_dict() if "icu_type" in ph.columns else {}

    ei = {ref: {"alpha": coef["intercept"]["beta"], "se": coef["intercept"]["se"],
                "n_patients": int(npt.get(ref, 0)), "is_reference": True, "stable": True}}
    coefs = dict(coef)
    for w in model["icu_categories"]:
        iw    = names.index(f"icu_{w}")
        alpha = float(coef["intercept"]["beta"] + coef[f"icu_{w}"]["beta"])
        se    = float(np.sqrt(max(0.0, cov[i0, i0] + cov[iw, iw] + 2 * cov[i0, iw])))
        ei[w] = {"alpha": alpha, "se": se, "n_patients": int(npt.get(w, 0)),
                 "is_reference": False, "stable": bool(se < MAX_STABLE_INTERCEPT_SE)}
        coefs[f"ward_{w}"] = coef[f"icu_{w}"]          # alias read by predict_p_structural
    for g, v in ei.items():
        print(f"    ICU type [{g}] n={v['n_patients']:,}  alpha={v['alpha']:+.4f} (SE={v['se']:.3f})"
              + ("  [reference]" if v["is_reference"] else "")
              + ("" if v["stable"] else "  [UNSTABLE - likely separation]"))
    if model.get("icu_absorbed"):
        print(f"    ICU types absorbed into reference (<{MIN_ICU_N_PATIENTS} patients): {model['icu_absorbed']}")

    out = {
        "site": model["site"], "cohort": model["cohort"],
        "n_patients": model["n_patients"], "n_ph_rows": model["n_ph_rows"],
        "n_vaso_on": model["n_vaso_on"],
        "means": model["means"], "sds": model["sds"],
        "device_categories": model["device_categories"],
        "nee_knots": model["nee_knots"], "time_knots": model["time_knots"],
        "ward_levels": list(model["icu_categories"]), "ward_ref": ref,
        "hosp_levels": [], "hosp_ref": None,
        "coefficients": coefs,
        "ward_effective_intercepts": ei,
        "hospital_effective_intercepts": {},
        "tau2_hosp": None, "icc_hosp": None, "mhr_hosp": None,
    }
    stable = {g: v for g, v in ei.items() if v["stable"] and v["se"] > 0}
    dl = _dl_pool(np.array([v["alpha"] for v in stable.values()]),
                  np.array([v["se"] for v in stable.values()]) ** 2) if len(stable) >= 2 else {}
    if dl:
        out.update({"tau2_ward": dl["tau2"], "icc_ward": dl["icc"], "mhr_ward": dl["mhr"]})
        print(f"    Within-site DL across ICU types: tau2={dl['tau2']:.4f}  ICC={dl['icc']:.3f}  "
              f"MHR={dl['mhr']:.3f}  (k={dl['k']})")
    else:
        out.update({"tau2_ward": None, "icc_ward": None, "mhr_ward": None})
        print("    Within-site DL across ICU types: <2 stable ICU types — tau2 not estimable")
    return out


def compute_covariate_variance_explained(
    ph_by_site: dict,
    site_models: dict,
    structural_models: dict,
    me_result: dict,
) -> dict:
    """Variance of each covariate group's linear-predictor contribution, per site.

    Every term of the site GLM is in a group (see _COVAR_GROUPS), ICU type included (its dummies are
    ordinary fixed effects). For group g:   σ²_g = Var_i( Σ_{j∈g} β_j x_ij )   over this site's patient-hours.

    Total latent variance = Σ_g σ²_g + σ²_site + π²/6
      • σ²_site  — between-site variance (DL τ² of the sites' reference-patient cloglog intercepts). It
                   cannot be estimated inside one site's model, so it is 0 here and added by 08.
      • π²/6 ≈ 1.64 — variance of the standard Gumbel distribution, the cloglog latent-variable residual
                      (a logit model would use π²/3 ≈ 3.29 here)

    R²_marginal    = Σ_g σ²_g / σ²_total
    R²_conditional = (Σ_g σ²_g + σ²_site) / σ²_total

    Reference: Nakagawa & Schielzeth (2013) Methods Ecol Evol 4:133-142. Σ_g Var(η_g) ignores the
    covariances between groups, so the group shares only approximately partition Var(η).
    """
    resid_var = RESID_VAR            # pi^2/6: cloglog latent residual variance
    results: dict = {}

    for site in site_models:
        if site not in ph_by_site:
            continue
        model = site_models[site]
        ph = ph_by_site[site].copy()
        _pbc = model.get("present_bin_cols", [])
        _bin_in_ph = [c for c in _pbc if c in ph.columns]
        _icu_col = ["icu_type"] if "icu_type" in ph.columns else []
        ph = (ph[["stay_id", "time_hour", "vaso_on", "nee", "rrt",
                   "device_category"] + _CONT_COLS + _bin_in_ph + _icu_col].dropna(
                       subset=["nee", "rrt", "device_category"] + _CONT_COLS).copy())
        if ph.empty:
            continue
        ph["device_category"] = ph["device_category"].where(
            ~ph["device_category"].isin(model.get("device_merged", [])), "Other")

        try:
            X, col_names, _, _, _, _, _, _, _ = _build_design_matrix(
                ph,
                nee_knots=np.array(model["nee_knots"]),
                time_knots=np.array(model["time_knots"]),
                means=model["means"],
                sds=model["sds"],
                device_categories=model["device_categories"],
                present_bin_cols=_pbc,
                icu_categories=model.get("icu_categories", []),
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

        # Between-site variance needs other sites: 0 inside a single-site model, added by 08.
        sigma2_site = float(me_result.get("dl_tau2") or 0.0)
        sigma2_total = sigma2_fixed + sigma2_site + resid_var

        def _pct(v: float) -> float:
            return v / sigma2_total * 100 if sigma2_total > 0 else 0.0

        results[site] = {
            "n_patients":    model["n_patients"],
            "group_sigma2":  group_sigma2,
            "sigma2_fixed":  sigma2_fixed,
            "sigma2_site":   sigma2_site,
            "sigma2_resid":  resid_var,
            "sigma2_total":  sigma2_total,
            "r2_marginal":   sigma2_fixed / sigma2_total if sigma2_total > 0 else float("nan"),
            "r2_conditional": (sigma2_fixed + sigma2_site) / sigma2_total if sigma2_total > 0 else float("nan"),
            "pct_groups":    {g: _pct(v) for g, v in group_sigma2.items()},
            "pct_site":      _pct(sigma2_site),
            "pct_resid":     _pct(resid_var),
        }
        print(f"  [{site}] R²_marginal={results[site]['r2_marginal']:.3f}  "
              f"R²_conditional={results[site]['r2_conditional']:.3f}")

    return results


def plot_group_intercepts(
    structural_model: dict, out_dir: Path, cohort_label: str, site: str, label: str
) -> None:
    """Forest plot of each ward/ICU-type or hospital's effective cloglog intercept
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
    ax.set_xlabel("Effective complementary log-log intercept ± 95% CI\n(ref patient at site-wide mean covariates)", fontsize=9)

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
    ax.set_ylabel("P(vasopressin starts in the hour)", fontsize=11)
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

    ax.set_xlabel("NEE of the previous hour (mcg/kg/min)", fontsize=11)
    ax.set_ylabel("P(vasopressin starts in the hour)", fontsize=11)
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
    # The cohort's first_vaso_time covers the whole hospitalization, including
    # vasopressin that started before t=0, so it is not interchangeable with an
    # on-grid initiation hour; the grid is the right source here.
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
    """One row per fitted-model term: beta, SE, hazard ratio, 95% CI, p-value.

    The link is cloglog, so exp(beta) is a hazard ratio for starting vasopressin in
    the hour, not an odds ratio."""
    from scipy.stats import norm

    rows = []
    for term, c in model["coefficients"].items():
        beta, se = c["beta"], c["se"]
        z = beta / se if se else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append({
            "term": term, "beta": beta, "se": se, "hr": c["hr"],
            "hr_ci_lo95": float(np.exp(beta - 1.96 * se)),
            "hr_ci_hi95": float(np.exp(beta + 1.96 * se)),
            "p_value": p,
        })
    return pd.DataFrame(rows)


def export_group_intercept_table(structural_model: dict, label: str = "ward") -> pd.DataFrame:
    """One row per ward/ICU-type or hospital: effective cloglog intercept (delta-method SE),
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


def run_site_diagnostics(cohort_label: str, site: str, out_dir: Path, tau2_ward=None) -> None:
    """Model diagnostics on the ONE fitted site cloglog GLM (same design matrix, same reference levels):
    convergence / singularity, residuals, VIF, covariate balance (vasopressin-on vs off hours),
    missing data, ICU-type cluster counts, leave-one-ICU-type-out sensitivity of the NEE spline term,
    plus a markdown summary. Written to <cohort>/diagnostics/."""
    cached = _FIT_CACHE.get((cohort_label, site))
    if cached is None:
        print("  [diagnostics] no cached fit — skipped")
        return
    spec = importlib.util.spec_from_file_location("model_diagnostics", Path(__file__).parent / "model_diagnostics.py")
    md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md)

    names, X, Y, df = cached["col_names"], cached["X"], cached["Y"], cached["df"]
    X_df = pd.DataFrame(X, columns=names, index=df.index)
    y = pd.Series(Y, index=df.index)
    diag_x = X_df.copy()
    diag_x["vaso_on"] = Y
    diag_x["icu_type"] = df["icu_type"].values if "icu_type" in df.columns else "Unknown"
    target = "rcs_nee_0"

    def _fit_without(df_sub):
        # leave-one-ICU-type-out refit on the same design columns (drop all-zero dummies)
        Xs = df_sub[names]
        Xs = Xs.loc[:, (Xs != 0).any(axis=0)]
        r = fit_cloglog(df_sub["_y"].values, Xs, maxiter=100)
        return {target: float(r.params.get(target, np.nan)), f"{target}_se": float(r.bse.get(target, np.nan))}

    md.run_model_diagnostics(
        sm_result=cached["fit"], X=diag_x, y=y,
        model_name="site_cloglog_vaso_start", script="05_site_variation_analysis.py",
        out_dir=out_dir, cohort=cohort_label, site=site,
        group_col="vaso_on",
        cont_cols=[c for c in names if c.endswith("_c")],
        bin_cols=[c for c in ["rrt"] if c in names],
        cluster_col="icu_type",
        tau2_dict=({"ward (ICU type)": float(tau2_ward)} if tau2_ward is not None else None),
        resid_var=RESID_VAR,
        fit_fn=_fit_without if target in names else None,
        target_coef=target if target in names else None,
        main_coef_override=float(cached["fit"].params[names.index(target)]) if target in names else None,
    )


def run_for_cohort(cohort_label: str):
    cohort_path = PATIENT_LEVEL_DIR / f"cohort_{cohort_label}.parquet"
    feat_path   = PATIENT_LEVEL_DIR / "features.parquet"

    if not cohort_path.exists():
        print(f"  Cohort file not found: {cohort_path}"); return
    if not feat_path.exists():
        print(f"  Features file not found: {feat_path}"); return

    out = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / cohort_label
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"                     # all figures in one folder (CSV/JSON stay in out)
    plots.mkdir(parents=True, exist_ok=True)

    site_cohort   = pd.read_parquet(cohort_path)
    site_features = pd.read_parquet(feat_path)
    site_features = site_features[site_features["stay_id"].isin(set(site_cohort["stay_id"]))].copy()

    # ── The one site GLM ──────────────────────────────────────────────────
    print(f"\n[{SITE_NAME}] Fitting site-specific discrete-time hazard GLM (cloglog link)...")
    ph = _prep_person_hours(site_cohort, site_features)
    print(f"  person-hours after prep: {ph.shape}")
    if ph.empty:
        print("  No complete person-hours — skipping."); return
    model = fit_site_logistic(ph, cohort_label, SITE_NAME)
    print(f"  n={model['n_patients']:,}  vaso_on={model['n_vaso_on']:,}  "
          f"alpha={model['coefficients']['intercept']['beta']:+.4f}  "
          f"P_ref={model['ref_patient_p_t5']:.1%}  ICU ref={model['icu_ref']}")

    site_models = {SITE_NAME: model}
    pal = {SITE_NAME: {k.upper(): v for k, v in LOGO_PAL.items()}.get(SITE_NAME.upper(), _DEFAULT_COLORS[0])}

    print("\n--- Reference-patient P(vaso) vs time / vs NEE ---")
    plot_approach2_time(site_models, plots, cohort_label, pal)
    plot_approach2_nee(site_models, {SITE_NAME: ph}, plots, cohort_label, pal)

    # ── ICU-type effective intercepts from the same fit ───────────────────
    print("\n--- ICU-type effective intercepts (from the site GLM) ---")
    struct = build_icu_structure(model, ph)
    plot_group_intercepts(struct, plots, cohort_label, SITE_NAME, "ward")
    plot_approach2_time_by_group(struct, plots, cohort_label, SITE_NAME, "ward")
    plot_approach2_nee_by_group(struct, ph, plots, cohort_label, SITE_NAME, "ward")

    # ── Model diagnostics on the fitted GLM ───────────────────────────────
    print("--- Model diagnostics ---")
    try:
        run_site_diagnostics(cohort_label, SITE_NAME, out / "diagnostics", struct.get("tau2_ward"))
    except Exception as _exc:
        print(f"  [diagnostics] failed: {_exc}")

    # ── Variance of each covariate group's linear-predictor contribution ──
    print("\n--- Covariate variance explained (site-level component added downstream) ---")
    covar_variance = compute_covariate_variance_explained(
        {SITE_NAME: ph}, site_models, {SITE_NAME: struct}, {}
    )

    # ── Location transitions ──────────────────────────────────────────────
    print("\n--- Location transition heatmaps ---")
    plot_location_transition_heatmap(site_cohort, plots, cohort_label, SITE_NAME)
    print("\n--- Location transition outcome heatmaps ---")
    plot_location_transition_outcome_heatmaps(site_cohort, site_features, plots, cohort_label, SITE_NAME)

    # ── Packet (this site only) ───────────────────────────────────────────
    packet = {
        "site":               SITE_NAME,
        "cohort":             cohort_label,
        "per_site_model":     model,                    # coefficients, SEs, full covariance, knots, scaling
        "per_site_models":    {SITE_NAME: model},       # same shape the coordinating scripts read
        "structural_models":  {SITE_NAME: struct},      # ICU-type effective intercepts (ward == ICU type)
        "covariate_variance": covar_variance,
    }
    pkt_path = out / f"site_variation_packet_{cohort_label}_{SITE_NAME}.json"
    with open(pkt_path, "w", encoding="utf-8") as fout:
        json.dump(packet, fout, indent=2)

    export_coefficient_table(model).to_csv(out / f"coefficient_table_{cohort_label}_{SITE_NAME}.csv", index=False)
    if struct.get("ward_effective_intercepts"):
        export_group_intercept_table(struct, "ward").to_csv(
            out / f"ward_intercepts_{cohort_label}_{SITE_NAME}.csv", index=False)
    print(f"\nSaved packet: {pkt_path}")


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
