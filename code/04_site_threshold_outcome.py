#!/usr/bin/env python3
"""
04_site_threshold_outcome.py

Per-site discrete-time survival analysis of threshold-based vasopressin rules.

Policy question: "If the clinician follows this rule (give vasopressin when
feature crosses the threshold), do patients have better mortality?"

Model (cf. OVISS discrete-time survival approach):
  logit(P(Y_{i,t}=1)) = β₀ + β₁V_{i,0} + β₂V_{i,t} + β₃A_{i,t} + f(t)

  Y_{i,t}  = 1 if patient i dies in hour t (0 otherwise; censored at discharge)
  A_{i,t}  = concordance: 1 if action_vaso matches the threshold rule at hour t
  V_{i,0}  = baseline: age, weight, sepsis_onset_sofa, initial_lactate
  V_{i,t}  = time-varying confounders at hour t (analyzed feature excluded)
  f(t)     = restricted cubic spline of time_hour (5 knots, 4 df)

Two estimators reported for each feature × threshold:

  adj (conditional OR)
    Covariate-adjusted pooled logistic regression.
    Confounders included as regressors; SE clustered by patient.

  iptw (marginal OR — MSM)
    IPTW-weighted pooled logistic regression (marginal structural model).
    Stabilized weights from sequential propensity score models:
      Denominator: P(A_{i,t} | V_{i,0}, V_{i,t}, A_{i,t-1})  — all confounders
      Numerator:   P(A_{i,t} | V_{i,0},          A_{i,t-1})  — baseline only
    Weights are the cumulative product across time within each patient,
    trimmed at 1st/99th percentile.
    Weighted outcome model uses same formula; SE via clustered sandwich estimator.

One threshold definition per feature:
  kappa_initiation — maximises Cohen's kappa on training split (vaso-naive at-risk steps only;
                     predicts clinician starting vasopressin, not maintaining it)

Outputs to output/upload_to_box_<SITE>/ (aggregate only, no patient rows):
  threshold_outcome_table.csv       — adj and iptw OR, 95% CI, p-value
  threshold_concordance_summary.csv — concordance prevalence + crude hazard (QC)

Usage:
    uv run python code/04_site_threshold_outcome.py
"""

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent


def _load_site_config():
    import importlib.util as _ilu
    cfg_path = BASE.parent / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit("ERROR: config/config.py not found.")
SITE_NAME   = getattr(_cfg, "SITE_NAME", "UCMC")
OUTPUT_ROOT = getattr(_cfg, "OUTPUT_ROOT", None)
if OUTPUT_ROOT is None:
    raise SystemExit("ERROR: OUTPUT_ROOT is not set in config/config.py.")
OUTPUT_ROOT = Path(OUTPUT_ROOT)

PATIENT_LEVEL_DIR = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"
UPLOAD_DIR        = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / "threshold"

SUPPRESS_K     = 11
ROUND_N        = 3
N_THRESH_GRID  = 100
RANDOM_SEED    = 42
TRAIN_FRAC     = 0.70
N_SPLINE_KNOTS = 5   # 5 knots → 4 df for f(t)

# Time-varying confounders (time_hour enters the model via spline only)
FEAT_CONF_CONT = ["sofa", "nee", "mbp", "lactate", "creatinine", "bun"]
FEAT_CONF_BIN  = ["ventil", "rrt", "steroid"]
COH_CONF       = ["age", "weight", "sepsis_onset_sofa", "initial_lactate"]
ALL_CONFOUNDERS = FEAT_CONF_CONT + FEAT_CONF_BIN + COH_CONF

# Columns to LOCF-impute across time (feature measurements with observation gaps)
LOCF_COLS = FEAT_CONF_CONT + FEAT_CONF_BIN + ["norepinephrine", "fluids"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(x):
    """Round to ROUND_N decimals; None for NaN."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, ROUND_N)
    except (TypeError, ValueError):
        return None


def get_train_ids(cohort: pd.DataFrame) -> set:
    """Deterministic 70% training split — identical to 02_site_summary.py."""
    ids = cohort["stay_id"].values.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    ids = ids[rng.permutation(len(ids))]
    return set(ids[: int(len(ids) * TRAIN_FRAC)])


def kappa_threshold_train(
    feat_train: pd.DataFrame, feature_col: str, is_binary: bool
) -> tuple[float, str, float] | None:
    """Return (threshold, direction, kappa_initiation) from vaso-naive training at-risk steps.

    kappa_initiation is Cohen's kappa at the chosen threshold on those same at-risk steps.
    Returns None if insufficient data.
    """
    df = feat_train.copy()
    df = df.sort_values(["stay_id", "time_hour"])
    df["prev_vaso"] = df.groupby("stay_id")["action_vaso"].shift(1).fillna(0).astype(int)
    at_risk = df[df["prev_vaso"] == 0]

    if feature_col not in at_risk.columns:
        return None

    vals   = at_risk[feature_col].to_numpy(dtype=float)
    labels = at_risk["action_vaso"].to_numpy(dtype=int)
    fin    = np.isfinite(vals)
    vals, labels = vals[fin], labels[fin]

    if len(vals) < 20 or labels.sum() == 0:
        return None

    if is_binary:
        pred = (vals >= 0.5).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            return None
        try:
            k_val = float(cohen_kappa_score(labels, pred))
        except Exception:
            k_val = float("nan")
        return 0.5, "gt", k_val

    lo, hi = np.percentile(vals, [5, 95])
    if lo >= hi:
        return None

    best_kappa, best_thresh, best_dir = -np.inf, None, None
    for t in np.linspace(lo, hi, N_THRESH_GRID):
        for pred, direc in [
            ((vals >= t).astype(int), "gt"),
            ((vals <= t).astype(int), "lt"),
        ]:
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            k = cohen_kappa_score(labels, pred)
            if k > best_kappa:
                best_kappa, best_thresh, best_dir = k, t, direc

    if best_thresh is None:
        return None
    # Re-evaluate kappa at the chosen threshold
    pred_best = ((vals >= best_thresh).astype(int) if best_dir == "gt"
                 else (vals <= best_thresh).astype(int))
    try:
        k_val = float(cohen_kappa_score(labels, pred_best))
    except Exception:
        k_val = float("nan")
    return float(best_thresh), best_dir, k_val


def crosses(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Boolean array: True where value satisfies the threshold condition."""
    if direction == "gt":
        return values >= threshold
    if direction == "lt":
        return values <= threshold
    if direction == "eq":
        return values == threshold
    raise ValueError(f"Unknown direction: {direction!r}")


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Restricted cubic spline basis (Harrell 2001).
    K knots → (K-1) columns: [x_linear, cubic_1, ..., cubic_{K-2}].
    Natural (linear) outside boundary knots.
    """
    K     = len(knots)
    t_K   = knots[-1]
    t_Km1 = knots[-2]
    scale = t_K - t_Km1

    cols = [x.copy()]
    for k in range(K - 2):
        u = (np.maximum(x - knots[k], 0.0) ** 3
             - (t_K   - knots[k]) / scale * np.maximum(x - t_Km1, 0.0) ** 3
             + (t_Km1 - knots[k]) / scale * np.maximum(x - t_K,   0.0) ** 3)
        cols.append(u)
    return np.column_stack(cols)


def _iptw_sandwich_or(X, Y, weights, groups, concordance_col_idx):
    """
    IPTW-weighted logistic regression with clustered sandwich SE.

    Fits via sklearn (sample_weight) then computes the clustered sandwich
    variance using the weighted score equations. Returns (OR, OR_lo, OR_hi,
    p-value) for the concordance column; NaN tuple on failure.

    X:                   (n, p) feature matrix without intercept
    Y:                   (n,) binary outcome
    weights:             (n,) IPTW weights
    groups:              (n,) patient IDs (cluster labels)
    concordance_col_idx: column index of the concordance variable in X
    """
    n, p = X.shape
    nan4 = (np.nan, np.nan, np.nan, np.nan)

    try:
        lr = LogisticRegression(C=1e4, max_iter=2000, solver="lbfgs", penalty="l2")
        lr.fit(X, Y, sample_weight=weights)
    except Exception:
        return nan4

    coef      = lr.coef_[0]          # (p,)
    intercept = lr.intercept_[0]     # scalar

    eta   = X @ coef + intercept
    probs = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)

    # Augmented design matrix [1 | X] for the sandwich
    X_aug = np.column_stack([np.ones(n), X])   # (n, p+1)

    # Weighted score contributions: w_i * (y_i - p_i) * x_i
    resid  = weights * (Y - probs)
    scores = resid[:, None] * X_aug            # (n, p+1)

    # Bread: -Σ w_i * p_i(1-p_i) * x_i x_i'
    w_var = weights * probs * (1.0 - probs)
    bread = -(X_aug * w_var[:, None]).T @ X_aug  # (p+1, p+1)

    # Meat: clustered sum-of-scores outer products
    order         = np.argsort(groups, kind="stable")
    groups_ord    = groups[order]
    scores_ord    = scores[order]
    unique_g      = np.unique(groups_ord)
    splits        = np.searchsorted(groups_ord, unique_g[1:])
    meat          = np.zeros((p + 1, p + 1))
    for chunk in np.split(scores_ord, splits):
        cs = chunk.sum(axis=0)
        meat += np.outer(cs, cs)

    try:
        bread_inv = np.linalg.inv(bread)
        V  = bread_inv @ meat @ bread_inv.T
        se = np.sqrt(np.diag(V))
    except np.linalg.LinAlgError:
        return nan4

    # Concordance is col concordance_col_idx in X → col concordance_col_idx+1 in X_aug
    c_idx = concordance_col_idx + 1
    c_coef = coef[concordance_col_idx]
    c_se   = se[c_idx]

    if not (np.isfinite(c_se) and c_se > 0):
        return nan4

    or_est = float(np.exp(c_coef))
    or_lo  = float(np.exp(c_coef - 1.96 * c_se))
    or_hi  = float(np.exp(c_coef + 1.96 * c_se))
    z      = c_coef / c_se
    p_val  = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    return or_est, or_lo, or_hi, p_val


# ---------------------------------------------------------------------------
# Core analysis: one feature × threshold
# ---------------------------------------------------------------------------

def analyze(
    feat: pd.DataFrame,
    cohort: pd.DataFrame,
    feature_col: str,
    threshold: float,
    direction: str,
    threshold_type: str,
    spline_knots: np.ndarray,
) -> tuple[dict, dict] | None:
    """
    Discrete-time survival analysis for one feature × threshold.

    Returns (result_row, summary_row) or None if insufficient data.
    All returned values are aggregate — no patient-level rows.
    """
    if feature_col not in feat.columns:
        return None

    f = feat.copy()

    # ── Concordance A_{i,t} ─────────────────────────────────────────────────
    vals = f[feature_col].to_numpy(dtype=float)
    fin  = np.isfinite(vals)
    rule = np.full(len(f), np.nan)
    rule[fin] = crosses(vals[fin], threshold, direction).astype(float)
    f["concordance"] = np.where(
        np.isfinite(rule),
        (f["action_vaso"].to_numpy(dtype=int) == rule.astype(np.int64)).astype(float),
        np.nan,
    )

    # ── Outcome Y_{i,t}: 1 at last row iff hospital_death=1 ─────────────────
    f = f.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)
    death_map = cohort.set_index("stay_id")["hospital_death"]
    f["hospital_death"] = f["stay_id"].map(death_map).fillna(0).astype(int)
    f["is_last"] = ~f.duplicated(subset="stay_id", keep="last")
    f["event"]   = (f["is_last"] & (f["hospital_death"] == 1)).astype(int)

    # ── Merge baseline covariates ────────────────────────────────────────────
    bl_cols = [c for c in COH_CONF if c in cohort.columns]
    f = f.merge(cohort[["stay_id"] + bl_cols], on="stay_id", how="left")

    # ── Time-varying confounders (exclude analyzed feature) ──────────────────
    tv_cols   = [c for c in FEAT_CONF_CONT + FEAT_CONF_BIN
                 if c in f.columns and c != feature_col]
    all_conf  = bl_cols + tv_cols
    cont_conf = [c for c in all_conf if c not in FEAT_CONF_BIN]

    # Impute remaining missing confounder values with column median
    for c in all_conf:
        if c in f.columns:
            med = float(f[c].median()) if f[c].notna().any() else 0.0
            f[c] = f[c].fillna(med)

    # ── Analysis dataset (sorted, no NaN concordance) ────────────────────────
    keep = ["stay_id", "time_hour", "event", "concordance"] + all_conf
    df   = f[[c for c in keep if c in f.columns]].dropna(subset=["concordance"])
    df   = df.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)

    n_patients = int(df["stay_id"].nunique())
    n_events   = int(df.groupby("stay_id")["event"].max().sum())

    if n_patients < SUPPRESS_K or n_events < SUPPRESS_K:
        return None

    Y           = df["event"].to_numpy(dtype=int)
    concordance = df["concordance"].to_numpy(dtype=float)
    groups      = df["stay_id"].to_numpy()
    t_vals      = df["time_hour"].to_numpy(dtype=float)

    # ── Spline basis f(t) ────────────────────────────────────────────────────
    T_basis = rcs_basis(t_vals, spline_knots)
    t_names = [f"_t{k}" for k in range(T_basis.shape[1])]

    # ── Standardise continuous confounders ───────────────────────────────────
    X_conf = df[all_conf].to_numpy(dtype=float)
    for i, c in enumerate(all_conf):
        if c in cont_conf:
            sd = X_conf[:, i].std()
            X_conf[:, i] = ((X_conf[:, i] - X_conf[:, i].mean()) / sd
                            if sd > 1e-8 else X_conf[:, i] - X_conf[:, i].mean())

    # Shared design matrix (concordance first, then confounders, then spline)
    X_df = pd.DataFrame(
        np.column_stack([concordance, X_conf, T_basis]),
        columns=["concordance"] + all_conf + t_names,
    )

    # ── Concordance summary (QC, aggregate) ──────────────────────────────────
    conc_rate         = float(concordance.mean())
    hazard_concordant = (float(Y[concordance == 1].mean())
                         if (concordance == 1).sum() > 0 else np.nan)
    hazard_discordant = (float(Y[concordance == 0].mean())
                         if (concordance == 0).sum() > 0 else np.nan)

    # =========================================================================
    # Estimator 1: Covariate-adjusted pooled logistic (conditional OR)
    # =========================================================================
    adj_or = adj_or_lo = adj_or_hi = adj_pval = np.nan
    adj_converged = False
    try:
        X_c    = sm.add_constant(X_df, has_constant="add")
        res    = sm.Logit(Y, X_c).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
            disp=False,
            maxiter=200,
        )
        adj_or      = float(np.exp(res.params["concordance"]))
        ci_vals     = res.conf_int().loc["concordance"].to_numpy()
        adj_or_lo   = float(np.exp(ci_vals[0]))
        adj_or_hi   = float(np.exp(ci_vals[1]))
        adj_pval    = float(res.pvalues["concordance"])
        adj_converged = bool(res.mle_retvals.get("converged", True))
    except Exception:
        pass

    # =========================================================================
    # Estimator 2: IPTW-weighted pooled logistic (marginal OR — MSM)
    # =========================================================================
    iptw_or = iptw_or_lo = iptw_or_hi = iptw_pval = np.nan
    iptw_mean_w = iptw_max_w = np.nan

    try:
        # Lagged concordance (treatment history)
        prev_conc = (df.groupby("stay_id")["concordance"]
                     .shift(1).fillna(0.5).to_numpy(dtype=float))

        # PS denominator: P(concordance | V_{i,0}, V_{i,t}, A_{i,t-1}, f(t))
        X_denom = np.column_stack([X_conf, prev_conc[:, None], T_basis])

        # PS numerator: P(concordance | V_{i,0}, A_{i,t-1})
        bl_idx  = [list(all_conf).index(c) for c in bl_cols if c in all_conf]
        X_bl    = X_conf[:, bl_idx]   # already standardised
        X_num   = np.column_stack([X_bl, prev_conc[:, None]])

        conc_int = concordance.astype(int)

        lr_d = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_denom, conc_int)
        lr_n = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_num,   conc_int)

        p_d   = lr_d.predict_proba(X_denom)[:, 1]
        p_n   = lr_n.predict_proba(X_num)[:, 1]

        # Stabilized weight per row
        p_d_obs = np.clip(np.where(concordance == 1, p_d, 1 - p_d), 0.01, 0.99)
        p_n_obs = np.where(concordance == 1, p_n, 1 - p_n)
        sw      = p_n_obs / p_d_obs

        # Cumulative product within patient (df is sorted by stay_id, time_hour)
        cw_series = (pd.Series(sw, index=df.index)
                     .groupby(df["stay_id"])
                     .cumprod())
        cw = cw_series.to_numpy(dtype=float)

        # Trim at 1st / 99th percentile
        fin = np.isfinite(cw)
        if fin.sum() > 0:
            lo, hi = np.percentile(cw[fin], [1, 99])
            cw = np.where(fin, np.clip(cw, lo, hi), 1.0)

        iptw_mean_w = float(cw.mean())
        iptw_max_w  = float(cw.max())

        # Weighted outcome model (same design matrix as adjusted)
        X_arr = X_df.to_numpy(dtype=float)
        conc_idx = list(X_df.columns).index("concordance")   # = 0

        iptw_or, iptw_or_lo, iptw_or_hi, iptw_pval = _iptw_sandwich_or(
            X_arr, Y, cw, groups, conc_idx
        )

    except Exception:
        pass

    # ── Assemble output rows ─────────────────────────────────────────────────
    result_row = {
        "feature":           feature_col,
        "threshold_type":    threshold_type,
        "threshold_value":   threshold,
        "direction":         direction,
        "n_patients":        n_patients,
        "n_events":          n_events,
        "n_patient_hours":   len(df),
        "concordance_rate":  _r(conc_rate),
        # Covariate-adjusted
        "adj_or":            _r(adj_or),
        "adj_or_lo":         _r(adj_or_lo),
        "adj_or_hi":         _r(adj_or_hi),
        "adj_pval":          _r(adj_pval),
        "adj_converged":     adj_converged,
        # IPTW-MSM
        "iptw_or":           _r(iptw_or),
        "iptw_or_lo":        _r(iptw_or_lo),
        "iptw_or_hi":        _r(iptw_or_hi),
        "iptw_pval":         _r(iptw_pval),
        "iptw_mean_weight":  _r(iptw_mean_w),
        "iptw_max_weight":   _r(iptw_max_w),
    }

    summary_row = {
        "feature":                   feature_col,
        "threshold_type":            threshold_type,
        "n_patients":                n_patients,
        "n_events":                  n_events,
        "concordance_rate":          _r(conc_rate),
        "hourly_hazard_concordant":  _r(hazard_concordant),
        "hourly_hazard_discordant":  _r(hazard_discordant),
    }

    return result_row, summary_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Site:   {SITE_NAME}")
    print(f"Input:  {PATIENT_LEVEL_DIR}")
    print(f"Output: {UPLOAD_DIR}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = PATIENT_LEVEL_DIR / "features.parquet"
    coh_path  = PATIENT_LEVEL_DIR / "cohort.parquet"

    for p in (feat_path, coh_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p}\nRun the extraction script first."
            )

    cohort = pd.read_parquet(coh_path)
    feat   = pd.read_parquet(feat_path)

    # Same t≤0 vaso exclusion as 02_site_summary.py
    feat = feat.sort_values(["stay_id", "time_hour"])
    vaso_t0 = feat[feat["time_hour"] <= 0].groupby("stay_id")["action_vaso"].max()
    excl = set(vaso_t0[vaso_t0 == 1].index)
    if excl:
        print(f"  Excluding {len(excl)} patients with action_vaso=1 at t<=0")
        cohort = cohort[~cohort["stay_id"].isin(excl)].copy()
        feat   = feat[~feat["stay_id"].isin(excl)].copy()

    # LOCF for feature measurements with observation gaps
    for col in LOCF_COLS:
        if col in feat.columns:
            feat[col] = feat.groupby("stay_id")[col].ffill()

    print(f"Patients: {cohort['stay_id'].nunique():,}")

    # Training split used only for threshold selection
    train_ids  = get_train_ids(cohort)
    feat_train = feat[feat["stay_id"].isin(train_ids)].copy()
    print(f"Train patients (threshold selection): {feat_train['stay_id'].nunique():,}")

    # Spline knots on full cohort time distribution
    all_t        = feat["time_hour"].dropna().to_numpy(dtype=float)
    spline_knots = np.percentile(all_t, [5, 27.5, 50, 72.5, 95])
    print(f"Spline knots f(t): {np.round(spline_knots, 1)}")

    FEATURE_DEFS = [
        ("time_hour",      False),
        ("norepinephrine", False),
        ("nee",            False),
        ("mbp",            False),
        ("sofa",           False),
        ("lactate",        False),
        ("creatinine",     False),
        ("bun",            False),
        ("fluids",         False),
        ("ventil",         True),
        ("rrt",            True),
        ("steroid",        True),
    ]

    # ── Build threshold list ──────────────────────────────────────────────────
    threshold_defs = []

    print("\nComputing kappa thresholds from training data...")
    for feat_col, is_bin in FEATURE_DEFS:
        if feat_col not in feat_train.columns:
            continue
        res = kappa_threshold_train(feat_train, feat_col, is_bin)
        if res is None:
            print(f"  [skip] {feat_col}: insufficient training data")
            continue
        val, direc, k_init = res
        print(f"  {feat_col:<18}: {direc} {val:.4g}  κ-init={k_init:.3f}")
        threshold_defs.append((feat_col, val, direc, "kappa_initiation", k_init))

    # ── Run analyses ──────────────────────────────────────────────────────────
    results, summaries = [], []
    print(f"\nRunning {len(threshold_defs)} analyses "
          f"(adj + IPTW-MSM, clustered sandwich SE, {N_SPLINE_KNOTS}-knot spline)...")

    for feature_col, threshold, direction, ttype, k_init in threshold_defs:
        label = f"{feature_col:<16} [{ttype:<18}]  {direction} {threshold:.4g}  κ-init={k_init:.3f}"
        print(f"  {label}...", end=" ", flush=True)

        out = analyze(feat, cohort, feature_col, threshold, direction, ttype, spline_knots)
        if out is None:
            print("skipped (insufficient data)")
            continue

        row, summary = out
        row["kappa_initiation"] = _r(k_init)
        results.append(row)
        summaries.append(summary)
        adj_flag  = "" if row["adj_converged"] else " [adj NOT CONVERGED]"
        print(
            f"n={row['n_patients']:,}  events={row['n_events']}  "
            f"conc={row['concordance_rate']}  "
            f"adj OR={row['adj_or']} [{row['adj_or_lo']}, {row['adj_or_hi']}] p={row['adj_pval']}  "
            f"iptw OR={row['iptw_or']} [{row['iptw_or_lo']}, {row['iptw_or_hi']}] p={row['iptw_pval']}"
            f"{adj_flag}"
        )

    # Save aggregate outputs — no patient-level data
    out_tbl = pd.DataFrame(results)
    out_tbl.to_csv(UPLOAD_DIR / "threshold_outcome_table.csv", index=False)
    print(f"\nSaved: {UPLOAD_DIR / 'threshold_outcome_table.csv'}  ({len(out_tbl)} rows)")

    out_sum = pd.DataFrame(summaries)
    out_sum.to_csv(UPLOAD_DIR / "threshold_concordance_summary.csv", index=False)
    print(f"Saved: {UPLOAD_DIR / 'threshold_concordance_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
