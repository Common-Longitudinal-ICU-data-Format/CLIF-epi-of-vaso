#!/usr/bin/env python3
"""
02_site_summary.py

Federated-safe aggregate summary for a septic shock cohort.

Reads (PHI, local only):
  output/patient_level_data_<SITE>/cohort_<cohort>.parquet   (sepsis3 or rhee)
  output/patient_level_data_<SITE>/features.parquet          (union; filtered here to cohort)
  output/patient_level_data_<SITE>/cohort_filter_counts_<cohort>.csv

Writes only aggregate CSVs to output/upload_to_box_<SITE>/<cohort>/ — no patient-level data leaves the site.

Privacy guarantees
  - No row-level values, identifiers, free text, or exact dates in any output.
  - Cell suppression: n < K=11 → count shown as "<11", derived stats blank.
  - Continuous summaries rounded to 3 decimal places.
  - ROC/threshold curves evaluated on a fixed 100-point grid derived from
    rounded [p5, p95] of training data — NOT one point per observation.

Outcome for analyses 4 and 5
  Per-timestep imminent initiation: at each "at-risk" hour (patient not yet on
  vasopressin, i.e. previous action_vaso = 0), outcome = action_vaso at that hour.
  Threshold selected by max Youden's J on TRAIN split; carried unchanged to val/test.

Usage:
    uv run python code/02_site_summary.py                     # runs both sepsis3 and rhee
    uv run python code/02_site_summary.py --cohort rhee
    uv run python code/02_site_summary.py --cohort sepsis3 --site MIMIC
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR         = Path(__file__).parent.parent

# Load SITE_NAME from config/config.py (by file path, not import, so the
# config/ directory is not mistaken for an empty namespace package)
def _load_site_config():
    import importlib.util as _ilu
    cfg_path = BASE_DIR / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--site",   default=None, help="Override SITE_NAME from config")
_parser.add_argument("--cohort", default="both", choices=["sepsis3", "rhee", "both"],
                     help="Which cohort file to read (cohort_<cohort>.parquet); "
                          "'both' (default) runs sepsis3 then rhee")
_cli_args, _passthrough_args = _parser.parse_known_args()

if _cli_args.cohort == "both":
    import subprocess
    for _c in ("sepsis3", "rhee"):
        _cmd = [sys.executable, __file__, "--cohort", _c]
        if _cli_args.site:
            _cmd += ["--site", _cli_args.site]
        _cmd += _passthrough_args
        print(f"[02_site_summary] $ {' '.join(_cmd)}", flush=True)
        _result = subprocess.run(_cmd)
        if _result.returncode != 0:
            raise SystemExit(_result.returncode)
    raise SystemExit(0)

_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit(
        "ERROR: config/config.py not found.\n"
        "Copy config/config.example.py to config/config.py and set SITE_NAME, CLIF_DIR, OUTPUT_ROOT."
    )
SITE_NAME   = _cli_args.site   if _cli_args.site   else getattr(_cfg, "SITE_NAME", "UCMC")
COHORT_NAME = _cli_args.cohort
OUTPUT_ROOT = getattr(_cfg, "OUTPUT_ROOT", None)
if OUTPUT_ROOT is None:
    raise SystemExit("ERROR: OUTPUT_ROOT is not set in config/config.py.")
OUTPUT_ROOT = Path(OUTPUT_ROOT)

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

SUPPRESS_K       = 11       # suppress cells derived from fewer than K patients
ROUND_N          = 3        # decimal places for continuous statistics
N_THRESH_GRID    = 100      # grid points per continuous feature

# Clinical features for analyses 4 and 5
# urine_output excluded: all zero in CLIF 2.1.0
ANALYSIS_FEATURES_CONT = [
    "time_hour", "norepinephrine", "nee", "mbp", "sofa",
    "lactate", "creatinine", "bun", "fluids",
]
ANALYSIS_FEATURES_BIN  = ["ventil", "rrt", "steroid"]
ANALYSIS_FEATURES      = ANALYSIS_FEATURES_CONT + ANALYSIS_FEATURES_BIN

# Baseline table
BL_CONTINUOUS  = ["age", "weight", "sepsis_onset_sofa", "initial_lactate", "traj_hours",
                   "icu_los_days", "hospital_los_days"]
BL_BINARY      = ["hospital_death"]
BL_CATEGORICAL = ["gender", "race", "location_category", "location_type", "hospital_type",
                   "traj_end_reason"]

# Continuous features aggregated in the new ICU-type / hospital initiation
# breakdown CSVs (subset of ANALYSIS_FEATURES_CONT — everything except
# fluids, which isn't part of the cross-site box/whisker-by-group plots).
GROUP_BREAKDOWN_FEATURES = ["time_hour", "norepinephrine", "nee", "mbp", "sofa", "lactate", "creatinine", "bun"]

# ICU-type canonicalization (same mapping used in 04_site_variation_analysis.py,
# kept in sync for consistent labels across baseline tables and variance analyses).
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


def _canon_icu_series(raw: pd.Series) -> pd.Series:
    """Canonicalize raw location_type/location_name strings into ICU-type labels."""
    low = raw.astype(str).str.lower().str.strip()
    is_missing = raw.isna() | low.isin(["none", "nan", ""])
    canon = low.map(_ICU_CANON).fillna("Other ICU")
    canon[is_missing] = np.nan
    return canon


def _canon_hospital_type(raw: pd.Series) -> pd.Series:
    """Canonicalize raw hospital_type strings into academic/community/unknown."""
    low = raw.astype(str).str.lower().str.strip()
    is_missing = raw.isna() | low.isin(["none", "nan", ""])
    canon = low.where(low.isin(["academic", "community"]), "unknown")
    canon[is_missing] = np.nan
    return canon

# ============================================================
# HELPERS
# ============================================================

def _r(x):
    """Round to ROUND_N decimals; pass through None/NaN."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, ROUND_N)
    except (TypeError, ValueError):
        return None


def _n_str(n):
    """Return count as string; suppress if < K."""
    return str(int(n)) if int(n) >= SUPPRESS_K else f"<{SUPPRESS_K}"


def _suppress(val, n):
    """Return val if n >= K, else None."""
    return val if int(n) >= SUPPRESS_K else None


def _smd_cont(m1, s1, m2, s2):
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    return _r((m1 - m2) / pooled) if pooled > 0 else None


def _smd_bin(p1, p2):
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    return _r((p1 - p2) / denom) if denom > 0 else None


# ============================================================
# DATA LOADING AND SPLITTING
# ============================================================

def load_and_split():
    """Load cohort + features; add ever_vaso flag; split 70/15/15 by patient."""
    coh_all = pd.read_parquet(INPUT_DIR / f"cohort_{COHORT_NAME}.parquet")
    feat_all = pd.read_parquet(INPUT_DIR / "features.parquet")
    # Filter features to this cohort's patients
    feat_all = feat_all[feat_all["stay_id"].isin(set(coh_all["stay_id"]))].copy()
    coh  = coh_all
    feat = feat_all

    # Exclude patients on vasopressin at or before t=0
    feat = feat.sort_values(["stay_id", "time_hour"])
    vaso_at_t0 = feat[feat["time_hour"] <= 0].groupby("stay_id")["action_vaso"].max()
    vaso_at_t0_ids = set(vaso_at_t0[vaso_at_t0 == 1].index)
    if vaso_at_t0_ids:
        n_excl = int(coh["stay_id"].isin(vaso_at_t0_ids).sum())
        print(f"  Excluding {n_excl} patients with action_vaso=1 at t<=0")
        coh  = coh[~coh["stay_id"].isin(vaso_at_t0_ids)].copy()
        feat = feat[~feat["stay_id"].isin(vaso_at_t0_ids)].copy()

    # LOCF per patient for continuous features before any analysis
    for col in ANALYSIS_FEATURES_CONT:
        if col in feat.columns:
            feat[col] = feat.groupby("stay_id")[col].ffill()

    # Patient-level: ever received vasopressin during trajectory
    ever_vaso = (
        feat.groupby("stay_id")["action_vaso"]
        .max()
        .rename("ever_vaso")
        .reset_index()
    )
    coh = coh.merge(ever_vaso, on="stay_id", how="left")
    coh["ever_vaso"] = coh["ever_vaso"].fillna(0).astype(int)

    # Patient-level split (deterministic)
    ids = coh["stay_id"].values.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    ids = ids[rng.permutation(len(ids))]
    n     = len(ids)
    n_tr  = int(n * TRAIN_FRAC)
    n_va  = int(n * VAL_FRAC)
    tr_ids = set(ids[:n_tr])
    va_ids = set(ids[n_tr : n_tr + n_va])

    def _split_label(s):
        if s in tr_ids:
            return "train"
        if s in va_ids:
            return "val"
        return "test"

    coh["split"] = coh["stay_id"].map(_split_label)
    feat = feat.merge(coh[["stay_id", "split", "ever_vaso"]], on="stay_id", how="left")

    return coh, feat


# ============================================================
# OUTPUT 1: cohort_filter_counts.csv (copy as-is)
# ============================================================

def write_filter_counts():
    src = INPUT_DIR / f"cohort_filter_counts_{COHORT_NAME}.csv"
    dst = OUTPUT_DIR / f"cohort_filter_counts_{COHORT_NAME}.csv"
    shutil.copy2(src, dst)
    print(f"  Copied  {dst.name}")


# ============================================================
# OUTPUT 2: split_counts.csv
# ============================================================

def write_split_counts(coh):
    rows = []
    for split in ["train", "val", "test", "all"]:
        sub = coh if split == "all" else coh[coh["split"] == split]
        n_tot  = len(sub)
        n_vaso = int((sub["ever_vaso"] == 1).sum())
        n_none = int((sub["ever_vaso"] == 0).sum())
        rows.extend([
            {"split": split, "outcome_group": "ever_vaso_yes", "n_patients": _n_str(n_vaso)},
            {"split": split, "outcome_group": "ever_vaso_no",  "n_patients": _n_str(n_none)},
            {"split": split, "outcome_group": "total",         "n_patients": _n_str(n_tot)},
        ])
    out = OUTPUT_DIR / "split_counts.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 3: baseline_table1.csv
# ============================================================

def _cont_row(variable, col, groups_df, group_n):
    stats = {}
    for g, df in groups_df.items():
        arr = pd.to_numeric(df[col], errors="coerce").dropna()
        n       = len(arr)
        n_miss  = group_n[g] - n
        if n >= SUPPRESS_K:
            stats[g] = {
                "n": n, "n_missing": n_miss,
                "mean":   _r(arr.mean()),   "sd":     _r(arr.std()),
                "median": _r(arr.median()), "q25":    _r(arr.quantile(0.25)),
                "q75":    _r(arr.quantile(0.75)),
                "min":    _r(arr.min()),    "max":    _r(arr.max()),
                "pct": None,
                "_mean": float(arr.mean()), "_sd": float(arr.std()), "_n": n,
            }
        else:
            stats[g] = {k: None for k in ("mean", "sd", "median", "q25", "q75", "min", "max", "pct")}
            stats[g].update({"n": _n_str(n), "n_missing": n_miss, "_mean": None, "_sd": None, "_n": n})

    # SMD (vaso vs no_vaso)
    m1, s1 = stats["vaso"].get("_mean"), stats["vaso"].get("_sd")
    m2, s2 = stats["no_vaso"].get("_mean"), stats["no_vaso"].get("_sd")
    smd = _smd_cont(m1, s1, m2, s2) if all(v is not None for v in [m1, s1, m2, s2]) else None

    row = {"variable": variable, "level": "", "type": "continuous", "smd": smd}
    for g in ("vaso", "no_vaso", "overall"):
        for k in ("n", "n_missing", "mean", "sd", "median", "q25", "q75", "min", "max", "pct", "n_pct"):
            row[f"{g}_{k}"] = stats[g].get(k)
    return row


def _bin_row(variable, col, groups_df, group_n):
    stats = {}
    for g, df in groups_df.items():
        arr    = pd.to_numeric(df[col], errors="coerce").dropna()
        n      = len(arr)
        n_miss = group_n[g] - n
        pos    = int(arr.sum())
        pct    = _suppress(round(pos / n * 100, ROUND_N) if n > 0 else None, pos)
        n_pct  = f"{_n_str(pos)} ({pct}%)" if pct is not None else None
        stats[g] = {"n": _n_str(n), "n_missing": n_miss, "pct": pct, "n_pct": n_pct,
                    "_n": n, "_pos": pos}

    p1 = stats["vaso"]["_pos"] / max(stats["vaso"]["_n"], 1)
    p2 = stats["no_vaso"]["_pos"] / max(stats["no_vaso"]["_n"], 1)
    smd = _smd_bin(p1, p2)

    row = {"variable": variable, "level": "=1", "type": "binary", "smd": smd}
    for g in ("vaso", "no_vaso", "overall"):
        row[f"{g}_n"]         = stats[g]["n"]
        row[f"{g}_n_missing"] = stats[g]["n_missing"]
        row[f"{g}_pct"]       = stats[g]["pct"]
        row[f"{g}_n_pct"]     = stats[g]["n_pct"]
        for k in ("mean", "sd", "median", "q25", "q75", "min", "max"):
            row[f"{g}_{k}"] = None
    return row


def _cat_rows(variable, col, groups_df, group_n):
    all_levels = sorted(
        set(groups_df["overall"][col].dropna().unique())
    )
    rows = []
    for level in all_levels:
        stats = {}
        for g, df in groups_df.items():
            n_grp  = group_n[g]
            cnt    = int((df[col].dropna() == level).sum())
            pct    = _suppress(round(cnt / n_grp * 100, ROUND_N) if n_grp > 0 else None, cnt)
            n_pct  = f"{_n_str(cnt)} ({pct}%)" if pct is not None else None
            stats[g] = {"n": _n_str(cnt), "n_missing": None, "pct": pct, "n_pct": n_pct,
                        "_cnt": cnt, "_n": n_grp}

        p1  = stats["vaso"]["_cnt"] / max(stats["vaso"]["_n"], 1)
        p2  = stats["no_vaso"]["_cnt"] / max(stats["no_vaso"]["_n"], 1)
        c1  = stats["vaso"]["_cnt"]
        c2  = stats["no_vaso"]["_cnt"]
        smd = _smd_bin(p1, p2) if (c1 >= SUPPRESS_K and c2 >= SUPPRESS_K) else None

        row = {"variable": variable, "level": level, "type": "categorical", "smd": smd}
        for g in ("vaso", "no_vaso", "overall"):
            row[f"{g}_n"]         = stats[g]["n"]
            row[f"{g}_n_missing"] = None
            row[f"{g}_pct"]       = stats[g]["pct"]
            row[f"{g}_n_pct"]     = stats[g]["n_pct"]
            for k in ("mean", "sd", "median", "q25", "q75", "min", "max"):
                row[f"{g}_{k}"] = None
        rows.append(row)
    return rows


def write_baseline_table1(coh):
    coh = coh.copy()
    if "location_type" in coh.columns:
        coh["location_type"] = _canon_icu_series(coh["location_type"])
    if "hospital_type" in coh.columns:
        coh["hospital_type"] = _canon_hospital_type(coh["hospital_type"])

    groups_df = {
        "vaso":    coh[coh["ever_vaso"] == 1],
        "no_vaso": coh[coh["ever_vaso"] == 0],
        "overall": coh,
    }
    group_n = {g: len(df) for g, df in groups_df.items()}

    rows = []
    for col in BL_CONTINUOUS:
        if col in coh.columns:
            rows.append(_cont_row(col, col, groups_df, group_n))
    for col in BL_BINARY:
        if col in coh.columns:
            rows.append(_bin_row(col, col, groups_df, group_n))
    for col in BL_CATEGORICAL:
        if col in coh.columns:
            rows.extend(_cat_rows(col, col, groups_df, group_n))

    out = OUTPUT_DIR / "baseline_table1.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 4: feature_at_initiation.csv
# ============================================================

def _initiation_rows(feat):
    """First 0→1 vasopressin-action transition per patient, shared by both
    the flat and the group-level feature-at-initiation writers."""
    fs = feat.sort_values(["stay_id", "time_hour"]).copy()
    fs["prev_vaso"] = fs.groupby("stay_id")["action_vaso"].shift(1).fillna(0)
    return (
        fs[(fs["action_vaso"] == 1) & (fs["prev_vaso"] == 0)]
        .drop_duplicates(subset=["stay_id"], keep="first")
    )


def _feature_stats_row(arr: pd.Series, n_total: int, extra: dict | None = None) -> dict:
    """One suppressed mean/sd/median/q25/q75/min/max row for a numeric array,
    given the (unfiltered) group total used for n_missing."""
    n      = len(arr)
    n_miss = n_total - n
    row = dict(extra or {})
    row.update({"n": _n_str(n), "n_missing": n_miss})
    if n >= SUPPRESS_K:
        row.update({
            "mean":   _r(arr.mean()),   "sd":     _r(arr.std()),
            "median": _r(arr.median()), "q25":    _r(arr.quantile(0.25)),
            "q75":    _r(arr.quantile(0.75)),
            "min":    _r(arr.min()),    "max":    _r(arr.max()),
        })
    else:
        row.update({k: None for k in ("mean", "sd", "median", "q25", "q75", "min", "max")})
    return row


def write_feature_at_initiation(feat):
    """Aggregate feature values at the first vasopressin initiation timestep."""
    init = _initiation_rows(feat)
    n_total = len(init)

    rows = []
    for col in ANALYSIS_FEATURES:
        if col not in init.columns:
            continue
        arr = pd.to_numeric(init[col], errors="coerce").dropna()
        rows.append(_feature_stats_row(arr, n_total, extra={"feature": col}))

    out = OUTPUT_DIR / "feature_at_initiation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


def write_feature_at_initiation_by_group(coh, feat):
    """Feature values at initiation, broken down by ICU type and by hospital —
    feeds the cross-site box/whisker-by-group plots (GROUP_BREAKDOWN_FEATURES,
    not the full ANALYSIS_FEATURES list). The hospital breakdown also carries
    each hospital's (canonicalized) hospital_type through, so the report can
    label/color hospitals by academic vs community."""
    init = _initiation_rows(feat)

    group_specs = [
        ("location_type", "feature_at_initiation_by_icu.csv",           _canon_icu_series,      False),
        ("hospital_id",   "feature_at_initiation_by_hospital.csv",       None,                   True),
        ("hospital_type", "feature_at_initiation_by_hospital_type.csv",  _canon_hospital_type,   False),
    ]
    for group_col, fname, canon_fn, with_hospital_type in group_specs:
        if group_col not in coh.columns:
            print(f"  SKIP {fname}: no '{group_col}' column for this site")
            continue

        keep_cols = ["stay_id", group_col]
        if with_hospital_type and "hospital_type" in coh.columns:
            keep_cols.append("hospital_type")
        group_map = coh[keep_cols].copy()
        if canon_fn is not None:
            group_map[group_col] = canon_fn(group_map[group_col])
        if "hospital_type" in group_map.columns:
            group_map["hospital_type"] = _canon_hospital_type(group_map["hospital_type"])
        group_map = group_map.dropna(subset=[group_col])

        merged = init.merge(group_map, on="stay_id", how="inner")
        if merged.empty:
            print(f"  SKIP {fname}: no initiating patients with a known '{group_col}'")
            continue

        rows = []
        for grp, sub in merged.groupby(group_col):
            n_grp_total = len(sub)
            extra = {"group_column": group_col, "group": grp}
            if "hospital_type" in sub.columns:
                modal = sub["hospital_type"].mode(dropna=True)
                extra["hospital_type"] = modal.iloc[0] if len(modal) else None
            for col in GROUP_BREAKDOWN_FEATURES:
                if col not in sub.columns:
                    continue
                arr = pd.to_numeric(sub[col], errors="coerce").dropna()
                rows.append(_feature_stats_row(arr, n_grp_total, extra={**extra, "feature": col}))

        out = OUTPUT_DIR / fname
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 5a+5b: feature_thresholds_youden.csv + feature_roc_curves.csv
# ============================================================

def _threshold_grid(train_vals, is_binary):
    """Fixed grid that cannot reconstruct individual observations."""
    if is_binary:
        # Three points give proper ROC corners for a binary predictor
        return np.array([-0.5, 0.5, 1.5])

    p5  = float(np.nanpercentile(train_vals[np.isfinite(train_vals)], 5))
    p95 = float(np.nanpercentile(train_vals[np.isfinite(train_vals)], 95))

    # Round to 2 significant figures so thresholds don't reveal individual values
    def _sig2(x):
        if x == 0:
            return 0.0
        mag = int(np.floor(np.log10(abs(x))))
        return round(x, -(mag - 1))

    lo, hi = _sig2(p5), _sig2(p95)
    if lo >= hi:
        hi = lo + abs(lo * 0.1) + 1e-9
    return np.linspace(lo, hi, N_THRESH_GRID)


def _eval_thresh(y_true, y_score, threshold, direction):
    pred = (y_score > threshold if direction == "gt" else y_score < threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"tpr": sens, "fpr": 1 - spec,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "youden_j": sens + spec - 1.0}


_integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _auc(fprs, tprs):
    """Trapezoidal AUC with (0,0) and (1,1) corner anchors added.

    The threshold grid covers only [p5, p95] of training values, so without
    explicit corners the computed area misses the tails.  Adding (0,0) and
    (1,1) gives the correct AUROC under the assumption that the threshold can
    extend to ±∞ (standard convention).
    """
    fprs_ext = list(fprs) + [0.0, 1.0]
    tprs_ext = list(tprs) + [0.0, 1.0]
    idx = np.argsort(fprs_ext)
    f   = np.array(fprs_ext)[idx]
    t   = np.array(tprs_ext)[idx]
    return float(_integrate(t, f))


def write_roc_outputs(feat):
    # Per-timestep "at-risk" dataset: hours where patient was not yet on vasopressin
    fs = feat.sort_values(["stay_id", "time_hour"]).copy()
    fs["prev_vaso"] = fs.groupby("stay_id")["action_vaso"].shift(1).fillna(0)
    pred = fs[fs["prev_vaso"] == 0].copy()
    pred["outcome"] = pred["action_vaso"].astype(int)

    thresh_rows = []
    roc_rows    = []

    for col in ANALYSIS_FEATURES:
        if col not in pred.columns:
            continue
        is_bin = col in ANALYSIS_FEATURES_BIN

        tr      = pred[pred["split"] == "train"]
        tr_vals = pd.to_numeric(tr[col], errors="coerce").fillna(0).values
        tr_y    = tr["outcome"].values.astype(int)

        grid = _threshold_grid(tr_vals, is_bin)

        # Evaluate all grid points on train; determine direction
        gt_fprs, gt_tprs = [], []
        for tau in grid:
            m = _eval_thresh(tr_y, tr_vals, tau, "gt")
            gt_fprs.append(m["fpr"])
            gt_tprs.append(m["tpr"])

        auc_gt = _auc(gt_fprs, gt_tprs)
        direction = "gt" if auc_gt >= 0.5 else "lt"

        if direction == "lt":
            lt_fprs, lt_tprs = [], []
            for tau in grid:
                m = _eval_thresh(tr_y, tr_vals, tau, "lt")
                lt_fprs.append(m["fpr"])
                lt_tprs.append(m["tpr"])
            fprs_for_j, tprs_for_j = lt_fprs, lt_tprs
        else:
            fprs_for_j, tprs_for_j = gt_fprs, gt_tprs

        # Optimal threshold by Youden's J on train
        j_vals  = [t - f for t, f in zip(tprs_for_j, fprs_for_j)]
        opt_i   = int(np.argmax(j_vals))
        opt_tau = float(grid[opt_i])

        # Report on each split
        for split in ("train", "val", "test"):
            sp      = pred[pred["split"] == split]
            sv      = pd.to_numeric(sp[col], errors="coerce").fillna(0).values
            sy      = sp["outcome"].values.astype(int)
            n       = len(sy)
            n_pos   = int(sy.sum())
            n_neg   = n - n_pos

            if n < SUPPRESS_K or n_pos < SUPPRESS_K or n_neg < SUPPRESS_K:
                thresh_rows.append({
                    "feature": col, "split": split,
                    "optimal_threshold": _r(opt_tau), "direction": direction,
                    "auc": None, "sensitivity": None, "specificity": None,
                    "youden_j": None, "ppv": None, "npv": None,
                    "n": _n_str(n), "n_pos": _n_str(n_pos), "n_neg": _n_str(n_neg),
                })
                continue

            # ROC curve on grid
            sp_fprs, sp_tprs = [], []
            for tau in grid:
                m = _eval_thresh(sy, sv, tau, direction)
                sp_fprs.append(m["fpr"])
                sp_tprs.append(m["tpr"])
                roc_rows.append({
                    "feature": col, "split": split,
                    "threshold":   _r(float(tau)),
                    "tpr":         _r(m["tpr"]),
                    "fpr":         _r(m["fpr"]),
                    "sensitivity": _r(m["sens"]),
                    "specificity": _r(m["spec"]),
                })

            auc = _auc(sp_fprs, sp_tprs)

            # Apply train-derived threshold
            m_opt = _eval_thresh(sy, sv, opt_tau, direction)
            thresh_rows.append({
                "feature": col, "split": split,
                "optimal_threshold": _r(opt_tau), "direction": direction,
                "auc":         _r(auc),
                "sensitivity": _r(m_opt["sens"]),
                "specificity": _r(m_opt["spec"]),
                "youden_j":    _r(m_opt["youden_j"]),
                "ppv":         _r(m_opt["ppv"]),
                "npv":         _r(m_opt["npv"]),
                "n": n, "n_pos": n_pos, "n_neg": n_neg,
            })

    out1 = OUTPUT_DIR / "feature_thresholds_youden.csv"
    out2 = OUTPUT_DIR / "feature_roc_curves.csv"
    pd.DataFrame(thresh_rows).to_csv(out1, index=False)
    pd.DataFrame(roc_rows).to_csv(out2, index=False)
    print(f"  Wrote   {out1.name}")
    print(f"  Wrote   {out2.name}")


# ============================================================
# README
# ============================================================

README = f"""# Site Summary Outputs — Federated Privacy Manifest

Generated by: analysis/site_summary.py

## Privacy guarantees
- No patient-level records, identifiers, free text, or exact dates.
- Cell suppression: any statistic derived from fewer than {SUPPRESS_K} patients is
  reported as "<{SUPPRESS_K}"; all derived statistics for that cell are blank.
- Continuous summaries rounded to {ROUND_N} decimal places.
- ROC/threshold curves evaluated on a fixed {N_THRESH_GRID}-point grid derived from
  2-significant-figure rounded [p5, p95] of training data — NOT one per observation.
  AUC is computed with standard (0,0) and (1,1) corner anchors added to the grid-based
  curve, so it reflects the full AUROC rather than only the [p5, p95] excerpt.

## Analysis design
- Patient-level 70/15/15 train/val/test split, seed={RANDOM_SEED}.
- Outcome (analyses 4-5): per-timestep imminent vasopressin initiation.
  At-risk set = patient-hours where vasopressin was NOT active in previous hour.
  Outcome = action_vaso at current hour. Threshold selected on TRAIN only.
- Features (analyses 4-5): {", ".join(ANALYSIS_FEATURES)}.

## File schemas

### cohort_filter_counts.csv
Cohort inclusion flowchart. Columns: step, n_hospitalizations.

### split_counts.csv
Patients per split by ever-vasopressin group.
Columns: split, outcome_group (ever_vaso_yes/no/total), n_patients.

### baseline_table1.csv
Baseline characteristics stratified by eventual vasopressin (vaso/no_vaso) and overall.
One row per variable (or per level for categoricals). location_type and hospital_type are
canonicalized (ICU-type / academic-vs-community labels) before counting so spelling/casing
differences across sites don't fragment categories.
Columns: variable, level, type, smd,
  {{group}}_n, {{group}}_n_missing, {{group}}_mean, {{group}}_sd,
  {{group}}_median, {{group}}_q25, {{group}}_q75, {{group}}_min, {{group}}_max, {{group}}_pct
where group ∈ {{vaso, no_vaso, overall}}.
Note: min/max are included per specification; consider suppressing in final sharing if group
sizes are near K.

### feature_at_initiation.csv
Feature values at the first vasopressin initiation timestep per initiating patient.
Columns: feature, n, n_missing, mean, sd, median, q25, q75, min, max.

### feature_at_initiation_by_icu.csv / feature_at_initiation_by_hospital.csv
Same feature-at-initiation statistics as feature_at_initiation.csv, but broken down by
ICU type (canonicalized location_type) or by hospital_id, restricted to
{', '.join(GROUP_BREAKDOWN_FEATURES)} — feeds the cross-site ICU-type/hospital box-and-whisker
at-initiation plots. Absent for sites without a usable location_type/hospital_id column,
or where a site has fewer than 2 groups (e.g. single-hospital sites for the hospital file).
The hospital file also carries a hospital_type column (academic/community/unknown,
canonicalized) alongside each hospital_id group.
Columns: group_column, group, hospital_type, feature, n, n_missing, mean, sd, median, q25, q75, min, max.

### feature_thresholds_youden.csv
Per-feature (× split) threshold performance for imminent initiation.
Optimal threshold fixed from TRAIN; applied unchanged to val and test.
Columns: feature, split, optimal_threshold, direction (gt/lt),
  auc, sensitivity, specificity, youden_j, ppv, npv, n, n_pos, n_neg.

### feature_roc_curves.csv
Full ROC curve for coordinating site to replot. No patient-level data.
Columns: feature, split, threshold, tpr, fpr, sensitivity, specificity.
"""


def write_readme():
    out = OUTPUT_DIR / "README.md"
    out.write_text(README, encoding="utf-8")
    print(f"  Wrote   {out.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    global INPUT_DIR, OUTPUT_DIR

    # Read patient-level intermediate (PHI, local); write shareable aggregates.
    INPUT_DIR  = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"
    OUTPUT_DIR = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / COHORT_NAME

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Site:   {SITE_NAME}")
    print(f"Cohort: {COHORT_NAME}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    print("[1/7] Filter counts")
    write_filter_counts()

    print("[2/7] Loading + splitting")
    coh, feat = load_and_split()
    tr, va, te = [(coh["split"] == s).sum() for s in ("train", "val", "test")]
    print(f"  {len(coh):,} patients  train={tr}  val={va}  test={te}")
    print(f"  Ever-vaso: {coh['ever_vaso'].sum():,}  "
          f"Never-vaso: {(coh['ever_vaso']==0).sum():,}")

    print("[3/7] Split counts")
    write_split_counts(coh)

    print("[4/7] Baseline table 1")
    write_baseline_table1(coh)

    print("[5/7] Feature at initiation")
    write_feature_at_initiation(feat)

    print("[6/7] Feature at initiation, by ICU type / by hospital")
    write_feature_at_initiation_by_group(coh, feat)

    print("[7/7] ROC / threshold analysis")
    write_roc_outputs(feat)

    write_readme()

    print(f"\nDone. Outputs in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>6.1f} KB")


if __name__ == "__main__":
    main()
