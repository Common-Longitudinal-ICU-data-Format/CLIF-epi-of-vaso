#!/usr/bin/env python3
"""
02_site_summary.py

Federated-safe aggregate summary for a septic shock cohort.

Reads (PHI, local only):
  output/patient_level_data_<SITE>/cohort_<cohort>.parquet   (sepsis3 or rhee)
  output/patient_level_data_<SITE>/features.parquet          (union; filtered here to cohort)
  output/patient_level_data_<SITE>/cohort_filter_counts_<cohort>.csv

Writes only aggregates (CSV + one JSON) to output/upload_to_box_<SITE>/<cohort>/ — no
patient-level data leaves the site. cohort_tables.json carries the per-cohort Table 1
(baseline at t=0), Table 2 (never- vs ever-vasopressin, incl. Mann-Whitney U for van Elteren
pooling) and Table 3 (at vasopressin initiation) statistics (formerly 03_cohort_comparison_summary.py).

Privacy guarantees
  - No row-level values, identifiers, free text, or exact dates in any output.
  - Cell suppression: n < K=11 → count shown as "<11", derived stats blank.
  - Continuous summaries rounded to 3 decimal places.

Usage:
    uv run python code/02_site_summary.py                     # runs both sepsis3 and rhee
    uv run python code/02_site_summary.py --cohort rhee
    uv run python code/02_site_summary.py --cohort sepsis3 --site MIMIC
"""

import json
import re
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

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
_parser.add_argument("--cohort", default="both",
                     choices=["sepsis3", "rhee", "rhee_clifpy", "both"],
                     help="Which cohort file to read (cohort_<cohort>.parquet); "
                          "'both' (default) runs sepsis3, rhee, and rhee_clifpy")
_cli_args, _passthrough_args = _parser.parse_known_args()

if _cli_args.cohort == "both":
    import subprocess
    for _c in ("sepsis3", "rhee", "rhee_clifpy"):
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

# Clinical features summarised at vasopressin initiation
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

# Fallback labels for location_category values when location_type is absent
# (matches 05_site_variation_analysis.py for consistent labels).
_LOC_CAT_CANON = {
    "icu":            "ICU (unspecified)",
    "ward":           "Ward",
    "ed":             "ED / Emergency",
    "emergency":      "ED / Emergency",
    "or":             "OR / Procedural",
    "operating_room": "OR / Procedural",
    "procedure_room": "OR / Procedural",
    "procedural":     "OR / Procedural",
    "stepdown":       "Step-down / IMC",
    "step_down":      "Step-down / IMC",
    "intermediate":   "Step-down / IMC",
    "imc":            "Step-down / IMC",
    "other":          "Other",
}


def _canon_icu_series(raw: pd.Series) -> pd.Series:
    """Canonicalize location strings into ICU-type labels.

    ICU subtypes (location_type) → abbreviated names via _ICU_CANON.
    Broad categories (location_category fallback) → readable labels via
    _LOC_CAT_CANON.  Genuinely missing → NaN.
    """
    low = raw.astype(str).str.lower().str.strip()
    is_missing = raw.isna() | low.isin(["none", "nan", ""])
    canon = low.map(_ICU_CANON)
    unmapped = canon.isna() & ~is_missing
    canon[unmapped] = low[unmapped].map(_LOC_CAT_CANON).fillna("Other ICU")
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

    return coh, feat, coh_all, feat_all


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
    # Derive effective location: location_type when non-null, else location_category fallback
    if "location_type" in coh.columns or "location_category" in coh.columns:
        eff = coh["location_type"].copy() if "location_type" in coh.columns else pd.Series(pd.NA, index=coh.index)
        if "location_category" in coh.columns:
            is_missing = eff.isna() | eff.astype(str).str.lower().str.strip().isin(["none", "nan", ""])
            eff[is_missing] = coh.loc[is_missing, "location_category"]
        coh["location_type"] = _canon_icu_series(eff)
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

    # Derive effective location: location_type when non-null, else location_category fallback
    # (ward/step-down patients only have location_category; location_type is NaN for them)
    eff = coh["location_type"].copy() if "location_type" in coh.columns else pd.Series(pd.NA, index=coh.index)
    if "location_category" in coh.columns:
        is_missing = eff.isna() | eff.astype(str).str.lower().str.strip().isin(["none", "nan", ""])
        eff[is_missing] = coh.loc[is_missing, "location_category"]
    coh = coh.copy()
    coh["_effective_location"] = _canon_icu_series(eff)

    group_specs = [
        ("_effective_location", "feature_at_initiation_by_icu.csv",         None,                   False),
        ("hospital_id",         "feature_at_initiation_by_hospital.csv",     None,                   True),
        ("hospital_type",       "feature_at_initiation_by_hospital_type.csv", _canon_hospital_type,  False),
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
# OUTPUT 5: cohort_tables.json  (Table 1 / 2 / 3 aggregates; formerly 03)
# ============================================================

_PRESSOR_COLS = ["norepinephrine", "epinephrine", "dopamine", "phenylephrine", "angiotensin ii"]

# variable -> (label unused here; kept only as column name) for each table.
TABLE1_CONT = ["age", "weight", "cci_score", "norepinephrine", "nee", "heart_rate", "mbp",
               "respiratory_rate", "temperature", "gcs", "sepsis_onset_sofa", "wbc",
               "initial_lactate", "creatinine", "bun", "hemoglobin", "platelet", "bilirubin",
               "icu_los_days", "hospital_los_days", "vaso_init_hour"]
TABLE1_BIN = ["_male", "ventil", "rrt"]

TABLE2_CONT = TABLE1_CONT
TABLE2_BIN = TABLE1_BIN + ["hospital_death"]

TABLE3_CONT = ["vaso_init_hour", "norepinephrine", "nee", "heart_rate", "mbp",
               "respiratory_rate", "temperature", "gcs", "sofa", "wbc", "lactate",
               "creatinine", "bun", "hemoglobin", "platelet", "bilirubin"]
TABLE3_BIN = ["ventil", "rrt"]
TABLE3_PREHOUR_CONT = ["_n_pressors"]


def _cont_stats(vals) -> dict | None:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    n = len(arr)
    if n < SUPPRESS_K:
        return None
    return {
        "n": n, "mean": _r(arr.mean()), "sd": _r(arr.std()),
        "median": _r(arr.median()), "q25": _r(arr.quantile(0.25)), "q75": _r(arr.quantile(0.75)),
        "min": _r(arr.min()), "max": _r(arr.max()),
    }


def _bin_stats(vals) -> dict | None:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    n = len(arr)
    pos = int(arr.sum())
    if n < SUPPRESS_K:
        return None
    return {"n": n, "pos": pos, "pct": _r(pos / n * 100)}


def _cat_stats(vals) -> dict:
    s = pd.Series(vals).dropna()
    out = {}
    for level, cnt in s.value_counts().items():
        n = len(s)
        out[str(level)] = ({"n": int(cnt), "pct": _r(int(cnt) / n * 100)}
                            if cnt >= SUPPRESS_K else None)
    return out


def _box_stats(vals) -> dict | None:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values
    n = len(arr)
    if n < SUPPRESS_K:
        return None
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within = arr[(arr >= lo_fence) & (arr <= hi_fence)]
    whislo = float(within.min()) if len(within) else float(q1)
    whishi = float(within.max()) if len(within) else float(q3)
    return {"n": int(n), "q1": _r(q1), "median": _r(med), "q3": _r(q3),
            "whislo": _r(whislo), "whishi": _r(whishi)}


def _mwu_stats(a, b) -> dict | None:
    """Return {U, n1, n2} for Mann-Whitney U test (vaso group = a, no-vaso = b).
    Stored alongside p-values so the coordinating site can compute the
    van Elteren (stratified Wilcoxon) pooled p-value without individual data.
    """
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    try:
        res = mannwhitneyu(a, b, alternative="two-sided")
        return {"U": _r(res.statistic), "n1": int(len(a)), "n2": int(len(b))}
    except Exception:
        return None


def _pval_cont(a, b) -> float | None:
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return _r(p)
    except Exception:
        return None


def _pval_bin(a, b) -> float | None:
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().astype(int)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().astype(int)
    if len(a) < SUPPRESS_K or len(b) < SUPPRESS_K:
        return None
    try:
        table = [[int(a.sum()), len(a) - int(a.sum())], [int(b.sum()), len(b) - int(b.sum())]]
        _, p, _, _ = chi2_contingency(table)
        return _r(p)
    except Exception:
        return None


def _year_range(vals) -> dict | None:
    years = []
    for v in pd.Series(vals).dropna():
        years.extend(int(y) for y in re.findall(r"\d{4}", str(v)))
    if not years:
        return None
    return {"min_year": min(years), "max_year": max(years)}


def _col(df, col):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return df[col]


def hour0_features(features_df):
    if features_df is None:
        return pd.DataFrame(columns=["stay_id"])
    return features_df[features_df["time_hour"] == 0]


def _prior_vaso_ids_from(cohort_df):
    """Return set of stay_ids with vasopressin before NE start, if the column exists."""
    if cohort_df is not None and "vaso_before_traj" in cohort_df.columns:
        return set(cohort_df.loc[cohort_df["vaso_before_traj"] == 1, "stay_id"])
    return set()


def compute_vaso_flags(features_df, exclude_ids=None):
    if features_df is None or len(features_df) == 0:
        return pd.DataFrame(columns=["stay_id", "ever_vaso", "vaso_init_hour"])
    _feat = features_df
    if exclude_ids:
        _feat = features_df[~features_df["stay_id"].isin(exclude_ids)]
    ever = (_feat.groupby("stay_id")["action_vaso"].max()
            .rename("ever_vaso").reset_index())
    first_hr = (_feat[_feat["action_vaso"] == 1]
                .groupby("stay_id")["time_hour"].min()
                .rename("vaso_init_hour").reset_index())
    return ever.merge(first_hr, on="stay_id", how="left")


def build_baseline_with_vaso(cohort_df, features_df):
    if cohort_df is None or len(cohort_df) == 0:
        return None
    hr0 = hour0_features(features_df)
    extra_cols = [c for c in hr0.columns if c not in ("stay_id", "time_hour")]
    base = cohort_df.merge(hr0[["stay_id"] + extra_cols], on="stay_id", how="left")
    flags = compute_vaso_flags(features_df)  # ever_vaso includes prior-vaso patients intentionally
    base = base.merge(flags[["stay_id", "ever_vaso", "vaso_init_hour"]], on="stay_id", how="left")
    base["ever_vaso"] = base["ever_vaso"].fillna(0).astype(int)
    base["_male"] = (base["gender"] == "M").astype(int) if "gender" in base.columns else np.nan
    return base


def add_pressor_count(df):
    if df is None:
        return None
    df = df.copy()
    active = pd.DataFrame({
        c: (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int) if c in df.columns else 0
        for c in _PRESSOR_COLS
    })
    df["_n_pressors"] = active.sum(axis=1)
    return df


def build_vaso_init_df(cohort_df, features_df):
    if cohort_df is None or len(cohort_df) == 0 or features_df is None:
        return None
    # Exclude prior-vaso patients: their vaso started before NE, so vaso_init_hour=0
    # reflects NE-start physiology rather than the true vasopressin-initiation state.
    _excl = _prior_vaso_ids_from(cohort_df)
    flags = compute_vaso_flags(features_df, exclude_ids=_excl)
    ever = flags[flags["ever_vaso"] == 1].dropna(subset=["vaso_init_hour"])
    ever_ids = set(cohort_df["stay_id"]) & set(ever["stay_id"])
    if not ever_ids:
        return None
    ever = ever[ever["stay_id"].isin(ever_ids)]
    at_init = features_df.merge(ever[["stay_id", "vaso_init_hour"]], on="stay_id")
    at_init = at_init[at_init["time_hour"] == at_init["vaso_init_hour"]]
    cohort_sub = cohort_df[cohort_df["stay_id"].isin(ever_ids)]
    return cohort_sub.merge(at_init, on="stay_id", how="inner")


def build_vaso_init_prehour_df(cohort_df, features_df):
    if cohort_df is None or len(cohort_df) == 0 or features_df is None:
        return None
    _excl = _prior_vaso_ids_from(cohort_df)
    flags = compute_vaso_flags(features_df, exclude_ids=_excl)
    ever = flags[flags["ever_vaso"] == 1].dropna(subset=["vaso_init_hour"]).copy()
    ever["prehour"] = ever["vaso_init_hour"] - 1
    ever = ever[ever["prehour"] >= 0]
    ever_ids = set(cohort_df["stay_id"]) & set(ever["stay_id"])
    if not ever_ids:
        return None
    ever = ever[ever["stay_id"].isin(ever_ids)]
    pre = features_df.merge(ever[["stay_id", "prehour"]], on="stay_id")
    pre = pre[pre["time_hour"] == pre["prehour"]]
    cohort_sub = cohort_df[cohort_df["stay_id"].isin(ever_ids)]
    merged = cohort_sub.merge(pre, on="stay_id", how="inner")
    return add_pressor_count(merged)


# ============================================================
# AGGREGATE ASSEMBLY
# ============================================================

def build_variable_block(df, cont_vars, bin_vars) -> dict:
    out = {}
    for col in cont_vars:
        out[col] = _cont_stats(_col(df, col))
    for col in bin_vars:
        out[col] = _bin_stats(_col(df, col))
    if df is not None and "race" in df.columns:
        out["race"] = _cat_stats(df["race"])
    return out


def build_table1(baseline_df) -> dict:
    if baseline_df is None:
        return {"n": 0}
    block = build_variable_block(baseline_df, TABLE1_CONT, TABLE1_BIN)
    block["n"] = len(baseline_df)
    block["enrollment_period"] = _year_range(_col(baseline_df, "anchor_year_group"))
    return block


def build_table2(baseline_df) -> dict:
    if baseline_df is None:
        return {"never": {"n": 0}, "ever": {"n": 0}, "pvalues": {}}
    never_df = baseline_df[baseline_df["ever_vaso"] == 0]
    ever_df = baseline_df[baseline_df["ever_vaso"] == 1]

    never_block = build_variable_block(never_df, TABLE2_CONT, TABLE2_BIN)
    never_block["n"] = len(never_df)
    ever_block = build_variable_block(ever_df, TABLE2_CONT, TABLE2_BIN)
    ever_block["n"] = len(ever_df)

    pvals = {}
    for col in TABLE2_CONT:
        pvals[col] = _pval_cont(_col(never_df, col), _col(ever_df, col))
    for col in TABLE2_BIN:
        pvals[col] = _pval_bin(_col(never_df, col), _col(ever_df, col))
    if "race" in baseline_df.columns:
        for level in sorted(set(baseline_df["race"].dropna().unique().tolist())):
            nv = (never_df["race"] == level).astype(int) if "race" in never_df.columns else pd.Series(dtype=int)
            ev = (ever_df["race"] == level).astype(int) if "race" in ever_df.columns else pd.Series(dtype=int)
            pvals[f"race:{level}"] = _pval_bin(nv, ev)

    # Mann-Whitney U statistics for van Elteren pooling at the coordinating site.
    # Keys match TABLE2_CONT column names; values are {U, n1, n2} where
    # n1 = ever-vaso size, n2 = never-vaso size.
    mwu = {}
    for col in TABLE2_CONT:
        stats = _mwu_stats(_col(ever_df, col), _col(never_df, col))
        if stats:
            mwu[col] = stats

    return {"never": never_block, "ever": ever_block, "pvalues": pvals, "mwu_stats": mwu}


def build_table3(vaso_init_df, vaso_prehour_df) -> dict:
    if vaso_init_df is None:
        return {"n": 0}
    block = build_variable_block(vaso_init_df, TABLE3_CONT, TABLE3_BIN)
    block["n"] = len(vaso_init_df)
    if vaso_prehour_df is not None:
        for col in TABLE3_PREHOUR_CONT:
            block[col] = _cont_stats(_col(vaso_prehour_df, col))
    else:
        for col in TABLE3_PREHOUR_CONT:
            block[col] = {"n": 0}
    return block


def write_cohort_tables(coh_raw, feat_raw):
    """Table 1 (baseline at t=0), Table 2 (never vs ever vasopressin; p-values and
    Mann-Whitney U for van Elteren pooling) and Table 3 (at vasopressin initiation).

    Uses the raw cohort + features (before 02's vaso-at-t0 exclusion and LOCF) so the
    numbers are identical to the previous 03_cohort_comparison_summary.py output.
    Cells with fewer than SUPPRESS_K patients are omitted."""
    baseline  = build_baseline_with_vaso(coh_raw, feat_raw)
    vaso_init = build_vaso_init_df(coh_raw, feat_raw)
    prehour   = build_vaso_init_prehour_df(coh_raw, feat_raw)

    payload = {
        "site":   SITE_NAME,
        "cohort": COHORT_NAME,
        "table1": build_table1(baseline),
        "table2": build_table2(baseline),
        "table3": build_table3(vaso_init, prehour),
    }
    out = OUTPUT_DIR / "cohort_tables.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote   {out.name}")


# ============================================================
# README
# ============================================================

README = f"""# Site Summary Outputs — Federated Privacy Manifest

Generated by: code/02_site_summary.py

## Privacy guarantees
- No patient-level records, identifiers, free text, or exact dates.
- Cell suppression: any statistic derived from fewer than {SUPPRESS_K} patients is
  reported as "<{SUPPRESS_K}"; all derived statistics for that cell are blank.
- Continuous summaries rounded to {ROUND_N} decimal places.

## Analysis design
- Patient-level 70/15/15 train/val/test split, seed={RANDOM_SEED} (split_counts.csv).
- Features summarised at vasopressin initiation: {", ".join(ANALYSIS_FEATURES)}.

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

### cohort_tables.json
Per-cohort aggregates for the paper tables (no patient-level data; any per-variable
statistic from fewer than {SUPPRESS_K} patients is omitted).
- `table1`: baseline (t=0) stats — {{n, enrollment_period, <variable>: {{n, mean, sd, median, q25,
  q75, min, max}} or {{n, pos, pct}}, race: {{level: {{n, pct}}}}}}
- `table2`: {{never, ever}} blocks shaped like table1, plus `pvalues` {{variable: p}} and
  `mwu_stats` {{variable: {{U, n1, n2}}}} (Mann-Whitney U + group sizes for van Elteren pooling
  at the coordinating site)
- `table3`: characteristics at vasopressin initiation (ever-vaso only), shaped like table1, plus
  `_n_pressors` (from the hour before initiation)
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
    coh, feat, coh_raw, feat_raw = load_and_split()
    tr, va, te = [(coh["split"] == s).sum() for s in ("train", "val", "test")]
    print(f"  {len(coh):,} patients  train={tr}  val={va}  test={te}")
    print(f"  Ever-vaso: {coh['ever_vaso'].sum():,}  "
          f"Never-vaso: {(coh['ever_vaso']==0).sum():,}")

    print("[3/7] Split counts")
    write_split_counts(coh)

    print("[4/7] Baseline table 1")
    write_baseline_table1(coh)

    print("[5/7] Feature at initiation")
    # Exclude patients whose vasopressin started before NE: for them the first
    # 0→1 action_vaso transition on the grid occurs at t=0, which reflects NE-start
    # physiology rather than true vasopressin-initiation state.
    _prior_vaso_ids_02 = (
        set(coh.loc[coh["vaso_before_traj"] == 1, "stay_id"])
        if "vaso_before_traj" in coh.columns else set()
    )
    feat_ne_first = feat[~feat["stay_id"].isin(_prior_vaso_ids_02)].copy() if _prior_vaso_ids_02 else feat
    write_feature_at_initiation(feat_ne_first)

    print("[6/7] Feature at initiation, by ICU type / by hospital")
    write_feature_at_initiation_by_group(coh, feat_ne_first)

    print("[7/7] Cohort tables (Table 1 / 2 / 3 aggregates)")
    write_cohort_tables(coh_raw, feat_raw)

    write_readme()

    print(f"\nDone. Outputs in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>6.1f} KB")


if __name__ == "__main__":
    main()
