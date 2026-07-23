#!/usr/bin/env python3
"""
02b_cohort_comparison_summary.py

Federated-safe aggregate summary comparing the Sepsis-3 (CMS) and Rhee/CDC
ASE cohorts at this site — everything 12_cohort_comparison_report.py needs
to build its tables/plots, computed locally and shared as aggregate-only
statistics (no patient-level rows).

Unlike 02_site_summary.py this script needs BOTH cohort files at once (to
compute their overlap), so it is not run per-cohort — it reads
cohort_sepsis3.parquet and cohort_rhee.parquet together.

Reads (PHI, local only):
  output/patient_level_data_<SITE>/cohort_sepsis3.parquet
  output/patient_level_data_<SITE>/cohort_rhee.parquet
  output/patient_level_data_<SITE>/features.parquet

Writes only aggregate JSON to output/upload_to_box_<SITE>/cohort_comparison/
— no patient-level data leaves the site.

Privacy guarantees (same as 02_site_summary.py)
  - No row-level values, identifiers, free text, or exact dates.
  - Cell suppression: any statistic derived from fewer than K=11 patients is
    omitted (null) for that group/variable.
  - Continuous summaries rounded to 3 decimal places.
  - Only p-values (single floats), never raw values, cross the site boundary
    for the never-vs-ever-vasopressin comparison.
  - Cohort overlap / filter-cascade counts are shared unsuppressed, matching
    the existing convention for cohort_filter_counts.csv (CONSORT-style flow
    counts, not per-variable statistics).

Usage:
    uv run python code/02b_cohort_comparison_summary.py
    uv run python code/02b_cohort_comparison_summary.py --site MIMIC
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent


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
_parser.add_argument("--site", default=None, help="Override SITE_NAME from config")
_cli_args, _ = _parser.parse_known_args()

_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit(
        "ERROR: config/config.py not found.\n"
        "Copy config/config.example.py to config/config.py and set SITE_NAME, CLIF_DIR, OUTPUT_ROOT."
    )
SITE_NAME = _cli_args.site if _cli_args.site else getattr(_cfg, "SITE_NAME", "UCMC")
OUTPUT_ROOT = getattr(_cfg, "OUTPUT_ROOT", None)
if OUTPUT_ROOT is None:
    raise SystemExit("ERROR: OUTPUT_ROOT is not set in config/config.py.")
OUTPUT_ROOT = Path(OUTPUT_ROOT)

SUPPRESS_K = 11
ROUND_N = 3

_PRESSOR_COLS = ["norepinephrine", "epinephrine", "dopamine", "phenylephrine", "angiotensin ii"]

# variable -> (label unused here; kept only as column name) for each table.
TABLE1_CONT = ["age", "weight", "cci_score", "norepinephrine", "nee", "heart_rate", "mbp",
               "respiratory_rate", "temperature", "gcs", "sepsis_onset_sofa", "wbc",
               "initial_lactate", "creatinine", "bun", "hemoglobin", "platelet", "bilirubin",
               "icu_los_days", "hospital_los_days"]
TABLE1_BIN = ["_male", "ventil", "rrt"]

TABLE2_CONT = TABLE1_CONT
TABLE2_BIN = TABLE1_BIN + ["hospital_death"]

TABLE3_CONT = ["vaso_init_hour", "norepinephrine", "nee", "heart_rate", "mbp",
               "respiratory_rate", "temperature", "gcs", "sofa", "wbc", "lactate",
               "creatinine", "bun", "hemoglobin", "platelet", "bilirubin"]
TABLE3_BIN = ["ventil", "rrt"]
TABLE3_PREHOUR_CONT = ["_n_pressors"]

BOX_FEATURES = ["sepsis_onset_sofa", "initial_lactate", "age", "traj_hours"]

COHORTS = ["sepsis3", "rhee"]


# ============================================================
# HELPERS
# ============================================================

def _r(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, ROUND_N)
    except (TypeError, ValueError):
        return None


def _cont_stats(vals) -> dict | None:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    n = len(arr)
    if n < SUPPRESS_K:
        return {"n": n}
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
        return {"n": n}
    return {"n": n, "pos": pos, "pct": _r(pos / n * 100)}


def _cat_stats(vals) -> dict:
    s = pd.Series(vals).dropna()
    out = {}
    for level, cnt in s.value_counts().items():
        n = len(s)
        out[str(level)] = ({"n": int(cnt), "pct": _r(int(cnt) / n * 100)}
                            if cnt >= SUPPRESS_K else {"n": int(cnt)})
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


# ============================================================
# BUILDERS (mirrors 12_cohort_comparison_report.py's local versions)
# ============================================================

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
    base = base.merge(flags[["stay_id", "ever_vaso"]], on="stay_id", how="left")
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

    return {"never": never_block, "ever": ever_block, "pvalues": pvals}


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


def build_boxplots(cohort_dfs: dict) -> dict:
    out = {}
    for cohort, df in cohort_dfs.items():
        out[cohort] = {}
        for col in BOX_FEATURES:
            out[cohort][col] = _box_stats(_col(df, col)) if df is not None else None
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    input_dir = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"
    output_dir = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / "cohort_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Site:   {SITE_NAME}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    cohort_dfs, feat_by_cohort = {}, {}
    features_all = None
    fp = input_dir / "features.parquet"
    if fp.exists():
        features_all = pd.read_parquet(fp)

    for cohort in COHORTS:
        pq = input_dir / f"cohort_{cohort}.parquet"
        cohort_dfs[cohort] = pd.read_parquet(pq) if pq.exists() else None
        if cohort_dfs[cohort] is not None and features_all is not None:
            ids = set(cohort_dfs[cohort]["stay_id"])
            feat_by_cohort[cohort] = features_all[features_all["stay_id"].isin(ids)].copy()
        else:
            feat_by_cohort[cohort] = None

    if all(df is None for df in cohort_dfs.values()):
        sys.exit(f"No cohort parquets found under {input_dir}. Run 01/01b extract first.")

    # ---- Overlap (unsuppressed, CONSORT-style counts) ----
    s3_ids = set(cohort_dfs["sepsis3"]["stay_id"]) if cohort_dfs["sepsis3"] is not None else set()
    rhee_ids = set(cohort_dfs["rhee"]["stay_id"]) if cohort_dfs["rhee"] is not None else set()
    overlap = {
        "n_sepsis3": len(s3_ids), "n_rhee": len(rhee_ids),
        "n_both": len(s3_ids & rhee_ids),
        "n_sepsis3_only": len(s3_ids - rhee_ids),
        "n_rhee_only": len(rhee_ids - s3_ids),
    }
    print(f"[1/4] Cohort overlap: {overlap}")

    # ---- Baseline / vaso-init tables per cohort ----
    baseline_by_cohort, vaso_init_by_cohort, vaso_prehour_by_cohort = {}, {}, {}
    for cohort in COHORTS:
        feat = feat_by_cohort[cohort]
        baseline_by_cohort[cohort] = build_baseline_with_vaso(cohort_dfs[cohort], feat)
        vaso_init_by_cohort[cohort] = build_vaso_init_df(cohort_dfs[cohort], feat)
        vaso_prehour_by_cohort[cohort] = build_vaso_init_prehour_df(cohort_dfs[cohort], feat)

    print("[2/4] Table 1 (baseline @ t0) + Table 2 (never vs ever vaso)")
    table1 = {c: build_table1(baseline_by_cohort[c]) for c in COHORTS}
    table2 = {c: build_table2(baseline_by_cohort[c]) for c in COHORTS}

    print("[3/4] Table 3 (at vasopressin initiation)")
    table3 = {c: build_table3(vaso_init_by_cohort[c], vaso_prehour_by_cohort[c]) for c in COHORTS}

    print("[4/4] Boxplot 5-number summaries")
    boxplots = build_boxplots(cohort_dfs)

    payload = {
        "site": SITE_NAME,
        "overlap": overlap,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "boxplots": boxplots,
    }

    out = output_dir / "cohort_comparison_stats.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    readme = output_dir / "README.md"
    readme.write_text(
        "# Cohort Comparison Summary — Federated Privacy Manifest\n\n"
        "Generated by: code/02b_cohort_comparison_summary.py\n\n"
        "## Privacy guarantees\n"
        f"- No patient-level records; every per-variable statistic derived from fewer than "
        f"{SUPPRESS_K} patients is omitted (object contains only `n`).\n"
        f"- Continuous summaries rounded to {ROUND_N} decimal places.\n"
        "- `table2.*.pvalues` are single floats (Mann-Whitney U / chi-square) computed locally "
        "— no raw values cross the site boundary.\n"
        "- `overlap` counts (cohort sizes / intersection) are shared unsuppressed, matching the "
        "existing convention for cohort_filter_counts.csv (CONSORT-style flow counts).\n\n"
        "## cohort_comparison_stats.json schema\n"
        "- `overlap`: {n_sepsis3, n_rhee, n_both, n_sepsis3_only, n_rhee_only}\n"
        "- `table1.<cohort>`: baseline (t=0) stats — {n, enrollment_period, <variable>: {n, mean, sd, "
        "median, q25, q75, min, max} or {n, pos, pct}, race: {level: {n, pct}}}\n"
        "- `table2.<cohort>`: {never, ever} each shaped like table1's per-cohort block, plus "
        "`pvalues`: {variable: p}\n"
        "- `table3.<cohort>`: characteristics at vasopressin initiation (ever-vaso only), same "
        "shape as table1, plus `_n_pressors` (from the hour before initiation)\n"
        "- `boxplots.<cohort>.<feature>`: {n, q1, median, q3, whislo, whishi} — standard Tukey "
        "5-number summary (whis=1.5*IQR), directly usable with matplotlib's `ax.bxp()`\n",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
