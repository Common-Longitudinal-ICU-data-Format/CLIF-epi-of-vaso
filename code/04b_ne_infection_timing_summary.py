#!/usr/bin/env python3
"""
04b_ne_infection_timing_summary.py

Per-site, federated-safe summary of clinical event timing (blood culture ->
antibiotic -> norepinephrine), read directly from this site's raw CLIF
tables (config.py's CLIF_DIR/SITE_NAME — same as 01_clif_extract.py).

Computes the same underlying deltas as the old 09_ne_infection_timing.py did
(when it read UCMC/NU raw data directly at a single coordinating machine),
but instead of plotting from patient-level values, writes only aggregate
statistics to upload_to_box_<SITE>/timing/ — no patient-level rows leave the
site. 09_ne_infection_timing.py (coordinating-site) rebuilds the four plots
from these aggregates alone.

Privacy guarantees
  - No row-level values, identifiers, free text, or exact dates.
  - Histograms use a fixed grid (6h bins over a fixed range); any bin with
    fewer than K=11 patients is suppressed (set to 0).
  - Only median/Q25/Q75 (not raw values) are shared for Plot B's dot
    positions — no per-patient connecting lines leave the site.
  - Plot C (event timing vs Sepsis-3 status) is shared as suppressed 1D
    histograms per anchor x sepsis3-status, not a per-patient scatter.

Like 01_clif_extract.py, this script has no --site override: SITE_NAME and
CLIF_DIR both come from config/config.py as currently set, since CLIF_DIR is
what actually determines which site's raw data gets read (an override that
changed only the output folder name would silently mislabel data — see
run_pipeline.py, which edits config.py per site before calling this script).

Usage:
    uv run python code/04b_ne_infection_timing_summary.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from clifpy import ClifOrchestrator
    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE
    HAS_CLIFPY = True
except ModuleNotFoundError:
    HAS_CLIFPY = False

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


_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit("ERROR: config/config.py not found.")
SITE_NAME = getattr(_cfg, "SITE_NAME", "UCMC")
CLIF_DIR = getattr(_cfg, "CLIF_DIR", None)
OUTPUT_ROOT = getattr(_cfg, "OUTPUT_ROOT", None)
if OUTPUT_ROOT is None:
    raise SystemExit("ERROR: OUTPUT_ROOT is not set in config/config.py.")
OUTPUT_ROOT = Path(OUTPUT_ROOT)

if CLIF_DIR is None:
    sys.exit(
        f"ERROR: CLIF_DIR is not configured for site {SITE_NAME}.\n"
        "This script needs raw CLIF tables (same as 01_clif_extract.py) — "
        "it does not apply to MIMIC (see 01b_mimic_extract.py's own MIMIC_CLIF_DIR)."
    )
CLIF_DIR = Path(CLIF_DIR)

SUPPRESS_K = 11
ROUND_N = 3

INFECTION_WINDOW_HOURS = 24
SOFA_THRESHOLD = 2.0
LACTATE_THRESHOLD = 2.0
ANCHOR_WINDOW_H = 24

PLOT_A_X_MIN, PLOT_A_X_MAX = -72, 120
BIN_WIDTH_H = 6
PLOT_C_X_MAX = 120
PLOT_D_X_MIN, PLOT_D_X_MAX = -120, 120

ANCHORS = [("culture", "culture_dttm"), ("abx", "abx_dttm"), ("ne", "first_ne_dttm")]


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


def to_naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def _quantile_stats(vals) -> dict | None:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    n = len(arr)
    if n < SUPPRESS_K:
        return {"n": n}
    return {"n": n, "median": _r(arr.median()), "q25": _r(arr.quantile(0.25)), "q75": _r(arr.quantile(0.75))}


def _suppressed_hist(vals, lo, hi, width) -> dict:
    """Fixed-grid histogram; bins with < K patients suppressed to 0."""
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().clip(lo, hi).values
    edges = np.arange(lo, hi + width, width)
    counts, _ = np.histogram(arr, bins=edges)
    counts = np.where(counts < SUPPRESS_K, 0, counts)
    return {"n": int(len(arr)), "bin_edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


# ============================================================
# DATA EXTRACTION (mirrors old 09_ne_infection_timing.py, site-parametrized)
# ============================================================

def _find_qualifying_pairs(clif_dir: Path) -> pd.DataFrame:
    meds_i = pd.read_parquet(clif_dir / "clif_medication_admin_intermittent.parquet")
    cultures = pd.read_parquet(clif_dir / "clif_microbiology_culture.parquet")

    abx = meds_i[meds_i["med_group"] == "CMS_sepsis_qualifying_antibiotics"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    abx["admin_dttm"] = to_naive_utc(abx["admin_dttm"])

    blood_cx = cultures[
        (cultures["fluid_category"] == "blood_buffy") &
        cultures["collect_dttm"].notna() &
        (cultures["method_category"] == "culture")
    ][["hospitalization_id", "collect_dttm"]].copy()
    blood_cx["collect_dttm"] = to_naive_utc(blood_cx["collect_dttm"])

    merged = abx.merge(blood_cx, on="hospitalization_id", how="inner")
    merged["time_diff"] = (
        (merged["admin_dttm"] - merged["collect_dttm"]).dt.total_seconds().abs() / 3600
    )
    merged = merged[merged["time_diff"] <= INFECTION_WINDOW_HOURS].copy()
    merged["presumed_infection_dttm"] = merged[["admin_dttm", "collect_dttm"]].min(axis=1)
    return merged


def identify_suspected_infection(clif_dir: Path) -> pd.DataFrame:
    pairs = _find_qualifying_pairs(clif_dir)
    return (
        pairs.groupby("hospitalization_id")
             .agg(presumed_infection_dttm=("presumed_infection_dttm", "min"))
             .reset_index()
    )


def identify_qualifying_pair_timestamps(clif_dir: Path) -> pd.DataFrame:
    pairs = _find_qualifying_pairs(clif_dir)
    result = (
        pairs.sort_values("presumed_infection_dttm")
             .groupby("hospitalization_id")
             .first()
             .reset_index()
             [["hospitalization_id", "admin_dttm", "collect_dttm"]]
    )
    return result.rename(columns={"admin_dttm": "abx_dttm", "collect_dttm": "culture_dttm"})


def first_ne_anytime(clif_dir: Path, hosp_ids: set) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    ne = meds[
        (meds["med_category"] == "norepinephrine") &
        (meds["hospitalization_id"].isin(hosp_ids))
    ][["hospitalization_id", "admin_dttm"]].copy()
    ne["admin_dttm"] = to_naive_utc(ne["admin_dttm"])
    return (
        ne.groupby("hospitalization_id")
          .agg(first_ne_dttm=("admin_dttm", "min"))
          .reset_index()
    )


def compute_deltas(clif_dir: Path) -> pd.Series:
    inf = identify_suspected_infection(clif_dir)
    print(f"  Suspected infection: {len(inf):,}")
    ne = first_ne_anytime(clif_dir, set(inf["hospitalization_id"]))
    print(f"  Any NE during hosp: {len(ne):,}")
    merged = inf.merge(ne, on="hospitalization_id", how="inner")
    print(f"  With both: {len(merged):,}")
    delta_h = (merged["first_ne_dttm"] - merged["presumed_infection_dttm"]).dt.total_seconds() / 3600
    return delta_h.dropna().reset_index(drop=True)


def compute_timing_data(clif_dir: Path) -> pd.DataFrame:
    pairs = identify_qualifying_pair_timestamps(clif_dir)
    print(f"  Suspected infection: {len(pairs):,}")
    ne = first_ne_anytime(clif_dir, set(pairs["hospitalization_id"]))
    data = pairs.merge(ne, on="hospitalization_id", how="inner")
    print(f"  With NE: {len(data):,}")
    data["abx_delta_h"] = (data["abx_dttm"] - data["culture_dttm"]).dt.total_seconds() / 3600
    data["ne_delta_h"] = (data["first_ne_dttm"] - data["culture_dttm"]).dt.total_seconds() / 3600
    return data.reset_index(drop=True)


def _load_sofa_tables(co, hosp_ids: list) -> None:
    co.load_table("labs", filters={
        "hospitalization_id": hosp_ids,
        "lab_category": ["creatinine", "platelet_count", "po2_arterial",
                         "bilirubin_total", "lactate"],
    })
    co.load_table("vitals", filters={
        "hospitalization_id": hosp_ids,
        "vital_category": ["map", "spo2", "weight_kg"],
    })
    co.load_table("patient_assessments", filters={
        "hospitalization_id": hosp_ids,
        "assessment_category": ["gcs_total"],
    })
    co.load_table("medication_admin_continuous", filters={"hospitalization_id": hosp_ids})
    co.load_table("respiratory_support", filters={"hospitalization_id": hosp_ids})
    med_df = co.medication_admin_continuous.df.copy()
    med_df = med_df[med_df["med_dose"].notna() & med_df["med_dose_unit"].notna()]
    co.medication_admin_continuous.df = med_df
    co.medication_admin_continuous.df_converted = med_df.copy()
    co.medication_admin_continuous.df_converted["_convert_status"] = "success"


def _add_missing_med_cols(co) -> None:
    for col in ["norepinephrine_mcg_kg_min", "epinephrine_mcg_kg_min",
                "dopamine_mcg_kg_min", "dobutamine_mcg_kg_min"]:
        if col not in co.wide_df.columns:
            co.wide_df[col] = 0.0


def compute_sepsis3_at_anchors(clif_dir: Path, timing_df: pd.DataFrame, tmp_dir: Path) -> pd.DataFrame:
    hosp_ids = timing_df["hospitalization_id"].tolist()
    print(f"  Loading SOFA tables for {len(hosp_ids):,} patients...")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    co = ClifOrchestrator(
        data_directory=str(clif_dir), filetype="parquet", timezone="UTC",
        output_directory=str(tmp_dir),
    )
    _load_sofa_tables(co, hosp_ids)

    labs_lac = co.labs.df[co.labs.df["lab_category"] == "lactate"][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].copy()
    labs_lac["lab_result_dttm"] = to_naive_utc(labs_lac["lab_result_dttm"])
    labs_lac = labs_lac[labs_lac["lab_value_numeric"].notna()]

    results = timing_df[["hospitalization_id"]].copy()

    for label, col in ANCHORS:
        print(f"  SOFA + lactate at '{label}' anchor...")
        anchor_naive = pd.to_datetime(timing_df[col])
        anchor_aware = anchor_naive.dt.tz_localize("UTC")
        end_aware = anchor_aware + pd.Timedelta(hours=ANCHOR_WINDOW_H)

        cohort_df = timing_df[["hospitalization_id"]].copy()
        cohort_df["start_time"] = anchor_aware
        cohort_df["end_time"] = end_aware

        co.create_wide_dataset(
            category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
            cohort_df=cohort_df, return_dataframe=True,
        )
        _add_missing_med_cols(co)
        sofa_out = co.compute_sofa_scores(
            wide_df=co.wide_df, id_name="hospitalization_id",
            fill_na_scores_with_zero=True, remove_outliers=True, create_new_wide_df=False,
        )
        results = results.merge(
            sofa_out[["hospitalization_id", "sofa_total"]].rename(columns={"sofa_total": f"sofa_{label}"}),
            on="hospitalization_id", how="left",
        )

        win = pd.DataFrame({
            "hospitalization_id": timing_df["hospitalization_id"].values,
            "win_start": anchor_naive.values,
            "win_end": (anchor_naive + pd.Timedelta(hours=ANCHOR_WINDOW_H)).values,
        })
        lac_win = labs_lac.merge(win, on="hospitalization_id", how="inner")
        lac_win = lac_win[
            (lac_win["lab_result_dttm"] >= lac_win["win_start"]) &
            (lac_win["lab_result_dttm"] < lac_win["win_end"])
        ]
        lac_agg = (
            lac_win.groupby("hospitalization_id")["lab_value_numeric"].max()
            .reset_index().rename(columns={"lab_value_numeric": f"lac_{label}"})
        )
        results = results.merge(lac_agg, on="hospitalization_id", how="left")

    for label, _ in ANCHORS:
        s = results[f"sofa_{label}"].fillna(0)
        l = results[f"lac_{label}"].fillna(0)
        results[f"meets_sofa_{label}"] = (s >= SOFA_THRESHOLD).astype(int)
        results[f"meets_lac_{label}"] = (l > LACTATE_THRESHOLD).astype(int)
        results[f"meets_sepsis3_{label}"] = (
            results[f"meets_sofa_{label}"] & results[f"meets_lac_{label}"]
        ).astype(int)

    return results


def compute_ne_cohort_timing(clif_dir: Path) -> pd.DataFrame:
    cultures = pd.read_parquet(
        clif_dir / "clif_microbiology_culture.parquet",
        columns=["hospitalization_id", "fluid_category", "method_category", "collect_dttm"],
    )
    meds_i = pd.read_parquet(
        clif_dir / "clif_medication_admin_intermittent.parquet",
        columns=["hospitalization_id", "med_group", "admin_dttm"],
    )
    labs = pd.read_parquet(
        clif_dir / "clif_labs.parquet",
        columns=["hospitalization_id", "lab_category", "lab_value_numeric"],
    )
    meds_c = pd.read_parquet(
        clif_dir / "clif_medication_admin_continuous.parquet",
        columns=["hospitalization_id", "med_category", "admin_dttm"],
    )

    cx = cultures[
        (cultures["fluid_category"] == "blood_buffy") &
        (cultures["method_category"] == "culture") &
        cultures["collect_dttm"].notna()
    ].copy()
    cx["collect_dttm"] = to_naive_utc(cx["collect_dttm"])
    cx_first = (
        cx.groupby("hospitalization_id")["collect_dttm"].min()
        .reset_index().rename(columns={"collect_dttm": "culture_dttm"})
    )

    abx = meds_i[meds_i["med_group"] == "CMS_sepsis_qualifying_antibiotics"].copy()
    abx["admin_dttm"] = to_naive_utc(abx["admin_dttm"])
    abx_first = (
        abx.groupby("hospitalization_id")["admin_dttm"].min()
        .reset_index().rename(columns={"admin_dttm": "abx_dttm"})
    )

    lac_ids = set(labs.loc[
        (labs["lab_category"] == "lactate") & (labs["lab_value_numeric"] > LACTATE_THRESHOLD),
        "hospitalization_id",
    ])

    ne = meds_c[meds_c["med_category"] == "norepinephrine"].copy()
    ne["admin_dttm"] = to_naive_utc(ne["admin_dttm"])
    ne_first = (
        ne.groupby("hospitalization_id")["admin_dttm"].min()
        .reset_index().rename(columns={"admin_dttm": "ne_dttm"})
    )

    df = (
        cx_first
        .merge(abx_first, on="hospitalization_id", how="inner")
        .pipe(lambda d: d[d["hospitalization_id"].isin(lac_ids)])
        .merge(ne_first, on="hospitalization_id", how="inner")
    )
    print(f"  Cohort: {len(df):,} patients")

    df["delta_culture_h"] = (df["culture_dttm"] - df["ne_dttm"]).dt.total_seconds() / 3600
    df["delta_abx_h"] = (df["abx_dttm"] - df["ne_dttm"]).dt.total_seconds() / 3600
    return df[["hospitalization_id", "delta_culture_h", "delta_abx_h"]]


# ============================================================
# MAIN
# ============================================================

def main():
    output_dir = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / "timing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Site:     {SITE_NAME}")
    print(f"CLIF dir: {CLIF_DIR}")
    print(f"Output:   {output_dir}\n")

    print("[1/4] Plot A data: NE timing relative to suspected infection onset")
    deltas = compute_deltas(CLIF_DIR)
    plot_a = {
        "quantiles": _quantile_stats(deltas),
        "histogram": _suppressed_hist(deltas, PLOT_A_X_MIN, PLOT_A_X_MAX, BIN_WIDTH_H),
        "pct_outside_range": _r(100 * ((deltas < PLOT_A_X_MIN) | (deltas > PLOT_A_X_MAX)).mean()) if len(deltas) else None,
    }

    print("\n[2/4] Plot B data: culture -> abx -> NE timing (median/IQR only)")
    timing_df = compute_timing_data(CLIF_DIR)
    plot_b = {
        "n": len(timing_df),
        "abx_delta_h": _quantile_stats(timing_df["abx_delta_h"]),
        "ne_delta_h": _quantile_stats(timing_df["ne_delta_h"]),
    }

    print("\n[3/4] Plot C data: event timing vs Sepsis-3 status at each anchor")
    plot_c = None
    if not HAS_CLIFPY:
        print("  clifpy not found — skipping Plot C")
    else:
        tmp_dir = output_dir / "_clifpy_tmp"
        sepsis3_df = compute_sepsis3_at_anchors(CLIF_DIR, timing_df, tmp_dir)
        merged = timing_df.merge(sepsis3_df, on="hospitalization_id", how="inner")
        merged["t0"] = merged[["culture_dttm", "abx_dttm", "first_ne_dttm"]].min(axis=1)
        plot_c = {"n": len(merged), "anchors": {}}
        for label, col in ANCHORS:
            x = (merged[col] - merged["t0"]).dt.total_seconds() / 3600
            met_col = f"meets_sepsis3_{label}"
            hist_met0 = _suppressed_hist(x[merged[met_col] == 0], 0, PLOT_C_X_MAX, BIN_WIDTH_H)
            hist_met1 = _suppressed_hist(x[merged[met_col] == 1], 0, PLOT_C_X_MAX, BIN_WIDTH_H)
            plot_c["anchors"][label] = {
                "n_sepsis3_not_met": int((merged[met_col] == 0).sum()),
                "n_sepsis3_met": int((merged[met_col] == 1).sum()),
                "hist_not_met": hist_met0,
                "hist_met": hist_met1,
            }

    print("\n[4/4] Plot D data: NE-anchored cohort timing (culture/abx vs NE)")
    ne_timing = compute_ne_cohort_timing(CLIF_DIR)
    plot_d = {
        "n": len(ne_timing),
        "delta_culture_h": _suppressed_hist(ne_timing["delta_culture_h"], PLOT_D_X_MIN, PLOT_D_X_MAX, BIN_WIDTH_H),
        "delta_abx_h": _suppressed_hist(ne_timing["delta_abx_h"], PLOT_D_X_MIN, PLOT_D_X_MAX, BIN_WIDTH_H),
    }

    payload = {
        "site": SITE_NAME,
        "bin_width_h": BIN_WIDTH_H,
        "plot_a_ne_vs_infection": plot_a,
        "plot_b_connected_dot": plot_b,
        "plot_c_sepsis3_at_anchors": plot_c,
        "plot_d_ne_cohort_density": plot_d,
    }

    out = output_dir / "timing_stats.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    readme = output_dir / "README.md"
    readme.write_text(
        "# NE-Infection Timing Summary — Federated Privacy Manifest\n\n"
        "Generated by: code/04b_ne_infection_timing_summary.py\n\n"
        "## Privacy guarantees\n"
        f"- No patient-level records. Histograms use a fixed {BIN_WIDTH_H}h-wide grid; any bin "
        f"with fewer than K={SUPPRESS_K} patients is suppressed to 0.\n"
        "- Plot B shares only median/Q25/Q75 for each delta — no per-patient connecting lines.\n"
        "- Plot C shares suppressed 1D histograms per anchor x Sepsis-3-met status, not a "
        "per-patient scatter.\n\n"
        "## timing_stats.json schema\n"
        "- `plot_a_ne_vs_infection`: {quantiles: {n,median,q25,q75}, histogram: {n,bin_edges,counts}, "
        "pct_outside_range} — hours from suspected infection onset to first NE.\n"
        "- `plot_b_connected_dot`: {n, abx_delta_h: {n,median,q25,q75}, ne_delta_h: {...}} — hours "
        "relative to blood culture.\n"
        "- `plot_c_sepsis3_at_anchors`: null if clifpy unavailable at this site, else "
        "{n, anchors: {culture|abx|ne: {n_sepsis3_met, n_sepsis3_not_met, hist_met, hist_not_met}}} "
        "— hours from the earliest of the 3 events, histogram per Sepsis-3 status.\n"
        "- `plot_d_ne_cohort_density`: {n, delta_culture_h: {n,bin_edges,counts}, delta_abx_h: {...}} "
        "— hours relative to first NE, for the ever-NE + ever-culture + ever-abx + lactate>2 cohort.\n",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
