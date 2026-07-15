# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
CLIF 2.1.0 cohort identification and hourly feature extraction.

Two cohorts are identified, both anchored at t=0 = first NE administration:

  Sepsis-3 (CMS): CMS qualifying IV abx + blood culture within 24 h of each other
                  + lactate > 2 mmol/L, all within ±24 h of NE start.

  Rhee/CDC ASE:   Blood culture within ±24 h of NE start
                  + first qualifying IV abx within 2 calendar days of culture
                  + ≥4 consecutive qualifying antibiotic days (≤1-day gap allowed),
                    or antibiotic course running until ≤1 day before discharge/death
                  + lactate >= 2 mmol/L within ±24 h of NE start.

Outputs (OUTPUT_ROOT/output/patient_level_data_<SITE_NAME>/ — PHI, local only):
  cohort_sepsis3.parquet        — Sepsis-3 cohort demographics and outcomes
  cohort_rhee.parquet           — Rhee/CDC ASE cohort demographics and outcomes
  features.parquet              — hourly features for union of both cohorts
  cohort_filter_counts_sepsis3.csv
  cohort_filter_counts_rhee.csv
"""

import sys
import io
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from clifpy import ClifOrchestrator
    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE, DEVICE_RANK_DICT
    from clifpy.utils.comorbidity import calculate_cci
except ModuleNotFoundError:
    print("Install clifpy: pip install clifpy")
    sys.exit(1)

warnings.filterwarnings("ignore")


def _load_site_config():
    """Load config/config.py by file path to avoid namespace-package collision."""
    import importlib.util as _ilu
    cfg_path = Path(__file__).parent.parent / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Configuration — defaults (override in config/config.py)
# ---------------------------------------------------------------------------
CLIF_DIR    = None   # REQUIRED
OUTPUT_ROOT = None   # REQUIRED
SITE_NAME   = "UCMC"
TIMEZONE = "UTC"
TRAJECTORY_HOURS = 120
MIN_NE_RECORDS = 2     # ≥2 NE administrations
SOFA_THRESHOLD = 2.0   # kept for config compatibility; not used as inclusion criterion
LACTATE_THRESHOLD = 2.0
MAP_THRESHOLD = 65.0

STEROID_CATEGORIES = [
    "hydrocortisone", "dexamethasone", "methylprednisolone",
    "fludrocortisone", "prednisolone", "prednisone",
]
VASOPRESSOR_CATEGORIES = [
    "norepinephrine", "epinephrine", "phenylephrine",
    "vasopressin", "dopamine", "angiotensin ii",
]

_cfg = _load_site_config()
if _cfg is not None:
    for _k in (
        "CLIF_DIR", "OUTPUT_ROOT", "SITE_NAME", "TIMEZONE", "TRAJECTORY_HOURS",
        "MIN_NE_RECORDS", "SOFA_THRESHOLD", "LACTATE_THRESHOLD", "MAP_THRESHOLD",
        "STEROID_CATEGORIES", "VASOPRESSOR_CATEGORIES",
    ):
        if hasattr(_cfg, _k):
            globals()[_k] = getattr(_cfg, _k)
    del _k
del _cfg

if CLIF_DIR is None or OUTPUT_ROOT is None:
    sys.exit(
        "ERROR: CLIF_DIR and OUTPUT_ROOT are not configured.\n"
        "Copy config/config.example.py to config/config.py and set your site-specific paths."
    )

PATIENT_LEVEL_DIR = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def tz_coerce(s: pd.Series, tz: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        if s.dt.tz is None:
            return s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        return s.dt.tz_convert(tz)
    return s


def tz_strip(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s


def to_naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


# ---------------------------------------------------------------------------
# Phase A-1: NE starts — t=0 anchor
# ---------------------------------------------------------------------------
def get_all_ne_starts(clif_dir: Path) -> pd.DataFrame:
    """First NE administration per hospitalization with ≥MIN_NE_RECORDS records.

    No infection constraint — sepsis criteria are checked separately within
    a ±24 h window around this anchor.
    Returns: hospitalization_id, first_norepi_time
    """
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    ne = meds[meds["med_category"] == "norepinephrine"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    ne["admin_dttm"] = to_naive_utc(ne["admin_dttm"])
    ne_agg = (ne.groupby("hospitalization_id")
                .agg(first_norepi_time=("admin_dttm", "min"),
                     n_ne=("admin_dttm", "count"))
                .reset_index())
    return ne_agg[ne_agg["n_ne"] >= MIN_NE_RECORDS].drop(columns="n_ne")


# ---------------------------------------------------------------------------
# Phase A-2: Shared source data helpers
# ---------------------------------------------------------------------------
def get_abx_records(clif_dir: Path) -> pd.DataFrame:
    """CMS qualifying IV antibiotics with UTC timestamp and calendar date."""
    meds_i = pd.read_parquet(clif_dir / "clif_medication_admin_intermittent.parquet")
    abx = meds_i[meds_i["med_group"] == "CMS_sepsis_qualifying_antibiotics"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    abx["admin_dttm"] = to_naive_utc(abx["admin_dttm"])
    abx["abx_date"] = abx["admin_dttm"].dt.date
    return abx


def get_blood_cultures(clif_dir: Path) -> pd.DataFrame:
    """Blood buffy-coat culture records with UTC timestamp and calendar date."""
    cultures = pd.read_parquet(clif_dir / "clif_microbiology_culture.parquet")
    blood_cx = cultures[
        (cultures["fluid_category"] == "blood_buffy") &
        cultures["collect_dttm"].notna() &
        (cultures["method_category"] == "culture")
    ][["hospitalization_id", "collect_dttm"]].copy()
    blood_cx["collect_dttm"] = to_naive_utc(blood_cx["collect_dttm"])
    blood_cx["culture_date"] = blood_cx["collect_dttm"].dt.date
    return blood_cx


# ---------------------------------------------------------------------------
# Phase A-3a: Sepsis-3 (CMS) cohort
# ---------------------------------------------------------------------------
def identify_sepsis3_cohort(
    clif_dir: Path,
    ne_df: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    """Sepsis-3 (CMS) criteria within ±window_hours of NE start (t=0):
      1. CMS qualifying IV abx + blood culture within 24 h of each other
      2. Both events within ±window_hours of NE start
      3. Lactate > LACTATE_THRESHOLD within ±window_hours of NE start

    Returns: hospitalization_id, presumed_infection_dttm, initial_lactate
    """
    abx = get_abx_records(clif_dir)
    blood_cx = get_blood_cultures(clif_dir)

    ne = ne_df[["hospitalization_id", "first_norepi_time"]].copy()
    ne["t0"] = to_naive_utc(pd.to_datetime(ne["first_norepi_time"], utc=True))
    ne["win_start"] = ne["t0"] - pd.Timedelta(hours=window_hours)
    ne["win_end"]   = ne["t0"] + pd.Timedelta(hours=window_hours)

    # Filter abx to window
    abx_win = abx.merge(ne[["hospitalization_id", "win_start", "win_end"]], on="hospitalization_id")
    abx_win = abx_win[
        (abx_win["admin_dttm"] >= abx_win["win_start"]) &
        (abx_win["admin_dttm"] <= abx_win["win_end"])
    ]

    # Filter cultures to window
    cx_win = blood_cx.merge(ne[["hospitalization_id", "win_start", "win_end"]], on="hospitalization_id")
    cx_win = cx_win[
        (cx_win["collect_dttm"] >= cx_win["win_start"]) &
        (cx_win["collect_dttm"] <= cx_win["win_end"])
    ]

    # Pair abx + culture within 24 h of each other
    paired = abx_win[["hospitalization_id", "admin_dttm"]].merge(
        cx_win[["hospitalization_id", "collect_dttm"]], on="hospitalization_id"
    )
    paired["diff_h"] = (
        (paired["admin_dttm"] - paired["collect_dttm"]).dt.total_seconds().abs() / 3600
    )
    paired = paired[paired["diff_h"] <= 24].copy()
    paired["infection_anchor"] = paired[["admin_dttm", "collect_dttm"]].min(axis=1)

    infection = (paired.groupby("hospitalization_id")
                       .agg(presumed_infection_dttm=("infection_anchor", "min"))
                       .reset_index())

    if infection.empty:
        infection["initial_lactate"] = np.nan
        return infection

    # Lactate check
    labs = pd.read_parquet(clif_dir / "clif_labs.parquet")
    lac_df = labs[labs["lab_category"] == "lactate"][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].copy()
    lac_df["lab_result_dttm"] = to_naive_utc(lac_df["lab_result_dttm"])

    lac_merged = lac_df.merge(
        ne[["hospitalization_id", "win_start", "win_end"]], on="hospitalization_id"
    )
    lac_win = lac_merged[
        (lac_merged["lab_result_dttm"] >= lac_merged["win_start"]) &
        (lac_merged["lab_result_dttm"] <= lac_merged["win_end"]) &
        lac_merged["lab_value_numeric"].notna()
    ]

    initial_lac = (
        lac_win.sort_values("lab_result_dttm")
        .groupby("hospitalization_id")["lab_value_numeric"]
        .first()
        .reset_index()
        .rename(columns={"lab_value_numeric": "initial_lactate"})
    )

    elevated_ids = set(lac_win.loc[lac_win["lab_value_numeric"] > LACTATE_THRESHOLD, "hospitalization_id"])
    infection = infection[infection["hospitalization_id"].isin(elevated_ids)].copy()
    infection = infection.merge(initial_lac, on="hospitalization_id", how="left")
    return infection


# ---------------------------------------------------------------------------
# Phase A-3b: Rhee/CDC ASE cohort
# ---------------------------------------------------------------------------
def identify_rhee_cohort(
    clif_dir: Path,
    ne_df: pd.DataFrame,
    mortality_df: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    """Rhee/CDC Adult Sepsis Event criteria within ±window_hours of NE start (t=0):
      1. Blood culture within ±window_hours of NE start
      2. First qualifying IV abx within 2 calendar days of culture date
      3. ≥4 consecutive qualifying antibiotic calendar days (≤1-day gap allowed),
         OR antibiotic course runs until ≤1 day before discharge/death

    Returns: hospitalization_id, blood_culture_dttm
    """
    abx = get_abx_records(clif_dir)
    blood_cx = get_blood_cultures(clif_dir)

    ne = ne_df[["hospitalization_id", "first_norepi_time"]].copy()
    ne["t0"] = to_naive_utc(pd.to_datetime(ne["first_norepi_time"], utc=True))
    ne["win_start"] = ne["t0"] - pd.Timedelta(hours=window_hours)
    ne["win_end"]   = ne["t0"] + pd.Timedelta(hours=window_hours)

    # Step 1: Blood culture within ±window_hours of NE start
    cx_win = blood_cx.merge(
        ne[["hospitalization_id", "win_start", "win_end"]], on="hospitalization_id"
    )
    cx_win = cx_win[
        (cx_win["collect_dttm"] >= cx_win["win_start"]) &
        (cx_win["collect_dttm"] <= cx_win["win_end"])
    ].copy()
    if cx_win.empty:
        return pd.DataFrame(columns=["hospitalization_id", "blood_culture_dttm"])

    earliest_cx = (cx_win.sort_values("collect_dttm")
                         .groupby("hospitalization_id")
                         .first()
                         .reset_index()[["hospitalization_id", "collect_dttm", "culture_date"]])

    # Step 2: First qualifying IV abx within 2 calendar days of culture
    abx_cx = abx.merge(earliest_cx[["hospitalization_id", "culture_date"]], on="hospitalization_id")
    abx_cx["day_diff"] = (
        pd.to_datetime(abx_cx["abx_date"]) -
        pd.to_datetime(abx_cx["culture_date"])
    ).dt.days.abs()
    qualifying_abx = abx_cx[abx_cx["day_diff"] <= 2].copy()
    if qualifying_abx.empty:
        return pd.DataFrame(columns=["hospitalization_id", "blood_culture_dttm"])

    first_abx = (qualifying_abx.sort_values("abx_date")
                                .groupby("hospitalization_id")["abx_date"]
                                .first()
                                .reset_index()
                                .rename(columns={"abx_date": "first_abx_date"}))

    candidates = earliest_cx.merge(first_abx, on="hospitalization_id")
    if candidates.empty:
        return pd.DataFrame(columns=["hospitalization_id", "blood_culture_dttm"])

    # Step 3: ≥4 consecutive antibiotic calendar days (≤1-day gap allowed)
    cand_ids = set(candidates["hospitalization_id"])
    abx_cands = (abx[abx["hospitalization_id"].isin(cand_ids)]
                 [["hospitalization_id", "abx_date"]]
                 .drop_duplicates()
                 .merge(candidates[["hospitalization_id", "first_abx_date"]], on="hospitalization_id"))
    abx_cands = abx_cands[abx_cands["abx_date"] >= abx_cands["first_abx_date"]].copy()
    abx_cands["abx_date_dt"] = pd.to_datetime(abx_cands["abx_date"])
    abx_cands = abx_cands.sort_values(["hospitalization_id", "abx_date_dt"])

    abx_cands["prev_dt"] = (abx_cands.groupby("hospitalization_id")["abx_date_dt"]
                                      .shift(1))
    abx_cands["gap_days"] = (
        (abx_cands["abx_date_dt"] - abx_cands["prev_dt"]).dt.days.fillna(1)
    )
    # gap > 2 means >1 calendar-day gap → break in consecutive run
    abx_cands["new_run"] = (abx_cands["gap_days"] > 2).astype(int)
    first_row = abx_cands.groupby("hospitalization_id").cumcount() == 0
    abx_cands.loc[first_row, "new_run"] = 1
    abx_cands["run_id"] = abx_cands.groupby("hospitalization_id")["new_run"].cumsum()

    run_lengths = (abx_cands.groupby(["hospitalization_id", "run_id"])
                             .size()
                             .reset_index(name="run_len"))
    max_run = (run_lengths.groupby("hospitalization_id")["run_len"]
                          .max()
                          .reset_index(name="max_run_days"))
    last_abx = (abx_cands.groupby("hospitalization_id")["abx_date_dt"]
                          .max()
                          .reset_index(name="last_abx_dt"))

    # Discharge/death date for early-termination criterion
    discharge = mortality_df[["hospitalization_id", "discharge_dttm", "deathtime"]].copy()
    discharge["end_dt"] = pd.to_datetime(discharge.apply(
        lambda r: r["deathtime"] if pd.notna(r["deathtime"]) else r["discharge_dttm"],
        axis=1,
    ), utc=True).dt.tz_localize(None).dt.normalize()

    rhee_check = (max_run
                  .merge(last_abx, on="hospitalization_id")
                  .merge(discharge[["hospitalization_id", "end_dt"]], on="hospitalization_id", how="left"))

    rhee_check["days_before_end"] = (
        (rhee_check["end_dt"] - rhee_check["last_abx_dt"]).dt.days
    )
    # Criterion: ≥4 consecutive days, OR course ends within 1 day of discharge/death
    rhee_check["meets"] = (
        (rhee_check["max_run_days"] >= 4) |
        (rhee_check["days_before_end"].fillna(99) <= 1)
    )

    qualifying = set(rhee_check.loc[rhee_check["meets"], "hospitalization_id"])
    result = candidates[candidates["hospitalization_id"].isin(qualifying)].copy()
    return result[["hospitalization_id", "collect_dttm"]].rename(
        columns={"collect_dttm": "blood_culture_dttm"}
    )


# ---------------------------------------------------------------------------
# Phase A-4: ICU admission/discharge times from ADT
# ---------------------------------------------------------------------------
def get_icu_times(clif_dir: Path) -> pd.DataFrame:
    adt = pd.read_parquet(clif_dir / "clif_adt.parquet")
    icu_rows = adt[adt["location_category"] == "icu"].sort_values("in_dttm")
    agg = (icu_rows.groupby("hospitalization_id")
           .agg(icu_intime=("in_dttm", "first"), icu_outtime=("out_dttm", "last"))
           .reset_index())
    first_loc_type = (icu_rows.groupby("hospitalization_id")
                      .first()
                      .reset_index()
                      [["hospitalization_id"] +
                       (["location_type"] if "location_type" in icu_rows.columns else [])])
    icu = agg.merge(first_loc_type, on="hospitalization_id", how="left")
    if "location_type" not in icu.columns:
        icu["location_type"] = np.nan
    return icu


def get_any_location_times(clif_dir: Path) -> pd.DataFrame:
    adt = pd.read_parquet(clif_dir / "clif_adt.parquet")
    first = (adt.sort_values("in_dttm")
             .groupby("hospitalization_id")
             .first()
             .reset_index())
    cols = ["hospitalization_id", "in_dttm", "out_dttm"]
    for c in ["location_category", "location_type"]:
        if c in first.columns:
            cols.append(c)
    first = first[cols].rename(columns={"in_dttm": "first_loc_intime",
                                        "out_dttm": "first_loc_outtime"})
    for c in ["location_category", "location_type"]:
        if c not in first.columns:
            first[c] = np.nan
    return first


_LOC_T0_OPTIONAL_COLS = ["location_type", "hospital_id", "hospital_type"]


def get_location_at_t0(clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """ADT location (category, type, hospital) at t=0 (first_norepi_time) per
    patient. hospital_id/hospital_type identify the individual community or
    academic hospital within this CLIF site's data pool (a site may bundle
    multiple hospitals)."""
    adt = pd.read_parquet(clif_dir / "clif_adt.parquet")
    adt["in_dttm"]  = to_naive_utc(adt["in_dttm"])
    adt["out_dttm"] = to_naive_utc(adt["out_dttm"].fillna(pd.Timestamp("2100-01-01")))

    t0 = (cohort[["stay_id", "first_norepi_time"]]
          .rename(columns={"stay_id": "hospitalization_id"})
          .copy())
    t0["t0"] = to_naive_utc(t0["first_norepi_time"])

    loc_cols = ["hospitalization_id", "in_dttm", "out_dttm", "location_category"]
    loc_cols += [c for c in _LOC_T0_OPTIONAL_COLS if c in adt.columns]

    merged = t0.merge(adt[loc_cols], on="hospitalization_id", how="left")
    active = merged[
        (merged["t0"] >= merged["in_dttm"]) &
        (merged["t0"] <  merged["out_dttm"])
    ]
    active = (active.sort_values("in_dttm")
                    .groupby("hospitalization_id")
                    .last()
                    .reset_index())

    all_cols = ["location_category"] + _LOC_T0_OPTIONAL_COLS
    keep = ["hospitalization_id"] + [c for c in all_cols if c in active.columns]
    result = active[keep].rename(columns={"hospitalization_id": "stay_id"})
    for c in all_cols:
        if c not in result.columns:
            result[c] = np.nan
    return result


# ---------------------------------------------------------------------------
# Phase A-5: Mortality, demographics, trajectory helpers
# ---------------------------------------------------------------------------
def get_mortality(clif_dir: Path) -> pd.DataFrame:
    patient = pd.read_parquet(clif_dir / "clif_patient.parquet")[["patient_id", "death_dttm"]]
    hosp = pd.read_parquet(clif_dir / "clif_hospitalization.parquet")[
        ["patient_id", "hospitalization_id", "discharge_category", "discharge_dttm",
         "admission_dttm", "age_at_admission"]
    ]
    m = hosp.merge(patient, on="patient_id", how="left")
    m["hospital_death"] = (
        (m["discharge_category"] == "Expired") | m["death_dttm"].notna()
    ).astype(int)
    m["deathtime"] = m.apply(
        lambda r: r["death_dttm"] if pd.notna(r["death_dttm"])
        else (r["discharge_dttm"] if r["hospital_death"] == 1 else pd.NaT),
        axis=1,
    )
    return m[["hospitalization_id", "hospital_death", "deathtime",
              "discharge_dttm", "admission_dttm", "age_at_admission"]]


def get_demographics(clif_dir: Path, stay_ids: set) -> pd.DataFrame:
    patient = pd.read_parquet(clif_dir / "clif_patient.parquet")[
        ["patient_id", "sex_category", "race_category", "ethnicity_category"]
    ]
    hosp = pd.read_parquet(clif_dir / "clif_hospitalization.parquet")[
        ["patient_id", "hospitalization_id"]
    ]
    demo = hosp[hosp["hospitalization_id"].isin(stay_ids)].merge(patient, on="patient_id", how="left")
    demo = demo.rename(columns={"hospitalization_id": "stay_id"})
    demo["gender"] = demo["sex_category"].map({"Male": "M", "Female": "F"})
    demo["race"] = demo.apply(
        lambda r: "Hispanic" if r["ethnicity_category"] == "Hispanic" else r["race_category"],
        axis=1,
    )
    return demo[["stay_id", "gender", "race"]]


def get_vaso_pretraj(clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """1 if vasopressin was administered in the 24 h before trajectory_start, else 0."""
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    vaso = meds[meds["med_category"] == "vasopressin"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    vaso = vaso.rename(columns={"hospitalization_id": "stay_id"})
    vaso["admin_dttm"] = to_naive_utc(vaso["admin_dttm"])

    bounds = cohort[["stay_id", "trajectory_start"]].copy()
    bounds["traj_start"] = pd.to_datetime(bounds["trajectory_start"], utc=True).dt.tz_localize(None)
    vaso = vaso.merge(bounds[["stay_id", "traj_start"]], on="stay_id", how="inner")

    in_window = (
        (vaso["admin_dttm"] >= vaso["traj_start"] - pd.Timedelta(hours=24)) &
        (vaso["admin_dttm"] <  vaso["traj_start"] + pd.Timedelta(hours=1))
    )
    pretraj_ids = set(vaso.loc[in_window, "stay_id"])
    return pd.DataFrame({
        "stay_id": cohort["stay_id"],
        "vaso_before_traj": cohort["stay_id"].isin(pretraj_ids).astype(int),
    })


def get_weight_at_onset(clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """First recorded weight_kg per patient in the hospitalization."""
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    w = vitals[vitals["vital_category"] == "weight_kg"][
        ["hospitalization_id", "recorded_dttm", "vital_value"]
    ].rename(columns={"hospitalization_id": "stay_id", "vital_value": "weight"})
    w = w[w["weight"].notna() & w["stay_id"].isin(set(cohort["stay_id"]))].copy()
    w["recorded_dttm"] = to_naive_utc(w["recorded_dttm"])
    return (w.sort_values("recorded_dttm")
             .groupby("stay_id")["weight"]
             .first()
             .reset_index())


def get_cci(clif_dir: Path, hosp_ids: set) -> pd.DataFrame:
    """Charlson Comorbidity Index per hospitalization (via clifpy, ICD10CM only).

    Falls back to NaN if clif_hospital_diagnosis.parquet is absent (optional table).
    """
    path = clif_dir / "clif_hospital_diagnosis.parquet"
    if not path.exists():
        return pd.DataFrame({"stay_id": list(hosp_ids), "cci_score": np.nan})
    dx = pd.read_parquet(path)
    dx = dx[dx["hospitalization_id"].isin(hosp_ids)]
    if dx.empty:
        return pd.DataFrame({"stay_id": list(hosp_ids), "cci_score": np.nan})
    cci = calculate_cci(dx)
    return cci[["hospitalization_id", "cci_score"]].rename(
        columns={"hospitalization_id": "stay_id"}
    )


# ---------------------------------------------------------------------------
# Phase A-6: clifpy SOFA helpers
# ---------------------------------------------------------------------------
def _load_sofa_tables(co: ClifOrchestrator, hosp_ids: list) -> None:
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


def _add_missing_med_cols(co: ClifOrchestrator) -> None:
    for col in ["norepinephrine_mcg_kg_min", "epinephrine_mcg_kg_min",
                "dopamine_mcg_kg_min", "dobutamine_mcg_kg_min"]:
        if col not in co.wide_df.columns:
            co.wide_df[col] = 0.0


def compute_sofa_at_ne_start(ne_df: pd.DataFrame, co: ClifOrchestrator) -> pd.DataFrame:
    """SOFA score in the 24 h window starting at first_norepi_time.

    Returns: hospitalization_id, sepsis_onset_sofa
    """
    sofa_window = ne_df[["hospitalization_id", "first_norepi_time"]].copy()
    sofa_window["start_time"] = tz_coerce(
        pd.to_datetime(sofa_window["first_norepi_time"]), TIMEZONE
    )
    sofa_window["end_time"] = sofa_window["start_time"] + pd.Timedelta(hours=24)

    hosp_ids = sofa_window["hospitalization_id"].tolist()
    _load_sofa_tables(co, hosp_ids)
    co.create_wide_dataset(
        category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df=sofa_window,
        return_dataframe=True,
    )
    _add_missing_med_cols(co)
    sofa = co.compute_sofa_scores(
        wide_df=co.wide_df,
        id_name="hospitalization_id",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        create_new_wide_df=False,
    )
    return sofa[["hospitalization_id", "sofa_total"]].rename(
        columns={"sofa_total": "sepsis_onset_sofa"}
    )


# ---------------------------------------------------------------------------
# Phase A-7: Cohort DataFrame assembly helper
# ---------------------------------------------------------------------------
def _assemble_cohort_df(
    label_df: pd.DataFrame,
    ne_df: pd.DataFrame,
    icu_df: pd.DataFrame,
    mortality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge label_df (sepsis3 or rhee IDs) with NE times, ICU, and mortality;
    compute trajectory bounds and death_hour.

    label_df must have hospitalization_id as key; other columns are preserved.
    Returns a DataFrame with stay_id replacing hospitalization_id.
    """
    cohort = label_df.merge(ne_df, on="hospitalization_id", how="inner")
    cohort = cohort.merge(icu_df, on="hospitalization_id", how="left")
    cohort["icu_intime"] = tz_coerce(cohort["icu_intime"], TIMEZONE)
    cohort = cohort.merge(
        mortality_df.rename(columns={"age_at_admission": "age"}),
        on="hospitalization_id", how="left",
    )

    for col in ["icu_intime", "icu_outtime", "first_norepi_time", "deathtime",
                "admission_dttm", "discharge_dttm"]:
        if col in cohort.columns:
            cohort[col] = tz_coerce(cohort[col], TIMEZONE)

    # ICU/hospital length of stay (days) — full stay, independent of the
    # vasopressin trajectory window/cap below.
    cohort["icu_los_days"] = (
        (cohort["icu_outtime"] - cohort["icu_intime"]).dt.total_seconds() / 86400
    )
    cohort["hospital_los_days"] = (
        (cohort["discharge_dttm"] - cohort["admission_dttm"]).dt.total_seconds() / 86400
    )

    cohort["trajectory_start"] = cohort["first_norepi_time"]
    traj_cap = cohort["trajectory_start"] + pd.Timedelta(hours=TRAJECTORY_HOURS)

    def _to_tz(s):
        s = pd.to_datetime(s, utc=False)
        if s.dt.tz is None:
            s = s.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        else:
            s = s.dt.tz_convert(TIMEZONE)
        return s

    # Column order = tie-break priority (death > discharge > hour_cap) for the
    # rare case two candidates land on the exact same timestamp.
    _end_candidates = pd.DataFrame({
        "death":     _to_tz(cohort["deathtime"]),
        "discharge": _to_tz(cohort["icu_outtime"]),
        "hour_cap":  _to_tz(traj_cap),
    })
    cohort["trajectory_end"] = _end_candidates.min(axis=1)
    cohort["traj_end_reason"] = _end_candidates.idxmin(axis=1)

    cohort["traj_hours"] = (
        (cohort["trajectory_end"] - cohort["trajectory_start"])
        .dt.total_seconds() / 3600
    ).clip(lower=0).astype(int)

    _dt = to_naive_utc(pd.to_datetime(cohort["deathtime"], utc=True))
    _ts = to_naive_utc(pd.to_datetime(cohort["trajectory_start"], utc=True))
    cohort["death_hour"] = (
        ((_dt - _ts).dt.total_seconds() / 3600)
        .where(cohort["hospital_death"] == 1)
        .apply(lambda x: float(int(x)) if pd.notna(x) else float("nan"))
    )

    cohort["anchor_year_group"] = cohort["first_norepi_time"].dt.year.astype(str)
    cohort = cohort.rename(columns={"hospitalization_id": "stay_id"})
    return cohort


# ---------------------------------------------------------------------------
# Phase A: Full cohort assembly
# ---------------------------------------------------------------------------
def build_cohort(clif_dir: Path, co: ClifOrchestrator) -> tuple:
    """Identify Sepsis-3 and Rhee/CDC ASE cohorts with t=0 at NE start.

    Returns: (cohort_sepsis3, cohort_rhee, filter_log_sepsis3, filter_log_rhee)
    Each cohort DataFrame has the same schema as the old cohort.parquet.
    """
    filter_s3   = []
    filter_rhee = []

    print("\nStep 0: Total hospitalizations in CLIF site...")
    hosp_all = pd.read_parquet(clif_dir / "clif_hospitalization.parquet")[["hospitalization_id"]]
    n_total = len(hosp_all)
    filter_s3.append({"step": "Total hospitalizations in CLIF site", "n_hospitalizations": n_total})
    filter_rhee.append({"step": "Total hospitalizations in CLIF site", "n_hospitalizations": n_total})
    print(f"  {n_total:,} total hospitalizations")

    print(f"\nStep 1: All NE starts (≥{MIN_NE_RECORDS} records, t=0 anchor)...")
    ne_df = get_all_ne_starts(clif_dir)
    for fl in [filter_s3, filter_rhee]:
        fl.append({"step": f"NE started (>={MIN_NE_RECORDS} records)", "n_hospitalizations": len(ne_df)})
    print(f"  {len(ne_df):,} patients with NE")

    print("\nStep 2: Mortality/discharge info...")
    mortality = get_mortality(clif_dir)

    print("\nStep 3a: Sepsis-3 (CMS) criteria within ±24 h of NE start...")
    print("         [abx + blood culture within 24 h + lactate > 2 mmol/L]")
    sepsis3_df = identify_sepsis3_cohort(clif_dir, ne_df, window_hours=24)
    filter_s3.append({
        "step": "Sepsis-3 (CMS): abx + culture within 24 h + lactate > 2 (within ±24 h of NE start)",
        "n_hospitalizations": len(sepsis3_df),
    })
    print(f"  {len(sepsis3_df):,} meet Sepsis-3 (CMS) criteria")

    print("\nStep 3b: Rhee/CDC ASE criteria within ±24 h of NE start...")
    print("         [blood culture + ≥4 consecutive antibiotic days + lactate >= 2 mmol/L]")
    rhee_df = identify_rhee_cohort(clif_dir, ne_df, mortality, window_hours=24)
    filter_rhee.append({
        "step": "Rhee/CDC ASE: blood culture + ≥4 consecutive abx days (within ±24 h of NE start)",
        "n_hospitalizations": len(rhee_df),
    })
    print(f"  {len(rhee_df):,} meet Rhee/CDC ASE criteria")

    union_ids = (set(sepsis3_df["hospitalization_id"]) |
                 set(rhee_df["hospitalization_id"]))
    ne_union = ne_df[ne_df["hospitalization_id"].isin(union_ids)].copy()
    print(f"\n  Union: {len(ne_union):,} unique patients in either cohort")

    print("\nStep 4: ICU times...")
    icu = get_icu_times(clif_dir)

    print("\nStep 5: Assembling cohort DataFrames with trajectory bounds...")
    cohort_s3   = _assemble_cohort_df(sepsis3_df, ne_union, icu, mortality)
    cohort_rhee = _assemble_cohort_df(rhee_df,   ne_union, icu, mortality)

    print("\nStep 6: SOFA at NE start (union cohort via clifpy)...")
    sofa_df = compute_sofa_at_ne_start(ne_union, co)
    sofa_map = sofa_df.set_index("hospitalization_id")["sepsis_onset_sofa"]
    cohort_s3["sepsis_onset_sofa"]   = cohort_s3["stay_id"].map(sofa_map)
    cohort_rhee["sepsis_onset_sofa"] = cohort_rhee["stay_id"].map(sofa_map)

    print("\nStep 7: Initial lactate within ±24 h of NE start...")
    # Sepsis-3 initial_lactate already computed in identify_sepsis3_cohort
    lac_s3_map = sepsis3_df.set_index("hospitalization_id")["initial_lactate"]
    cohort_s3["initial_lactate"] = cohort_s3["stay_id"].map(lac_s3_map)

    # Rhee: first lactate within ±24 h of NE start; must be >= LACTATE_THRESHOLD
    labs = pd.read_parquet(clif_dir / "clif_labs.parquet")
    lac_df = labs[labs["lab_category"] == "lactate"][
        ["hospitalization_id", "lab_result_dttm", "lab_value_numeric"]
    ].copy()
    lac_df["lab_result_dttm"] = to_naive_utc(lac_df["lab_result_dttm"])
    rhee_ids = set(cohort_rhee["stay_id"])
    ne_win = ne_union[ne_union["hospitalization_id"].isin(rhee_ids)].copy()
    ne_win["t0"] = to_naive_utc(pd.to_datetime(ne_win["first_norepi_time"], utc=True))
    ne_win["win_start"] = ne_win["t0"] - pd.Timedelta(hours=24)
    ne_win["win_end"]   = ne_win["t0"] + pd.Timedelta(hours=24)
    lac_rhee = lac_df.merge(
        ne_win[["hospitalization_id", "win_start", "win_end"]], on="hospitalization_id"
    )
    lac_rhee = lac_rhee[
        (lac_rhee["lab_result_dttm"] >= lac_rhee["win_start"]) &
        (lac_rhee["lab_result_dttm"] <= lac_rhee["win_end"]) &
        lac_rhee["lab_value_numeric"].notna()
    ]
    rhee_lac = (lac_rhee.sort_values("lab_result_dttm")
                .groupby("hospitalization_id")["lab_value_numeric"]
                .first()
                .reset_index()
                .rename(columns={"lab_value_numeric": "initial_lactate",
                                 "hospitalization_id": "stay_id"}))
    cohort_rhee = cohort_rhee.merge(rhee_lac, on="stay_id", how="left")

    n_before_lac = len(cohort_rhee)
    cohort_rhee = cohort_rhee[cohort_rhee["initial_lactate"] >= LACTATE_THRESHOLD].copy()
    filter_rhee.append({
        "step": f"Lactate < {LACTATE_THRESHOLD} or missing (within ±24 h of NE start) (excluded)",
        "n_hospitalizations": n_before_lac - len(cohort_rhee),
    })
    print(f"  {len(cohort_rhee):,} meet Rhee/CDC ASE + lactate >= {LACTATE_THRESHOLD} criteria")

    print("\nStep 8: Demographics, weight, vasopressin pre-trajectory, location, CCI...")
    all_stay_ids = set(cohort_s3["stay_id"]) | set(cohort_rhee["stay_id"])
    union_for_shared = (pd.concat([
        cohort_s3[["stay_id", "trajectory_start", "trajectory_end"]],
        cohort_rhee[["stay_id", "trajectory_start", "trajectory_end"]],
    ]).drop_duplicates(subset=["stay_id"]))

    demo = get_demographics(clif_dir, all_stay_ids)
    weight_df = get_weight_at_onset(clif_dir, union_for_shared)
    vaso_pre = get_vaso_pretraj(clif_dir, union_for_shared)
    cci_df = get_cci(clif_dir, all_stay_ids)

    for cohort in [cohort_s3, cohort_rhee]:
        cohort.merge(demo, on="stay_id", how="left")  # don't reassign yet; done below
    cohort_s3   = cohort_s3.merge(demo, on="stay_id", how="left")
    cohort_s3   = cohort_s3.merge(weight_df, on="stay_id", how="left")
    cohort_s3   = cohort_s3.merge(vaso_pre, on="stay_id", how="left")
    cohort_s3   = cohort_s3.merge(cci_df, on="stay_id", how="left")
    cohort_rhee = cohort_rhee.merge(demo, on="stay_id", how="left")
    cohort_rhee = cohort_rhee.merge(weight_df, on="stay_id", how="left")
    cohort_rhee = cohort_rhee.merge(vaso_pre, on="stay_id", how="left")
    cohort_rhee = cohort_rhee.merge(cci_df, on="stay_id", how="left")

    for cohort in [cohort_s3, cohort_rhee]:
        cohort["vaso_before_traj"] = cohort["vaso_before_traj"].fillna(0).astype(int)

    print("\nStep 9: Excluding patients with vasopressin before trajectory start...")
    for fl, cohort in [(filter_s3, cohort_s3), (filter_rhee, cohort_rhee)]:
        n_excl = int((cohort["vaso_before_traj"] == 1).sum())
        fl.append({"step": "Vasopressin in 24 h before trajectory start (excluded)",
                   "n_hospitalizations": n_excl})

    cohort_s3   = cohort_s3[cohort_s3["vaso_before_traj"] == 0].copy()
    cohort_rhee = cohort_rhee[cohort_rhee["vaso_before_traj"] == 0].copy()

    print("\nStep 10: Location at t=0 (ADT row active at NE start)...")
    union_for_loc = (pd.concat([
        cohort_s3[["stay_id", "first_norepi_time"]],
        cohort_rhee[["stay_id", "first_norepi_time"]],
    ]).drop_duplicates(subset=["stay_id"]))
    loc_t0 = get_location_at_t0(clif_dir, union_for_loc)

    cohort_s3   = cohort_s3.drop(columns=_LOC_T0_OPTIONAL_COLS, errors="ignore")
    cohort_s3   = cohort_s3.merge(loc_t0, on="stay_id", how="left")
    cohort_rhee = cohort_rhee.drop(columns=_LOC_T0_OPTIONAL_COLS, errors="ignore")
    cohort_rhee = cohort_rhee.merge(loc_t0, on="stay_id", how="left")

    print("\nStep 11: Excluding patients in OR/procedural areas at NE start...")
    _OR_CATS = frozenset({"or", "procedure_room", "procedural", "pacu", "operating_room"})
    for _lbl, _fl, _c in [("Sepsis-3", filter_s3, cohort_s3),
                           ("Rhee",     filter_rhee, cohort_rhee)]:
        _loc = _c["location_category"].fillna("").str.lower().str.strip()
        _n   = int(_loc.isin(_OR_CATS).sum())
        _fl.append({"step": "Location at t=0 = OR/procedural/PACU (excluded)",
                    "n_hospitalizations": _n})
        print(f"  {_lbl}: excluding {_n} patients in OR/procedural/PACU at t=0")
    cohort_s3   = cohort_s3[
        ~cohort_s3["location_category"].fillna("").str.lower().str.strip().isin(_OR_CATS)
    ].copy()
    cohort_rhee = cohort_rhee[
        ~cohort_rhee["location_category"].fillna("").str.lower().str.strip().isin(_OR_CATS)
    ].copy()

    filter_s3.append({"step": "Final Sepsis-3 cohort", "n_hospitalizations": len(cohort_s3)})
    filter_rhee.append({"step": "Final Rhee cohort",    "n_hospitalizations": len(cohort_rhee)})
    print(f"  Sepsis-3 final: {len(cohort_s3):,} | Rhee final: {len(cohort_rhee):,}")

    return cohort_s3, cohort_rhee, filter_s3, filter_rhee


# ---------------------------------------------------------------------------
# Phase B: Hourly feature extraction
# (Functions below are unchanged from original — NE-first t=0 anchor means
#  trajectory_start = first_norepi_time for all cohort patients.)
# ---------------------------------------------------------------------------
def build_hourly_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build per-patient hourly rows. All timestamps are tz-naive UTC."""
    rows = []
    for _, row in cohort.iterrows():
        ts_raw = row["trajectory_start"]
        traj_start = pd.to_datetime(ts_raw, utc=True).tz_localize(None)
        for h in range(int(row["traj_hours"]) + 1):
            rows.append({
                "stay_id":    row["stay_id"],
                "time_hour":  h,
                "start_time": traj_start + pd.Timedelta(hours=h),
                "end_time":   traj_start + pd.Timedelta(hours=h + 1),
            })
    return pd.DataFrame(rows)


def add_ne_dose(grid: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    ne = meds[
        (meds["med_category"] == "norepinephrine") &
        (meds["_convert_status"] == "success") &
        meds["med_dose_converted"].notna()
    ][["stay_id", "admin_dttm", "med_dose_converted"]].copy()
    ne = ne.rename(columns={"med_dose_converted": "med_dose"})
    ne = ne.sort_values(["stay_id", "admin_dttm"])
    ne["end_dttm"] = (ne.groupby("stay_id")["admin_dttm"]
                        .shift(-1)
                        .fillna(pd.Timestamp("2100-01-01")))
    ne.loc[ne["med_dose"] == 0.0, "end_dttm"] = ne.loc[ne["med_dose"] == 0.0, "admin_dttm"]

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    ne_g = ne.merge(g, on="stay_id")
    ne_hr = ne_g[(ne_g["admin_dttm"] < ne_g["end_time"]) &
                 (ne_g["end_dttm"]   > ne_g["start_time"])]
    ne_agg = (ne_hr.groupby(["stay_id", "time_hour"])["med_dose"]
                   .mean().reset_index()
                   .rename(columns={"med_dose": "norepinephrine"}))
    grid = grid.merge(ne_agg, on=["stay_id", "time_hour"], how="left")
    grid["norepinephrine"] = grid["norepinephrine"].fillna(0.0)
    return grid


def add_vaso_dose(grid: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    vaso = meds[
        (meds["med_category"] == "vasopressin") &
        (meds["_convert_status"] == "success") &
        meds["med_dose_converted"].notna()
    ][["stay_id", "admin_dttm", "med_dose_converted"]].copy()
    vaso = vaso.rename(columns={"med_dose_converted": "med_dose"})
    vaso = vaso.sort_values(["stay_id", "admin_dttm"])
    vaso["end_dttm"] = (vaso.groupby("stay_id")["admin_dttm"]
                            .shift(-1)
                            .fillna(pd.Timestamp("2100-01-01")))
    vaso.loc[vaso["med_dose"] == 0.0, "end_dttm"] = vaso.loc[vaso["med_dose"] == 0.0, "admin_dttm"]

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    vaso_g = vaso.merge(g, on="stay_id")
    vaso_hr = vaso_g[(vaso_g["admin_dttm"] < vaso_g["end_time"]) &
                     (vaso_g["end_dttm"]   > vaso_g["start_time"])]
    vaso_agg = (vaso_hr.groupby(["stay_id", "time_hour"])["med_dose"]
                       .mean().reset_index()
                       .rename(columns={"med_dose": "vaso_dose"}))
    grid = grid.merge(vaso_agg, on=["stay_id", "time_hour"], how="left")
    grid["vaso_dose"] = grid["vaso_dose"].fillna(0.0)
    return grid


def add_mbp(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    mbp = vitals[vitals["vital_category"] == "map"][
        ["hospitalization_id", "recorded_dttm", "vital_value"]
    ].copy()
    mbp = mbp.rename(columns={"hospitalization_id": "stay_id"})
    mbp["recorded_dttm"] = to_naive_utc(mbp["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    mbp_g = mbp.merge(g, on="stay_id")
    mbp_hr = mbp_g[(mbp_g["recorded_dttm"] >= mbp_g["start_time"]) &
                   (mbp_g["recorded_dttm"] < mbp_g["end_time"])]
    mbp_agg = (mbp_hr.groupby(["stay_id", "time_hour"])["vital_value"]
                     .mean().reset_index()
                     .rename(columns={"vital_value": "mbp"}))
    grid = grid.merge(mbp_agg, on=["stay_id", "time_hour"], how="left")
    return grid


def add_ventil(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    resp = pd.read_parquet(clif_dir / "clif_respiratory_support.parquet")
    imv = resp[resp["device_category"].isin(["IMV"])][
        ["hospitalization_id", "recorded_dttm"]
    ].copy()
    imv = imv.rename(columns={"hospitalization_id": "stay_id"})
    imv["recorded_dttm"] = to_naive_utc(imv["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    imv_g = imv.merge(g, on="stay_id")
    imv_hr = (imv_g[(imv_g["recorded_dttm"] >= imv_g["start_time"]) &
                    (imv_g["recorded_dttm"] < imv_g["end_time"])]
              .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n"))
    imv_hr["ventil"] = 1
    grid = grid.merge(imv_hr[["stay_id", "time_hour", "ventil"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["ventil"] = grid["ventil"].fillna(0).astype(int)
    return grid


def add_rrt(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    rrt_all = pd.read_parquet(clif_dir / "clif_crrt_therapy.parquet")[
        ["hospitalization_id", "recorded_dttm"]
    ].copy()
    rrt_all = rrt_all.rename(columns={"hospitalization_id": "stay_id"})
    rrt_all["recorded_dttm"] = to_naive_utc(rrt_all["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    rrt_g = rrt_all.merge(g, on="stay_id")
    rrt_hr = (rrt_g[(rrt_g["recorded_dttm"] >= rrt_g["start_time"]) &
                    (rrt_g["recorded_dttm"] < rrt_g["end_time"])]
              .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n"))
    rrt_hr["rrt"] = 1
    grid = grid.merge(rrt_hr[["stay_id", "time_hour", "rrt"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["rrt"] = grid["rrt"].fillna(0).astype(int)
    return grid


def add_steroids(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds_i = pd.read_parquet(clif_dir / "clif_medication_admin_intermittent.parquet")
    steroids = meds_i[meds_i["med_category"].isin(STEROID_CATEGORIES)][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    steroids = steroids.rename(columns={"hospitalization_id": "stay_id"})
    steroids["admin_dttm"] = to_naive_utc(steroids["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    st_g = steroids.merge(g, on="stay_id")
    st_hr = (
        st_g[
            (st_g["admin_dttm"] >= st_g["start_time"]) &
            (st_g["admin_dttm"] <  st_g["end_time"])
        ]
        .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n")
    )
    st_hr["_given"] = 1

    grid = grid.sort_values(["stay_id", "time_hour"])
    grid = grid.merge(st_hr[["stay_id", "time_hour", "_given"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["_given"] = grid["_given"].fillna(0).astype(int)
    grid["steroid"] = (
        (grid["_given"] | grid.groupby("stay_id")["_given"].shift(1, fill_value=0))
        .astype(int)
    )
    grid = grid.drop(columns="_given")
    return grid


def add_vitals(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    target = vitals[vitals["vital_category"].isin(
        ["heart_rate", "spo2", "temp_c", "respiratory_rate"]
    )][
        ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]
    ].copy()
    target = target.rename(columns={"hospitalization_id": "stay_id"})
    target["recorded_dttm"] = to_naive_utc(target["recorded_dttm"])
    target = target.dropna(subset=["recorded_dttm"])

    traj_start = (grid[grid["time_hour"] == 0][["stay_id", "start_time"]]
                  .rename(columns={"start_time": "traj_start"}))
    target = target.merge(traj_start, on="stay_id", how="inner")
    target["time_hour"] = ((target["recorded_dttm"] - target["traj_start"])
                           .dt.total_seconds() / 3600).astype(int)
    max_hour = grid.groupby("stay_id")["time_hour"].max().rename("max_hour")
    target = target.merge(max_hour, on="stay_id", how="inner")
    target = target[(target["time_hour"] >= 0) & (target["time_hour"] <= target["max_hour"])]

    v_last = (target.sort_values("recorded_dttm")
              .groupby(["stay_id", "time_hour", "vital_category"])["vital_value"]
              .last().reset_index())
    v_wide = (v_last.pivot_table(index=["stay_id", "time_hour"],
                                  columns="vital_category",
                                  values="vital_value")
              .reset_index())
    v_wide.columns.name = None
    v_wide = v_wide.rename(columns={"temp_c": "temperature"})
    grid = grid.merge(v_wide, on=["stay_id", "time_hour"], how="left")
    return grid


_NEE_FACTORS_MCG = {
    "norepinephrine": 1.0,
    "epinephrine":    1.0,
    "phenylephrine":  0.1,
    "dopamine":       0.01,
    "angiotensin ii": 10.0,
}
_VASO_FACTOR_U_MIN = 2.5
_ASSUMED_WEIGHT_KG = 70.0

_PREFERRED_UNITS_CONT = {
    "norepinephrine": "mcg/kg/min",
    "epinephrine":    "mcg/kg/min",
    "phenylephrine":  "mcg/kg/min",
    "dopamine":       "mcg/kg/min",
    "vasopressin":    "u/min",
    "vasopressine":   "u/min",
    "angiotensin ii": "mcg/kg/min",
}


def _load_and_convert_meds(co: ClifOrchestrator, stay_ids: list) -> pd.DataFrame:
    co.load_table("medication_admin_continuous", filters={"hospitalization_id": stay_ids})
    co.load_table("vitals", filters={
        "hospitalization_id": stay_ids,
        "vital_category": ["weight_kg"],
    })

    if co.vitals is not None and co.vitals.df is not None:
        w_df = (
            co.vitals.df[co.vitals.df["vital_category"] == "weight_kg"]
            [["hospitalization_id", "recorded_dttm", "vital_value"]]
            .dropna(subset=["vital_value"])
            .rename(columns={"vital_value": "weight_kg"})
            .sort_values("recorded_dttm")
        )
        if not w_df.empty:
            med_df = co.medication_admin_continuous.df.copy().sort_values("admin_dttm")
            med_keys = med_df[["hospitalization_id", "admin_dttm"]].drop_duplicates()

            wt_bwd = pd.merge_asof(
                med_keys, w_df,
                left_on="admin_dttm", right_on="recorded_dttm",
                by="hospitalization_id", direction="backward",
            )[["hospitalization_id", "admin_dttm", "weight_kg"]]

            wt_fwd = pd.merge_asof(
                med_keys, w_df,
                left_on="admin_dttm", right_on="recorded_dttm",
                by="hospitalization_id", direction="forward",
            )[["hospitalization_id", "admin_dttm", "weight_kg"]].rename(
                columns={"weight_kg": "_wt_fwd"}
            )

            wt = wt_bwd.merge(wt_fwd, on=["hospitalization_id", "admin_dttm"])
            wt["weight_kg"] = wt["weight_kg"].fillna(wt["_wt_fwd"])
            wt = wt[["hospitalization_id", "admin_dttm", "weight_kg"]]

            co.medication_admin_continuous.df = med_df.merge(
                wt, on=["hospitalization_id", "admin_dttm"], how="left"
            )

    co.convert_dose_units_for_continuous_meds(
        preferred_units=_PREFERRED_UNITS_CONT,
        save_to_table=True,
        override=True,
    )
    df = co.medication_admin_continuous.df_converted.copy()
    df = df.rename(columns={"hospitalization_id": "stay_id"})
    df["admin_dttm"] = to_naive_utc(df["admin_dttm"])
    return df


def add_nee(grid: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    converted = meds[meds["_convert_status"] == "success"].copy()

    cath = converted[
        converted["med_category"].isin(_NEE_FACTORS_MCG) &
        converted["med_dose_converted"].notna()
    ][["stay_id", "admin_dttm", "med_category", "med_dose_converted"]].copy()
    cath["nee_contrib"] = cath["med_dose_converted"] * cath["med_category"].map(_NEE_FACTORS_MCG)

    vaso_cats = {"vasopressin", "vasopressine", "terlipressin"}
    vaso = converted[
        converted["med_category"].str.lower().isin(vaso_cats) &
        converted["med_dose_converted"].notna()
    ][["stay_id", "admin_dttm", "med_dose_converted"]].copy()
    vaso["nee_contrib"] = vaso["med_dose_converted"] * _VASO_FACTOR_U_MIN
    vaso["med_category"] = "vasopressin"

    contrib = pd.concat([
        cath[["stay_id", "admin_dttm", "med_category", "nee_contrib"]],
        vaso[["stay_id", "admin_dttm", "med_category", "nee_contrib"]],
    ], ignore_index=True)
    contrib = contrib.sort_values(["stay_id", "med_category", "admin_dttm"])
    contrib["end_dttm"] = (contrib.groupby(["stay_id", "med_category"])["admin_dttm"]
                                   .shift(-1)
                                   .fillna(pd.Timestamp("2100-01-01")))
    contrib.loc[contrib["nee_contrib"] == 0.0, "end_dttm"] = (
        contrib.loc[contrib["nee_contrib"] == 0.0, "admin_dttm"]
    )

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    c_g = contrib.merge(g, on="stay_id")
    c_hr = c_g[(c_g["admin_dttm"] < c_g["end_time"]) &
               (c_g["end_dttm"]   > c_g["start_time"])]
    nee_agg = (c_hr.groupby(["stay_id", "time_hour"])["nee_contrib"]
                   .sum().reset_index()
                   .rename(columns={"nee_contrib": "nee"}))
    grid = grid.merge(nee_agg, on=["stay_id", "time_hour"], how="left")
    grid["nee"] = grid["nee"].fillna(0.0)
    return grid


_NEE_COMPONENT_CATS = ["epinephrine", "phenylephrine", "dopamine", "angiotensin ii"]


def add_nee_components(grid: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    converted = meds[meds["_convert_status"] == "success"].copy()
    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]

    for med_cat in _NEE_COMPONENT_CATS:
        sub = converted[
            (converted["med_category"] == med_cat) &
            converted["med_dose_converted"].notna()
        ][["stay_id", "admin_dttm", "med_dose_converted"]].copy()

        if sub.empty:
            grid[med_cat] = 0.0
            continue

        sub = sub.sort_values(["stay_id", "admin_dttm"])
        sub["end_dttm"] = (sub.groupby("stay_id")["admin_dttm"]
                           .shift(-1)
                           .fillna(pd.Timestamp("2100-01-01")))
        sub.loc[sub["med_dose_converted"] == 0.0, "end_dttm"] = (
            sub.loc[sub["med_dose_converted"] == 0.0, "admin_dttm"]
        )

        sub_g = sub.merge(g, on="stay_id")
        sub_hr = sub_g[(sub_g["admin_dttm"] < sub_g["end_time"]) &
                       (sub_g["end_dttm"]   > sub_g["start_time"])]
        agg = (sub_hr.groupby(["stay_id", "time_hour"])["med_dose_converted"]
               .mean().reset_index()
               .rename(columns={"med_dose_converted": med_cat}))
        grid = grid.merge(agg, on=["stay_id", "time_hour"], how="left")
        grid[med_cat] = grid[med_cat].fillna(0.0)

    return grid


def add_labs(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    labs = pd.read_parquet(clif_dir / "clif_labs.parquet")
    target = labs[labs["lab_category"].isin(
        ["bun", "creatinine", "lactate", "wbc", "platelet_count",
         "hemoglobin", "bilirubin_total", "po2_arterial"]
    )][["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"]].copy()
    target = target.rename(columns={"hospitalization_id": "stay_id"})
    target["lab_result_dttm"] = to_naive_utc(target["lab_result_dttm"])

    g = grid[["stay_id", "time_hour", "end_time"]]
    lab_g = target.merge(g, on="stay_id")
    lab_win = lab_g[lab_g["lab_result_dttm"] < lab_g["end_time"]]
    lab_last = (lab_win.sort_values("lab_result_dttm")
                .groupby(["stay_id", "time_hour", "lab_category"])["lab_value_numeric"]
                .last().reset_index())
    lab_wide = (lab_last.pivot_table(index=["stay_id", "time_hour"],
                                     columns="lab_category",
                                     values="lab_value_numeric")
                .reset_index())
    lab_wide.columns.name = None
    lab_wide = lab_wide.rename(columns={
        "platelet_count": "platelet",
        "bilirubin_total": "bilirubin",
    })
    grid = grid.merge(lab_wide, on=["stay_id", "time_hour"], how="left")
    return grid


def add_gcs(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    pa_path = clif_dir / "clif_patient_assessments.parquet"
    if not pa_path.exists():
        grid["gcs"] = np.nan
        return grid
    pa = pd.read_parquet(pa_path)
    gcs = pa[pa["assessment_category"] == "gcs_total"][
        ["hospitalization_id", "recorded_dttm", "numerical_value"]
    ].copy()
    gcs = gcs.rename(columns={"hospitalization_id": "stay_id"})
    gcs["recorded_dttm"] = to_naive_utc(gcs["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "end_time"]]
    gcs_g = gcs.merge(g, on="stay_id")
    gcs_win = gcs_g[gcs_g["recorded_dttm"] < gcs_g["end_time"]]
    gcs_last = (gcs_win.sort_values("recorded_dttm")
                .groupby(["stay_id", "time_hour"])["numerical_value"]
                .last().reset_index()
                .rename(columns={"numerical_value": "gcs"}))
    grid = grid.merge(gcs_last, on=["stay_id", "time_hour"], how="left")
    return grid


# Canonical respiratory device categories (SOFA-respiratory axis), Room Air
# is the reference/no-support level for the regression covariate.
_DEVICE_CATEGORIES = list(DEVICE_RANK_DICT.keys())


def add_resp_support(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """Hourly device_category + fio2_set (last observed within the hour,
    then forward-filled per stay_id — these are persistent device/ventilator
    settings, not one-off measurements, so LOCF avoids dropping hours where
    nothing was re-documented)."""
    resp_path = clif_dir / "clif_respiratory_support.parquet"
    if not resp_path.exists():
        grid["device_category"] = np.nan
        grid["fio2_set"] = np.nan
        return grid

    resp = pd.read_parquet(resp_path)[
        ["hospitalization_id", "recorded_dttm", "device_category", "fio2_set"]
    ].copy()
    resp = resp.rename(columns={"hospitalization_id": "stay_id"})
    resp["recorded_dttm"] = to_naive_utc(resp["recorded_dttm"])
    resp["device_category"] = (
        resp["device_category"].where(resp["device_category"].isin(_DEVICE_CATEGORIES))
    )

    g = grid[["stay_id", "time_hour", "end_time"]]
    resp_g = resp.merge(g, on="stay_id")
    resp_win = resp_g[resp_g["recorded_dttm"] < resp_g["end_time"]]
    resp_last = (resp_win.sort_values("recorded_dttm")
                 .groupby(["stay_id", "time_hour"])[["device_category", "fio2_set"]]
                 .last().reset_index())
    grid = grid.merge(resp_last, on=["stay_id", "time_hour"], how="left")

    grid = grid.sort_values(["stay_id", "time_hour"])
    grid["device_category"] = grid.groupby("stay_id")["device_category"].ffill()
    grid["fio2_set"] = grid.groupby("stay_id")["fio2_set"].ffill()
    return grid


def add_fluids(grid: pd.DataFrame, clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")

    all_fl = meds[meds["med_group"] == "fluids_electrolytes"][
        ["hospitalization_id", "admin_dttm", "med_category",
         "med_dose", "med_dose_unit", "mar_action_category"]
    ].copy()
    all_fl = all_fl.rename(columns={"hospitalization_id": "stay_id"})
    all_fl["admin_dttm"] = to_naive_utc(all_fl["admin_dttm"])
    all_fl = all_fl.sort_values(["stay_id", "med_category", "admin_dttm"])

    all_fl["end_dttm"] = (
        all_fl.groupby(["stay_id", "med_category"])["admin_dttm"]
        .shift(-1)
        .fillna(pd.Timestamp("2100-01-01"))
    )

    action_lo = all_fl["mar_action_category"].str.lower().fillna("").str.strip()
    is_stop = (action_lo == "stop") | (all_fl["med_dose"].fillna(-1) == 0.0)
    all_fl.loc[is_stop, "end_dttm"] = all_fl.loc[is_stop, "admin_dttm"]

    unit_norm = all_fl["med_dose_unit"].str.lower().str.strip()
    fluids = all_fl[
        all_fl["med_dose"].notna() &
        unit_norm.isin(["ml/hr", "ml/kg/hr"]) &
        ~is_stop
    ].copy()

    weight_map = cohort.set_index("stay_id")["weight"].fillna(_ASSUMED_WEIGHT_KG)
    fluids["weight_kg"] = fluids["stay_id"].map(weight_map).fillna(_ASSUMED_WEIGHT_KG)
    per_kg = fluids["med_dose_unit"].str.lower().str.strip() == "ml/kg/hr"
    fluids.loc[per_kg, "med_dose"] = fluids.loc[per_kg, "med_dose"] * fluids.loc[per_kg, "weight_kg"]
    fluids = fluids.drop(columns=["med_dose_unit", "weight_kg", "mar_action_category"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    f_g = fluids.merge(g, on="stay_id")
    f_hr = f_g[
        (f_g["admin_dttm"] < f_g["end_time"]) &
        (f_g["end_dttm"]   > f_g["start_time"])
    ].copy()

    overlap_start = f_hr[["admin_dttm", "start_time"]].max(axis=1)
    overlap_end   = f_hr[["end_dttm",   "end_time"]].min(axis=1)
    f_hr["volume_ml"] = (overlap_end - overlap_start).dt.total_seconds() / 3600 * f_hr["med_dose"]

    fluids_agg = (
        f_hr.groupby(["stay_id", "time_hour"])["volume_ml"]
        .sum().reset_index()
        .rename(columns={"volume_ml": "fluids"})
    )
    grid = grid.merge(fluids_agg, on=["stay_id", "time_hour"], how="left")
    grid["fluids"] = grid["fluids"].fillna(0.0)
    return grid


def add_explicitly_stopped(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")[
        ["hospitalization_id", "admin_dttm", "med_category", "med_dose", "mar_action_name"]
    ].copy()
    meds = meds.rename(columns={"hospitalization_id": "stay_id"})
    meds["admin_dttm"] = to_naive_utc(meds["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]

    for med_cat, col_name in [
        ("norepinephrine", "norepi_explicitly_stopped"),
        ("vasopressin",    "vaso_explicitly_stopped"),
    ]:
        stopped = meds[
            (meds["med_category"] == med_cat) &
            (
                meds["mar_action_name"].str.lower().str.contains(
                    "stop|discontinu|held|off|cancel|ended", na=False
                ) |
                (meds["med_dose"].fillna(-1) == 0)
            )
        ].copy()
        stopped_g = stopped.merge(g, on="stay_id")
        stopped_hr = stopped_g[
            (stopped_g["admin_dttm"] >= stopped_g["start_time"]) &
            (stopped_g["admin_dttm"] <  stopped_g["end_time"])
        ].groupby(["stay_id", "time_hour"]).size().reset_index(name="_n")
        stopped_hr[col_name] = 1
        grid = grid.merge(stopped_hr[["stay_id", "time_hour", col_name]],
                          on=["stay_id", "time_hour"], how="left")
        grid[col_name] = grid[col_name].fillna(0).astype(int)
    return grid


def add_mar_actions(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")

    action_col = next(
        (c for c in ["mar_action_group", "mar_action_name"] if c in meds.columns),
        None,
    )
    if action_col is None:
        grid["ne_mar_action"]   = np.nan
        grid["vaso_mar_action"] = np.nan
        return grid

    meds = meds[["hospitalization_id", "admin_dttm", "med_category", action_col]].copy()
    meds = meds.rename(columns={"hospitalization_id": "stay_id"})
    meds["admin_dttm"] = to_naive_utc(meds["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]

    for med_cat, col_name in [
        ("norepinephrine", "ne_mar_action"),
        ("vasopressin",    "vaso_mar_action"),
    ]:
        med_sub = meds[meds["med_category"] == med_cat]
        med_g   = med_sub.merge(g, on="stay_id")
        med_hr  = med_g[
            (med_g["admin_dttm"] >= med_g["start_time"]) &
            (med_g["admin_dttm"] <  med_g["end_time"])
        ]
        last_action = (
            med_hr.sort_values("admin_dttm")
            .groupby(["stay_id", "time_hour"])[action_col]
            .last()
            .reset_index()
            .rename(columns={action_col: col_name})
        )
        grid = grid.merge(last_action, on=["stay_id", "time_hour"], how="left")

    return grid


def add_hourly_sofa(grid: pd.DataFrame, co: ClifOrchestrator) -> pd.DataFrame:
    cohort_ids = grid["stay_id"].unique().tolist()

    traj = (grid.groupby("stay_id")
            .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
            .reset_index()
            .rename(columns={"stay_id": "hospitalization_id"}))
    traj["start_time"] = pd.to_datetime(traj["start_time"], utc=True)
    traj["end_time"]   = pd.to_datetime(traj["end_time"],   utc=True)

    _load_sofa_tables(co, cohort_ids)
    co.create_wide_dataset(
        category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df=traj,
        return_dataframe=True,
    )
    _add_missing_med_cols(co)

    sofa_scores = co.compute_sofa_scores(
        wide_df=co.wide_df,
        id_name="hospitalization_id",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        create_new_wide_df=False,
    )
    sofa_scores = (sofa_scores[["hospitalization_id", "sofa_total"]]
                   .rename(columns={"hospitalization_id": "stay_id", "sofa_total": "sofa"}))
    grid = grid.merge(sofa_scores, on="stay_id", how="left")
    grid["sofa"] = grid["sofa"].fillna(0.0)
    return grid


def build_features(cohort: pd.DataFrame, co: ClifOrchestrator, clif_dir: Path) -> pd.DataFrame:
    print("\nBuilding hourly grid...")
    grid = build_hourly_grid(cohort)
    print(f"  {len(grid):,} patient-hours across {cohort['stay_id'].nunique()} patients")

    print("Converting medication units via clifpy...")
    meds = _load_and_convert_meds(co, cohort["stay_id"].tolist())

    print("Adding NE dose...")
    grid = add_ne_dose(grid, meds)

    print("Adding vasopressin dose...")
    grid = add_vaso_dose(grid, meds)

    print("Adding MBP...")
    grid = add_mbp(grid, clif_dir)

    print("Adding ventilation (IMV)...")
    grid = add_ventil(grid, clif_dir)

    print("Adding RRT (CRRT + IHD)...")
    grid = add_rrt(grid, clif_dir)

    print("Adding steroids...")
    grid = add_steroids(grid, clif_dir)

    print("Adding vitals (heart rate, SpO2, temperature, respiratory rate)...")
    grid = add_vitals(grid, clif_dir)

    print("Adding NEE (norepinephrine equivalent dose)...")
    grid = add_nee(grid, meds)

    print("Adding NEE component doses (epinephrine, phenylephrine, dopamine)...")
    grid = add_nee_components(grid, meds)

    print("Adding labs (BUN, creatinine, lactate, WBC, platelet, hemoglobin, bilirubin)...")
    grid = add_labs(grid, clif_dir)

    print("Adding GCS (last observed per hour)...")
    grid = add_gcs(grid, clif_dir)

    print("Adding respiratory support (device_category, FiO2)...")
    grid = add_resp_support(grid, clif_dir)
    po2_ok  = grid["po2_arterial"].between(0, 700)
    fio2_ok = grid["fio2_set"].between(0.21, 1.0)
    grid["p_f_ratio"] = np.where(
        po2_ok & fio2_ok, grid["po2_arterial"] / grid["fio2_set"], np.nan
    )

    print("Adding explicit cessation flags (NE/vasopressin stopped events)...")
    grid = add_explicitly_stopped(grid, clif_dir)

    print("Adding MAR action group (NE/vasopressin, last per hour)...")
    grid = add_mar_actions(grid, clif_dir)

    print("Adding SOFA via clifpy (per patient, broadcast to all hours)...")
    grid = add_hourly_sofa(grid, co)

    print("Adding fluids (mL/hr × overlap hours from fluids_electrolytes meds)...")
    grid = add_fluids(grid, clif_dir, cohort)

    grid["urine_output"] = 0.0
    grid["action_vaso"] = (grid["vaso_dose"] > 0).astype(int)

    grid = grid.merge(cohort[["stay_id", "death_hour"]], on="stay_id", how="left")
    grid["death"] = (grid["time_hour"] == grid["death_hour"]).astype(int)
    grid = grid.drop(columns=["death_hour"])

    grid = grid.drop(columns=["start_time", "end_time"])
    grid = grid.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_COHORT_COLS = [
    "stay_id", "hospital_death", "anchor_year_group",
    "traj_hours", "death_hour", "first_norepi_time", "trajectory_start",
    "age", "gender", "race", "weight",
    "sepsis_onset_sofa", "initial_lactate", "cci_score",
    "vaso_before_traj", "location_category", "location_type",
    "hospital_id", "hospital_type",
    "icu_los_days", "hospital_los_days", "traj_end_reason",
]


def main():
    PATIENT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)

    co = ClifOrchestrator(
        data_directory=str(CLIF_DIR),
        filetype="parquet",
        timezone=TIMEZONE,
        output_directory=str(PATIENT_LEVEL_DIR),
    )

    print("=" * 60)
    print("PHASE A: COHORT IDENTIFICATION")
    print("=" * 60)
    cohort_s3, cohort_rhee, filter_s3, filter_rhee = build_cohort(CLIF_DIR, co)

    print(f"\nSepsis-3 cohort: {len(cohort_s3):,} patients  "
          f"(mortality {cohort_s3['hospital_death'].mean():.1%})")
    print(f"Rhee cohort:     {len(cohort_rhee):,} patients  "
          f"(mortality {cohort_rhee['hospital_death'].mean():.1%})")

    pd.DataFrame(filter_s3).to_csv(
        PATIENT_LEVEL_DIR / "cohort_filter_counts_sepsis3.csv", index=False
    )
    pd.DataFrame(filter_rhee).to_csv(
        PATIENT_LEVEL_DIR / "cohort_filter_counts_rhee.csv", index=False
    )
    print("Saved filter count CSVs.")

    print("\n" + "=" * 60)
    print("PHASE B: FEATURE EXTRACTION (union of both cohorts)")
    print("=" * 60)

    # Union cohort for features: one row per stay_id (prefer Sepsis-3 row where duplicated)
    union_cohort = (pd.concat([cohort_s3, cohort_rhee])
                    .drop_duplicates(subset=["stay_id"])
                    .reset_index(drop=True))

    features = build_features(union_cohort, co, CLIF_DIR)

    # Validate: NE > 0 at t=0 for every patient
    ne_at_t0 = features[features["time_hour"] == 0].set_index("stay_id")["norepinephrine"]
    no_ne_ids = set(ne_at_t0[ne_at_t0 == 0].index)
    if no_ne_ids:
        print(f"\n  WARNING: {len(no_ne_ids)} patients have NE=0 at t=0 — excluded.")
        features   = features[~features["stay_id"].isin(no_ne_ids)].copy()
        cohort_s3  = cohort_s3[~cohort_s3["stay_id"].isin(no_ne_ids)].copy()
        cohort_rhee = cohort_rhee[~cohort_rhee["stay_id"].isin(no_ne_ids)].copy()

    print(f"\nFeatures: {len(features):,} rows | "
          f"{features['stay_id'].nunique():,} patients")

    # Save — select only the standard schema columns that exist
    def _save_cohort(df, path):
        cols = [c for c in _COHORT_COLS if c in df.columns]
        df[cols].to_parquet(path, index=False)

    _save_cohort(cohort_s3,   PATIENT_LEVEL_DIR / "cohort_sepsis3.parquet")
    _save_cohort(cohort_rhee, PATIENT_LEVEL_DIR / "cohort_rhee.parquet")
    features.to_parquet(PATIENT_LEVEL_DIR / "features.parquet", index=False)

    print(f"\nOutputs written to {PATIENT_LEVEL_DIR}/")
    print("  cohort_sepsis3.parquet")
    print("  cohort_rhee.parquet")
    print("  features.parquet  (union — filter to cohort IDs in downstream scripts)")


if __name__ == "__main__":
    main()
