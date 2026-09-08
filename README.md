# Epidemiology of Vasopressin in Septic Shock

Federated multi-site analysis of vasopressin initiation patterns in ICU patients meeting septic shock criteria under three cohort definitions: Sepsis-3 (CMS), Rhee/CDC Adult Sepsis Event (hand-coded), and Rhee/CDC ASE (clifpy implementation).

## CLIF VERSION

2.1.0

## Objective

Characterize clinician vasopressin initiation behavior across sites — baseline characteristics, timing relative to shock onset, vasopressor combinations, and how much of the between-site variation in initiation rates is attributable to case-mix differences versus practice variation (federated ICC decomposition). The project supports federated execution: each site runs extraction and summary scripts locally and shares only aggregate outputs.

Feature-threshold decision-rule identification, optimal-threshold testing, and subphenotype/subgroup analysis (formerly part of this repo, before its own renumbering) have moved to the sibling **CLIF-OPT-VASO** repo, which shares this repo's extraction scripts (01/01b) as a starting point.

## Required CLIF tables and fields

The following tables are required:

1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`
3. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = `'map'`
4. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value`
   - `lab_category` = `'lactate'`, `'creatinine'`, `'bun'`
5. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`
   - `med_category` = `'norepinephrine'`, `'vasopressin'`, `'epinephrine'`, `'phenylephrine'`, `'dopamine'`, `'angiotensin'`, `'hydrocortisone'`, `'dexamethasone'`, `'methylprednisolone'`
6. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`
7. **patient_assessments** (SOFA): `hospitalization_id`, `recorded_dttm`, `numerical_value`

The [clifpy](https://common-longitudinal-icu-data-format.github.io/clifpy/) package is used for SOFA score computation and outlier handling.

## Cohort identification

All three cohorts share the same t=0 anchor: the **first norepinephrine administration** (≥`MIN_NE_RECORDS` = 2 NE records in `medication_admin_continuous`).

Each definition first identifies an **infection anchor** from antibiotics and blood cultures, then requires **t=0 to fall within ±`NE_WINDOW_HOURS` (24 h) of that anchor**. The window is two-sided and inclusive: NE may start before or after the infection event. When a patient has several qualifying infection anchors, candidates are restricted to those inside the window and the earliest remaining one is kept — so a patient qualifies on *any* in-window anchor, not only the earliest overall.

| Cohort | Key criteria |
|--------|-------------|
| **sepsis3** | CMS qualifying IV abx + blood culture within 24 h of each other (anchor = earlier of the pair); NE start within ±24 h of the anchor; lactate > 2 mmol/L within ±24 h of the anchor |
| **rhee** | Blood culture within ±24 h of NE start (anchor = earliest in-window culture); first qualifying IV abx within 2 calendar days of the culture; ≥4 consecutive qualifying antibiotic calendar days (≤1-day gap) or course ends ≤1 day before discharge/death; lactate ≥ 2 mmol/L within ±24 h of NE start |
| **rhee_clifpy** | clifpy `compute_ase` (include_lactate=True, apply_rit=True); ASE episode's blood culture within ±24 h of NE start; lactate ≥ 2 is one of six optional organ-dysfunction criteria rather than a hard filter |

**Shared exclusion:** patients whose ADT `location_category` at t=0 is an OR / procedure room / PACU are dropped from all three cohorts.

**Not exclusions** (both are retained and flagged, so they can be analysed rather than silently dropped):

- **Vasopressin before NE.** `vaso_before_traj` = 1 if vasopressin was given in the 24 h immediately before t=0; `vaso_before_ne` = 1 for any lead time; `vaso_to_ne_hours` gives the lead time in hours (positive = vasopressin first). Reported as `NOTE:` rows in the filter-count CSVs and as the `drug=NE, direction=after` rows of `vaso_timing_summary.csv`.
- **NE = 0 at t=0.** Since t=0 *is* the first NE administration, NE should never be 0 there; any violation is a plumbing artifact, not a clinical finding. `01_clif_extract.py` verifies the invariant and attributes each violation to a cause in `ne_zero_at_t0_diagnostic.csv` (`zero_dose_at_t0` — EHR zero-rate start marker; `dose_null_at_t0`; `unit_conversion_failed`; `anchor_record_missing`; `unexplained` — a genuine bug in the interval logic, which should be zero). Affected patients stay in every cohort.

**Trajectory:** from t=0 to the earliest of death, ICU discharge, or a 120-hour cap (`traj_end_reason` records which), sampled hourly.

## Detailed instructions for running the project

### 1. Configure `config/config.py`

```bash
cp config/config.example.py config/config.py
# Edit config/config.py: set CLIF_DIR and OUTPUT_ROOT for your site
```

The scripts create per-site subfolders under `<OUTPUT_ROOT>/output/` automatically. See [`config/README.md`](config/README.md) for details.

### 2. Set up the Python environment

```bash
uv sync
```

### 3. Extract cohort data

```bash
uv run python code/01_clif_extract.py
```

Writes `cohort_sepsis3.parquet`, `cohort_rhee.parquet`, `cohort_rhee_clifpy.parquet`, `features.parquet`, three `cohort_filter_counts_<cohort>.csv` files, and `ne_zero_at_t0_diagnostic.csv` to `output/patient_level_data_<SITE>/`. **These never leave the site.**

### 4. Run federated summary

```bash
uv run python code/02_site_summary.py
```

Writes aggregate CSVs to `output/upload_to_box_<SITE>/`.

### 5. Run epidemiological analysis

```bash
uv run python code/04_epi_analysis.py
```

Writes figures and aggregate CSVs to `output/upload_to_box_<SITE>/<cohort>/epi_analysis/`.

### 6. Run eligible-but-untreated analysis

```bash
uv run python code/06_eligible_untreated_analysis.py
```

Classifies patients who sustained high NEE without vasopressin (comfort care / too brief / MAP recovered / unexpectedly untreated). Writes figures + summary CSV to `output/upload_to_box_<SITE>/<cohort>/eligible_untreated/`. Reads `clif_code_status.parquet` (beta — skips comfort-care classification gracefully if absent).

### 7. Share your upload folder

**Share only `output/upload_to_box_<SITE>/`** with the coordinating site. This folder contains no patient-level data. Or use `run_pipeline.py` to run all steps (01–06) in sequence.

See [`code/README.md`](code/README.md) for full script documentation.

## Output structure

```
output/
  patient_level_data_<SITE>/         # PHI intermediate — NEVER share
    cohort_sepsis3.parquet
    cohort_rhee.parquet
    cohort_rhee_clifpy.parquet
    cohort_filter_counts_sepsis3.csv
    cohort_filter_counts_rhee.csv
    cohort_filter_counts_rhee_clifpy.csv
    ne_zero_at_t0_diagnostic.csv       # NE>0 at t=0 invariant audit
    features.parquet
  upload_to_box_<SITE>/              # Aggregate results — SHARE THIS FOLDER
    cohort_comparison/                ← 03_cohort_comparison_summary.py
      cohort_comparison_stats.json
    <cohort>/                         # sepsis3/ or rhee/ or rhee_clifpy/
      cohort_filter_counts.csv        ← 02_site_summary.py
      split_counts.csv                ← 02_site_summary.py
      baseline_table1.csv             ← 02_site_summary.py
      feature_at_initiation.csv       ← 02_site_summary.py
      feature_thresholds_youden.csv   ← 02_site_summary.py
      feature_roc_curves.csv          ← 02_site_summary.py
      site_variation_packet_<cohort>_<SITE>.json  ← 05_site_variation_analysis.py
      epi_analysis/                   ← 04_epi_analysis.py (CSVs + figures)
      eligible_untreated/             ← 06_eligible_untreated_analysis.py (figures + summary CSV)
        km_cif_by_nee_bin.csv
        km_survival_by_nee_bin.csv
        km_survival_ever_never_vaso.csv
        nee_proportion_on_vaso.csv
        nee_vaso_state_hours.csv
        feature_dist_nee_vaso.csv
        tod_init_features_binned.csv
        tod_init_features_lowess.csv
        time_to_vaso_hist.csv
        wait_time_histograms.csv
        init_features_by_quartile.csv
        init_features_by_nee_bin.csv
        vasopressor_combinations.csv
        vaso_timing_summary.csv
        vaso_receipt_logreg.csv
        <site>_analysis*.png          ← figures alongside CSVs
```

Threshold/rule-optimality outputs (`<cohort>/threshold/`, `global_rules_<cohort>.json`, etc.) are produced by the sibling **CLIF-OPT-VASO** repo's pipeline instead.

## Directory structure

```
.
├── code/                              # All analysis scripts
│   ├── 00_mimic_extract_duckdb.py     # Builds intermediate MIMIC DuckDB (optional, MIMIC only)
│   ├── 01_clif_extract.py             # CLIF 2.1.0 cohort extraction (per CLIF site)
│   ├── 01b_mimic_extract.py           # MIMIC-CLIF cohort extraction (MIMIC only)
│   ├── 02_site_summary.py             # Federated aggregate summary (per site, per cohort)
│   ├── 03_cohort_comparison_summary.py  # sepsis3 vs rhee cohort comparison stats (per site)
│   ├── 04_epi_analysis.py             # Epidemiological characterization (per site)
│   ├── 05_site_variation_analysis.py  # Patient/ward/hospital variance decomposition (per site)
│   ├── 06_eligible_untreated_analysis.py  # Eligible-but-untreated classification + characterization (per site)
│   ├── 07_multisite_epi_plots.py      # Multi-site epi comparison figures (coordinating site)
│   ├── 08_cross_site_variation_analysis.py  # Pooled GEE + DL meta-analysis (coordinating site)
│   ├── 09_cross_site_vasopressin_analysis.py  # Cross-site epi comparison tables + plots (coordinating site)
│   ├── 10_consolidated_report.py      # One HTML report, 4 sections (coordinating site)
│   ├── 11_ne_infection_timing.py      # NE/infection timing report (coordinating site)
│   ├── run_pipeline.py                # Orchestrates 01–05 at each site
│   ├── run_coordinating_pipeline.py   # Orchestrates 07–11 at the coordinating site
│   └── README.md
├── config/                      # Configuration
│   ├── config.example.py        # Copy to config/config.py and fill in site paths
│   ├── config.py                # Site-specific config (gitignored)
│   └── README.md
├── docs/                        # Documentation
│   └── clif_extract.md
├── output/                      # Generated outputs (gitignored)
├── pyproject.toml               # Dependencies and project metadata
└── uv.lock                      # Pinned, reproducible dependency versions
```

Feature-threshold/rule-optimality analysis and its own copy of the extraction
scripts live in the sibling **CLIF-OPT-VASO** repo.
