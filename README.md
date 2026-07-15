# Epidemiology of Vasopressin in Septic Shock

Federated multi-site analysis of vasopressin initiation patterns in ICU patients meeting septic shock criteria (Sepsis-3 + norepinephrine + lactate > 2 mmol/L).

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

**Inclusion:**
- First ICU stay per patient
- Sepsis-3 criteria: suspected infection + SOFA ≥ 2 at or near ICU admission
- Norepinephrine started within 24 hours of ICU admission (≥ 2 administration records)
- Lactate > 2.0 mmol/L within 24 hours of suspected infection

**Exclusion:** Patients already on vasopressin in the 24 hours before trajectory start.

**Trajectory:** Up to 120 hours from shock onset (norepinephrine start), sampled hourly.

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

Writes `cohort.parquet`, `features.parquet`, `cohort_filter_counts.csv` to `output/patient_level_data_<SITE>/`. **These never leave the site.**

### 4. Run federated summary

```bash
uv run python code/02_site_summary.py
```

Writes aggregate CSVs to `output/upload_to_box_<SITE>/`.

### 5. Run epidemiological analysis

```bash
uv run python code/03_epi_analysis.py
```

Writes figures and aggregate CSVs to `output/upload_to_box_<SITE>/epi_analysis/`, including a federated ICC return packet (`site_packet_<SITE>.json`).

**UCMC runs first** (with `FEDERATED_ICC_ANCHOR = None` in config). **All other sites** uncomment the pre-filled `FEDERATED_ICC_ANCHOR` block in `config.example.py` before running. See [`config/README.md`](config/README.md) for details.

See `code/run_pipeline.py` to run the full per-site pipeline (01-04) in one command.

### 6. Share your upload folder

**Share only `output/upload_to_box_<SITE>/`** with the coordinating site. This folder contains no patient-level data.

See [`code/README.md`](code/README.md) for full script documentation.

## Output structure

```
output/
  patient_level_data_<SITE>/         # PHI intermediate — NEVER share
    cohort_sepsis3.parquet
    cohort_rhee.parquet
    features.parquet
  upload_to_box_<SITE>/              # Aggregate results — SHARE THIS FOLDER
    cohort_comparison/                ← 02b_cohort_comparison_summary.py
      cohort_comparison_stats.json
    timing/                           ← 04b_ne_infection_timing_summary.py (CLIF sites only)
      timing_stats.json
    <cohort>/                         # sepsis3/ or rhee/
      cohort_filter_counts.csv        ← 02_site_summary.py
      split_counts.csv                ← 02_site_summary.py
      baseline_table1.csv             ← 02_site_summary.py
      feature_at_initiation.csv       ← 02_site_summary.py
      feature_thresholds_youden.csv   ← 02_site_summary.py
      feature_roc_curves.csv          ← 02_site_summary.py
      epi_analysis/                   ← 03_epi_analysis.py (CSVs + figures + ICC packet)
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
        vaso_receipt_logreg.csv
        site_packet_<SITE>.json       ← federated ICC return packet
        <site>_analysis*.png          ← figures alongside CSVs
```

Threshold/rule-optimality outputs (`<cohort>/threshold/`, `global_rules_<cohort>.json`, etc.) are produced by the sibling **CLIF-OPT-VASO** repo's pipeline instead.

## Directory structure

```
.
├── code/                              # All analysis scripts
│   ├── 00_mimic_extract_duckdb.py     # Builds intermediate MIMIC DuckDB (optional, MIMIC only)
│   ├── 01_clif_extract.py             # CLIF 2.1.0 cohort extraction (per CLIF site)
│   ├── 01b_mimic_extract.py           # MIMIC-CLIF cohort extraction (per site)
│   ├── 02_site_summary.py             # Federated aggregate summary (per site, per cohort)
│   ├── 02b_cohort_comparison_summary.py  # sepsis3 vs rhee cohort comparison stats (per site)
│   ├── 03_epi_analysis.py             # Epidemiological characterization + ICC packet (per site)
│   ├── 04_site_variation_analysis.py  # Patient/ward/hospital variance decomposition (per site)
│   ├── 04b_ne_infection_timing_summary.py  # NE-vs-infection timing stats (per site, CLIF only)
│   ├── 05_multisite_epi_plots.py      # Multi-site epi comparison figures (coordinating site)
│   ├── 06_cross_site_variation_analysis.py  # Pooled GEE + DL meta-analysis (coordinating site)
│   ├── 07_cross_site_vasopressin_analysis.py  # Cross-site epi comparison tables + plots (coordinating site)
│   ├── 08_consolidated_report.py      # One HTML report, 4 sections (coordinating site)
│   ├── 09_ne_infection_timing.py      # NE-vs-infection timing report (coordinating site)
│   ├── run_pipeline.py                # Orchestrates 01-04/04b for one or more sites
│   ├── run_coordinating_pipeline.py   # Orchestrates 05-09 at the coordinating site
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
