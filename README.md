# Epidemiology of Vasopressin in Septic Shock

Federated multi-site analysis of vasopressin initiation patterns in ICU patients meeting septic shock criteria (Sepsis-3 + norepinephrine + lactate > 2 mmol/L).

## CLIF VERSION

2.1.0

## Objective

Characterize clinician vasopressin initiation behavior across sites, identify feature-threshold decision rules that explain initiation timing, and compare clinician practice to a reinforcement-learning (RL) policy. The project supports federated execution: each site runs extraction and summary scripts locally and shares only aggregate outputs.

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
uv run python code/clif_extract.py
```

Writes `cohort.parquet`, `features.parquet`, `cohort_filter_counts.csv` to `output/patient_level_data_<SITE>/`. **These never leave the site.**

### 4. Run federated summary

```bash
uv run python code/site_summary.py
```

Writes aggregate CSVs to `output/upload_to_box_<SITE>/`.

### 5. Run threshold analysis

```bash
uv run python code/site_threshold_sweep.py
```

Writes `threshold_comparison_table.csv`, `patient_level_table.csv`, and plots to `output/upload_to_box_<SITE>/threshold/`.

### 6. Run threshold outcome analysis (optional)

```bash
uv run python code/site_threshold_outcome.py
```

Writes `threshold_outcome_table.csv` and `threshold_concordance_summary.csv` to `output/upload_to_box_<SITE>/threshold/`.

### 7. Run epidemiological analysis

```bash
uv run python code/epi_analysis.py
uv run python code/epi_analysis.py --site MIMIC   # override site name
```

Writes figures and 12 federated-safe CSVs to `output/upload_to_box_<SITE>/epi_analysis/`.

### 8. Share your upload folder

**Share only `output/upload_to_box_<SITE>/`** with the coordinating site. This folder contains no patient-level data.

### 9. Cross-site aggregation (coordinating site only)

Place all received `upload_to_box_<SITE>/` folders inside your `output/` directory, then:

```bash
uv run python code/cross_site_vasopressin_analysis.py
uv run python code/make_summary_report.py
uv run python code/make_summary_report.py --embed   # portable single-file HTML
```

`cross_site_vasopressin_analysis.py` auto-detects all `upload_to_box_*` directories in `output/`. `make_summary_report.py` reads figures from each site's `upload_to_box_*/epi_analysis/` folder.

See [`code/README.md`](code/README.md) for full script documentation.

## Output structure

```
output/
  patient_level_data_<SITE>/         # PHI intermediate — NEVER share
    cohort.parquet
    features.parquet
  epi_analysis_<SITE>/               # PHI figures — NEVER share
    <site>_analysis*.png
  upload_to_box_<SITE>/              # Aggregate results — SHARE THIS FOLDER
    cohort_filter_counts.csv         ← site_summary.py
    split_counts.csv                 ← site_summary.py
    baseline_table1.csv              ← site_summary.py
    feature_at_initiation.csv        ← site_summary.py
    feature_thresholds_youden.csv    ← site_summary.py
    feature_roc_curves.csv           ← site_summary.py
    threshold/                       ← site_threshold_sweep.py + site_threshold_outcome.py
      threshold_comparison_table.csv
      patient_level_table.csv
      patient_level_confounders.csv
      threshold_sweep_data.csv
      threshold_outcome_table.csv    ← site_threshold_outcome.py (optional)
      threshold_concordance_summary.csv
      plots/
        threshold_sweep.png
        decision_tree_fidelity.png
        threshold_sweep_individual/
    epi_analysis/                    ← epi_analysis.py (CSVs + figures together)
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
      <site>_analysis*.png           ← figures alongside CSVs

cross-site output/
  plots/                             ← cross_site_vasopressin_analysis.py outputs
    consort_flowchart.png
    baseline_comparison_combined.png
    initiation_features_combined.png
    kappa_initiation_step.png
    ...
  cross_site_summary.csv
```

## Directory structure

```
.
├── code/                        # All analysis scripts
│   ├── clif_extract.py          # CLIF 2.1.0 cohort extraction
│   ├── site_summary.py          # Federated aggregate summary (run at each site)
│   ├── site_threshold_sweep.py  # Per-feature threshold sweep (run at each site)
│   ├── site_threshold_outcome.py# Discrete-time survival analysis (optional, each site)
│   ├── epi_analysis.py          # Epidemiological characterization (run at each site)
│   ├── cross_site_vasopressin_analysis.py  # Cross-site combined analysis (coordinating site)
│   ├── make_summary_report.py   # HTML summary report (coordinating site)
│   └── README.md
├── config/                      # Configuration
│   ├── config.example.py        # Copy to config/config.py and fill in site paths
│   ├── config.py                # Site-specific config (gitignored)
│   └── README.md
├── docs/                        # Documentation
│   └── clif_extract.md
├── output/                      # Generated outputs (gitignored)
├── cross-site output/           # Cross-site aggregation outputs
├── pyproject.toml               # Dependencies and project metadata
└── uv.lock                      # Pinned, reproducible dependency versions
```
