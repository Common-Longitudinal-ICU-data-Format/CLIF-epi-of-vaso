# code/

Per-site analysis scripts for the vasopressin epidemiology project.

## Cohort structure

Two cohorts are identified relative to first norepinephrine (NE) administration (t=0, ±24h window):

| Cohort | Criteria |
|--------|----------|
| **sepsis3** | CMS qualifying IV abx + blood culture within 24h of each other, both within ±24h of NE start; lactate > 2 mmol/L within ±24h of NE start |
| **rhee** | Blood culture within ±24h of NE start; first qualifying IV abx within 2 calendar days; ≥4 consecutive qualifying antibiotic days (≤1-day gap) or course ends ≤1 day before discharge/death; lactate ≥ 2 mmol/L within ±24h of NE start |

Each extraction script produces `cohort_sepsis3.parquet`, `cohort_rhee.parquet`, and a shared `features.parquet` (hourly features for the union of both cohorts). Downstream scripts accept `--cohort sepsis3|rhee`.

---

## Scripts

All scripts run at each participating site.

| Script | Purpose |
|--------|---------|
| `01_clif_extract.py` | Extract dual-cohort septic shock cohort + hourly features from CLIF 2.1.0 parquet tables |
| `02_site_summary.py` | Compute federated-safe aggregate statistics; write shareable CSVs to `upload_to_box_<SITE>/<cohort>/` |
| `03_cohort_comparison_summary.py` | Federated-safe Sepsis-3 vs Rhee cohort-comparison stats; write `upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` |
| `04_epi_analysis.py` | Epidemiological characterization; write figures + CSVs to `upload_to_box_<SITE>/<cohort>/epi_analysis/` |
| `05_site_variation_analysis.py` | Site-specific hospital/ICU variation analysis: GEE logistic, discrete-time hazard (ICC/MOR), MELR moments; write `site_variation_packet_<cohort>_<SITE>.json` |
| `06_ne_infection_timing_summary.py` | Federated-safe NE-vs-suspected-infection timing stats; write `upload_to_box_<SITE>/timing/timing_stats.json` |

---

## Usage

```bash
# Extraction (writes PHI intermediate to output/patient_level_data_<SITE>/)
uv run python code/01_clif_extract.py

# Federated summary (writes shareable CSVs to output/upload_to_box_<SITE>/<cohort>/)
uv run python code/02_site_summary.py --cohort sepsis3
uv run python code/02_site_summary.py --cohort rhee

# Cohort-comparison summary (sepsis3 vs rhee; needs both cohort files)
uv run python code/03_cohort_comparison_summary.py

# Epidemiological analysis
uv run python code/04_epi_analysis.py --cohort sepsis3
uv run python code/04_epi_analysis.py --cohort rhee

# Site variation analysis (ICC/MOR/GEE + MELR packet)
uv run python code/05_site_variation_analysis.py --cohort both

# NE-infection timing summary
uv run python code/06_ne_infection_timing_summary.py
```

Or drive all of the above with `run_pipeline.py`.

---

## Outputs

| Script | Output location | Notes |
|--------|----------------|-------|
| `01_clif_extract.py` | `output/patient_level_data_<SITE>/cohort_sepsis3.parquet`<br>`output/patient_level_data_<SITE>/cohort_rhee.parquet`<br>`output/patient_level_data_<SITE>/features.parquet` | PHI — never shared |
| `02_site_summary.py` | `output/upload_to_box_<SITE>/<cohort>/` | Share |
| `03_cohort_comparison_summary.py` | `output/upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` | Share |
| `04_epi_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/epi_analysis/` | Share |
| `05_site_variation_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/site_variation_packet_<cohort>_<SITE>.json` | Share |
| `06_ne_infection_timing_summary.py` | `output/upload_to_box_<SITE>/timing/timing_stats.json` | Share |

---

## Prerequisites

### CLIF 2.1.0 parquet files

Required at `CLIF_DIR` (set in `config/config.py`):

| Table | Used for |
|-------|----------|
| `clif_adt.parquet` | ICU admission/discharge times, location at t=0 |
| `clif_patient.parquet` | Demographics (sex, race, death date) |
| `clif_hospitalization.parquet` | Discharge disposition, age at admission |
| `clif_medication_admin_intermittent.parquet` | CMS qualifying antibiotics (sepsis criteria) |
| `clif_medication_admin_continuous.parquet` | NE, vasopressin, all vasopressors, fluids |
| `clif_microbiology_culture.parquet` | Blood cultures (sepsis criteria) |
| `clif_vitals.parquet` | MAP, HR, SpO2, temperature, weight |
| `clif_labs.parquet` | Creatinine, BUN, lactate, WBC, platelet |
| `clif_respiratory_support.parquet` | IMV (ventilation flag) |
| `clif_crrt_therapy.parquet` | Continuous RRT |
| `clif_patient_assessments.parquet` | GCS (optional; zero-filled if absent) |

### Python environment

```bash
uv sync
```

### Configuration

Copy `config/config.example.py` to `config/config.py` and fill in your site's paths:

```bash
cp config/config.example.py config/config.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIF_DIR` | *(set per site)* | Root directory of CLIF parquet files |
| `OUTPUT_ROOT` | *(set per site)* | Root for outputs |
| `SITE_NAME` | `"UCMC"` | Site identifier used in output filenames |
| `TRAJECTORY_HOURS` | 120 | Maximum trajectory length (hours) |
| `MIN_NE_RECORDS` | 2 | Minimum NE administration records required |
| `LACTATE_THRESHOLD` | 2.0 | Lactate cutoff (mmol/L) — Sepsis-3 requires strictly greater than; Rhee requires greater than or equal to |

---

## `01_clif_extract.py` — cohort identification

### Design anchor

t=0 = first norepinephrine administration. All sepsis criteria are checked within a ±24h window around this anchor.

### Cohort 1: Sepsis-3 (CMS)

| Criterion | Source | Window |
|-----------|--------|--------|
| CMS qualifying IV antibiotic | `medication_admin_intermittent` (`med_group == "CMS_sepsis_qualifying_antibiotics"`) | ±24h of NE start |
| Blood buffy culture | `microbiology_culture` (`fluid_category == "blood_buffy"`, `method_category == "culture"`) | ±24h of NE start |
| Abx + culture within 24h of each other | — | — |
| Lactate > 2 mmol/L | `clif_labs` (`lab_category == "lactate"`) | ±24h of NE start |

### Cohort 2: Rhee / CDC Adult Sepsis Event

| Criterion | Source | Window / Logic |
|-----------|--------|---------------|
| Blood culture | `microbiology_culture` | ±24h of NE start |
| First qualifying IV abx within 2 calendar days of culture | `medication_admin_intermittent` | — |
| ≥4 consecutive qualifying antibiotic calendar days (≤1-day gap allowed) OR course ends ≤1 day before discharge/death | `medication_admin_intermittent` | From first abx date |
| Lactate ≥ 2 mmol/L | `clif_labs` (`lab_category == "lactate"`) | ±24h of NE start |

### Shared exclusion

Patients on vasopressin in the 24h before trajectory start are excluded from both cohorts.

### Phase A outputs: `cohort_*.parquet`

| Column | Description |
|--------|-------------|
| `stay_id` | Hospitalization ID |
| `hospital_death` | 1 if died in hospital |
| `anchor_year_group` | Year of NE start |
| `first_norepi_time` | Timestamp of first NE administration (= t=0) |
| `trajectory_start` | Same as `first_norepi_time` |
| `traj_hours` | Trajectory length (integer hours, capped at 120) |
| `death_hour` | Hour of death within trajectory (NaN if survived) |
| `age` | Age at admission |
| `gender` | M / F |
| `race` | Race category (normalized) |
| `weight` | Median weight (kg) during trajectory |
| `sepsis_onset_sofa` | SOFA score in 24h window from NE start |
| `initial_lactate` | First lactate value within ±24h of NE start |
| `vaso_before_traj` | 1 if vasopressin in 24h before trajectory start (all 0 — excluded) |
| `location_category` | ADT location category at t=0 |
| `location_type` | ADT location type at t=0 |
| `hospital_id` | ADT hospital identifier at t=0 (if present in source data) |
| `hospital_type` | ADT hospital type (academic/community) at t=0 (if present in source data) |
| `icu_los_days` | ICU LOS (days) — full ICU stay, independent of the trajectory window/cap |
| `hospital_los_days` | Hospital LOS (days) — full hospitalization, independent of the trajectory window/cap |
| `traj_end_reason` | Which of death / ICU discharge / 120h cap bound the trajectory (tie-break: death > discharge > hour_cap) |

Filter counts CSVs are written separately per cohort: `cohort_filter_counts_sepsis3.csv` and `cohort_filter_counts_rhee.csv`.

### Phase B outputs: `features.parquet`

One row per patient-hour for the **union** of both cohorts. Downstream scripts join with `cohort_sepsis3.parquet` or `cohort_rhee.parquet` to restrict to one cohort.

| Column | Source | Aggregation |
|--------|--------|-------------|
| `norepinephrine` | Continuous meds (mcg/kg/min) | Mean dose over hour; 0 if no record |
| `vaso_dose` | Continuous meds (u/min) | Mean dose over hour; 0 if no record |
| `nee` | NE + Epi + Phe/10 + Dopa/100 + AngII×10 + Vaso×2.5 | Sum of time-overlapping contributions |
| `action_vaso` | Binary: `vaso_dose > 0` | — |
| `mbp` | Vitals (`map`) | Mean over hour |
| `heart_rate` | Vitals | Last value in hour |
| `spo2` | Vitals | Last value in hour |
| `temperature` | Vitals (`temp_c`) | Last value in hour |
| `ventil` | Respiratory support (IMV) | 1 if any record in hour |
| `rrt` | CRRT + HD | 1 if any record in hour |
| `steroid` | Intermittent meds (hydrocortisone, etc.) | 1 if given this or prior epoch |
| `fluids` | Continuous meds (`fluids_electrolytes`) | Rate × overlap hours (mL) |
| `bun`, `creatinine`, `lactate`, `wbc`, `platelet` | Labs | Last observed up to end of hour |
| `gcs` | Patient assessments | Last observed up to end of hour; zero-filled if table absent |
| `sofa` | clifpy (per patient) | Single value broadcast to all hours |
| `epinephrine`, `phenylephrine`, `dopamine`, `angiotensin ii` | Continuous meds | Mean dose per hour |
| `norepi_explicitly_stopped` | MAR stop action or dose=0 | 1 if any cessation event |
| `vaso_explicitly_stopped` | MAR stop action or dose=0 | 1 if any cessation event |
| `ne_mar_action` / `vaso_mar_action` | `mar_action_group` / `mar_action_name` | Last action string in hour; NaN-filled if column absent |
| `urine_output` | Zero-filled if absent | Summed per hour |
| `death` | From cohort `death_hour` | 1 at the hour of death; 0 otherwise |

---

## `04_epi_analysis.py` analyses

| ID | Name | Description |
|----|------|-------------|
| 0 | Time-to-vaso by pre-vaso NEE bin | Cumulative incidence of vasopressin initiation, stratified by max NEE before vaso start |
| 1 | KM survival by pre-vaso NEE bin | Kaplan–Meier survival curves, 7 NEE-dose strata |
| 1.5 | KM ever vs never vaso | Kaplan–Meier survival — ever-vaso vs never-vaso |
| 2B | NEE vs vaso proportion | Proportion of patient-hours on vasopressin by NEE bin |
| 3 | Feature distributions by NEE × vaso | Boxplots of clinical features per NEE bin × ever/never vaso |
| 4a | Time-to-vaso histogram | Distribution of first vasopressin hour |
| 4d | Waiting time histograms | Hours above NE/lactate/MAP thresholds before vasopressin |
| 5A–5E | Patient profile by NEE | Features at initiation by NEE quartile; rate of change; time-of-day |

ICC/hazard/effects models → **`05_site_variation_analysis.py`**

## `05_site_variation_analysis.py` analyses

| Analysis | Model | Output |
|----------|-------|--------|
| GEE logistic (time-varying) | `vaso_on ~ SOFA + age + rcs(NEE, 4) + rcs(time_hour, 4)`, clustering by patient | Coefficients + OR in JSON packet |
| Discrete-time hazard | Logit GLMM: `h(t) = alpha_t + beta_SOFA + beta_NEE + u_icu`; clog-log GLM | ICC, MOR, baseline hazard plot |
| MELR moments | 3rd-order moment statistics for federated MELR pooling | JSON packet |
