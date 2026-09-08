# code/

Per-site analysis scripts for the vasopressin epidemiology project.

## Cohort structure

Three cohorts, all anchored at t=0 = first norepinephrine (NE) administration. Each definition identifies an **infection anchor** from antibiotics and blood cultures, then requires **t=0 within ±`NE_WINDOW_HOURS` (24 h) of that anchor** — two-sided and inclusive, so NE may start before or after the infection event.

| Cohort | Criteria |
|--------|----------|
| **sepsis3** | CMS qualifying IV abx + blood culture within 24 h of each other (anchor = earlier of the pair); NE start within ±24 h of the anchor; lactate > 2 mmol/L within ±24 h of the anchor |
| **rhee** | Blood culture within ±24 h of NE start; first qualifying IV abx within 2 calendar days of that culture; ≥4 consecutive qualifying antibiotic calendar days (≤1-day gap) or course ends ≤1 day before discharge/death; lactate ≥ 2 mmol/L within ±24 h of NE start |
| **rhee_clifpy** | clifpy `compute_ase(include_lactate=True, apply_rit=True)`; ASE episode's blood culture within ±24 h of NE start; lactate ≥ 2 is one of six optional organ-dysfunction criteria (not a hard filter); 14-day RIT de-duplication applied |

When a patient has several qualifying infection anchors, candidates are restricted to those inside the ±24 h window and the **earliest remaining** one becomes `infection_dttm`. A patient therefore qualifies on *any* in-window anchor, not only the earliest overall — an early out-of-window infection does not disqualify a patient whose later infection sits next to NE start.

`01_clif_extract.py` produces `cohort_sepsis3.parquet`, `cohort_rhee.parquet`, `cohort_rhee_clifpy.parquet`, three `cohort_filter_counts_<cohort>.csv` files, `ne_zero_at_t0_diagnostic.csv`, and a shared `features.parquet` (hourly features for the union of all three cohorts). Downstream scripts accept `--cohort sepsis3|rhee|rhee_clifpy|both`; `02_site_summary.py`, `04_epi_analysis.py`, `05_site_variation_analysis.py`, and `06_eligible_untreated_analysis.py` default to `both`, running each cohort in turn.

---

## Scripts

### Per-site scripts (run at each participating site)

| Script | Purpose |
|--------|---------|
| `01_clif_extract.py` | Extract dual-cohort septic shock cohort + hourly features from CLIF 2.1.0 parquet tables |
| `02_site_summary.py` | Compute federated-safe aggregate statistics; write shareable CSVs to `upload_to_box_<SITE>/<cohort>/` |
| `03_cohort_comparison_summary.py` | Federated-safe three-cohort comparison stats (Sepsis-3, Rhee, Rhee-clifpy); write `upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` |
| `04_epi_analysis.py` | Epidemiological characterization; write figures + CSVs to `upload_to_box_<SITE>/<cohort>/epi_analysis/` |
| `05_site_variation_analysis.py` | Site-specific hospital/ICU variation analysis: fixed-effects intercept comparison, DL mixed-effects pooling (ICC/MOR), patient/ward/hospital/site variance decomposition; write `site_variation_packet_<cohort>_<SITE>.json` |
| `06_eligible_untreated_analysis.py` | "Eligible but untreated" analysis: among patients sustaining high NEE (±MAP < 65) without vasopressin, classify mechanism (comfort care / too brief / MAP recovered / unexpectedly untreated) and characterize the unexpectedly untreated group; write figures + summary CSV to `upload_to_box_<SITE>/<cohort>/eligible_untreated/`. Requires `clif_code_status.parquet` (beta — degrades gracefully if absent). |

Use `run_pipeline.py` to run scripts 01–06 in sequence at a site.

### Coordinating-site scripts (run once, after collecting all `upload_to_box_<SITE>/` folders)

| Script | Purpose |
|--------|---------|
| `07_multisite_epi_plots.py` | Multi-site epi comparison figures (KM curves, etc.) |
| `08_cross_site_variation_analysis.py` | Pooled GEE logistic regression + DL meta-analysis across sites |
| `09_cross_site_vasopressin_analysis.py` | Cross-site vasopressor combination and timing tables + plots |
| `10_consolidated_report.py` | Single HTML report combining all cross-site results |
| `11_ne_infection_timing.py` | NE vs infection timing density plots from per-site `timing_stats.json` aggregates |

Use `run_coordinating_pipeline.py` to run scripts 07–11 in sequence.

---

## Usage

### Run all per-site scripts at once

```bash
uv run python code/run_pipeline.py
```

### Or run individually

```bash
# Extraction (writes PHI intermediate to output/patient_level_data_<SITE>/)
uv run python code/01_clif_extract.py

# Federated summary (writes shareable CSVs to output/upload_to_box_<SITE>/<cohort>/)
uv run python code/02_site_summary.py                  

# Cohort-comparison summary (sepsis3 vs rhee; needs both cohort files)
uv run python code/03_cohort_comparison_summary.py

# Epidemiological analysis
uv run python code/04_epi_analysis.py           

# Site variation analysis (fixed-effects + DL mixed-effects pooling + variance decomposition)
uv run python code/05_site_variation_analysis.py

# Eligible-but-untreated analysis (needs CLIF_DIR set in config.py for code_status)
uv run python code/06_eligible_untreated_analysis.py
```

---

## Outputs

| Script | Output location | Notes |
|--------|----------------|-------|
| `01_clif_extract.py` | `output/patient_level_data_<SITE>/cohort_sepsis3.parquet`<br>`output/patient_level_data_<SITE>/cohort_rhee.parquet`<br>`output/patient_level_data_<SITE>/cohort_rhee_clifpy.parquet`<br>`output/patient_level_data_<SITE>/cohort_filter_counts_sepsis3.csv`<br>`output/patient_level_data_<SITE>/cohort_filter_counts_rhee.csv`<br>`output/patient_level_data_<SITE>/cohort_filter_counts_rhee_clifpy.csv`<br>`output/patient_level_data_<SITE>/ne_zero_at_t0_diagnostic.csv`<br>`output/patient_level_data_<SITE>/features.parquet` | PHI — never shared |
| `02_site_summary.py` | `output/upload_to_box_<SITE>/<cohort>/` | Share |
| `03_cohort_comparison_summary.py` | `output/upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` | Share |
| `04_epi_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/epi_analysis/` | Share |
| `05_site_variation_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/site_variation_packet_<cohort>_<SITE>.json` | Share |
| `06_eligible_untreated_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/eligible_untreated/` | Share |

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
| `clif_code_status.parquet` | Goals-of-care / DNR status for `06_eligible_untreated_analysis.py` (beta table — optional; comfort_care classification disabled if absent) |

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
| `SITE_NAME` | *(set per site)*  | Site identifier used in output filenames |
| `TRAJECTORY_HOURS` | 120 | Maximum trajectory length (hours) |
| `NE_WINDOW_HOURS` | 24 | NE start (t=0) must fall within ±this many hours of the infection anchor |
| `MIN_NE_RECORDS` | 2 | Minimum NE administration records required |
| `LACTATE_THRESHOLD` | 2.0 | Lactate cutoff (mmol/L) — Sepsis-3 requires strictly greater than; Rhee requires greater than or equal to |

---

## `01_clif_extract.py` — cohort identification

### Design anchor

t=0 = first norepinephrine administration (≥`MIN_NE_RECORDS` NE records). Infection criteria are searched across the whole hospitalization; the resulting infection anchor must then sit within **±`NE_WINDOW_HOURS` of t=0**. The window is two-sided and inclusive (exactly 24 h apart qualifies), and a patient qualifies on *any* in-window anchor — candidates are filtered to the window first, then the earliest remaining becomes `infection_dttm`.

### Cohort 1: Sepsis-3 (CMS)

| Criterion | Source | Window |
|-----------|--------|--------|
| CMS qualifying IV antibiotic | `medication_admin_intermittent` (`med_group == "CMS_sepsis_qualifying_antibiotics"`) | anywhere in hospitalization |
| Blood buffy culture | `microbiology_culture` (`fluid_category == "blood_buffy"`, `method_category == "culture"`) | anywhere in hospitalization |
| Abx + culture within 24h of each other → anchor = earlier of the pair | — | — |
| **NE start within ±24h of the anchor** | `medication_admin_continuous` | ±24h of presumed infection |
| Lactate > 2 mmol/L | `clif_labs` (`lab_category == "lactate"`) | ±24h of presumed infection |

### Cohort 2: Rhee / CDC Adult Sepsis Event

| Criterion | Source | Window / Logic |
|-----------|--------|---------------|
| Blood culture | `microbiology_culture` | **±24h of NE start**; earliest in-window culture is the anchor |
| First qualifying IV abx within 2 calendar days of culture | `medication_admin_intermittent` | — |
| ≥4 consecutive qualifying antibiotic calendar days (≤1-day gap allowed) OR course ends ≤1 day before discharge/death | `medication_admin_intermittent` | From first abx date |
| Lactate ≥ 2 mmol/L | `clif_labs` (`lab_category == "lactate"`) | ±24h of NE start |

### Cohort 3: Rhee / CDC ASE via clifpy

`compute_ase(include_lactate=True, apply_rit=True)`, `sepsis == 1`. Episodes are restricted to those whose `blood_culture_dttm` is within ±24h of NE start; the earliest remaining episode is kept. Lactate ≥ 2 is one of six organ-dysfunction criteria (vasopressor, IMV, AKI, thrombocytopenia, hyperbilirubinemia, lactate), each reported as a non-exclusive `NOTE:` row in the filter counts.

### Shared exclusion

Patients whose ADT `location_category` at t=0 is an **OR / procedure room / procedural / PACU** are excluded from all three cohorts. This is the only hard exclusion beyond the cohort criteria.

### Retained-and-flagged (NOT exclusions)

| Situation | Handling |
|-----------|----------|
| Vasopressin before NE start | Retained. `vaso_before_traj` (24 h pre-window), `vaso_before_ne` (any lead time), and `vaso_to_ne_hours` record it; two `NOTE:` rows appear in the filter counts. `04_epi_analysis.py` reports the frequency and median/IQR lead time. |
| NE = 0 at t=0 | Retained. t=0 *is* the first NE administration, so NE should never be 0 there — every violation is audited and attributed in `ne_zero_at_t0_diagnostic.csv` (see below). |

### `ne_zero_at_t0_diagnostic.csv`

Verifies the invariant `norepinephrine > 0` at `time_hour == 0` and classifies each violation, so the artifact is quantified rather than tolerated. One row per cause with `n_patients`, `n_total_patients`, `pct_of_patients`:

| Cause | Meaning |
|-------|---------|
| `zero_dose_at_t0` | The anchoring MAR record itself carries dose 0 — an EHR "start" marker written before the actual rate. `add_ne_dose` gives zero-dose records a zero-length interval, so nothing overlaps hour 0. |
| `dose_null_at_t0` | Anchoring record has a null `med_dose` and is dropped before dose intervals are built. |
| `unit_conversion_failed` | Real non-zero dose, but clifpy could not convert the unit to mcg/kg/min (usually a missing weight or unrecognised `med_dose_unit`). The cohort anchor uses raw records while the feature dose uses converted ones, so conversion failures show up here. |
| `anchor_record_missing` | No raw NE row at `first_norepi_time`. Should never occur; indicates the anchor and the feature build disagree about the source table. |
| `unexplained` | A convertible, non-zero NE record exists at t=0 but the hourly mean is still 0 — a genuine bug in the interval-overlap logic. **Expected to be 0**; a non-zero count prints a WARNING. |

### Phase A outputs: `cohort_*.parquet`

Schema is fixed by `_COHORT_COLS`; columns absent at a site are dropped on write.

| Column | Description |
|--------|-------------|
| `stay_id` | Hospitalization ID |
| `hospital_death` | 1 if `discharge_category == "Expired"` or `death_dttm` is non-null |
| `anchor_year_group` | Year of NE start |
| `first_norepi_time` | Timestamp of first NE administration (= t=0) |
| `trajectory_start` | Same as `first_norepi_time` |
| `traj_hours` | Trajectory length (integer hours, capped at 120) |
| `death_hour` | Hour of death within trajectory (NaN if survived) |
| `traj_end_reason` | Which of death / ICU discharge / 120h cap bound the trajectory (tie-break: death > discharge > hour_cap) |
| `age` | Age at admission |
| `gender` | M / F |
| `race` | `race_category`, overwritten with "Hispanic" when `ethnicity_category == "Hispanic"` |
| `weight` | **First** recorded `weight_kg` of the hospitalization |
| `sepsis_onset_sofa` | clifpy SOFA over the 24h window starting at NE start |
| `initial_lactate` | First lactate in the cohort's lactate window |
| `cci_score` | Charlson Comorbidity Index (clifpy `calculate_cci`, ICD-10-CM); NaN if `clif_hospital_diagnosis.parquet` absent |
| `infection_dttm` | Infection anchor — presumed-infection time (sepsis3) or blood-culture time (both Rhee variants) |
| `vaso_before_traj` | 1 if vasopressin in the 24h immediately before t=0 (prior-vaso group) |
| `vaso_before_ne` | 1 if the first vasopressin precedes t=0 at **any** lead time (superset of `vaso_before_traj`) |
| `vaso_to_ne_hours` | t=0 − `first_vaso_time`, in hours. Positive = vasopressin started before NE |
| `first_vaso_time` | First vasopressin administration **anywhere in the hospitalization** (NaT if never) |
| `first_vaso_time_pretraj` | First vasopressin inside the 24h pre-t=0 window only (NaT otherwise) — the episode adjacent to NE start |
| `location_category` / `location_type` | ADT location at t=0 |
| `hospital_id` / `hospital_type` | ADT hospital identifier and academic/community type at t=0 (if present in source data) |
| `location_category_end` / `location_type_end` | ADT location at `trajectory_end` (last ADT row entered on or before it) |
| `icu_los_days` | ICU LOS (days) — full ICU stay, independent of the trajectory window/cap |
| `hospital_los_days` | Hospital LOS (days) — full hospitalization, independent of the trajectory window/cap |
| `cvc_during_hosp` | 1 if a qualifying CVC was billed any time during the hospitalization |
| `first_cvc_dttm` | Timestamp of first qualifying CVC procedure (NaT if none) |
| `cvc_before_ne_start` | 1 if `first_cvc_dttm` < `first_norepi_time` |
| `comorbid_cirrhosis` | Present-on-admission ICD-10-CM cirrhosis flag |
| `comorbid_liver_disease` | Broader chronic liver disease flag — Table 1 descriptives only, **not** a model fixed effect (overlaps cirrhosis) |
| `comorbid_liver_nocirrh` | `comorbid_liver_disease AND NOT comorbid_cirrhosis` — the non-collinear version used in models |
| `comorbid_cad` | Chronic ischemic heart disease (I25) |
| `comorbid_aortic_stenosis` | Aortic stenosis (rheumatic, non-rheumatic, congenital) |
| `comorbid_carotid_stenosis` | Carotid artery occlusion/stenosis (I65.2) |

CVC detection covers only **CPT 36556** (non-tunneled CVC) and **ICD-10-PCS 02HV33Z** (SVC infusion device insertion) from `clif_patient_procedures.parquet` — tunneled lines, implanted ports, and PICCs are not captured. Comorbidity flags use `poa_present == 1`; if `poa_present` is entirely null the script falls back to all ICD-10-CM diagnoses and prints a NOTE, so pre-existing and ICU-acquired conditions cannot be distinguished at that site.

Filter counts CSVs are written separately per cohort: `cohort_filter_counts_sepsis3.csv`, `cohort_filter_counts_rhee.csv`, and `cohort_filter_counts_rhee_clifpy.csv`. All three are written after the NE=0 verification step, so counts match the saved parquet files.

### Phase B outputs: `features.parquet`

One row per patient-hour for the **union** of all three cohorts. Downstream scripts join with a `cohort_*.parquet` to restrict to one cohort.

| Column | Source | Aggregation |
|--------|--------|-------------|
| `norepinephrine` | Continuous meds (mcg/kg/min) | Mean dose over hour; 0 if no record |
| `vaso_dose` | Continuous meds (u/min) | Mean dose over hour; 0 if no record |
| `nee` | NE×1 + Epi×1 + Phe×0.1 + Dopa×0.01 + AngII×10 + Vaso×2.5 | **Sum** of time-overlapping contributions |
| `epinephrine`, `phenylephrine`, `dopamine`, `angiotensin ii` | Continuous meds | Mean dose per hour; 0 if no record |
| `action_vaso` | Binary: `vaso_dose > 0` | — |
| `mbp` | Vitals (`map`) | **Mean** over hour |
| `heart_rate`, `spo2`, `temperature` (`temp_c`), `respiratory_rate` | Vitals | Last value in hour |
| `ventil` | Respiratory support (`device_category == "IMV"`) | 1 if any record in hour |
| `device_category`, `fio2_set` | Respiratory support | Last in hour, then forward-filled per patient (persistent settings) |
| `p_f_ratio` | `po2_arterial / fio2_set` | Computed only where po2 ∈ [0,700] and fio2 ∈ [0.21,1.0]; NaN otherwise |
| `rrt` | `clif_crrt_therapy` (CRRT only — intermittent HD is **not** included) | 1 if any record in hour |
| `steroid` | Intermittent meds (hydrocortisone, etc.) | 1 if given this or prior hour |
| `fluids` | Continuous meds (`fluids_electrolytes`, ml/hr or ml/kg/hr) | Rate × overlap hours (mL), summed |
| `bun`, `creatinine`, `lactate`, `wbc`, `platelet`, `hemoglobin`, `bilirubin`, `po2_arterial` | Labs | Last observed **at any point up to end of hour** (carries forward across the trajectory) |
| `gcs` | Patient assessments (`gcs_total`) | Last observed up to end of hour; NaN if table absent |
| `sofa` | clifpy (per patient) | Single value broadcast to all hours — **not** time-varying |
| `norepi_explicitly_stopped` | MAR stop action or dose=0 | 1 if any cessation event in hour |
| `vaso_explicitly_stopped` | MAR stop action or dose=0 | 1 if any cessation event in hour |
| `ne_mar_action` / `vaso_mar_action` | `mar_action_group`, falling back to `mar_action_name` | Last action string in hour; NaN-filled if column absent |
| `urine_output` | **Hardcoded 0.0** — placeholder, never populated | — |
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
| 13 | Vasopressor combinations | `vasopressor_combinations.csv` — any-use, single-agent, first-agent, and pairwise co-use counts per drug |
| 13b | Vasopressor timing | `vaso_timing_summary.csv` — per co-drug n and median [IQR] hours relative to vasopressin start |

### `vaso_timing_summary.csv` — sign convention and the NE rows

Rows are `drug` × `direction` (`before` / `after` / `all`) with `n`, `suppressed`, `median_h`, `q25_h`, `q75_h`, `basis`. The delta is always **first co-drug dose − first vasopressin dose**, so negative = co-drug started before vasopressin. `direction` is relative to vasopressin: `before` holds the negative deltas, `after` the positive ones. Cells with n < 11 are suppressed.

`basis` distinguishes how the delta was measured:

- `hourly_grid` (PHENYL, DOPA, EPI, ANGII) — from `features.time_hour`, which begins at t=0 = NE start.
- `cohort_timestamps` (NE) — from `first_norepi_time` and `first_vaso_time` in the cohort parquet, covering the whole hospitalization.

The NE rows must use cohort timestamps: on the hourly grid the first NE hour is 0 for essentially every patient, so a grid-based NE row is degenerate and **structurally cannot see vasopressin that started before NE**. With cohort timestamps the NE rows read as:

| Row | Meaning |
|-----|---------|
| `drug=NE, direction=before` | NE first, then vasopressin — the usual order. `median_h` is negative. |
| `drug=NE, direction=after` | **Vasopressin given before NE.** `n` = how many patients, `median_h`/`q25_h`/`q75_h` = the lead time in hours. |
| `drug=NE, direction=all` | All vasopressin recipients; also the denominator for the two rows above. |

Denominators are vasopressin recipients. Note that `09_cross_site_vasopressin_analysis.py` computes its percentages against `n_any_use` for VASO from `vasopressor_combinations.csv`, which counts on-grid vasopressin exposure — a patient whose vasopressin started and stopped entirely before t=0 is in the numerator but not that denominator.

ICC/hazard/effects models → **`05_site_variation_analysis.py`**

## `05_site_variation_analysis.py` analyses

| Analysis | Model | Output |
|----------|-------|--------|
| Approach 1 — fixed-effects intercept comparison | Shared logistic spec (`vaso_on ~ age + resp/renal/coag/liver/neuro components + rcs(NEE,4) + rcs(time,4)`), refit independently per site | Site-specific intercepts; P(vasopressor) for a reference patient at each site |
| Approach 2 — time/dose-varying probability | Per-site coefficients, all components but one held at reference values | P(vaso_on) vs. time; P(vaso_on) vs. NEE dose |
| Mixed-effects pooling | DL random-effects meta-analysis of site intercepts; pooled logistic GLM with site dummies; optional GLMM with site random intercept | τ², ICC, MOR; forest plot |
| Approach 3 — ward/hospital variance decomposition | Same fixed-effects + DL pooling technique applied within-site across ICU/ward type and hospital_id | 4-level (patient/ward/hospital/site) variance decomposition; JSON packet |

Note: the pooled GEE logistic across directly-accessible sites (coordinating-site step) lives in `08_cross_site_variation_analysis.py`, not here.

## `06_eligible_untreated_analysis.py` analyses

Fixed eligibility threshold: norepinephrine ≥ **0.15 µg/kg/min** (raw NE dose, not NEE). Patients are split into four groups based on peak NE and vasopressin receipt.

**4-group classification:**

| Group | Criterion |
|-------|-----------|
| `high_ne_ever_vaso` | Peak NE ≥ 0.15 µg/kg/min AND received vasopressin |
| `high_ne_never_vaso` | Peak NE ≥ 0.15 µg/kg/min AND never received vasopressin ← primary interest |
| `low_ne_ever_vaso` | Peak NE < 0.15 µg/kg/min AND received vasopressin |
| `low_ne_never_vaso` | Peak NE < 0.15 µg/kg/min AND never received vasopressin |

**Mechanism classification** (priority order) within `high_ne_never_vaso`:

| Mechanism | Definition |
|-----------|-----------|
| `comfort_care` | Goals-of-care limitation at/before first high-NE hour (from `clif_code_status.parquet`) |
| `too_brief` | Died within 4 h of first high-NE hour |
| `map_recovered` | MAP ≥ 65 sustained ≥ 2 h after first high-NE hour (without vasopressin) |
| `unexpectedly_untreated` | None of the above |

**Outputs per cohort:**

| File | Description |
|------|-------------|
| `{site}_{cohort}_four_group_distribution.png` | Stacked bar: % of patients by 4-group classification |
| `{site}_{cohort}_mechanism_breakdown.png` | Stacked bar: mechanism breakdown within `high_ne_never_vaso` |
| `{site}_{cohort}_map_recovery_wait.png` | Histogram of hours until MAP recovery for `map_recovered` patients |
| `{site}_{cohort}_smd_love.png` | Love plot: SMD between `unexpectedly_untreated` vs `high_ne_ever_vaso` |
| `{site}_{cohort}_eligible_duration.png` | Violin: hours in high-NE state by mechanism |
| `{site}_{cohort}_eligible_untreated_summary.csv` | One row with all N counts and rates |
| `{site}_{cohort}_group_comparison_ne015.csv` | Feature comparison: `high_ne_ever_vaso` vs `high_ne_never_vaso` |
