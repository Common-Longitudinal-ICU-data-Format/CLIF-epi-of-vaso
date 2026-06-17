# code/

All analysis scripts for the vasopressin epidemiology project.

## Script order

| Script | Purpose | Run at |
|--------|---------|--------|
| `01_clif_extract.py` | Extract septic shock cohort and hourly features from CLIF 2.1.0 parquet tables | Each CLIF site |
| `02_site_summary.py` | Compute federated-safe aggregate statistics; write shareable CSVs to `upload_to_box_<SITE>/` | Each site |
| `03_site_threshold_sweep.py` | Per-feature threshold sweep vs clinician vasopressin action | Each site |
| `04_site_threshold_outcome.py` | Discrete-time survival analysis of threshold-based rules (optional) | Each site |
| `05_epi_analysis.py` | Epidemiological characterization; write figures + CSVs to `upload_to_box_<SITE>/epi_analysis/` | Each site |

The site is read from `SITE_NAME` in `config/config.py`. Pass `--site <NAME>` to `05_epi_analysis.py` to override (e.g. for MIMIC-IV).
The site is read from `SITE_NAME` in `config/config.py`. Pass `--site <NAME>` to `05_epi_analysis.py` to override (e.g. for MIMIC-IV).

## Usage

```bash
# Extraction — writes PHI intermediate to output/patient_level_data_<SITE>/
uv run python code/01_clif_extract.py
uv run python code/01_clif_extract.py

# Federated summary — writes shareable CSVs to output/upload_to_box_<SITE>/
uv run python code/02_site_summary.py
uv run python code/02_site_summary.py

# Threshold sweep — writes to output/upload_to_box_<SITE>/threshold/
uv run python code/03_site_threshold_sweep.py
uv run python code/03_site_threshold_sweep.py

# Threshold outcome (optional) — writes to output/upload_to_box_<SITE>/threshold/
uv run python code/04_site_threshold_outcome.py
uv run python code/04_site_threshold_outcome.py

# Epidemiological analysis — writes figures + 12 aggregate CSVs to upload_to_box_<SITE>/epi_analysis/
uv run python code/05_epi_analysis.py
uv run python code/05_epi_analysis.py --site MIMIC   # override site name
```

## Outputs

| Script | Output location | Notes |
|--------|----------------|-------|
| `01_clif_extract.py` | `output/patient_level_data_<SITE>/` | PHI — never shared |
| `02_site_summary.py` | `output/upload_to_box_<SITE>/` (root CSVs) | Share |
| `03_site_threshold_sweep.py` | `output/upload_to_box_<SITE>/threshold/` | Share |
| `04_site_threshold_outcome.py` | `output/upload_to_box_<SITE>/threshold/` | Share |
| `05_epi_analysis.py` | `output/epi_analysis_<SITE>/` (PHI copy) + `output/upload_to_box_<SITE>/epi_analysis/` (CSVs + plots) | Share the `epi_analysis/` folder inside `upload_to_box_<SITE>/` |

## 05_epi_analysis.py analyses
## 05_epi_analysis.py analyses

| ID | Name | Description |
|----|------|-------------|
| 0 | Time-to-vaso by pre-vaso NEE bin | Cumulative incidence of vasopressin initiation, stratified by max NEE before vaso start |
| 1 | KM survival by pre-vaso NEE bin | Kaplan–Meier survival curves, 7 NEE-dose strata (pre-vaso max NEE); event = death within 120h trajectory (`death_in_window`) |
| 1.5 | KM ever vs never vaso | Kaplan–Meier survival — ever-vasopressin vs never-vasopressin; event = death within 120h trajectory |
| 2B | NEE vs vaso proportion | Proportion of patient-hours on vasopressin by NEE bin (0.1 μg/kg/min bins, Wilson 95% CI) |
| 2B_stratified | NEE vs vaso by care unit | Same proportion, stratified by location_type/location_category if available |
| 2C | Stacked bar — vaso state by NEE | Patient-hours: never on vaso (grey) / on vaso (blue) / came off (red) |
| 2D | NEE component drugs pre-vaso | Vasopressor drug mix in 24h before vasopressin initiation |
| 3 | Feature distributions by NEE × vaso | Boxplots of 7 clinical features per NEE bin × ever/never vasopressin |
| 3_TOD | Time-of-day at initiation | 2-D density heatmaps (continuous) and proportion bars (binary) by clock hour of vaso start |
| 4a | Time-to-vaso histogram | Distribution of first vasopressin hour (2 h bins) |
| 4d | Waiting time histograms | Hours above NE > 0.25 / lactate > 2 / MAP < 65 thresholds before vasopressin |
| 5A | Patient profile by NEE quartile (violin) | 11 features at initiation, Kruskal-Wallis + Q1 vs Q4 Mann-Whitney (Bonferroni) |
| 5B | Patient profile by NEE bin (ribbon) | Median + IQR ribbon across 7 NEE-at-initiation bins |
| 5D | Rate of change pre-vaso | Linear slope of SOFA/lactate/MAP in final 6h before vasopressin (by NEE quartile) |
| 5E | TOD vs SOFA at initiation | Time of day of vasopressin start vs SOFA at initiation (LOWESS) |

### NEE bins

`[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, ∞)` → labels `<0.1`, `0.1–0.2`, `0.2–0.3`, `0.3–0.5`, `0.5–0.7`, `0.7–1.0`, `>1.0` (μg/kg/min)

### Aggregate CSV files written to `upload_to_box_{SITE}/epi_analysis/`

| File | Columns |
|------|---------|
| `km_cif_by_nee_bin.csv` | `nee_bin, time_hour, cif, ci_lo, ci_hi, n_at_risk, n_total, n_events` |
| `km_survival_by_nee_bin.csv` | `nee_bin, time_hour, km_survival, ci_lo, ci_hi, n_at_risk, n_total` |
| `km_survival_ever_never_vaso.csv` | `group, time_hour, km_survival, ci_lo, ci_hi, n_at_risk, n_total` |
| `nee_proportion_on_vaso.csv` | `nee_bin_mid, n_obs, prop, ci_lo, ci_hi` |
| `nee_vaso_state_hours.csv` | `nee_bin_mid, n_total, prop_never_on, prop_on_vaso, prop_came_off` |
| `feature_dist_nee_vaso.csv` | `feature, nee_bin, vaso_status, n, p5, q1, median, q3, p95` |
| `tod_init_features_binned.csv` | `feature, clock_hour, n, q1, median, q3, mean` |
| `tod_init_features_lowess.csv` | `feature, clock_hour, lowess_y` |
| `time_to_vaso_hist.csv` | `bin_left_hour, bin_right_hour, count, median_hours, p75_hours` |
| `wait_time_histograms.csv` | `metric, hours, count, median_hours` |
| `init_features_by_quartile.csv` | `feature, quartile_label, nee_lo, nee_hi, n, p5, q1, median, q3, p95, kw_p, mw_q1q4_p_bonf` |
| `init_features_by_nee_bin.csv` | `feature, nee_bin, n, q1, median, q3` |
