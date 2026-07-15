# code/

All analysis scripts for the vasopressin epidemiology project.

## Cohort structure

Two cohorts are identified relative to first norepinephrine (NE) administration (t=0, ±24h window):

| Cohort | Criteria |
|--------|----------|
| **sepsis3** | CMS qualifying IV abx + blood culture within 24h of each other, both within ±24h of NE start; lactate > 2 mmol/L within ±24h of NE start |
| **rhee** | Blood culture within ±24h of NE start; first qualifying IV abx within 2 calendar days; ≥4 consecutive qualifying antibiotic days (≤1-day gap) or course ends ≤1 day before discharge/death; lactate ≥ 2 mmol/L within ±24h of NE start |

Each extraction script produces `cohort_sepsis3.parquet`, `cohort_rhee.parquet`, and a shared `features.parquet` (hourly features for the union of both cohorts). Downstream scripts accept `--cohort sepsis3|rhee`.

---

## Related repo: CLIF-OPT-VASO

Feature-threshold decision-rule identification, optimal-threshold testing, and
subphenotype/subgroup clustering (formerly steps `03`, `11`, `14`, `15` in this repo, run
through a 4-phase federated pipeline) have moved to the sibling **CLIF-OPT-VASO** repo. It
carries its own copy of the `00`/`01a`/`01b` extraction scripts plus
`09b_threshold_cross_site_comparison.py` (the kappa/AUROC/feature-ranking half of what used
to be this repo's `07_cross_site_vasopressin_analysis.py`).

## Script order

### Per-site scripts (run at each participating institution)

| Script | Purpose |
|--------|---------|
| `01_clif_extract.py` | Extract dual-cohort septic shock cohort + hourly features from CLIF 2.1.0 parquet tables |
| `01b_mimic_extract.py` | Same as 01 but for MIMIC-IV (uses `prescriptions.csv.gz` + `microbiologyevents.csv.gz`) |
| `02_site_summary.py` | Compute federated-safe aggregate statistics; write shareable CSVs to `upload_to_box_<SITE>/<cohort>/` |
| `02b_cohort_comparison_summary.py` | Federated-safe Sepsis-3 vs Rhee cohort-comparison stats (overlap, baseline/vaso-init tables, boxplot 5-number summaries); write `upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` — feeds `08_consolidated_report.py` |
| `03_epi_analysis.py` | Epidemiological characterization; write figures + CSVs to `upload_to_box_<SITE>/<cohort>/epi_analysis/` |
| `04_site_variation_analysis.py` | Site-specific hospital/ICU variation analysis: GEE logistic, discrete-time hazard (ICC/MOR), MELR moments; write `site_variation_packet_<cohort>_<SITE>.json` |
| `04b_ne_infection_timing_summary.py` | CLIF sites only (UCMC/NU-style raw CLIF layout, not MIMIC): federated-safe NE-vs-suspected-infection timing stats (suppressed histograms, median/IQR, Sepsis-3-at-anchor histograms); write `upload_to_box_<SITE>/timing/timing_stats.json` — feeds `09_ne_infection_timing.py` |

### Coordinating-site scripts (run at UCMC/coordinating institution only)

| Script | Purpose |
|--------|---------|
| `05_multisite_epi_plots.py` | Multi-site epi comparison figures from per-site aggregate CSVs |
| `06_cross_site_variation_analysis.py` | Pooled GEE (UCMC+NU+MIMIC patient-level) + DL meta-analysis of all site packets; sensitivity: sepsis3 vs rhee |
| `07_cross_site_vasopressin_analysis.py` | Cross-site epi comparison tables + forest plots (baseline, initiation features, CONSORT, vasopressor combinations/timing, federated MELR/DTH) |
| `08_consolidated_report.py` | One HTML report, 4 sections (who's in the cohort, who gets vasopressin, when, sources of variation) — supersedes the former `10_make_summary_report.py` and `12_cohort_comparison_report.py` |
| `09_ne_infection_timing.py` | NE-to-infection timing analysis — reads only `upload_to_box_<SITE>/timing/` (from `04b_ne_infection_timing_summary.py`), no PHI |

`08` and `09` used to read `output/patient_level_data_<SITE>/` (and, for `09`, raw CLIF tables via
hardcoded UCMC/NU paths) directly, which only worked when run on a machine with access to every
site's raw data. They're now split into a per-site aggregate writer (`02b`/`04b`, PHI in, aggregate
out) and a coordinating-site reader (`08`/`09`, aggregate in only) — the same pattern `02` /
`04_site_variation_analysis.py` already used. See `run_pipeline.py` and `run_coordinating_pipeline.py`.

---

## Usage

```bash
# ── PER-SITE ────────────────────────────────────────────────────────────────

# Extraction (writes PHI intermediate to output/patient_level_data_<SITE>/)
uv run python code/01_clif_extract.py            # CLIF sites
uv run python code/01b_mimic_extract.py           # MIMIC-IV

# Federated summary (writes shareable CSVs to output/upload_to_box_<SITE>/<cohort>/)
uv run python code/02_site_summary.py --cohort sepsis3
uv run python code/02_site_summary.py --cohort rhee

# Cohort-comparison summary (sepsis3 vs rhee; needs both cohort files, no --cohort flag)
uv run python code/02b_cohort_comparison_summary.py

# Epidemiological analysis
uv run python code/03_epi_analysis.py --cohort sepsis3
uv run python code/03_epi_analysis.py --cohort rhee

# Site variation analysis (ICC/MOR/GEE + MELR packet)
uv run python code/04_site_variation_analysis.py --cohort both

# NE-infection timing summary (CLIF sites only, not MIMIC; no --site override — reads
# config.py's CLIF_DIR directly, same as 01)
uv run python code/04b_ne_infection_timing_summary.py

# ── COORDINATING SITE (after collecting all upload_to_box_<SITE>/ folders) ──

uv run python code/05_multisite_epi_plots.py
uv run python code/06_cross_site_variation_analysis.py --cohort both
uv run python code/07_cross_site_vasopressin_analysis.py
uv run python code/08_consolidated_report.py --cohort both --embed
uv run python code/09_ne_infection_timing.py
```

Or drive all of the above with the two orchestrator scripts:
`run_pipeline.py` (per-site) -> `run_coordinating_pipeline.py` (coordinating site).

See the sibling **CLIF-OPT-VASO** repo for the threshold/rule-optimality pipeline
(its own `run_pipeline.py` / `run_coordinating_pipeline.py` / `run_validation_pipeline.py`
/ `run_coordinating_validation_pipeline.py`, 4 phases).

---

## Outputs

| Script | Output location | Notes |
|--------|----------------|-------|
| `01_clif_extract.py` | `output/patient_level_data_<SITE>/cohort_sepsis3.parquet`<br>`output/patient_level_data_<SITE>/cohort_rhee.parquet`<br>`output/patient_level_data_<SITE>/features.parquet` | PHI — never shared |
| `02_site_summary.py` | `output/upload_to_box_<SITE>/<cohort>/` | Share |
| `02b_cohort_comparison_summary.py` | `output/upload_to_box_<SITE>/cohort_comparison/cohort_comparison_stats.json` | Share |
| `03_epi_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/epi_analysis/` | Share |
| `04_site_variation_analysis.py` | `output/upload_to_box_<SITE>/<cohort>/site_variation_packet_<cohort>_<SITE>.json` | Share |
| `04b_ne_infection_timing_summary.py` | `output/upload_to_box_<SITE>/timing/timing_stats.json` | Share |
| `06_cross_site_variation_analysis.py` | `output/cross_site_results/<cohort>/` | Coordinating site |

---

## `03_epi_analysis.py` analyses

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

ICC/hazard/effects models → **`04_site_variation_analysis.py`**

## `04_site_variation_analysis.py` analyses

| Analysis | Model | Output |
|----------|-------|--------|
| GEE logistic (time-varying) | `vaso_on ~ SOFA + age + rcs(NEE, 4) + rcs(time_hour, 4)`, clustering by patient | Coefficients + OR in JSON packet |
| Discrete-time hazard | Logit GLMM: `h(t) = alpha_t + beta_SOFA + beta_NEE + u_icu`; clog-log GLM | ICC, MOR, baseline hazard plot |
| MELR moments | 3rd-order moment statistics for federated MELR pooling | JSON packet |

## `06_cross_site_variation_analysis.py` analyses

| Analysis | Data | Output |
|----------|------|--------|
| Pooled GEE | Patient-level UCMC + NU + MIMIC | OR for SOFA and NEE with site fixed effects |
| DL meta-analysis | All site `site_variation_packet_*.json` files | Pooled OR, I², τ², ICC/MOR across all sites |
| Sensitivity | Sepsis-3 vs Rhee side-by-side | CSV comparison table |
