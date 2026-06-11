# code/

All analysis scripts for the vasopressin epidemiology project.

## Script order

| Script | Purpose | Run at |
|--------|---------|--------|
| `clif_extract.py` | Extract septic shock cohort and hourly features from CLIF 2.1.0 parquet tables | Each CLIF site |
| `site_summary.py` | Compute federated-safe aggregate statistics; write shareable CSVs | Each site |
| `site_threshold_sweep.py` | Per-feature threshold sweep vs clinician vasopressin action; optional RL comparison | Each site |

The site is read from `SITE_NAME` in `config/config.py` — no `--dataset` flag.

## Usage

```bash
# Extraction — writes PHI intermediate to output/patient_level_data_<SITE>/
uv run python code/clif_extract.py

# Federated summary — writes shareable CSVs to output/upload_to_box_<SITE>/
uv run python code/site_summary.py

# Analysis — run at each site after extraction
uv run python code/site_threshold_sweep.py
```

## Outputs

- `clif_extract.py` → `output/patient_level_data_<SITE>/` — **PHI, never shared**
- `site_summary.py` and `site_threshold_sweep.py` → `output/upload_to_box_<SITE>/` — **shareable aggregates**
