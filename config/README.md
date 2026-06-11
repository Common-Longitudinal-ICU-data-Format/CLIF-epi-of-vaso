# config/

Site-specific configuration for CLIF extraction.

## Setup

Copy `config.example.py` to `config.py` **in this directory** and fill in your paths:

```bash
cp config/config.example.py config/config.py
```

Then edit `config/config.py`:

```python
# Root directory containing your CLIF 2.1.0 parquet files
CLIF_DIR = Path(r"C:\path\to\your\clif\2.1.0")

# Root for all generated outputs (the scripts create the subfolders below)
OUTPUT_ROOT = Path(r"C:\path\to\your\project")
```

The scripts create two per-site subfolders under `<OUTPUT_ROOT>/output/`:

- `patient_level_data_<SITE_NAME>/` — PHI intermediate (cohort/features parquet); **never shared**
- `upload_to_box_<SITE_NAME>/` — aggregate CSVs + plots; **share this folder** for federation

`config/config.py` is gitignored — it stays local and is never committed.

## Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLIF_DIR` | Yes | — | Root of CLIF 2.1.0 parquet tables |
| `OUTPUT_ROOT` | Yes | — | Root for generated outputs; per-site subfolders are created under `output/` |
| `TIMEZONE` | No | `"UTC"` | Timezone for datetime parsing |
| `TRAJECTORY_HOURS` | No | `120` | Max trajectory length per patient |
| `NE_WINDOW_HOURS` | No | `24` | Hours from ICU admit for NE start |
| `MIN_NE_RECORDS` | No | `2` | Minimum NE administration records |
| `SOFA_THRESHOLD` | No | `2.0` | Minimum SOFA at sepsis onset |
| `LACTATE_THRESHOLD` | No | `2.0` | Minimum lactate (mmol/L) |
| `MAP_THRESHOLD` | No | `65.0` | MAP threshold (mmHg) |
