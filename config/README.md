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
| `FEDERATED_ICC_ANCHOR` | No | `None` | Anchor logistic parameters for federated ICC (see below) |

## Federated ICC setup

`05_epi_analysis.py` decomposes between-site variation in vasopressin initiation into case-mix vs. practice components (ICC/MOR/PCV). This requires all sites to evaluate their local likelihood at a common anchor point fitted at UCMC.

**If your site is UCMC:** leave `FEDERATED_ICC_ANCHOR = None`. The script will fit the anchor model and print the values to share with other sites.

**If your site is not UCMC:** the anchor values are pre-filled in `config.example.py` as a commented block. When you copy `config.example.py` to `config.py`, uncomment the `FEDERATED_ICC_ANCHOR` dict and delete the `None` line:

```python
# Before (default — UCMC only):
FEDERATED_ICC_ANCHOR = None

# After (all other sites):
FEDERATED_ICC_ANCHOR = {
    "theta0_m0": [-0.21931012169108077],
    "theta0_m1": [0.3928185635521517, 0.3890583712166622, 0.15303058399567868,
                  0.08127689712282007, 5.540782586985465, -0.09205483666618856,
                  0.10110298382000114],
    "mu":        [11.299216454456415, 5.593530852105779, 60.53036238981391,
                  1.1962261255207427, 61.61624477295528, 0.7957884427032321],
    "sd":        [3.4862384238400184, 4.391257053796251, 16.788664753855542,
                  4.4268377422816085, 30.29313903197279, 0.4031242949304802],
}
# Covariate order: sepsis_onset_sofa, initial_lactate, age, peak_nee_12h, map_t0, ventil_ever
```

Then run `05_epi_analysis.py` as normal. The script will use UCMC's fitted parameters to standardize your site's covariates and compute score/Hessian on a common scale, enabling the coordinating site to pool results across all sites.
