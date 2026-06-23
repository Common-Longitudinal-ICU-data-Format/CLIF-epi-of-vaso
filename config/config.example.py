# config.example.py
#
# Copy this file to config/config.py and fill in your site's paths.
# The analysis scripts automatically load config/config.py at startup.
#
# Only CLIF_DIR and OUTPUT_ROOT need to change per site.
# All other values reflect the OVISS inclusion criteria and
# should remain at their defaults unless your protocol differs.

from pathlib import Path

# --- Required: set these for your site ---

# Short identifier for your site — used in the per-site output folder names
# (e.g. output/patient_level_data_UCMC/, output/upload_to_box_UCMC/)
SITE_NAME = "UCMC"

# Root directory containing all CLIF 2.1.0 parquet files
CLIF_DIR = Path("/path/to/your/clif/2.1.0")
# e.g. Windows: Path(r"C:\data\clif\2.1.0")

# Root directory for all generated outputs. The scripts create two subfolders
# under <OUTPUT_ROOT>/output/ automatically:
#   patient_level_data_<SITE_NAME>/  — PHI intermediate (cohort/features); NEVER shared
#   upload_to_box_<SITE_NAME>/       — aggregate CSVs + plots; share these for federation
OUTPUT_ROOT = Path("/path/to/your/project")
# e.g. Windows: Path(r"C:\projects\epi-of-vaso")

# --- Optional: adjust only if your protocol differs ---

TIMEZONE = "UTC"          # Timezone for all datetime parsing
TRAJECTORY_HOURS = 120    # Max hours of trajectory per patient
NE_WINDOW_HOURS = 24      # NE must start within this many hours of ICU admit
MIN_NE_RECORDS = 2        # Min NE administration records required
SOFA_THRESHOLD = 2.0      # Min SOFA score at sepsis onset
LACTATE_THRESHOLD = 2.0   # Min lactate (mmol/L) within 24h of infection
MAP_THRESHOLD = 65.0      # MAP threshold (mmHg) for vasopressor indication

# --- Federated ICC anchor parameters ---
#
# After UCMC runs 05_epi_analysis.py, the coordinating site will provide a
# FEDERATED_ICC_ANCHOR dict to distribute to all other sites. Paste it here.
#
# Leave as None if your site IS the anchor (UCMC). The script will fit the
# anchor model locally and print the dict for you to share with other sites.
# Non-anchor sites: uncomment and use the values below (fitted at UCMC).
# Leave as None if your site IS the anchor (UCMC).
FEDERATED_ICC_ANCHOR = None
# FEDERATED_ICC_ANCHOR = {
#     "theta0_m0": [-0.21931012169108077],
#     "theta0_m1": [0.3928185635521517, 0.3890583712166622, 0.15303058399567868,
#                   0.08127689712282007, 5.540782586985465, -0.09205483666618856,
#                   0.10110298382000114],
#     "mu":        [11.299216454456415, 5.593530852105779, 60.53036238981391,
#                   1.1962261255207427, 61.61624477295528, 0.7957884427032321],
#     "sd":        [3.4862384238400184, 4.391257053796251, 16.788664753855542,
#                   4.4268377422816085, 30.29313903197279, 0.4031242949304802],
# }
# Covariate order: sepsis_onset_sofa, initial_lactate, age, peak_nee_12h, map_t0, ventil_ever

# Medication categories — must match your site's CLIF med_category values exactly
STEROID_CATEGORIES = [
    "hydrocortisone", "dexamethasone", "methylprednisolone",
    "fludrocortisone", "prednisolone", "prednisone",
]
VASOPRESSOR_CATEGORIES = [
    "norepinephrine", "epinephrine", "phenylephrine",
    "vasopressin", "dopamine", "angiotensin",
]
