#!/usr/bin/env python3
"""
05_epi_analysis.py

Epidemiological characterization of vasopressin use in septic shock.

Reads (PHI, local only):
  output/patient_level_data_<SITE>/cohort.parquet
  output/patient_level_data_<SITE>/features.parquet

Writes figures to:
  output/upload_to_box_<SITE>/epi_analysis/

Analyses:
  1    KM survival curves by max NEE dose bin (0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 μg/kg/min)
  1.5  KM survival curves: ever-vasopressin vs never-vasopressin
  2    NEE dose vs vasopressin use — raw scatter, hexbin [A], binned proportion [B]
  3    Love plot: who gets vasopressin? SMD for baseline + trajectory features
  3.5  Feature trajectories centered on first vasopressin hour (vs never-vaso anchor)
  4a   Distribution of time-to-vasopressin initiation
  4b   NEE dose at vasopressin initiation (violin)
  4c   Heatmap: patient × hour, colored by NEE, sorted by initiation time
  4d   Waiting time: hours above NEE/lactate/MAP thresholds before vasopressin

Usage:
    uv run python code/05_epi_analysis.py
"""

import sys
import argparse
import warnings
from pathlib import Path

import matplotlib
from scipy import stats as scipy_stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--site", default=None,
                 help="Override SITE_NAME from config (e.g. MIMIC, UCMC)")
_args, _ = _ap.parse_known_args()


def _load_site_config():
    import importlib.util as _ilu
    cfg_path = BASE_DIR / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit("ERROR: config/config.py not found.")
SITE_NAME   = _args.site if _args.site else getattr(_cfg, "SITE_NAME", "UCMC")
OUTPUT_ROOT = Path(getattr(_cfg, "OUTPUT_ROOT", "."))

PATIENT_LEVEL_DIR = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"
OUT_DIR = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / "epi_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_LOWER = SITE_NAME.lower()

# Clip NEE at this value for visualizations (true max is a data artefact ~5001)
NEE_PLOT_CAP = 5.0

# Bins for Analysis 1 — edges and display labels
NEE_BIN_EDGES  = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, np.inf]
NEE_BIN_LABELS = ["<0.1", "0.1–0.2", "0.2–0.3", "0.3–0.5",
                  "0.5–0.7", "0.7–1.0", ">1.0"]

NE_NEE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

# NEE conversion factors: NEE = NE + EPI + PE/10 + DA/100 + VASO×2.5 + ANG×10
NEE_WEIGHTS = {
    "norepinephrine": 1.0,
    "epinephrine":    1.0,
    "phenylephrine":  0.1,
    "dopamine":       0.01,
    "vasopressin":    2.5,
    "angiotensin ii": 10.0,
}

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
           "#59a14f", "#edc948", "#b07aa1"]

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
cohort   = pd.read_parquet(PATIENT_LEVEL_DIR / "cohort.parquet")
features = pd.read_parquet(PATIENT_LEVEL_DIR / "features.parquet")
print(f"  {len(cohort):,} patients, {len(features):,} patient-hours")

# traj_hours: UCMC clif_extract computes it; MIMIC stores trajectory_start/end instead
if "traj_hours" not in cohort.columns:
    if "trajectory_start" in cohort.columns and "trajectory_end" in cohort.columns:
        _ts = pd.to_datetime(cohort["trajectory_start"], utc=True)
        _te = pd.to_datetime(cohort["trajectory_end"],   utc=True)
        cohort["traj_hours"] = (
            (_te - _ts).dt.total_seconds() / 3600
        ).clip(lower=0).astype(int)
    else:
        cohort = cohort.merge(
            features.groupby("stay_id")["time_hour"].max()
                    .rename("traj_hours").reset_index(),
            on="stay_id", how="left",
        )

# ── patient-level summaries ───────────────────────────────────────────────────
pat = features.groupby("stay_id").agg(
    max_nee       = ("nee",         "max"),
    mean_nee      = ("nee",         "mean"),
    mean_mbp      = ("mbp",         "mean"),
    mean_sofa     = ("sofa",        "mean"),
    max_lactate   = ("lactate",     "max"),
    ever_vaso     = ("action_vaso", "max"),
    ventil_ever   = ("ventil",      "max"),
    rrt_ever      = ("rrt",         "max"),
    steroid_ever  = ("steroid",     "max"),
).reset_index()

first_vaso = (
    features[features["action_vaso"] == 1]
    .sort_values("time_hour")
    .groupby("stay_id")
    .first()
    .reset_index()
    [["stay_id", "time_hour", "nee", "norepinephrine", "mbp", "sofa", "lactate",
      "creatinine", "bun", "fluids", "ventil", "rrt", "steroid"]]
    .rename(columns={
        "time_hour":      "first_vaso_hour",
        "nee":            "nee_at_init",
        "norepinephrine": "ne_at_init",
        "mbp":            "mbp_at_init",
        "sofa":           "sofa_at_init",
        "lactate":        "lac_at_init",
        "creatinine":     "creatinine_at_init",
        "bun":            "bun_at_init",
        "fluids":         "fluids_at_init",
        "ventil":         "ventil_at_init",
        "rrt":            "rrt_at_init",
        "steroid":        "steroid_at_init",
    })
)

pat = pat.merge(first_vaso, on="stay_id", how="left")
_cohort_cols = ["stay_id", "hospital_death", "traj_hours",
                "age", "gender", "race", "weight",
                "sepsis_onset_sofa", "initial_lactate", "first_norepi_time"]
if "trajectory_start" in cohort.columns:
    _cohort_cols.append("trajectory_start")
pat = pat.merge(cohort[_cohort_cols], on="stay_id")
pat["gender_female"] = (pat["gender"] == "F").astype(float)

# Pre-vaso max NEE:
#   ever-vaso  → max NEE in hours strictly before first_vaso_hour
#   never-vaso → max NEE over full trajectory (no vaso to exclude)
pre_vaso_nee = features.merge(
    pat[["stay_id", "first_vaso_hour"]], on="stay_id", how="left"
)
pre_vaso_nee = pre_vaso_nee[
    pre_vaso_nee["first_vaso_hour"].isna() |
    (pre_vaso_nee["time_hour"] < pre_vaso_nee["first_vaso_hour"])
]
pre_vaso_max = pre_vaso_nee.groupby("stay_id")["nee"].max().rename("pre_vaso_max_nee")
pat = pat.merge(pre_vaso_max, on="stay_id", how="left")

# death_in_window: patient died within the 120h trajectory (vs. hospital_death which
# includes deaths after the window, incorrectly placed at t=120 in KM)
death_in_window = features.groupby("stay_id")["death"].max().rename("death_in_window")
pat = pat.merge(death_in_window, on="stay_id", how="left")
pat["death_in_window"] = pat["death_in_window"].fillna(0).astype(int)

# Approximate clock hour of vasopressin initiation.
# trajectory_start (exact) used if available; else first_norepi_time is a proxy
# (trajectory_start = max(icu_intime, first_norepi_time, presumed_infection_dttm))
_ref_col = "trajectory_start" if "trajectory_start" in pat.columns else "first_norepi_time"
if _ref_col == "first_norepi_time":
    print("  Note: trajectory_start not in cohort — using first_norepi_time as clock-hour proxy")
_ref_dt  = pd.to_datetime(pat[_ref_col], utc=True, errors="coerce")
_fvh     = pat["first_vaso_hour"].fillna(0)
_vaso_clk = _ref_dt + pd.to_timedelta(_fvh.astype(float), unit="h")
pat["vaso_clock_hour"] = _vaso_clk.dt.hour.where(pat["ever_vaso"] == 1)

# NEE dose bins (Analysis 1 uses full-trajectory max_nee; Analysis 0 uses pre_vaso_max_nee)
pat["nee_dose_group"] = pd.cut(
    pat["max_nee"],
    bins=NEE_BIN_EDGES,
    labels=NEE_BIN_LABELS,
    right=False,
    include_lowest=True,
)
pat["nee_dose_group"] = pd.Categorical(
    pat["nee_dose_group"], categories=NEE_BIN_LABELS, ordered=True
)

pat["pre_vaso_nee_group"] = pd.cut(
    pat["pre_vaso_max_nee"],
    bins=NEE_BIN_EDGES,
    labels=NEE_BIN_LABELS,
    right=False,
    include_lowest=True,
)
pat["pre_vaso_nee_group"] = pd.Categorical(
    pat["pre_vaso_nee_group"], categories=NEE_BIN_LABELS, ordered=True
)

ever_vaso_ids  = set(pat.loc[pat["ever_vaso"] == 1, "stay_id"])
never_vaso_ids = set(pat.loc[pat["ever_vaso"] == 0, "stay_id"])
print(f"  {len(ever_vaso_ids):,} ever-vaso ({len(ever_vaso_ids)/len(cohort):.1%}), "
      f"{len(never_vaso_ids):,} never-vaso")


# ── helpers ───────────────────────────────────────────────────────────────────
from lifelines import KaplanMeierFitter
def smd_continuous(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0


def smd_binary(a, b):
    a, b = a.dropna(), b.dropna()
    p1, p2  = a.mean(), b.mean()
    p_pool  = (a.sum() + b.sum()) / (len(a) + len(b))
    denom   = np.sqrt(p_pool * (1 - p_pool))
    return (p1 - p2) / denom if denom > 0 else 0.0


def mean_ci(df, col, group_col="rel_hour"):
    g = df.groupby(group_col)[col].agg(["mean", "sem"]).reset_index()
    g["ci_lo"] = g["mean"] - 1.96 * g["sem"]
    g["ci_hi"] = g["mean"] + 1.96 * g["sem"]
    return g


# =============================================================================
# Analysis 0: Cumulative incidence of vasopressin initiation by pre-vaso NEE bin
# =============================================================================
print("\nAnalysis 0: Time-to-vasopressin initiation by pre-vaso NEE bin...")

# Event time  = first_vaso_hour for initiators
# Censor time = traj_hours for never-vaso patients
# Binning     = pre_vaso_max_nee (max NEE before vaso started, or full traj for never-vaso)
km0 = pat[["stay_id", "ever_vaso", "first_vaso_hour", "traj_hours",
           "pre_vaso_nee_group"]].copy()
km0["event_time"] = np.where(
    km0["ever_vaso"] == 1,
    km0["first_vaso_hour"],
    km0["traj_hours"],
)
km0["event"] = km0["ever_vaso"].astype(int)

fig, ax = plt.subplots(figsize=(11, 6))

for i, grp in enumerate(NEE_BIN_LABELS):
    sub = km0[km0["pre_vaso_nee_group"] == grp]
    if len(sub) < 5:
        continue
    n_init = sub["event"].sum()
    kmf = KaplanMeierFitter()
    kmf.fit(sub["event_time"], event_observed=sub["event"])

    # Plot 1 - KM_estimate (cumulative incidence)
    sf  = kmf.survival_function_["KM_estimate"]
    ci_lo = kmf.confidence_interval_["KM_estimate_lower_0.95"]
    ci_hi = kmf.confidence_interval_["KM_estimate_upper_0.95"]
    ax.fill_between(sf.index, 1 - ci_hi, 1 - ci_lo, alpha=0.12, color=PALETTE[i])
    ax.step(sf.index, 1 - sf, where="post", color=PALETTE[i], linewidth=2,
            label=f"{grp} μg/kg/min  (n={len(sub)}, {n_init} initiated)")

ax.set_xlabel("Hours from trajectory start", fontsize=12)
ax.set_ylabel("Cumulative proportion started on vasopressin", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title(
    f"{SITE_NAME}: Time to vasopressin initiation by pre-vaso max NEE bin\n"
    f"(event = vaso start; censored at trajectory end; binned by NEE before vaso)",
    fontsize=12,
)
ax.set_xlim(0)
ax.set_ylim(0)
ax.legend(title="Pre-vaso max NEE", bbox_to_anchor=(1.02, 1),
          loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis0_time_to_vaso_by_nee.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis0")

# =============================================================================
# Analysis 1: Kaplan–Meier by max NEE dose bin
# =============================================================================
print("\nAnalysis 1: KM by pre-vaso NEE dose bin...")

fig, ax = plt.subplots(figsize=(11, 6))
for i, grp in enumerate(NEE_BIN_LABELS):
    sub = pat[pat["pre_vaso_nee_group"] == grp]
    if len(sub) < 5:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(sub["traj_hours"], event_observed=sub["death_in_window"],
            label=f"{grp} μg/kg/min (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i], linewidth=2)

ax.set_xlabel("Hours from trajectory start", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title(f"{SITE_NAME}: Kaplan–Meier survival by pre-vaso max NEE dose", fontsize=13)
ax.set_xlim(0)
ax.set_ylim(0, 1.02)
ax.legend(title="Pre-vaso max NEE bin", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis1_km_nee_dose.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis1")

# =============================================================================
# Analysis 1.5: Kaplan–Meier ever-vaso vs never-vaso
# =============================================================================
print("Analysis 1.5: KM ever-vaso vs never-vaso...")

fig, ax = plt.subplots(figsize=(9, 6))
for label, mask, color in [
    ("Ever vasopressin",  pat["ever_vaso"] == 1, PALETTE[2]),
    ("Never vasopressin", pat["ever_vaso"] == 0, PALETTE[0]),
]:
    sub = pat[mask]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["traj_hours"], event_observed=sub["death_in_window"],
            label=f"{label} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.5)

ax.set_xlabel("Hours from trajectory start", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title(f"{SITE_NAME}: Kaplan–Meier — ever vs never vasopressin", fontsize=13)
ax.set_xlim(0)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis1_5_km_evervaso.png", dpi=150)
plt.close(fig)
print("  Saved analysis1_5")

# =============================================================================
# Analysis 2: NEE dose vs vasopressin use
# =============================================================================
print("Analysis 2: NEE dose vs vaso use...")

scat = features[["nee", "action_vaso"]].copy()
scat["nee_clipped"] = scat["nee"].clip(upper=NEE_PLOT_CAP)

# ── B: binned proportion with Wilson 95% CI ───────────────────────────────────
bin_edges = np.arange(0, NEE_PLOT_CAP + 0.05, 0.1)
scat["nee_bin"] = pd.cut(scat["nee_clipped"], bins=bin_edges)
binned = (scat.groupby("nee_bin", observed=True)["action_vaso"]
          .agg(["sum", "count"]).reset_index())
binned = binned[binned["count"] >= 20]
binned["prop"] = binned["sum"] / binned["count"]
z = 1.96
n, p = binned["count"].values, binned["prop"].values
denom = 1 + z**2 / n
centre = p + z**2 / (2 * n)
spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
binned["ci_lo"] = np.maximum((centre - spread) / denom, 0.0)
binned["ci_hi"] = np.minimum((centre + spread) / denom, 1.0)
binned["x"] = binned["nee_bin"].apply(lambda b: (b.left + b.right) / 2)

fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(
    binned["x"], binned["prop"],
    yerr=[np.maximum(binned["prop"] - binned["ci_lo"], 0),
          np.maximum(binned["ci_hi"] - binned["prop"], 0)],
    fmt="o", color=PALETTE[0], ecolor="#aaaaaa", capsize=3, markersize=5,
)
ax.set_xlabel("NEE dose bin midpoint (μg/kg/min, bin width = 0.1)", fontsize=11)
ax.set_ylabel("Proportion of patient-hours on vasopressin", fontsize=11)
ax.set_title(f"{SITE_NAME}: Proportion on vasopressin by NEE dose (≥20 obs per bin, Wilson 95% CI)",
             fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2_B.png", dpi=150)
plt.close(fig)
print("  Saved analysis2_B")

# ── 2A: proportion of patients ever on vasopressin by pre-vaso max NEE bin ───
print("Analysis 2A: proportion of patients ever on vasopressin by NEE bin...")
_pat_2a = (
    pat.groupby("pre_vaso_nee_group", observed=True, sort=True)
    .agg(n_total=("stay_id", "count"), n_vaso=("ever_vaso", "sum"))
    .reset_index()
)
_pat_2a = _pat_2a[_pat_2a["n_total"] >= 5].copy()
_pat_2a["prop"] = _pat_2a["n_vaso"] / _pat_2a["n_total"]
_z2a = 1.96
_n2a, _p2a = _pat_2a["n_total"].values.astype(float), _pat_2a["prop"].values
_den2a = 1 + _z2a**2 / _n2a
_cen2a = _p2a + _z2a**2 / (2 * _n2a)
_spr2a = _z2a * np.sqrt(_p2a * (1 - _p2a) / _n2a + _z2a**2 / (4 * _n2a**2))
_pat_2a["ci_lo"] = np.maximum((_cen2a - _spr2a) / _den2a, 0.0)
_pat_2a["ci_hi"] = np.minimum((_cen2a + _spr2a) / _den2a, 1.0)

_x2a = np.arange(len(_pat_2a))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(_x2a, _pat_2a["prop"], color=PALETTE[0], alpha=0.8, width=0.6)
ax.errorbar(
    _x2a, _pat_2a["prop"],
    yerr=[np.maximum(_pat_2a["prop"] - _pat_2a["ci_lo"], 0),
          np.maximum(_pat_2a["ci_hi"] - _pat_2a["prop"], 0)],
    fmt="none", ecolor="black", capsize=4,
)
for _xi, _r in zip(_x2a, _pat_2a.itertuples()):
    ax.text(_xi, float(_r.prop) + 0.02, f"n={int(_r.n_total):,}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(_x2a)
ax.set_xticklabels(
    [str(g) for g in _pat_2a["pre_vaso_nee_group"]],
    rotation=30, ha="right",
)
ax.set_xlabel("Pre-vaso max NEE bin (μg/kg/min)", fontsize=11)
ax.set_ylabel("Proportion of patients ever on vasopressin", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0)
ax.set_title(
    f"{SITE_NAME}: Proportion of patients ever on vasopressin by pre-vaso max NEE bin\n"
    f"(patient-level; Wilson 95% CI; annotated with n per bin)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2_A.png", dpi=150)
plt.close(fig)
print("  Saved analysis2_A")

# ── 2B_annotated: same as 2B with obs count annotated on each point ──────────
print("Analysis 2B_annotated: annotating n obs on 2B plot...")
fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(
    binned["x"], binned["prop"],
    yerr=[np.maximum(binned["prop"] - binned["ci_lo"], 0),
          np.maximum(binned["ci_hi"] - binned["prop"], 0)],
    fmt="o", color=PALETTE[0], ecolor="#aaaaaa", capsize=3, markersize=5,
)
for _, _row in binned.iterrows():
    ax.annotate(
        f"n={int(_row['count']):,}", (_row["x"], _row["prop"]),
        textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=6, color="#555555",
    )
ax.set_xlabel("NEE dose bin midpoint (μg/kg/min, bin width = 0.1)", fontsize=11)
ax.set_ylabel("Proportion of patient-hours on vasopressin", fontsize=11)
ax.set_title(
    f"{SITE_NAME}: Proportion on vasopressin by NEE dose (n patient-hours per bin)",
    fontsize=12,
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2_B_annotated.png", dpi=150)
plt.close(fig)
print("  Saved analysis2_B_annotated")

# ── 2B_stratified: proportion on vaso by NEE bin, coloured by location ───────
print("Analysis 2B_stratified: proportion on vaso by NEE, stratified by location...")

# Requires location_type / location_category in cohort
_loc_col = None
for _c in ["location_type", "location_category"]:
    if _c in cohort.columns:
        _loc_col = _c
        break

if _loc_col is not None:
    _loc_pat = pat.merge(cohort[["stay_id", _loc_col]], on="stay_id", how="left")
    _loc_pat[_loc_col] = _loc_pat[_loc_col].fillna("unknown")
    _loc_groups = sorted(_loc_pat[_loc_col].unique())
    _loc_colors = PALETTE[:len(_loc_groups)]

    fig, ax = plt.subplots(figsize=(11, 5))
    _z = 1.96
    for _lg, _lc in zip(_loc_groups, _loc_colors):
        _lp = _loc_pat[_loc_pat[_loc_col] == _lg]
        _feat_loc = features[features["stay_id"].isin(_lp["stay_id"])].copy()
        _feat_loc["nee_clipped"] = _feat_loc["nee"].clip(upper=NEE_PLOT_CAP)
        _feat_loc["nee_bin"] = pd.cut(_feat_loc["nee_clipped"], bins=bin_edges)
        _bl = (_feat_loc.groupby("nee_bin", observed=True)["action_vaso"]
               .agg(["sum", "count"]).reset_index())
        _bl = _bl[_bl["count"] >= 10]
        _bl["prop"] = _bl["sum"] / _bl["count"]
        _bl["x"] = _bl["nee_bin"].apply(lambda b: (b.left + b.right) / 2)
        _n_a = _bl["count"].values.astype(float)
        _p_a = _bl["prop"].values
        _den = 1 + _z**2 / _n_a
        _cen = _p_a + _z**2 / (2 * _n_a)
        _spr = _z * np.sqrt(_p_a * (1 - _p_a) / _n_a + _z**2 / (4 * _n_a**2))
        _ci_lo = np.maximum((_cen - _spr) / _den, 0.0)
        _ci_hi = np.minimum((_cen + _spr) / _den, 1.0)
        ax.errorbar(
            _bl["x"], _bl["prop"],
            yerr=[np.maximum(_bl["prop"] - _ci_lo, 0), np.maximum(_ci_hi - _bl["prop"], 0)],
            fmt="o", color=_lc, ecolor=_lc, capsize=3, markersize=5, alpha=0.85,
            label=f"{_lg} (n={len(_lp):,})",
        )

    ax.set_xlabel("NEE dose bin midpoint (μg/kg/min)", fontsize=11)
    ax.set_ylabel("Proportion of patient-hours on vasopressin", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title=_loc_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title(
        f"{SITE_NAME}: Proportion on vasopressin by NEE dose, stratified by {_loc_col}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2B_stratified.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved analysis2B_stratified")
else:
    print("  Skipped analysis2B_stratified (location_type/location_category not in cohort)")

# ── C: Stacked bar — patient-hours by NEE × vasopressin state ─────────────────
print("Analysis 2_C: Stacked bar by NEE bin and vaso state...")

# Hour-level classification (based on this hour's state, not overall patient fate):
#   never_on  — action_vaso == 0 AND never been on vaso yet (cummax == 0) → grey
#   on_vaso   — action_vaso == 1 (currently on vaso this hour)            → blue
#   came_off  — action_vaso == 0 BUT was on vaso at an earlier hour       → red
feat_2c = features[["stay_id", "time_hour", "nee", "action_vaso"]].copy()
feat_2c = feat_2c.sort_values(["stay_id", "time_hour"])
feat_2c["cummax_vaso"] = feat_2c.groupby("stay_id")["action_vaso"].cummax()

feat_2c["vaso_state"] = "never_on"   # grey default
feat_2c.loc[feat_2c["action_vaso"] == 1, "vaso_state"] = "on_vaso"    # blue
feat_2c.loc[
    (feat_2c["action_vaso"] == 0) & (feat_2c["cummax_vaso"] == 1),
    "vaso_state"
] = "came_off"                                                          # red

feat_2c["nee_clipped"] = feat_2c["nee"].clip(upper=NEE_PLOT_CAP)
bin_edges_2c = np.arange(0, NEE_PLOT_CAP + 0.05, 0.1)
feat_2c["nee_bin"] = pd.cut(feat_2c["nee_clipped"], bins=bin_edges_2c)

binned_2c = (
    feat_2c.groupby(["nee_bin", "vaso_state"], observed=True)
    .size().unstack(fill_value=0).reset_index()
)
for _cat in ["never_on", "on_vaso", "came_off"]:
    if _cat not in binned_2c.columns:
        binned_2c[_cat] = 0

binned_2c["total"] = binned_2c[["never_on", "on_vaso", "came_off"]].sum(axis=1)
binned_2c = binned_2c[binned_2c["total"] >= 20]
binned_2c["prop_never_on"] = binned_2c["never_on"]  / binned_2c["total"]
binned_2c["prop_on_vaso"]  = binned_2c["on_vaso"]   / binned_2c["total"]
binned_2c["prop_came_off"] = binned_2c["came_off"]  / binned_2c["total"]
binned_2c["x"] = binned_2c["nee_bin"].apply(lambda b: (b.left + b.right) / 2)

fig, ax = plt.subplots(figsize=(11, 5))
_w = 0.09
ax.bar(binned_2c["x"], binned_2c["prop_never_on"],
       width=_w, color="lightgrey", label="Not on vaso, never been on",
       align="center")
ax.bar(binned_2c["x"], binned_2c["prop_on_vaso"],
       bottom=binned_2c["prop_never_on"],
       width=_w, color=PALETTE[0], label="On vasopressin this hour",
       align="center")
ax.bar(binned_2c["x"], binned_2c["prop_came_off"],
       bottom=binned_2c["prop_never_on"] + binned_2c["prop_on_vaso"],
       width=_w, color=PALETTE[2], label="Was on vaso, came off",
       align="center")

ax.set_xlabel("NEE dose bin midpoint (μg/kg/min, bin width = 0.1)", fontsize=11)
ax.set_ylabel("Proportion of patient-hours", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
ax.set_title(
    f"{SITE_NAME}: Patient-hours by NEE dose and vasopressin state  (≥20 obs per bin)\n"
    f"grey = never been on vaso  |  blue = on vaso this hour  |  red = was on vaso, came off",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2_C.png", dpi=150)
plt.close(fig)
print("  Saved analysis2_C")

# =============================================================================
# Analysis 2_D: NEE component drugs stacked bar — proportion of patients on
#   each drug by hour BEFORE vasopressin initiation (vaso patients only).
#   X-axis = hour relative to vaso start (negative), y-axis = proportion.
# =============================================================================
print("Analysis 2_D: NEE component drugs pre-vaso stacked bar...")

_COMP_COLS = ["norepinephrine", "epinephrine", "phenylephrine", "dopamine", "angiotensin ii"]
_COMP_LABELS = ["Norepinephrine", "Epinephrine", "Phenylephrine", "Dopamine", "Angiotensin II"]
_COMP_COLORS = [PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[3], PALETTE[4]]

# Check which component columns exist in features
_comp_avail = [c for c in _COMP_COLS if c in features.columns]
_comp_labels_avail = [_COMP_LABELS[_COMP_COLS.index(c)] for c in _comp_avail]
_comp_colors_avail = [_COMP_COLORS[_COMP_COLS.index(c)] for c in _comp_avail]

if _comp_avail and len(ever_vaso_ids) > 0:
    _fv = features[features["stay_id"].isin(ever_vaso_ids)].merge(
        pat[["stay_id", "first_vaso_hour"]], on="stay_id"
    ).copy()
    # Hours before vaso initiation only (rel_hour < 0)
    _fv["rel_hour"] = _fv["time_hour"] - _fv["first_vaso_hour"]
    _pre = _fv[(_fv["rel_hour"] >= -24) & (_fv["rel_hour"] < 0)].copy()

    if len(_pre) > 0:
        _WINDOW_PRE = 24
        _prop_rows = []
        for _rh in range(-_WINDOW_PRE, 0):
            _slice = _pre[_pre["rel_hour"] == _rh]
            _n = len(_slice)
            if _n == 0:
                continue
            _row = {"rel_hour": _rh, "n": _n}
            for _c in _comp_avail:
                _row[_c] = (_slice[_c] > 0).mean()
            _prop_rows.append(_row)
        _prop_df = pd.DataFrame(_prop_rows)

        fig, ax = plt.subplots(figsize=(12, 5))
        # Full-height lightgrey background = "no vasopressor" visible above drug stacks
        ax.bar(_prop_df["rel_hour"], np.ones(len(_prop_df)), width=0.85,
               color="lightgrey", label="No vasopressor")
        _bottom = np.zeros(len(_prop_df))
        for _c, _lbl, _clr in zip(_comp_avail, _comp_labels_avail, _comp_colors_avail):
            _vals = _prop_df[_c].values
            ax.bar(_prop_df["rel_hour"], _vals, bottom=_bottom,
                   width=0.85, color=_clr, label=_lbl)
            _bottom = _bottom + _vals

        ax.set_xlabel("Hours before vasopressin initiation (0 = vaso start)", fontsize=11)
        ax.set_ylabel("Proportion of patients", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_xlim(-_WINDOW_PRE - 0.5, -0.5)
        ax.set_ylim(0, 1.0)
        ax.axvline(-1, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Hour before vaso start")
        ax.legend(title="Drug", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_title(
            f"{SITE_NAME}: Vasopressor drug mix in 24h before vasopressin initiation\n"
            f"(ever-vaso patients, n={len(ever_vaso_ids):,}; stacked = proportion on each drug per hour)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2D_nee_components_prevaso.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved analysis2D_nee_components_prevaso")

        # ── 2D_dose: mean NEE-equivalent contribution by rel_hour (same x-axis) ─
        print("Analysis 2D_dose: mean NEE-equivalent contribution vs rel_hour...")
        _dose_time_rows = []
        for _rh in range(-_WINDOW_PRE, 0):
            _s = _pre[_pre["rel_hour"] == _rh]
            if len(_s) == 0:
                continue
            _rd = {"rel_hour": _rh, "n": len(_s)}
            for _c in _comp_avail:
                _rd[f"nee_{_c}"] = _s[_c].mean() * NEE_WEIGHTS.get(_c, 1.0)
            _dose_time_rows.append(_rd)
        _dose_time_df = pd.DataFrame(_dose_time_rows)

        if len(_dose_time_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            _bot_d = np.zeros(len(_dose_time_df))
            for _c, _lbl, _clr in zip(_comp_avail, _comp_labels_avail, _comp_colors_avail):
                _vals_d = _dose_time_df[f"nee_{_c}"].values
                ax.bar(_dose_time_df["rel_hour"], _vals_d, bottom=_bot_d,
                       width=0.85, color=_clr, label=_lbl)
                _bot_d = _bot_d + _vals_d
            ax.set_xlabel("Hours before vasopressin initiation (0 = vaso start)", fontsize=11)
            ax.set_ylabel("Mean NEE-equivalent contribution (μg/kg/min)", fontsize=11)
            ax.set_xlim(-_WINDOW_PRE - 0.5, -0.5)
            ax.set_ylim(bottom=0)
            ax.axvline(-1, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.legend(title="Drug", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            ax.set_title(
                f"{SITE_NAME}: Mean NEE-equivalent contribution per drug in 24h before vasopressin initiation\n"
                f"(ever-vaso patients, n={len(ever_vaso_ids):,}; stacked; weights: NE×1, EPI×1, PE×0.1, DA×0.01)",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2D_dose_prevaso.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved analysis2D_dose_prevaso")
    else:
        print("  Skipped analysis2D (no pre-vaso hours found)")
else:
    print("  Skipped analysis2D (NEE component columns not in features)")

# ── 2D_dose_by_nee / 2D_prop_by_nee: drug mix by NEE dose bin (all hours) ────
print("Analysis 2D by NEE dose bin: dose contribution and proportion per bin...")
if _comp_avail:
    _feat_all_d = features[["nee"] + _comp_avail].copy()
    _feat_all_d["nee_clipped"] = _feat_all_d["nee"].clip(upper=NEE_PLOT_CAP)
    _feat_all_d["nee_bin"] = pd.cut(
        _feat_all_d["nee_clipped"], bins=bin_edges, right=False, include_lowest=True,
    )

    _dose_nee_rows, _prop_nee_rows = [], []
    for _nee_b, _grp in _feat_all_d.groupby("nee_bin", observed=True):
        if len(_grp) < 20 or not hasattr(_nee_b, "left"):
            continue
        _xmid = (_nee_b.left + _nee_b.right) / 2
        _rd = {"nee_bin": _nee_b, "x": _xmid}
        _rp = {"nee_bin": _nee_b, "x": _xmid}
        for _c in _comp_avail:
            _rd[_c] = _grp[_c].mean() * NEE_WEIGHTS.get(_c, 1.0)  # NEE-equivalent
            _rp[_c] = (_grp[_c] > 0).mean()
        _dose_nee_rows.append(_rd)
        _prop_nee_rows.append(_rp)
    _dose_nee_df = pd.DataFrame(_dose_nee_rows)
    _prop_nee_df = pd.DataFrame(_prop_nee_rows)

    if len(_dose_nee_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        _w_nee = 0.09
        _bot_n = np.zeros(len(_dose_nee_df))
        for _c, _lbl, _clr in zip(_comp_avail, _comp_labels_avail, _comp_colors_avail):
            _vals = _dose_nee_df[_c].values
            ax.bar(_dose_nee_df["x"], _vals, bottom=_bot_n,
                   width=_w_nee, color=_clr, label=_lbl, align="center")
            _bot_n = _bot_n + _vals
        ax.set_xlabel("NEE dose bin midpoint (μg/kg/min, bin width = 0.1)", fontsize=11)
        ax.set_ylabel("Mean NEE-equivalent contribution (μg/kg/min)", fontsize=11)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(title="Drug", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_title(
            f"{SITE_NAME}: Mean NEE-equivalent contribution per drug by NEE dose bin\n"
            f"(all patient-hours; stacked; weights: NE×1, EPI×1, PE×0.1, DA×0.01)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2D_dose_by_nee.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved analysis2D_dose_by_nee")

    if len(_prop_nee_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        _bot_p = np.zeros(len(_prop_nee_df))
        for _c, _lbl, _clr in zip(_comp_avail, _comp_labels_avail, _comp_colors_avail):
            _vals = _prop_nee_df[_c].values
            ax.bar(_prop_nee_df["x"], _vals, bottom=_bot_p,
                   width=0.09, color=_clr, label=_lbl, align="center")
            _bot_p = _bot_p + _vals
        ax.set_xlabel("NEE dose bin midpoint (μg/kg/min, bin width = 0.1)", fontsize=11)
        ax.set_ylabel("Proportion of patient-hours", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.legend(title="Drug", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_title(
            f"{SITE_NAME}: Proportion of patient-hours on each vasopressor drug by NEE dose bin\n"
            f"(all patient-hours; stacked proportions)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis2D_prop_by_nee.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved analysis2D_prop_by_nee")
else:
    print("  Skipped analysis2D by NEE dose (no drug component columns)")

# =============================================================================
# Analysis 3: Boxplots — feature distributions by NEE bin × vaso status
#   X-axis order per feature: <0.1 never, <0.1 ever, 0.1-0.2 never, 0.1-0.2 ever …
# =============================================================================
print("Analysis 3: Boxplots by NEE bin × vaso status...")

from matplotlib.patches import Patch

BOX3_FEATURES = [
    ("age",               "Age (years)"),
    ("weight",            "Weight (kg)"),
    ("sepsis_onset_sofa", "SOFA at sepsis onset"),
    ("initial_lactate",   "Initial lactate (mmol/L)"),
    ("mean_mbp",          "Mean MAP (mmHg)"),
    ("mean_sofa",         "Mean SOFA"),
    ("max_lactate",       "Max lactate (mmol/L)"),
]

NCOLS3 = 2
NROWS3 = (len(BOX3_FEATURES) + NCOLS3 - 1) // NCOLS3
fig, axes = plt.subplots(NROWS3, NCOLS3, figsize=(16, 4.5 * NROWS3))
axes = np.array(axes).flatten()

BIN_STEP = 3.2   # distance from the start of one bin-pair to the next

for ax_idx, (col, label) in enumerate(BOX3_FEATURES):
    ax = axes[ax_idx]
    tick_positions, tick_labels = [], []

    for bin_idx, grp in enumerate(NEE_BIN_LABELS):
        sub   = pat[pat["pre_vaso_nee_group"] == grp]
        n_dat = sub[sub["ever_vaso"] == 0][col].dropna().values
        v_dat = sub[sub["ever_vaso"] == 1][col].dropna().values

        x_never = bin_idx * BIN_STEP
        x_ever  = bin_idx * BIN_STEP + 1.1

        def _draw_box(dat, pos, color, ax=ax):
            if len(dat) < 5:
                return
            ax.boxplot(
                dat, positions=[pos], widths=0.8,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.8),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
            )

        _draw_box(n_dat, x_never, PALETTE[0])   # blue = never vaso
        _draw_box(v_dat, x_ever,  PALETTE[2])   # red  = ever vaso

        tick_positions.append(bin_idx * BIN_STEP + 0.55)
        tick_labels.append(grp)

        if bin_idx < len(NEE_BIN_LABELS) - 1:
            ax.axvline(bin_idx * BIN_STEP + 2.1, color="lightgray",
                       linewidth=0.8, linestyle="-")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(
        handles=[Patch(facecolor=PALETTE[0], alpha=0.8, label="Never vasopressin"),
                 Patch(facecolor=PALETTE[2], alpha=0.8, label="Ever vasopressin")],
        fontsize=8, loc="upper right",
    )

for idx in range(len(BOX3_FEATURES), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"{SITE_NAME}: Feature distributions by pre-vaso NEE bin and vasopressin status\n"
    f"Each bin: blue (never-vaso) then red (ever-vaso)  |  median + IQR + whiskers",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis3.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis3")

# =============================================================================
# Analysis 3_TOD: Features at vasopressin initiation by time of day
#   Continuous features: 2-D density heatmap, column-normalised per clock hour
#   Binary features:     proportion bar chart per clock hour
# =============================================================================
print("Analysis 3_TOD: Features at vaso initiation by time of day...")

from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

# (col, display_label, y_bin_size, y_max)
TOD_FEATURES_CONT = [
    ("ne_at_init",         "NE at initiation (μg/kg/min)",      0.10,  3.0),
    ("nee_at_init",        "NEE at initiation (μg/kg/min)",     0.20,  5.0),
    ("mbp_at_init",        "MAP at initiation (mmHg)",           5.0, 160.0),
    ("sofa_at_init",       "SOFA at initiation",                 1.0,  24.0),
    ("lac_at_init",        "Lactate at initiation (mmol/L)",     0.5,  15.0),
    ("creatinine_at_init", "Creatinine at initiation (mg/dL)",   0.5,  10.0),
    ("bun_at_init",        "BUN at initiation (mg/dL)",          5.0, 100.0),
    ("fluids_at_init",     "Fluids at initiation (mL/hr)",     250.0, 3000.0),
]
TOD_FEATURES_BIN = [
    ("ventil_at_init",  "Ventilated at initiation"),
    ("rrt_at_init",     "On RRT at initiation"),
    ("steroid_at_init", "On steroid at initiation"),
]
# Combined list (col, label) used in save_aggregates CSVs
TOD_FEATURES = [(c, l) for c, l, *_ in TOD_FEATURES_CONT] + TOD_FEATURES_BIN

tod_init = pat[pat["ever_vaso"] == 1].copy()
tod_init = tod_init.dropna(subset=["vaso_clock_hour"])
tod_init["vaso_clock_hour"] = tod_init["vaso_clock_hour"].astype(int)

NCOLS_TOD = 3
NROWS_TOD = (len(TOD_FEATURES) + NCOLS_TOD - 1) // NCOLS_TOD   # 4 rows

fig, axes = plt.subplots(NROWS_TOD, NCOLS_TOD,
                         figsize=(5.5 * NCOLS_TOD, 4.5 * NROWS_TOD))
axes = np.array(axes).flatten()

x_edges = np.arange(-0.5, 24.5, 1.0)   # 24 bins centred on hours 0–23

# ── continuous: column-normalised 2-D density heatmap ────────────────────────
for ax_idx, (col, label, bin_size, y_max) in enumerate(TOD_FEATURES_CONT):
    ax = axes[ax_idx]
    sub = tod_init[["vaso_clock_hour", col]].dropna()
    if len(sub) < 10:
        ax.set_title(label + "\n(insufficient data)", fontsize=9)
        continue

    y_edges = np.arange(0, y_max + bin_size, bin_size)
    x = sub["vaso_clock_hour"].values.astype(float)
    y = sub[col].values.astype(float).clip(0, y_max)

    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    col_sums = H.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1
    H_norm = H / col_sums   # each clock-hour column sums to 1

    im = ax.pcolormesh(x_edges, y_edges, H_norm.T, cmap="Blues", shading="flat")
    fig.colorbar(im, ax=ax, label="Density\n(col-norm)", fraction=0.03, pad=0.03)

    # Annotate top of each clock-hour column with n (patients that hour)
    hr_n = H.sum(axis=1).astype(int)   # shape: (24,)
    y_top = y_edges[-1]
    for _hr, _n in enumerate(hr_n):
        if _n > 0:
            ax.text(_hr, y_top, str(_n), ha="center", va="bottom",
                    fontsize=5, color="black", clip_on=False)

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xticklabels(["0\n(midnight)", "6", "12\n(noon)", "18"], fontsize=7)
    ax.set_xlabel("Hour of day at vaso initiation", fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10)

# ── binary: proportion bar chart per clock hour ───────────────────────────────
for rel_idx, (col, label) in enumerate(TOD_FEATURES_BIN):
    ax = axes[len(TOD_FEATURES_CONT) + rel_idx]
    sub = tod_init[["vaso_clock_hour", col]].dropna()
    hours = np.arange(24)
    props = np.array([
        sub[sub["vaso_clock_hour"] == h][col].mean()
        if (sub["vaso_clock_hour"] == h).sum() >= 3 else np.nan
        for h in hours
    ])
    ax.bar(hours, np.nan_to_num(props), width=0.8, color=PALETTE[0], alpha=0.75)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xticklabels(["0\n(midnight)", "6", "12\n(noon)", "18"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("Hour of day at vaso initiation", fontsize=8)
    ax.set_ylabel("Proportion", fontsize=9)
    ax.set_title(label, fontsize=10)

for idx in range(len(TOD_FEATURES), len(axes)):
    axes[idx].set_visible(False)

_ref_note = ("trajectory_start" if "trajectory_start" in pat.columns
             else "first_norepi_time (proxy)")
fig.suptitle(
    f"{SITE_NAME}: Features at vasopressin initiation by time of day\n"
    f"(n={len(tod_init):,} initiators; clock hour from {_ref_note}; "
    f"continuous = column-normalised density  |  binary = proportion)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis3_TOD.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis3_TOD")

# =============================================================================
# Analysis 4a: Distribution of time-to-vasopressin initiation
# =============================================================================
print("Analysis 4a: Time-to-vaso histogram...")

vaso_timing = pat[pat["ever_vaso"] == 1]["first_vaso_hour"].dropna()
med_h = vaso_timing.median()
p75_h = vaso_timing.quantile(0.75)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(vaso_timing, bins=np.arange(0, 122, 2),
        color=PALETTE[0], edgecolor="white", linewidth=0.4)
ax.axvline(med_h, color=PALETTE[2], linewidth=2, linestyle="--",
           label=f"Median: {med_h:.0f} h")
ax.axvline(p75_h, color=PALETTE[3], linewidth=1.5, linestyle=":",
           label=f"75th pct: {p75_h:.0f} h")
ax.set_xlabel("Hour of first vasopressin administration", fontsize=11)
ax.set_ylabel("Number of patients", fontsize=11)
ax.set_title(f"{SITE_NAME}: When is vasopressin started? (n={len(vaso_timing):,} patients)", fontsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis4a_time_to_vaso.png", dpi=150)
plt.close(fig)
print("  Saved analysis4a")

# =============================================================================
# Analysis 4d: Waiting time — hours above thresholds before vasopressin
#   Multiple NE/NEE thresholds; SOFA delta (6h before → at start); SOFA >5 hours
# =============================================================================
print("Analysis 4d: Waiting time before vaso (per-patient loop)...")

vaso_pat_df = pat[pat["ever_vaso"] == 1].set_index("stay_id")
feat_vaso_only = features[features["stay_id"].isin(ever_vaso_ids)].copy()

_HAS_NE_COL = "norepinephrine" in features.columns

wait_rows = []
for stay_id, row in vaso_pat_df.iterrows():
    fvh = row["first_vaso_hour"]
    if pd.isna(fvh):
        continue
    pre = feat_vaso_only[
        (feat_vaso_only["stay_id"] == stay_id) &
        (feat_vaso_only["time_hour"] < fvh)
    ]
    _row = {"stay_id": stay_id, "first_vaso_hour": fvh}

    for _t in NE_NEE_THRESHOLDS:
        _tkey = f"{round(_t * 100):03d}"
        if _HAS_NE_COL:
            _row[f"hrs_ne_gt{_tkey}"]  = int((pre["norepinephrine"] > _t).sum())
        _row[f"hrs_nee_gt{_tkey}"] = int((pre["nee"] > _t).sum())

    _row["hrs_map_lt65"] = int((pre["mbp"]     < 65.0).sum())
    _row["hrs_lac_gt2"]  = int((pre["lactate"] > 2.0).sum())
    _row["hrs_sofa_gt5"] = int((pre["sofa"]    > 5.0).sum())

    # Delta SOFA: sofa at vaso start − sofa at earliest hour in final 6h window
    _sofa_init_vals = feat_vaso_only[
        (feat_vaso_only["stay_id"] == stay_id) &
        (feat_vaso_only["time_hour"] == fvh)
    ]["sofa"].dropna().values
    _sofa_at_init = float(_sofa_init_vals[0]) if len(_sofa_init_vals) > 0 else np.nan

    _pre_6h = pre[pre["time_hour"] >= fvh - 6].sort_values("time_hour")
    _sofa_6h_vals = _pre_6h["sofa"].dropna().values
    _sofa_6h_before = float(_sofa_6h_vals[0]) if len(_sofa_6h_vals) > 0 else np.nan

    _row["delta_sofa_6h"] = (
        _sofa_at_init - _sofa_6h_before
        if not (np.isnan(_sofa_at_init) or np.isnan(_sofa_6h_before))
        else np.nan
    )

    wait_rows.append(_row)

wait_df = pd.DataFrame(wait_rows)

# ── Figure A: NE/NEE thresholds — boxplots across threshold levels ────────────
_nee_threshold_data = [
    wait_df[f"hrs_nee_gt{round(t*100):03d}"].dropna().values
    for t in NE_NEE_THRESHOLDS
]
_n_thresh_panels = 2 if _HAS_NE_COL else 1
fig, axes_th = plt.subplots(1, _n_thresh_panels,
                             figsize=(6 * _n_thresh_panels, 5), sharey=False)
if _n_thresh_panels == 1:
    axes_th = [axes_th]

# NEE panel
_ax_nee = axes_th[-1]
_bp_nee = _ax_nee.boxplot(
    _nee_threshold_data,
    positions=range(len(NE_NEE_THRESHOLDS)),
    patch_artist=True, showfliers=False, widths=0.55,
)
for _patch in _bp_nee["boxes"]:
    _patch.set_facecolor(PALETTE[0])
    _patch.set_alpha(0.7)
_ax_nee.set_xticks(range(len(NE_NEE_THRESHOLDS)))
_ax_nee.set_xticklabels([f">{t}" for t in NE_NEE_THRESHOLDS], rotation=30, ha="right")
_ax_nee.set_xlabel("NEE threshold (μg/kg/min)", fontsize=10)
_ax_nee.set_ylabel("Hours above threshold before vasopressin", fontsize=10)
_ax_nee.set_title("Hours with NEE above threshold\nbefore vasopressin initiation", fontsize=10)

# NE panel (if available)
if _HAS_NE_COL:
    _ne_threshold_data = [
        wait_df[f"hrs_ne_gt{round(t*100):03d}"].dropna().values
        for t in NE_NEE_THRESHOLDS
    ]
    _ax_ne = axes_th[0]
    _bp_ne = _ax_ne.boxplot(
        _ne_threshold_data,
        positions=range(len(NE_NEE_THRESHOLDS)),
        patch_artist=True, showfliers=False, widths=0.55,
    )
    for _patch in _bp_ne["boxes"]:
        _patch.set_facecolor(PALETTE[1])
        _patch.set_alpha(0.7)
    _ax_ne.set_xticks(range(len(NE_NEE_THRESHOLDS)))
    _ax_ne.set_xticklabels([f">{t}" for t in NE_NEE_THRESHOLDS], rotation=30, ha="right")
    _ax_ne.set_xlabel("NE threshold (μg/kg/min)", fontsize=10)
    _ax_ne.set_ylabel("Hours above threshold before vasopressin", fontsize=10)
    _ax_ne.set_title("Hours with NE above threshold\nbefore vasopressin initiation", fontsize=10)

fig.suptitle(
    f"{SITE_NAME}: Hours above NE/NEE threshold before vasopressin\n"
    f"(n={len(wait_df):,} vasopressin patients; median + IQR + whiskers)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis4d_nee_thresholds.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis4d_nee_thresholds")

# ── Figure B: MAP, lactate, SOFA>5, delta SOFA ───────────────────────────────
panels = [
    ("hrs_map_lt65",  "Hours with MAP < 65 mmHg\nbefore vasopressin"),
    ("hrs_lac_gt2",   "Hours with lactate > 2 mmol/L\nbefore vasopressin"),
    ("hrs_sofa_gt5",  "Hours with SOFA > 5\nbefore vasopressin"),
    ("delta_sofa_6h", "ΔSOFA (at start − 6h before)\nat vasopressin initiation"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (col, label) in zip(axes, panels):
    data = wait_df[col].dropna()
    if len(data) == 0:
        ax.set_visible(False)
        continue
    if col == "delta_sofa_6h":
        _lo = max(data.min() - 1, -15)
        _hi = min(data.max() + 1, 15)
        ax.hist(data, bins=np.arange(_lo - 0.5, _hi + 1.5, 1),
                color=PALETTE[3], edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    else:
        max_bin = min(int(data.max()) + 2, 60)
        ax.hist(data, bins=range(0, max_bin + 1),
                color=PALETTE[0], edgecolor="white", linewidth=0.3)
    med_val = data.median()
    ax.axvline(med_val, color=PALETTE[2], linewidth=2, linestyle="--",
               label=f"Median: {med_val:.1f}")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Patients", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle(
    f"{SITE_NAME}: How long did clinicians wait? "
    f"(n={len(wait_df):,} vasopressin patients)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis4d_wait_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis4d")

# =============================================================================
# Analysis 4d_tod: KM survival curves by time-of-day of vasopressin initiation
#   TOD bins: 4-hour chunks (midnight–4am, 4–8am, ..., 8pm–midnight)
#   Population: vasopressin initiators only
# =============================================================================
print("Analysis 4d_tod: KM survival by TOD of vasopressin initiation...")

TOD_BIN_EDGES  = [0, 4, 8, 12, 16, 20, 24]
TOD_BIN_LABELS_KM = [
    "midnight–4am", "4–8am", "8am–noon",
    "noon–4pm", "4–8pm", "8pm–midnight",
]

tod_km = pat[pat["ever_vaso"] == 1].dropna(subset=["vaso_clock_hour"]).copy()
tod_km["tod_bin"] = pd.cut(
    tod_km["vaso_clock_hour"],
    bins=TOD_BIN_EDGES,
    labels=TOD_BIN_LABELS_KM,
    include_lowest=True,
    right=False,
)

fig, ax = plt.subplots(figsize=(11, 6))
_tod_pal = PALETTE[:len(TOD_BIN_LABELS_KM)]
for _lbl, _clr in zip(TOD_BIN_LABELS_KM, _tod_pal):
    _sub = tod_km[tod_km["tod_bin"] == _lbl]
    if len(_sub) < 5:
        continue
    _kmf_tod = KaplanMeierFitter()
    _kmf_tod.fit(_sub["traj_hours"], event_observed=_sub["death_in_window"],
                 label=f"{_lbl} (n={len(_sub)})")
    _kmf_tod.plot_survival_function(ax=ax, ci_show=True, color=_clr, linewidth=2)

ax.set_xlabel("Hours from trajectory start", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title(
    f"{SITE_NAME}: KM survival by time-of-day of vasopressin initiation\n"
    f"(ever-vaso patients, n={len(tod_km):,}; 4-hour bins)",
    fontsize=12,
)
ax.set_xlim(0)
ax.set_ylim(0, 1.02)
ax.legend(title="TOD of vaso start", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis4d_tod_km.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis4d_tod_km")

# =============================================================================
# Analysis 5_A/B/C: Why do some patients start vasopressin at low NEE?
#   Population : vasopressin initiators only (n ≈ 1,150)
#   Grouping   : quartiles of NEE at the moment vasopressin was started (nee_at_init)
#                Q1 = started at lowest NE dose  →  Q4 = waited for highest NE dose
#
#   5_A  Violin plots — one panel per feature, 4 violins (Q1–Q4)
#   5_B  Binned median + IQR ribbon — across 7 NEE bins, one panel per feature
#   5_C  Heatmap — rows = features, cols = Q1–Q4, color = row-z-scored median
#        Annotated with actual medians; compact single-figure summary
# =============================================================================
print("Analysis 5_A/B/C: Patient profile by NEE dose at vasopressin initiation...")

# ── build initiator-level dataset ─────────────────────────────────────────────
initiators = pat[pat["ever_vaso"] == 1].copy()

# GCS at the initiation hour (not stored in pat; pull from features)
gcs_init = (
    features[features["stay_id"].isin(ever_vaso_ids)]
    .merge(pat[["stay_id", "first_vaso_hour"]], on="stay_id")
)
gcs_init = gcs_init[gcs_init["time_hour"] == gcs_init["first_vaso_hour"]]
gcs_init = (
    gcs_init.groupby("stay_id")["gcs"].first()
    .reset_index().rename(columns={"gcs": "gcs_at_init"})
)
initiators = initiators.merge(gcs_init, on="stay_id", how="left")

# Quartile bins of nee_at_init
q25, q50, q75 = initiators["nee_at_init"].quantile([0.25, 0.50, 0.75])
Q_EDGES  = [-np.inf, q25, q50, q75, np.inf]
Q_LABELS = [
    f"Q1  ≤{q25:.2f}",
    f"Q2  {q25:.2f}–{q50:.2f}",
    f"Q3  {q50:.2f}–{q75:.2f}",
    f"Q4  ≥{q75:.2f}",
]
initiators["nee_init_q"] = pd.Categorical(
    pd.cut(initiators["nee_at_init"], bins=Q_EDGES,
           labels=Q_LABELS, right=True, include_lowest=True),
    categories=Q_LABELS, ordered=True,
)

# 7-bin version for Option B
initiators["nee_init_bin"] = pd.cut(
    initiators["nee_at_init"],
    bins=NEE_BIN_EDGES, labels=NEE_BIN_LABELS,
    right=False, include_lowest=True,
)

INIT_FEATURES = [
    ("ne_at_init",          "NE at initiation (μg/kg/min)"),
    ("nee_at_init",         "NEE at initiation (μg/kg/min)"),
    ("mbp_at_init",         "MAP at initiation (mmHg)"),
    ("sofa_at_init",        "SOFA at initiation"),
    ("lac_at_init",         "Lactate at initiation (mmol/L)"),
    ("creatinine_at_init",  "Creatinine at initiation (mg/dL)"),
    ("bun_at_init",         "BUN at initiation (mg/dL)"),
    ("fluids_at_init",      "Fluids at initiation (mL/hr)"),
    ("ventil_at_init",      "Ventilated at initiation"),
    ("rrt_at_init",         "On RRT at initiation"),
    ("steroid_at_init",     "On steroid at initiation"),
]

Q_COLORS = [PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[2]]


def fmt_p(p):
    """Format p-value for annotation."""
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    if p < 0.05:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def bracket(ax, x1, x2, y_top, h_rel, text, fontsize=8.5):
    """Draw a significance bracket spanning x1→x2 at y_top; h_rel is tick height."""
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h_rel, y_top + h_rel, y_top],
            lw=1.1, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y_top + h_rel * 1.3, text,
            ha="center", va="bottom", fontsize=fontsize)


# ─────────────────────────────────────────────────────────────────────────────
# 5_A: Violin plots by NEE-at-initiation quartile + statistical tests
#   Overall: Kruskal-Wallis H (p shown in panel title)
#   Pairwise Q1 vs Q4: Mann-Whitney U (bracket annotation, Bonferroni × 9)
# ─────────────────────────────────────────────────────────────────────────────
N_TESTS = len(INIT_FEATURES)   # Bonferroni denominator
NCOLS5  = 3
NROWS5  = (len(INIT_FEATURES) + NCOLS5 - 1) // NCOLS5
fig, axes = plt.subplots(NROWS5, NCOLS5, figsize=(5.5 * NCOLS5, 5.0 * NROWS5))
axes = np.array(axes).flatten()

rng5 = np.random.default_rng(42)

for ax_idx, (col, label) in enumerate(INIT_FEATURES):
    ax = axes[ax_idx]
    grp_data  = [initiators[initiators["nee_init_q"] == q][col].dropna().values
                 for q in Q_LABELS]
    valid_idx = [i for i, d in enumerate(grp_data) if len(d) >= 5]

    if valid_idx:
        parts = ax.violinplot(
            [grp_data[i] for i in valid_idx],
            positions=[i + 1 for i in valid_idx],
            showmedians=True, showextrema=False, widths=0.7,
        )
        for pc, i in zip(parts["bodies"], valid_idx):
            pc.set_facecolor(Q_COLORS[i])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2.5)

        # ── individual points (jittered) ──────────────────────────────────────
        for i in valid_idx:
            jitter = rng5.uniform(-0.12, 0.12, size=len(grp_data[i]))
            ax.scatter(
                np.full(len(grp_data[i]), i + 1) + jitter,
                grp_data[i],
                s=6, color="black", alpha=0.7, zorder=3, linewidths=0,
            )

    # ── Kruskal-Wallis across all valid groups ────────────────────────────────
    kw_groups = [grp_data[i] for i in valid_idx if len(grp_data[i]) >= 5]
    kw_sig    = False
    kw_label  = label
    if len(kw_groups) >= 2:
        _, kw_p = scipy_stats.kruskal(*kw_groups)
        if kw_p < 0.05:
            kw_sig   = True
            kw_label = f"{label}\nKruskal-Wallis {fmt_p(kw_p)}"

    # ── Mann-Whitney Q1 vs Q4 (Bonferroni-corrected, only if KW significant) ──
    d1, d4 = grp_data[0], grp_data[3]
    all_vals = np.concatenate([grp_data[i] for i in valid_idx])
    y_max    = np.nanpercentile(all_vals, 98)
    y_range  = np.nanpercentile(all_vals, 98) - np.nanpercentile(all_vals, 2)
    h        = y_range * 0.05

    if kw_sig and len(d1) >= 5 and len(d4) >= 5:
        _, mw_p  = scipy_stats.mannwhitneyu(d1, d4, alternative="two-sided")
        mw_p_adj = min(mw_p * N_TESTS, 1.0)
        if mw_p_adj < 0.05:
            bracket(ax, 1, 4, y_max + h * 0.5, h,
                    f"Q1 vs Q4: {fmt_p(mw_p_adj)} (Bonf.)")
            ax.set_ylim(top=y_max + h * 5)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        [f"Q{i+1}\n{q.split('  ')[1]}" for i, q in enumerate(Q_LABELS)],
        fontsize=8,
    )
    ax.set_xlabel("NEE at vaso initiation (μg/kg/min)", fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(kw_label, fontsize=9)

for idx in range(len(INIT_FEATURES), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"{SITE_NAME}  —  5_A: Patient features by NEE dose at vasopressin initiation\n"
    f"Q1 = started at lowest NE  →  Q4 = highest NE  (n={len(initiators):,} initiators)  |  "
    f"brackets = Q1 vs Q4 Mann-Whitney, Bonferroni × {N_TESTS}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis5_A.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis5_A")

# ─────────────────────────────────────────────────────────────────────────────
# 5_B: Binned median + IQR ribbon across 7 NEE bins at initiation
# ─────────────────────────────────────────────────────────────────────────────
valid_bins_5B = [g for g in NEE_BIN_LABELS
                 if (initiators["nee_init_bin"] == g).sum() >= 5]

fig, axes = plt.subplots(NROWS5, NCOLS5, figsize=(5.5 * NCOLS5, 4.0 * NROWS5))
axes = np.array(axes).flatten()

for ax_idx, (col, label) in enumerate(INIT_FEATURES):
    ax = axes[ax_idx]
    xs, meds, q1s, q3s = [], [], [], []

    for b_idx, grp in enumerate(valid_bins_5B):
        sub = initiators[initiators["nee_init_bin"] == grp][col].dropna()
        if len(sub) < 5:
            continue
        xs.append(b_idx)
        meds.append(sub.median())
        q1s.append(sub.quantile(0.25))
        q3s.append(sub.quantile(0.75))

    xs, meds, q1s, q3s = (np.array(a) for a in (xs, meds, q1s, q3s))
    ax.fill_between(xs, q1s, q3s, alpha=0.22, color=PALETTE[0])
    ax.plot(xs, meds, "o-", color=PALETTE[0], linewidth=2.2, markersize=7)

    ax.set_xticks(range(len(valid_bins_5B)))
    ax.set_xticklabels(valid_bins_5B, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("NEE bin at vaso initiation (μg/kg/min)", fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10)

for idx in range(len(INIT_FEATURES), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"{SITE_NAME}  —  5_B: Median (IQR ribbon) by NEE dose at vasopressin initiation\n"
    f"(vasopressin initiators only, n={len(initiators):,}; bins with <5 patients suppressed)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis5_B.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved analysis5_B")

# =============================================================================
# Analysis 5_D: Rate of change in key features vs NEE-at-initiation quartile
#   For each initiator, compute slope (hrs -6 to 0 before vaso start) for
#   SOFA, lactate, MAP. Compare across Q1–Q4 to see if late initiators (Q4)
#   were rapidly deteriorating.
# =============================================================================
print("Analysis 5_D: Rate of change in features before vaso initiation by quartile...")

_RC_FEATURES = [
    ("sofa",    "SOFA score"),
    ("lactate", "Lactate (mmol/L)"),
    ("mbp",     "MAP (mmHg)"),
]
_RC_WINDOW = 6   # hours before vaso initiation

_slope_rows = []
_feat_rc = features[features["stay_id"].isin(ever_vaso_ids)].merge(
    initiators[["stay_id", "first_vaso_hour", "nee_init_q"]], on="stay_id"
)
for _sid, _grp in _feat_rc.groupby("stay_id"):
    _fvh = _grp["first_vaso_hour"].iloc[0]
    _q   = _grp["nee_init_q"].iloc[0]
    _pre = _grp[(_grp["time_hour"] >= _fvh - _RC_WINDOW) &
                (_grp["time_hour"] < _fvh)].sort_values("time_hour")
    _row = {"stay_id": _sid, "nee_init_q": _q}
    for _col, _ in _RC_FEATURES:
        _y = _pre[_col].dropna().values
        if len(_y) >= 2:
            _x = np.arange(len(_y), dtype=float)
            _slope, *_ = np.polyfit(_x, _y, 1)
            _row[f"slope_{_col}"] = _slope
        else:
            _row[f"slope_{_col}"] = np.nan
    _slope_rows.append(_row)

_slope_df = pd.DataFrame(_slope_rows)

if len(_slope_df) > 0:
    _nc5d = len(_RC_FEATURES)
    fig, axes = plt.subplots(1, _nc5d, figsize=(5 * _nc5d, 5))
    if _nc5d == 1:
        axes = [axes]

    for _ax, (_col, _lbl) in zip(axes, _RC_FEATURES):
        _scol = f"slope_{_col}"
        _grp_data = [_slope_df[_slope_df["nee_init_q"] == _q][_scol].dropna().values
                     for _q in Q_LABELS]
        _valid = [i for i, d in enumerate(_grp_data) if len(d) >= 5]
        if _valid:
            _ax.violinplot(
                [_grp_data[i] for i in _valid],
                positions=[i + 1 for i in _valid],
                showmedians=True, showextrema=False, widths=0.7,
            )
        _ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        _ax.set_xticks([1, 2, 3, 4])
        _ax.set_xticklabels(
            [f"Q{i+1}\n{q.split('  ')[1]}" for i, q in enumerate(Q_LABELS)],
            fontsize=8,
        )
        _ax.set_xlabel("NEE at vaso initiation (μg/kg/min)", fontsize=9)
        _ax.set_ylabel(f"Δ{_lbl} per hour\n(slope in {_RC_WINDOW}h pre-vaso)", fontsize=9)
        _ax.set_title(_lbl, fontsize=10)

    fig.suptitle(
        f"{SITE_NAME}  —  5_D: Rate of change in features before vasopressin initiation\n"
        f"(linear slope over final {_RC_WINDOW}h; positive = worsening for SOFA/lactate, "
        f"negative = worsening for MAP; n={len(_slope_df):,} initiators)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis5_D_rate_of_change.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved analysis5_D")

# =============================================================================
# Analysis 5_E: Time of day of vasopressin initiation vs SOFA at initiation
# =============================================================================
print("Analysis 5_E: TOD of vaso initiation vs SOFA at initiation...")

_tod_sofa = pat[pat["ever_vaso"] == 1][["vaso_clock_hour", "sofa_at_init"]].dropna()
if len(_tod_sofa) >= 10:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
    fig, ax = plt.subplots(figsize=(10, 5))
    rng_e = np.random.default_rng(99)
    _jit = rng_e.uniform(-0.3, 0.3, size=len(_tod_sofa))
    ax.scatter(_tod_sofa["vaso_clock_hour"] + _jit, _tod_sofa["sofa_at_init"],
               s=8, alpha=0.35, color=PALETTE[0], rasterized=True, label="Patient")
    # LOWESS smoothed line
    _sm = _sm_lowess(_tod_sofa["sofa_at_init"].values,
                     _tod_sofa["vaso_clock_hour"].values,
                     frac=0.4, return_sorted=True)
    ax.plot(_sm[:, 0], _sm[:, 1], color=PALETTE[2], linewidth=2.5,
            label="LOWESS", zorder=5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["0\n(midnight)", "6", "12\n(noon)", "18", "23"], fontsize=9)
    ax.set_xlabel("Clock hour of vasopressin initiation", fontsize=11)
    ax.set_ylabel("SOFA score at vasopressin initiation", fontsize=11)
    ax.set_title(
        f"{SITE_NAME}  —  5_E: Time of day vs SOFA at vasopressin initiation\n"
        f"(n={len(_tod_sofa):,} initiators; jittered ±0.3 h; LOWESS frac=0.4)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis5_E_tod_vs_sofa.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved analysis5_E")
else:
    print("  Skipped analysis5_E (insufficient data)")

# =============================================================================
# Analysis: Dwell time at NE dose before changing — and MAP correlation
#   For each ever-vaso patient, identify consecutive "plateaus" at the same
#   NE dose.  Measure dwell duration and MAP change over each plateau.
# =============================================================================
print("\nAnalysis dwell_time: how long do clinicians wait at each NE dose?")

_DWELL_DOSE_COL = "norepinephrine" if "norepinephrine" in features.columns else None

if _DWELL_DOSE_COL is not None:
    _fd = (
        features[features["stay_id"].isin(ever_vaso_ids)]
        [["stay_id", "time_hour", _DWELL_DOSE_COL, "mbp"]]
        .sort_values(["stay_id", "time_hour"])
        .reset_index(drop=True)
        .copy()
    )
    _fd["dose_r"] = _fd[_DWELL_DOSE_COL].round(2)
    # Mark start of each new plateau (dose change OR new patient)
    _fd["_changed"] = (
        (_fd["dose_r"] != _fd["dose_r"].shift(1)) |
        (_fd["stay_id"] != _fd["stay_id"].shift(1))
    )
    _fd["plateau_id"] = _fd["_changed"].cumsum()

    # Aggregate plateau-level stats (excluding dose == 0)
    _plat_base = (
        _fd.groupby("plateau_id")
        .agg(
            stay_id      =("stay_id",   "first"),
            dose         =("dose_r",    "first"),
            duration_hrs =("time_hour", "count"),
        )
        .reset_index()   # plateau_id becomes a regular column
    )
    # MAP start / end from non-null values
    _map_agg = (
        _fd.dropna(subset=["mbp"])
        .groupby("plateau_id")["mbp"]
        .agg(map_start="first", map_end="last")
        .reset_index()
    )
    _plat = _plat_base.merge(
        _map_agg[["plateau_id", "map_start", "map_end"]],
        on="plateau_id", how="left",
    ).reset_index(drop=True)
    _plat["delta_map"] = _plat["map_end"] - _plat["map_start"]
    _plat = _plat[_plat["dose"] > 0].copy()

    if len(_plat) > 0:
        _plat["dose_bin"] = pd.cut(
            _plat["dose"],
            bins=NEE_BIN_EDGES,
            labels=NEE_BIN_LABELS,
            right=False, include_lowest=True,
        )
        _valid_dbins = [g for g in NEE_BIN_LABELS
                        if (_plat["dose_bin"] == g).sum() >= 5]

        # ── Plot A: dwell time distribution by NE dose bin ───────────────────
        _ncols_dw = min(len(_valid_dbins), 4)
        _nrows_dw = (len(_valid_dbins) + _ncols_dw - 1) // _ncols_dw
        fig, axes_dw = plt.subplots(_nrows_dw, _ncols_dw,
                                     figsize=(4.5 * _ncols_dw, 4 * _nrows_dw),
                                     sharey=False)
        axes_dw = np.array(axes_dw).flatten()

        for _ax_dw, _bl in zip(axes_dw, _valid_dbins):
            _d = _plat[_plat["dose_bin"] == _bl]["duration_hrs"].dropna()
            _max_b = min(int(_d.max()) + 2, 48)
            _ax_dw.hist(_d, bins=range(1, _max_b + 1),
                       color=PALETTE[0], edgecolor="white", linewidth=0.3)
            _ax_dw.axvline(_d.median(), color=PALETTE[2], linewidth=1.8,
                          linestyle="--", label=f"Median: {_d.median():.0f} h")
            _ax_dw.set_title(f"NE {_bl} μg/kg/min\n(n={len(_d):,} episodes)", fontsize=9)
            _ax_dw.set_xlabel("Dwell time (h)", fontsize=8)
            _ax_dw.set_ylabel("Episodes", fontsize=8)
            _ax_dw.legend(fontsize=7)

        for _idx_dw in range(len(_valid_dbins), len(axes_dw)):
            axes_dw[_idx_dw].set_visible(False)

        fig.suptitle(
            f"{SITE_NAME}: How long do clinicians wait at each NE dose before changing?\n"
            f"(consecutive same-dose plateaus; ever-vaso patients, n={len(pat[pat['ever_vaso']==1]):,})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis_dwell_time_by_dose.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved analysis_dwell_time_by_dose")

        # ── Plot B: dwell time vs MAP change ─────────────────────────────────
        _sc_df = _plat[
            (_plat["duration_hrs"] >= 2) &
            _plat["delta_map"].notna()
        ].copy()

        if len(_sc_df) >= 50:
            fig, axes_sc = plt.subplots(1, 2, figsize=(14, 5))

            # Panel A: scatter dwell time vs delta MAP, colored by NE dose
            _sc = axes_sc[0].scatter(
                _sc_df["duration_hrs"].clip(upper=48),
                _sc_df["delta_map"].clip(-50, 50),
                c=_sc_df["dose"].clip(upper=NEE_PLOT_CAP),
                cmap="viridis", alpha=0.3, s=5, rasterized=True,
            )
            plt.colorbar(_sc, ax=axes_sc[0], label="NE dose (μg/kg/min)")
            axes_sc[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes_sc[0].set_xlabel("Dwell time (h, capped at 48)", fontsize=10)
            axes_sc[0].set_ylabel("ΔMAP over dwell period (mmHg)", fontsize=10)
            axes_sc[0].set_title("Dwell time vs MAP change\n(colored by NE dose)", fontsize=10)

            # Panel B: median delta MAP by dose bin
            _dmap_grp = _sc_df.groupby("dose_bin", observed=True)["delta_map"]
            _med_dmap = pd.concat([
                _dmap_grp.median().rename("median"),
                _dmap_grp.quantile(0.25).rename("q25"),
                _dmap_grp.quantile(0.75).rename("q75"),
                _dmap_grp.count().rename("n"),
            ], axis=1).reset_index()
            _med_dmap = _med_dmap[_med_dmap["dose_bin"].isin(_valid_dbins)]
            _x_idx = np.arange(len(_med_dmap))
            axes_sc[1].bar(_x_idx, _med_dmap["median"], color=PALETTE[0], alpha=0.8,
                           width=0.6)
            axes_sc[1].errorbar(
                _x_idx, _med_dmap["median"],
                yerr=[np.maximum(_med_dmap["median"] - _med_dmap["q25"], 0),
                      np.maximum(_med_dmap["q75"] - _med_dmap["median"], 0)],
                fmt="none", ecolor="black", capsize=4,
            )
            axes_sc[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes_sc[1].set_xticks(_x_idx)
            axes_sc[1].set_xticklabels(
                [str(b) for b in _med_dmap["dose_bin"]], rotation=30, ha="right", fontsize=8,
            )
            axes_sc[1].set_xlabel("NE dose bin (μg/kg/min)", fontsize=10)
            axes_sc[1].set_ylabel("Median ΔMAP (mmHg, IQR bars)", fontsize=10)
            axes_sc[1].set_title("Median MAP change by NE dose bin\n(IQR bars)", fontsize=10)

            fig.suptitle(
                f"{SITE_NAME}: NE dose dwell time and MAP change\n"
                f"(plateau episodes with ≥2h at same dose, n={len(_sc_df):,})",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{SITE_LOWER}_analysis_dwell_vs_map.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved analysis_dwell_vs_map")
        else:
            print("  Skipped dwell vs MAP scatter (insufficient plateaus)")
    else:
        print("  Skipped dwell-time analysis (no valid plateau data)")
else:
    print("  Skipped dwell-time analysis (norepinephrine column not in features)")

print(f"\nAll figures written to: {OUT_DIR}")

# =============================================================================
# Save aggregates — federated-safe CSVs + plot copies for upload_to_box_{SITE}
# =============================================================================
print("\nSaving aggregate CSVs...")
AGG_DIR = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}" / "epi_analysis"
AGG_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. km_cif_by_nee_bin.csv ─────────────────────────────────────────────────
_rows = []
for _grp in NEE_BIN_LABELS:
    _sub = km0[km0["pre_vaso_nee_group"] == _grp]
    if len(_sub) < 5:
        continue
    _n_total  = len(_sub)
    _n_events = int(_sub["event"].sum())
    _kmf = KaplanMeierFitter()
    _kmf.fit(_sub["event_time"], event_observed=_sub["event"])
    _sf   = _kmf.survival_function_["KM_estimate"]
    _cilo = _kmf.confidence_interval_["KM_estimate_lower_0.95"]
    _cihi = _kmf.confidence_interval_["KM_estimate_upper_0.95"]
    _nar  = _kmf.event_table["at_risk"].reindex(_sf.index, method="ffill")
    for _t in _sf.index:
        _rows.append(dict(nee_bin=_grp, time_hour=_t,
                          cif=1-_sf[_t], ci_lo=1-_cihi[_t], ci_hi=1-_cilo[_t],
                          n_at_risk=int(_nar[_t]), n_total=_n_total, n_events=_n_events))
pd.DataFrame(_rows).to_csv(AGG_DIR / "km_cif_by_nee_bin.csv", index=False)
print("  1/12 km_cif_by_nee_bin.csv")

# ── 2. km_survival_by_nee_bin.csv ────────────────────────────────────────────
_rows = []
for _grp in NEE_BIN_LABELS:
    _sub = pat[pat["pre_vaso_nee_group"] == _grp]
    if len(_sub) < 5:
        continue
    _n_total = len(_sub)
    _kmf = KaplanMeierFitter()
    _kmf.fit(_sub["traj_hours"], event_observed=_sub["hospital_death"])
    _sf   = _kmf.survival_function_["KM_estimate"]
    _cilo = _kmf.confidence_interval_["KM_estimate_lower_0.95"]
    _cihi = _kmf.confidence_interval_["KM_estimate_upper_0.95"]
    _nar  = _kmf.event_table["at_risk"].reindex(_sf.index, method="ffill")
    for _t in _sf.index:
        _rows.append(dict(nee_bin=_grp, time_hour=_t,
                          km_survival=_sf[_t], ci_lo=_cilo[_t], ci_hi=_cihi[_t],
                          n_at_risk=int(_nar[_t]), n_total=_n_total))
pd.DataFrame(_rows).to_csv(AGG_DIR / "km_survival_by_nee_bin.csv", index=False)
print("  2/12 km_survival_by_nee_bin.csv")

# ── 3. km_survival_ever_never_vaso.csv ───────────────────────────────────────
_rows = []
for _label, _mask in [("ever_vaso", pat["ever_vaso"] == 1),
                       ("never_vaso", pat["ever_vaso"] == 0)]:
    _sub = pat[_mask]
    _n_total = len(_sub)
    _kmf = KaplanMeierFitter()
    _kmf.fit(_sub["traj_hours"], event_observed=_sub["hospital_death"])
    _sf   = _kmf.survival_function_["KM_estimate"]
    _cilo = _kmf.confidence_interval_["KM_estimate_lower_0.95"]
    _cihi = _kmf.confidence_interval_["KM_estimate_upper_0.95"]
    _nar  = _kmf.event_table["at_risk"].reindex(_sf.index, method="ffill")
    for _t in _sf.index:
        _rows.append(dict(group=_label, time_hour=_t,
                          km_survival=_sf[_t], ci_lo=_cilo[_t], ci_hi=_cihi[_t],
                          n_at_risk=int(_nar[_t]), n_total=_n_total))
pd.DataFrame(_rows).to_csv(AGG_DIR / "km_survival_ever_never_vaso.csv", index=False)
print("  3/12 km_survival_ever_never_vaso.csv")

# ── 4. nee_proportion_on_vaso.csv ────────────────────────────────────────────
(binned[["x", "count", "prop", "ci_lo", "ci_hi"]]
 .rename(columns={"x": "nee_bin_mid", "count": "n_obs"})
 .to_csv(AGG_DIR / "nee_proportion_on_vaso.csv", index=False))
print("  4/12 nee_proportion_on_vaso.csv")

# ── 5. nee_vaso_state_hours.csv ──────────────────────────────────────────────
(binned_2c[["x", "total", "prop_never_on", "prop_on_vaso", "prop_came_off"]]
 .rename(columns={"x": "nee_bin_mid", "total": "n_total"})
 .to_csv(AGG_DIR / "nee_vaso_state_hours.csv", index=False))
print("  5/12 nee_vaso_state_hours.csv")

# ── 6. feature_dist_nee_vaso.csv ─────────────────────────────────────────────
_rows = []
for _col, _label in BOX3_FEATURES:
    for _grp in NEE_BIN_LABELS:
        _sub = pat[pat["pre_vaso_nee_group"] == _grp]
        for _status, _smask in [("never_vaso", _sub["ever_vaso"] == 0),
                                 ("ever_vaso",  _sub["ever_vaso"] == 1)]:
            _d = _sub[_smask][_col].dropna()
            if len(_d) < 5:
                continue
            _rows.append(dict(feature=_col, nee_bin=_grp, vaso_status=_status,
                              n=len(_d),
                              p5=_d.quantile(0.05), q1=_d.quantile(0.25),
                              median=_d.median(), q3=_d.quantile(0.75),
                              p95=_d.quantile(0.95)))
pd.DataFrame(_rows).to_csv(AGG_DIR / "feature_dist_nee_vaso.csv", index=False)
print("  6/12 feature_dist_nee_vaso.csv")

# ── 7. tod_init_features_binned.csv ──────────────────────────────────────────
_rows = []
for _col, _label in TOD_FEATURES:
    for _h in range(24):
        _d = tod_init[tod_init["vaso_clock_hour"] == _h][_col].dropna()
        if len(_d) < 3:
            continue
        _rows.append(dict(feature=_col, clock_hour=_h, n=len(_d),
                          q1=_d.quantile(0.25), median=_d.median(),
                          q3=_d.quantile(0.75), mean=_d.mean()))
pd.DataFrame(_rows).to_csv(AGG_DIR / "tod_init_features_binned.csv", index=False)
print("  7/12 tod_init_features_binned.csv")

# ── 8. tod_init_features_lowess.csv ──────────────────────────────────────────
_rows = []
for _col, _label in TOD_FEATURES:
    _sub = tod_init[["vaso_clock_hour", _col]].dropna()
    if len(_sub) < 20:
        continue
    _x = _sub["vaso_clock_hour"].values.astype(float)
    _y = _sub[_col].values.astype(float)
    try:
        _smooth = sm_lowess(_y, _x, frac=0.4, return_sorted=True)
        for _xi, _yi in _smooth:
            _rows.append(dict(feature=_col, clock_hour=_xi, lowess_y=_yi))
    except Exception:
        pass
pd.DataFrame(_rows).to_csv(AGG_DIR / "tod_init_features_lowess.csv", index=False)
print("  8/12 tod_init_features_lowess.csv")

# ── 9. time_to_vaso_hist.csv ─────────────────────────────────────────────────
_vt = pat[pat["ever_vaso"] == 1]["first_vaso_hour"].dropna()
_hist_bins = np.arange(0, 122, 2)
_counts, _edges = np.histogram(_vt, bins=_hist_bins)
pd.DataFrame(dict(
    bin_left_hour=_edges[:-1],
    bin_right_hour=_edges[1:],
    count=_counts,
    median_hours=_vt.median(),
    p75_hours=_vt.quantile(0.75),
)).to_csv(AGG_DIR / "time_to_vaso_hist.csv", index=False)
print("  9/12 time_to_vaso_hist.csv")

# ── 10. wait_time_histograms.csv ─────────────────────────────────────────────
_wait_metric_cols = (
    [f"hrs_nee_gt{round(t*100):03d}" for t in NE_NEE_THRESHOLDS] +
    ([f"hrs_ne_gt{round(t*100):03d}" for t in NE_NEE_THRESHOLDS] if _HAS_NE_COL else []) +
    ["hrs_map_lt65", "hrs_lac_gt2", "hrs_sofa_gt5", "delta_sofa_6h"]
)
_rows = []
for _col in _wait_metric_cols:
    if _col not in wait_df.columns:
        continue
    _d = wait_df[_col].dropna()
    if len(_d) == 0:
        continue
    _is_delta = _col == "delta_sofa_6h"
    if _is_delta:
        _lo_b = max(int(_d.min()) - 1, -15)
        _hi_b = min(int(_d.max()) + 2, 15)
        _bins = list(range(_lo_b, _hi_b + 1))
    else:
        _max_bin = min(int(_d.max()) + 2, 60)
        _bins = list(range(0, _max_bin + 1))
    _counts, _edges = np.histogram(_d, bins=_bins)
    _med = _d.median()
    for _edge, _cnt in zip(_edges[:-1], _counts):
        _rows.append(dict(metric=_col, hours=float(_edge), count=int(_cnt),
                          median=_med, mean=_d.mean()))
pd.DataFrame(_rows).to_csv(AGG_DIR / "wait_time_histograms.csv", index=False)
print("  10/12 wait_time_histograms.csv")

# ── 11. init_features_by_quartile.csv ────────────────────────────────────────
_rows = []
for _col, _label in INIT_FEATURES:
    _grp_data = [initiators[initiators["nee_init_q"] == _q][_col].dropna().values
                 for _q in Q_LABELS]
    _valid = [_d for _d in _grp_data if len(_d) >= 5]
    _kw_p = np.nan
    _mw_bonf = np.nan
    if len(_valid) >= 2:
        _, _kw_p = scipy_stats.kruskal(*_valid)
    if len(_grp_data[0]) >= 5 and len(_grp_data[3]) >= 5:
        _, _mw_p = scipy_stats.mannwhitneyu(_grp_data[0], _grp_data[3],
                                             alternative="two-sided")
        _mw_bonf = min(_mw_p * N_TESTS, 1.0)
    for _i, (_q, _d) in enumerate(zip(Q_LABELS, _grp_data)):
        if len(_d) < 5:
            continue
        _lo = Q_EDGES[_i];   _hi = Q_EDGES[_i + 1]
        _sem_q = np.std(_d, ddof=1) / np.sqrt(len(_d)) if len(_d) > 1 else np.nan
        _rows.append(dict(
            feature=_col, quartile_label=_q,
            nee_lo=None if not np.isfinite(_lo) else _lo,
            nee_hi=None if not np.isfinite(_hi) else _hi,
            n=len(_d),
            mean=float(np.mean(_d)),
            ci_lo_mean=float(np.mean(_d) - 1.96 * _sem_q) if not np.isnan(_sem_q) else np.nan,
            ci_hi_mean=float(np.mean(_d) + 1.96 * _sem_q) if not np.isnan(_sem_q) else np.nan,
            min_val=float(np.min(_d)), max_val=float(np.max(_d)),
            p5=np.percentile(_d, 5), q1=np.percentile(_d, 25),
            median=np.median(_d),    q3=np.percentile(_d, 75),
            p95=np.percentile(_d, 95),
            kw_p=_kw_p, mw_q1q4_p_bonf=_mw_bonf,
        ))
pd.DataFrame(_rows).to_csv(AGG_DIR / "init_features_by_quartile.csv", index=False)
print("  11/12 init_features_by_quartile.csv")

# ── 12. init_features_by_nee_bin.csv ─────────────────────────────────────────
_rows = []
for _col, _label in INIT_FEATURES:
    for _grp in NEE_BIN_LABELS:
        _d = initiators[initiators["nee_init_bin"] == _grp][_col].dropna()
        if len(_d) < 5:
            continue
        _sem_b = _d.std(ddof=1) / np.sqrt(len(_d)) if len(_d) > 1 else np.nan
        _rows.append(dict(
            feature=_col, nee_bin=_grp, n=len(_d),
            mean=_d.mean(),
            ci_lo_mean=_d.mean() - 1.96 * _sem_b if not np.isnan(_sem_b) else np.nan,
            ci_hi_mean=_d.mean() + 1.96 * _sem_b if not np.isnan(_sem_b) else np.nan,
            min_val=_d.min(), max_val=_d.max(),
            p5=_d.quantile(0.05), q1=_d.quantile(0.25),
            median=_d.median(), q3=_d.quantile(0.75),
            p95=_d.quantile(0.95),
        ))
pd.DataFrame(_rows).to_csv(AGG_DIR / "init_features_by_nee_bin.csv", index=False)
print("  12/14 init_features_by_nee_bin.csv")

# ── 13. vasopressor_combinations.csv ─────────────────────────────────────────
_DRUG_DEFS_13 = [
    ("action_vaso",    "VASO"),
    ("norepinephrine", "NE"),
    ("phenylephrine",  "PHENYL"),
    ("dopamine",       "DOPA"),
    ("epinephrine",    "EPI"),
    ("angiotensin",    "ANGII"),
]
_avail_drugs_13 = [(col, name) for col, name in _DRUG_DEFS_13 if col in features.columns]
if _avail_drugs_13:
    _N13 = len(pat)
    _combo13 = pat[["stay_id"]].copy()
    for _col13, _name13 in _avail_drugs_13:
        _any13 = features.groupby("stay_id")[_col13].max() > 0
        _combo13[f"any_{_name13}"] = _combo13["stay_id"].map(_any13).fillna(False)
    _ac13 = [f"any_{n}" for _, n in _avail_drugs_13]
    _combo13["n_drug_types"] = _combo13[_ac13].sum(axis=1)
    _fh13 = {}
    for _col13, _name13 in _avail_drugs_13:
        _pos13 = features.loc[features[_col13] > 0].groupby("stay_id")["time_hour"].min()
        _fh13[_name13] = _combo13["stay_id"].map(_pos13)
    _first13 = pd.DataFrame(_fh13, index=_combo13.index)
    _combo13["first_drug"] = _first13.idxmin(axis=1)
    _rows13 = []
    for _col13, _name13 in _avail_drugs_13:
        _ac_i = f"any_{_name13}"
        _n_any13    = int(_combo13[_ac_i].sum())
        _n_single13 = int((_combo13[_ac_i] & (_combo13["n_drug_types"] == 1)).sum())
        _n_first13  = int((_combo13["first_drug"] == _name13).sum())
        _row13 = dict(
            drug=_name13,
            n_any_use=_n_any13,         pct_any_use=round(_n_any13/_N13*100, 1) if _N13 else None,
            n_single_agent=_n_single13, pct_single_agent=round(_n_single13/_N13*100, 1) if _N13 else None,
            n_first_agent=_n_first13,   pct_first_agent=round(_n_first13/_N13*100, 1) if _N13 else None,
            n_total_patients=_N13,
        )
        for _col2_13, _name2_13 in _avail_drugs_13:
            _ac_j = f"any_{_name2_13}"
            if _name2_13 == _name13:
                _row13[f"n_combined_{_name2_13}"]   = None
                _row13[f"pct_combined_{_name2_13}"] = None
            else:
                _nb13 = int((_combo13[_ac_i] & _combo13[_ac_j]).sum())
                _row13[f"n_combined_{_name2_13}"]   = _nb13
                _row13[f"pct_combined_{_name2_13}"] = round(_nb13/_N13*100, 1) if _N13 else None
        _rows13.append(_row13)
    pd.DataFrame(_rows13).to_csv(AGG_DIR / "vasopressor_combinations.csv", index=False)
    print("  13/14 vasopressor_combinations.csv")

    # ── 13b. vaso_timing_individual.csv ──────────────────────────────────────
    # For vasopressin recipients: hours from each co-drug's first dose to vaso
    # (negative = co-drug started before vasopressin)
    if "VASO" in _first13.columns:
        _vaso_mask13 = _combo13["any_VASO"]
        _timing_ind_rows = []
        for _, _name13 in _avail_drugs_13:
            if _name13 == "VASO":
                continue
            _ac_j13 = f"any_{_name13}"
            if _ac_j13 not in _combo13.columns:
                continue
            _both_mask13 = _vaso_mask13 & _combo13[_ac_j13]
            _diff13 = (
                _first13.loc[_both_mask13, _name13] - _first13.loc[_both_mask13, "VASO"]
            ).dropna()
            for _dh in _diff13.values:
                _timing_ind_rows.append({"drug": _name13, "diff_hours": round(float(_dh), 3)})
        if _timing_ind_rows:
            pd.DataFrame(_timing_ind_rows).to_csv(AGG_DIR / "vaso_timing_individual.csv", index=False)
            print("  13b vaso_timing_individual.csv")
        else:
            print("  13b vaso_timing_individual empty — skipped")
    else:
        print("  13b skipped vaso_timing_individual (no VASO feature column)")
else:
    print("  13/14 skipped vasopressor_combinations (no drug columns in features)")

# ── 14. vaso_receipt_logreg.csv ───────────────────────────────────────────────
try:
    import re as _re14
    import statsmodels.formula.api as _smf14
    _lr14 = pat[["stay_id", "ever_vaso", "age", "gender", "race"]].copy()
    if "anchor_year_group" in cohort.columns:
        _lr14 = _lr14.merge(cohort[["stay_id", "anchor_year_group"]], on="stay_id", how="left")
    else:
        _lr14["anchor_year_group"] = "unknown"
    _lr14["age_cat"] = pd.cut(
        _lr14["age"],
        bins=[0, 50, 65, 75, 85, np.inf],
        labels=["<50", "50-64", "65-74", "75-84", ">=85"],
        right=False,
    ).astype(str)
    _lr14["female"] = (_lr14["gender"].astype(str).str.upper().str.startswith("F")).astype(int)
    _RACE_MAP_14 = {
        "White": "White",                                        "WHITE": "White",
        "Black or African American": "Black or African American",
        "BLACK/AFRICAN AMERICAN": "Black or African American",  "BLACK/AFRICAN": "Black or African American",
        "Hispanic": "Hispanic",                                  "HISPANIC OR LATINO": "Hispanic",
        "Asian": "Asian",                                        "ASIAN": "Asian",
    }
    _lr14["race_cat"] = _lr14["race"].map(_RACE_MAP_14).fillna("Other/Unknown")
    _lr14 = _lr14.dropna(subset=["age", "ever_vaso"])
    _lr14 = _lr14[_lr14["age_cat"] != "nan"].copy()
    _yr_vals14 = sorted(_lr14["anchor_year_group"].dropna().unique().tolist())
    _has_year14 = len(_yr_vals14) > 1 and "unknown" not in _yr_vals14
    _formula14 = (
        "ever_vaso ~ C(age_cat, Treatment('<50')) + female"
        " + C(race_cat, Treatment('White'))"
        + (" + C(anchor_year_group)" if _has_year14 else "")
    )
    _mod14 = _smf14.logit(_formula14, data=_lr14).fit(disp=0)
    _ci14  = _mod14.conf_int()
    _ci14.columns = ["ci_lo", "ci_hi"]

    def _param_label_14(p):
        m = _re14.search(r"age_cat.*\[T\.(.+?)\]", p)
        if m: return f"Age: {m.group(1)} vs <50"
        if p == "female": return "Female vs Male"
        m = _re14.search(r"race_cat.*\[T\.(.+?)\]", p)
        if m: return f"Race: {m.group(1)} vs White"
        m = _re14.search(r"anchor_year_group.*\[T\.(.+?)\]", p)
        if m: return f"Year: {m.group(1)}"
        return p

    _lr_rows14 = []
    for _p, _coef, _cilo, _cihi, _pv in zip(
        _mod14.params.index, _mod14.params.values,
        _ci14["ci_lo"].values, _ci14["ci_hi"].values,
        _mod14.pvalues.values,
    ):
        if _p == "Intercept":
            continue
        _lr_rows14.append(dict(
            param=_p,
            label=_param_label_14(_p),
            or_est=round(float(np.exp(_coef)), 3),
            or_lo=round(float(np.exp(_cilo)), 3),
            or_hi=round(float(np.exp(_cihi)), 3),
            pval=round(float(_pv), 4),
            n_obs=int(_mod14.nobs),
        ))
    pd.DataFrame(_lr_rows14).to_csv(AGG_DIR / "vaso_receipt_logreg.csv", index=False)
    print("  14/14 vaso_receipt_logreg.csv")
except Exception as _e14:
    print(f"  14/14 skipped vaso_receipt_logreg: {_e14}")

print(f"\nAggregate CSVs written to: {AGG_DIR}")

# =============================================================================
# Section 16: Federated ICC — Site Return Packet
#
# Three outcomes measuring distinct aspects of vasopressin practice:
#   1. ever_vaso         (binary)     — WHO gets vasopressin?
#   2. first_vaso_hour   (continuous) — HOW QUICKLY is it started?
#   3. nee_at_init       (continuous) — AT WHAT NE BURDEN is it started?
#
# M0 (null):  intercept only        → raw between-site ICC
# M1 (adj):   6 case-mix covariates → residual ICC after patient-factor adjustment
# PCV = (τ²_M0 − τ²_M1) / τ²_M0   → fraction explained by case-mix
#
# Binary model: score/Hessian at θ₀ (one-shot Newton, aggregated centrally)
# Linear models: XᵀX / Xᵀy sufficient statistics (exact pooled OLS)
# =============================================================================
try:
    import json as _json_icc
    import scipy.special as _sp_spec
    import scipy.optimize as _sp_opt

    print("\n" + "=" * 60)
    print("SECTION 16: FEDERATED ICC SITE PACKET")
    print("=" * 60)

    _COVARIATES_M1 = [
        "sepsis_onset_sofa",
        "initial_lactate",
        "age",
        "peak_nee_12h",   # computed below from features
        "map_t0",         # computed below from features
        "ventil_ever",
    ]

    # ── New covariates from features ──────────────────────────────────────────
    _icc_nee_12h = (
        features[features["time_hour"] <= 12]
        .groupby("stay_id")["nee"].max()
        .reset_index().rename(columns={"nee": "peak_nee_12h"})
    )
    _icc_map_t0 = (
        features[features["time_hour"] <= 1]
        .sort_values(["stay_id", "time_hour"])
        .groupby("stay_id")["mbp"].first()
        .reset_index().rename(columns={"mbp": "map_t0"})
    )

    # ── Build analysis dataframe ──────────────────────────────────────────────
    _icc_df = pat[
        ["stay_id", "ever_vaso", "first_vaso_hour", "nee_at_init", "ventil_ever"]
    ].copy()
    _icc_df = _icc_df.merge(
        cohort[["stay_id", "sepsis_onset_sofa", "initial_lactate", "age"]],
        on="stay_id", how="left",
    )
    _icc_df = _icc_df.merge(_icc_nee_12h, on="stay_id", how="left")
    _icc_df = _icc_df.merge(_icc_map_t0, on="stay_id", how="left")

    # ── Off-hours in local time (initiators only) ─────────────────────────────
    _icc_tz = getattr(_cfg, "TIMEZONE", "UTC")  # _cfg still in scope from top
    _icc_ref = "trajectory_start" if "trajectory_start" in pat.columns else "first_norepi_time"
    _icc_ts = pat[["stay_id", _icc_ref, "first_vaso_hour", "ever_vaso"]].copy()
    _icc_ts["_ref_utc"] = pd.to_datetime(_icc_ts[_icc_ref], utc=True, errors="coerce")
    _icc_ts["_vaso_utc"] = (
        _icc_ts["_ref_utc"]
        + pd.to_timedelta(_icc_ts["first_vaso_hour"].fillna(0).astype(float), unit="h")
    )
    try:
        _icc_ts["_vaso_local"] = _icc_ts["_vaso_utc"].dt.tz_convert(_icc_tz)
    except Exception:
        _icc_ts["_vaso_local"] = _icc_ts["_vaso_utc"]

    _icc_ts["shock_hour"]   = _icc_ts["_vaso_local"].dt.hour
    _icc_ts["shock_dow"]    = _icc_ts["_vaso_local"].dt.dayofweek   # 0=Mon, 6=Sun
    _icc_ts["is_weekend"]   = (_icc_ts["shock_dow"] >= 5).astype(int)
    _icc_ts["is_off_hours"] = (
        (_icc_ts["is_weekend"] == 1)
        | (_icc_ts["shock_hour"] < 7)
        | (_icc_ts["shock_hour"] >= 18)
    ).astype(int)
    for _c16 in ["shock_hour", "shock_dow", "is_weekend", "is_off_hours"]:
        _icc_ts.loc[_icc_ts["ever_vaso"] == 0, _c16] = np.nan
    _icc_df = _icc_df.merge(
        _icc_ts[["stay_id", "shock_hour", "is_weekend", "is_off_hours"]],
        on="stay_id", how="left",
    )

    # ── Validation checks ─────────────────────────────────────────────────────
    _icc_init = _icc_df[_icc_df["ever_vaso"] == 1]
    _off_rate = _icc_init["is_off_hours"].mean()
    print(f"  Off-hours initiation rate: {_off_rate:.1%}")
    if not np.isnan(_off_rate) and not (0.35 <= _off_rate <= 0.75):
        print("  WARNING: off-hours rate outside expected range 35–75% — check TIMEZONE setting")
    for _c16 in _COVARIATES_M1:
        _miss = _icc_df[_c16].isna().mean()
        if _miss > 0.30:
            print(f"  WARNING: {_c16} has {_miss:.0%} missing in ICC dataset")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _icc_fit_logistic(X, y):
        """Pooled logistic MLE via BFGS. Returns coef vector."""
        def _nll(th):
            logit = X @ th
            return -float(np.sum(y * logit - np.logaddexp(0, logit)))
        def _grad(th):
            return -(X.T @ (y - _sp_spec.expit(X @ th)))
        return _sp_opt.minimize(
            _nll, x0=np.zeros(X.shape[1]), jac=_grad, method="BFGS",
            options={"maxiter": 1000},
        ).x

    def _icc_score_hess(X, y, theta):
        """Score and Hessian of logistic log-likelihood at theta."""
        p = _sp_spec.expit(X @ theta)
        score = X.T @ (y - p)
        W = p * (1.0 - p)
        hessian = -(X.T * W) @ X
        return score, hessian

    def _icc_lm_suff(outcome_col, anchor_mu, anchor_sd):
        """XᵀX / Xᵀy / yᵀy for linear model, standardized with anchor params."""
        _sub = _icc_init[["stay_id", outcome_col] + _COVARIATES_M1].dropna()
        if len(_sub) < 11:
            return {"n": len(_sub), "suppressed": True}
        _y  = _sub[outcome_col].values.astype(float)
        _Xr = _sub[_COVARIATES_M1].values.astype(float)
        _X  = np.column_stack([np.ones(len(_sub)), (_Xr - anchor_mu) / anchor_sd])
        return {
            "n":     int(len(_sub)),
            "y_bar": float(_y.mean()),
            "y_sd":  float(_y.std()),
            "XtX":   (_X.T @ _X).tolist(),
            "Xty":   (_X.T @ _y).tolist(),
            "yty":   float(_y @ _y),
        }

    # ── M0: intercept-only (binary) ───────────────────────────────────────────
    _m0_df   = _icc_df[["stay_id", "ever_vaso"]].dropna()
    _X_m0    = np.ones((len(_m0_df), 1))
    _y_m0    = _m0_df["ever_vaso"].values.astype(float)

    # ── M1: case-mix covariates (binary) ─────────────────────────────────────
    _m1_df   = _icc_df[["stay_id", "ever_vaso"] + _COVARIATES_M1].dropna()
    _m1_Xraw = _m1_df[_COVARIATES_M1].values.astype(float)
    _local_mu = _m1_Xraw.mean(axis=0)
    _local_sd = _m1_Xraw.std(axis=0)
    _local_sd[_local_sd == 0] = 1.0
    _y_m1    = _m1_df["ever_vaso"].values.astype(float)

    # ── Anchor parameters: config → local fit ────────────────────────────────
    _anchor_cfg = getattr(_cfg, "FEDERATED_ICC_ANCHOR", None)
    _theta0_m0, _theta0_m1, _anchor_mu, _anchor_sd = None, None, _local_mu, _local_sd
    _IS_ANCHOR = _anchor_cfg is None

    if _anchor_cfg is not None:
        _theta0_m0 = np.array(_anchor_cfg["theta0_m0"])
        _theta0_m1 = np.array(_anchor_cfg["theta0_m1"])
        _anchor_mu = np.array(_anchor_cfg["mu"])
        _anchor_sd = np.array(_anchor_cfg["sd"])
        _anchor_sd[_anchor_sd == 0] = 1.0
        print("  Loaded anchor theta0 from config.FEDERATED_ICC_ANCHOR")
    else:
        print("  Anchor site: fitting local logistic theta0 for M0 and M1...")
        _theta0_m0 = _icc_fit_logistic(_X_m0, _y_m0)
        _X_m1_fit  = np.column_stack([np.ones(len(_m1_df)),
                                      (_m1_Xraw - _local_mu) / _local_sd])
        _theta0_m1 = _icc_fit_logistic(_X_m1_fit, _y_m1)
        _anchor_mu, _anchor_sd = _local_mu, _local_sd
        _anchor_sd[_anchor_sd == 0] = 1.0

        def _fmt(arr):
            return "[" + ", ".join(f"{v:.6g}" for v in arr) + "]"

        print("\n  ── Anchor fitted. Add to config.py for all other sites: ──")
        print("  FEDERATED_ICC_ANCHOR = {")
        print(f'      "theta0_m0": {_fmt(_theta0_m0)},')
        print(f'      "theta0_m1": {_fmt(_theta0_m1)},')
        print(f'      "mu":        {_fmt(_anchor_mu)},')
        print(f'      "sd":        {_fmt(_anchor_sd)},')
        print("  }")
        print("  ─" * 28)

    # ── Score/Hessian for M0 at theta0 ───────────────────────────────────────
    _s_m0, _h_m0 = _icc_score_hess(_X_m0, _y_m0, _theta0_m0)

    # ── Score/Hessian for M1 at theta0 (using anchor standardization) ─────────
    _X_m1_anch = np.column_stack(
        [np.ones(len(_m1_df)), (_m1_Xraw - _anchor_mu) / _anchor_sd]
    )
    if _theta0_m1 is not None:
        _s_m1, _h_m1 = _icc_score_hess(_X_m1_anch, _y_m1, _theta0_m1)
    else:
        _s_m1, _h_m1 = None, None

    # ── Sufficient statistics for continuous outcomes (anchor standardization) ─
    _suff_time = _icc_lm_suff("first_vaso_hour", _anchor_mu, _anchor_sd)
    _suff_nee  = _icc_lm_suff("nee_at_init",     _anchor_mu, _anchor_sd)

    # ── Assemble packet ───────────────────────────────────────────────────────
    def _sf16(x):
        try:
            v = float(x)
            return None if np.isnan(v) else v
        except Exception:
            return None

    _n_m0, _n_m1 = len(_m0_df), len(_m1_df)

    _icc_pkt = {
        "site_id":    SITE_NAME,
        "is_anchor":  _IS_ANCHOR,
        "covariates_m1": _COVARIATES_M1,
        "anchor_standardization": {
            "mu": _anchor_mu.tolist(),
            "sd": _anchor_sd.tolist(),
        },
        # ── Cohort descriptives ───────────────────────────────────────────────
        "n_total":          int(len(_icc_df)),
        "n_initiators":     int(_icc_df["ever_vaso"].sum()),
        "ever_vaso_rate":   _sf16(_icc_df["ever_vaso"].mean()),
        # ── Case-mix (Table 1 reconstruction) ────────────────────────────────
        "covariate_stats": {
            c: {
                "mean":        _sf16(_icc_df[c].mean()),
                "sd":          _sf16(_icc_df[c].std()),
                "missing_pct": _sf16(_icc_df[c].isna().mean() * 100),
            }
            for c in _COVARIATES_M1
        },
        # ── Continuous outcome descriptives ───────────────────────────────────
        "time_to_init_hrs": {
            "median": _sf16(_icc_init["first_vaso_hour"].median()),
            "q1":     _sf16(_icc_init["first_vaso_hour"].quantile(0.25)),
            "q3":     _sf16(_icc_init["first_vaso_hour"].quantile(0.75)),
        },
        "nee_at_init_desc": {
            "median": _sf16(_icc_init["nee_at_init"].median()),
            "q1":     _sf16(_icc_init["nee_at_init"].quantile(0.25)),
            "q3":     _sf16(_icc_init["nee_at_init"].quantile(0.75)),
        },
        # ── Off-hours descriptives ────────────────────────────────────────────
        "off_hours": {
            "off_hours_rate":   _sf16(_icc_init["is_off_hours"].mean()),
            "weekend_rate":     _sf16(_icc_init["is_weekend"].mean()),
            "time_to_init_off": _sf16(
                _icc_init.loc[_icc_init["is_off_hours"] == 1, "first_vaso_hour"].mean()
            ),
            "time_to_init_on":  _sf16(
                _icc_init.loc[_icc_init["is_off_hours"] == 0, "first_vaso_hour"].mean()
            ),
            "nee_at_init_off":  _sf16(
                _icc_init.loc[_icc_init["is_off_hours"] == 1, "nee_at_init"].mean()
            ),
            "nee_at_init_on":   _sf16(
                _icc_init.loc[_icc_init["is_off_hours"] == 0, "nee_at_init"].mean()
            ),
        },
        # ── M0: binary null model ─────────────────────────────────────────────
        "m0_binary": {
            "n_model":  int(_n_m0),
            "y_bar":    _sf16(_y_m0.mean()),
            "theta0":   _theta0_m0.tolist() if _IS_ANCHOR else None,
            "score":    _s_m0.tolist() if _n_m0 >= 11 else None,
            "hessian":  _h_m0.tolist() if _n_m0 >= 11 else None,
        },
        # ── M1: binary case-mix model ─────────────────────────────────────────
        "m1_binary": {
            "n_model": int(_n_m1),
            "y_bar":   _sf16(_y_m1.mean()) if _n_m1 > 0 else None,
            "score":   _s_m1.tolist() if (_s_m1 is not None and _n_m1 >= 11) else None,
            "hessian": _h_m1.tolist() if (_h_m1 is not None and _n_m1 >= 11) else None,
        },
        # ── Linear outcomes: sufficient statistics ────────────────────────────
        "m1_time_to_init": _suff_time,
        "m1_nee_at_init":  _suff_nee,
    }

    _pkt_path = AGG_DIR / f"site_packet_{SITE_NAME}.json"
    with open(_pkt_path, "w") as _fout16:
        _json_icc.dump(_icc_pkt, _fout16, indent=2,
                       default=lambda x: None if x is None else x)
    print(f"  Saved site packet → {_pkt_path}")
    if _IS_ANCHOR:
        print("  Anchor theta0 files written — distribute to other sites before they run this script")

except Exception as _e16:
    print(f"  Section 16 failed: {_e16}")
    import traceback as _tb16
    _tb16.print_exc()
