"""
generate_synthetic_adni.py
===========================
Generate SYNTHETIC, ADNI/OASIS-style longitudinal Alzheimer's biomarker data
for the Alzheimer's Biomarker Hypothesis Agent capstone.

*** THIS IS NOT REAL PATIENT DATA. ***
Values are simulated from a latent disease-progression model whose parameters
are chosen to mirror published directions/magnitudes (OASIS-3, ADNI, and the
Jack/Bateman amyloid cascade). Any "finding" recovered from this file is a
property of the generator, not evidence about real biology. Use only to build
and demonstrate the analysis pipeline; swap in real OASIS-3/ADNI data later.

Calibration anchors (OASIS-3; LaMontagne et al., 2019, medRxiv 2019.12.13.19014902):
  - Amyloid positivity threshold: Centiloid = 16.4
  - Baseline Centiloid by group ~ 8 (stable controls), 28 (converters), 72 (dementia)
  - Whole-brain annual atrophy ~ -1.1% / -1.7% / -2.0% (controls/converters/dementia)
  - APOE genotype frequencies in OASIS-3 (e4 allele freq ~ 0.23, e2 ~ 0.08)
Cascade ordering (Jack et al. 2010 Lancet Neurol; Bateman et al. 2012 NEJM):
  amyloid -> tau -> neurodegeneration -> cognition
  This ordering is encoded by the per-biomarker logistic onset centers below:
  amyloid moves at the LOWEST progression index p, then tau, then
  neurodegeneration, then cognition (a lower center => earlier onset).

Author: capstone build helper
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG  -- tune these; every effect direction is centralized here
# --------------------------------------------------------------------------- #
CONFIG = dict(
    seed=42,
    n_subjects=300,

    # APOE allele frequencies (OASIS-3-like, e4-enriched relative to general pop)
    apoe_allele_freq={"e2": 0.08, "e3": 0.69, "e4": 0.23},

    # TREM2 R47H (rs75932628): rare, enriched in AD cohorts
    trem2_carrier_rate=0.025,

    # demographics
    frac_female=0.55,
    age_mean=70.0, age_sd=8.0, age_min=50.0, age_max=90.0,

    # visit schedule (ADNI-style months) and how many visits per subject
    visit_schedule_months=[0, 6, 12, 24, 36, 48],
    n_visit_probs=[0.18, 0.22, 0.24, 0.18, 0.12, 0.06],  # for 1..6 visits

    # latent progression index p(t) = p0 + rate * years
    #   p0 driven by age + genetics; rate driven by genetics
    p0_age_coef=0.010,       # per year above 70 (see p0 construction)
    p0_e4_coef=0.14,         # per e4 allele
    p0_e2_coef=-0.10,        # per e2 allele (protective)
    p0_trem2_coef=0.10,
    p0_noise_sd=0.21,
    rate_base_mean=0.050, rate_base_sd=0.032, rate_min=0.004,
    rate_e4_mult=0.50,       # each e4 allele multiplies rate by (1 + this)
    rate_trem2_mult=0.40,

    # biomarker onset centers (lower center => earlier in the cascade).
    # amyloid < tau < neurodegeneration < cognition, per Jack 2010 / Bateman 2012.
    onset_amyloid=0.60,      # earliest to move (unchanged from original calibration)
    onset_amyloid_e4_shift=0.12,   # each e4 allele shifts amyloid onset earlier
    onset_tau=0.66,
    onset_neurodeg=0.72,
    onset_cdrsb=0.76,
    onset_cognition=0.80,    # MMSE; lags everything
)

# --------------------------------------------------------------------------- #
def _logistic(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def _sample_apoe(rng, freqs, n):
    alleles = list(freqs.keys())
    probs = np.array([freqs[a] for a in alleles])
    probs = probs / probs.sum()
    a1 = rng.choice(alleles, size=n, p=probs)
    a2 = rng.choice(alleles, size=n, p=probs)
    genotype, e4_count, e2_count = [], [], []
    for x, y in zip(a1, a2):
        pair = sorted([x, y])  # canonical order e2<e3<e4
        genotype.append(f"{pair[0]}/{pair[1]}")
        e4_count.append((x == "e4") + (y == "e4"))
        e2_count.append((x == "e2") + (y == "e2"))
    return np.array(genotype), np.array(e4_count), np.array(e2_count)


def _biomarkers_from_latent(p, age, e4, rng, n, c=CONFIG):
    """Map latent progression index p -> observable biomarkers, with staggered
    onsets so amyloid leads and cognition lags (amyloid < tau < neurodegeneration
    < cognition). Returns a dict of arrays."""
    # amyloid Centiloid: earliest to move; e4 shifts onset earlier + raises plateau
    amy_base, amy_plateau = 2.0, 100.0 + 8.0 * e4
    amyloid = amy_base + (amy_plateau - amy_base) * _logistic(
        p - c["onset_amyloid"] + c["onset_amyloid_e4_shift"] * e4, k=4.0)
    amyloid += rng.normal(0, 4.0, n)

    # tau SUVR: lags amyloid
    tau = 1.05 + (2.00 - 1.05) * _logistic(p - c["onset_tau"], k=5.0) + rng.normal(0, 0.05, n)

    # neurodegeneration block: hippocampus, entorhinal, FDG (lags tau)
    nd = _logistic(p - c["onset_neurodeg"], k=5.0)
    hipp_base = 7800.0 - 11.0 * (age - 50.0)                       # mild age atrophy
    hippocampal = hipp_base - 2000.0 * (1 + 0.10 * e4) * nd + rng.normal(0, 130, n)
    entorhinal = 3.40 - 0.004 * (age - 50.0) - 1.00 * nd + rng.normal(0, 0.08, n)
    fdg = 1.35 - 0.35 * nd + rng.normal(0, 0.03, n)

    # cognition: lags everything
    cog = _logistic(p - c["onset_cognition"], k=6.0)
    mmse = 30.0 - 13.0 * cog + rng.normal(0, 1.0, n)
    cdrsb_raw = 13.0 * _logistic(p - c["onset_cdrsb"], k=6.0) + rng.normal(0, 0.6, n)

    return dict(amyloid=amyloid, tau=tau, hippocampal=hippocampal,
                entorhinal=entorhinal, fdg=fdg, mmse=mmse, cdrsb_raw=cdrsb_raw)


def _diagnosis_from_cdrsb(cdrsb):
    if cdrsb == 0:
        return "CN"
    if cdrsb <= 4.0:
        return "MCI"
    return "AD"


def generate(config=CONFIG):
    c = config
    rng = np.random.default_rng(c["seed"])
    n = c["n_subjects"]

    # ---- subject-level static attributes ----
    genotype, e4_count, e2_count = _sample_apoe(rng, c["apoe_allele_freq"], n)
    trem2 = np.where(rng.random(n) < c["trem2_carrier_rate"], "R47H", "WT")
    sex = np.where(rng.random(n) < c["frac_female"], "F", "M")
    base_age = np.clip(rng.normal(c["age_mean"], c["age_sd"], n),
                       c["age_min"], c["age_max"])

    # latent baseline severity and progression rate
    p0 = (c["p0_age_coef"] * (base_age - 70.0)
          + c["p0_e4_coef"] * e4_count
          + c["p0_e2_coef"] * e2_count
          + c["p0_trem2_coef"] * (trem2 == "R47H")
          + rng.normal(0, c["p0_noise_sd"], n))
    rate = np.maximum(c["rate_min"],
                      rng.normal(c["rate_base_mean"], c["rate_base_sd"], n))
    rate = rate * (1 + c["rate_e4_mult"] * e4_count
                     + c["rate_trem2_mult"] * (trem2 == "R47H"))

    n_visits = rng.choice(range(1, len(c["visit_schedule_months"]) + 1),
                          size=n, p=c["n_visit_probs"])

    # ---- expand to long format (one row per subject-visit) ----
    rows = []
    for i in range(n):
        months = c["visit_schedule_months"][: n_visits[i]]
        for m in months:
            yrs = m / 12.0
            p = p0[i] + rate[i] * yrs
            age = base_age[i] + yrs
            bm = _biomarkers_from_latent(p, age, e4_count[i], rng, 1, c)
            cdrsb = float(np.clip(np.round(bm["cdrsb_raw"][0] * 2) / 2, 0, 18))
            if cdrsb < 0.25:
                cdrsb = 0.0
            mmse = int(np.clip(round(bm["mmse"][0]), 0, 30))
            rows.append(dict(
                subject_id=f"SYN{i+1:04d}",
                visit_month=m,
                diagnosis=_diagnosis_from_cdrsb(cdrsb),
                age=round(age, 1),
                sex=sex[i],
                apoe_e4_count=int(e4_count[i]),
                apoe_genotype=genotype[i],          # bonus column (drop if undesired)
                trem2_variant_status=trem2[i],
                hippocampal_volume=round(float(bm["hippocampal"][0]), 1),
                entorhinal_thickness=round(float(bm["entorhinal"][0]), 3),
                amyloid_centiloid=round(float(bm["amyloid"][0]), 1),
                tau_suvr=round(float(bm["tau"][0]), 3),
                fdg_suvr=round(float(bm["fdg"][0]), 3),
                mmse=mmse,
                cdrsb=cdrsb,
            ))

    cols = ["subject_id", "visit_month", "diagnosis", "age", "sex",
            "apoe_e4_count", "apoe_genotype", "trem2_variant_status",
            "hippocampal_volume", "entorhinal_thickness", "amyloid_centiloid",
            "tau_suvr", "fdg_suvr", "mmse", "cdrsb"]
    return pd.DataFrame(rows)[cols]


def validate(df):
    """Print a quick sanity check that effect directions came out right."""
    AMY_POS = 16.4  # OASIS-3 positivity threshold
    print("=" * 64)
    print(f"Rows: {len(df)}  |  Subjects: {df.subject_id.nunique()}")
    print(f"Diagnosis counts (rows): {df.diagnosis.value_counts().to_dict()}")
    print(f"APOE e4 count (subjects): "
          f"{df.drop_duplicates('subject_id').apoe_e4_count.value_counts().sort_index().to_dict()}")
    print("-" * 64)
    print("Baseline means by APOE e4 dose (expect: more e4 -> higher amyloid,")
    print("lower hippocampal volume, lower MMSE):")
    bl = df[df.visit_month == 0]
    summary = bl.groupby("apoe_e4_count").agg(
        n=("subject_id", "nunique"),
        amyloid_centiloid=("amyloid_centiloid", "mean"),
        pct_amyloid_pos=("amyloid_centiloid", lambda s: 100 * (s >= AMY_POS).mean()),
        hippocampal_volume=("hippocampal_volume", "mean"),
        mmse=("mmse", "mean"),
    ).round(1)
    print(summary.to_string())
    print("-" * 64)
    # cascade check: median progression point at which each marker crosses its midpoint
    print("Cascade onset order (amyloid should be earliest, cognition latest):")
    for name, ctr in sorted({"amyloid": CONFIG["onset_amyloid"], "tau": CONFIG["onset_tau"],
                             "neurodegeneration": CONFIG["onset_neurodeg"],
                             "cdrsb": CONFIG["onset_cdrsb"],
                             "cognition(mmse)": CONFIG["onset_cognition"]}.items(),
                            key=lambda kv: kv[1]):
        print(f"   onset p={ctr:.2f}  {name}")
    print("=" * 64)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic ADNI-style dataset.")
    parser.add_argument("--n", type=int, default=CONFIG["n_subjects"], help="number of subjects")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="random seed")
    parser.add_argument("--out", default="synthetic_adni_style.csv", help="output CSV path")
    args = parser.parse_args()

    config = dict(CONFIG, n_subjects=args.n, seed=args.seed)
    df = generate(config)
    validate(df)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
