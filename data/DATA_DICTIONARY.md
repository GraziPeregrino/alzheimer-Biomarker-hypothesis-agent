# Synthetic ADNI/OASIS-style Dataset — Data Dictionary

**File:** `synthetic_adni_style.csv` · **Generator:** `generate_synthetic_adni.py`
**Rows:** one per subject-visit (long format) · **Default size:** 300 subjects, ~876 visits

> ⚠️ **This is synthetic data — not real patient data.** Values are simulated
> from a latent disease-progression model. Parameters were chosen to reproduce
> the *directions and rough magnitudes* reported in the literature, so any
> "result" recovered from this file is built into the generator by design. Use
> it only to build and demonstrate the analysis pipeline. The pipeline is
> **ADNI-/OASIS-ready**: swap in real data with the same column names to run a
> genuine analysis (subject to the relevant Data Use Agreement).

## Columns

| Column | Type | Units / values | Description |
|---|---|---|---|
| `subject_id` | str | `SYN0001`… | Stable per-subject identifier |
| `visit_month` | int | 0, 6, 12, 24, 36, 48 | Months from baseline (ADNI-style cadence) |
| `diagnosis` | str | `CN`, `MCI`, `AD` | Derived from CDR-SB at that visit |
| `age` | float | years | Age at visit (baseline age + elapsed time) |
| `sex` | str | `M`, `F` | Biological sex (~55% F) |
| `apoe_e4_count` | int | 0, 1, 2 | Number of APOE ε4 alleles (primary genetic variable) |
| `apoe_genotype` | str | e.g. `e3/e4` | Full genotype (**bonus column**; drop if you want the locked schema exactly) |
| `trem2_variant_status` | str | `WT`, `R47H` | TREM2 R47H (rs75932628) carrier status (~2.5% carriers) |
| `hippocampal_volume` | float | mm³ | Total hippocampal volume (FreeSurfer-style) |
| `entorhinal_thickness` | float | mm | Entorhinal cortical thickness |
| `amyloid_centiloid` | float | Centiloid (0–~120) | Global amyloid burden; **positivity threshold = 16.4** |
| `tau_suvr` | float | SUVR (~1.0–2.2) | Tau-PET standardized uptake value ratio (meta-temporal) |
| `fdg_suvr` | float | SUVR (~1.0–1.35) | FDG-PET metabolism (lower = more hypometabolism) |
| `mmse` | int | 0–30 | Mini-Mental State Exam (higher = better) |
| `cdrsb` | float | 0–18, 0.5 steps | Clinical Dementia Rating – Sum of Boxes (higher = worse) |

`diagnosis` rule: `cdrsb == 0` → CN; `0.5 ≤ cdrsb ≤ 4.0` → MCI; `cdrsb ≥ 4.5` → AD.

## How the data is generated (defensible modeling)

Each subject has a hidden **disease-progression index** `p(t) = p0 + rate·years`.
`p0` rises with age, ε4 dose, and TREM2 status, and falls with ε2; `rate` rises
with ε4 dose and TREM2. Every biomarker is a function of `p` with a **staggered
onset**, so they move in the order: **amyloid → tau → neurodegeneration
(hippocampus / entorhinal / FDG) → cognition (MMSE / CDR-SB)**. This reproduces
the Jack/Bateman amyloid cascade and makes ε4 effects flow through all markers.

## Calibration anchors

- **Amyloid positivity = 16.4 Centiloid**; baseline amyloid by group ~8 / 28 / 72 — OASIS-3 (LaMontagne et al., 2019, medRxiv 2019.12.13.19014902).
- **APOE allele frequencies** ε2 0.08 / ε3 0.69 / ε4 0.23 (OASIS-3-like, ε4-enriched).
- **Cascade ordering** — Jack et al., 2010, *Lancet Neurol*; Bateman et al., 2012, *NEJM*.
- **APOE biology** — ε4 raises risk and accelerates accumulation; ε2 protective (Corder 1993; Corder 1994).

## Built-in effect directions (validated on the default seed)

Baseline means by ε4 dose (0 / 1 / 2):
- amyloid Centiloid ↑: 11 / 22 / 59; amyloid-positive %: 20 / 61 / 100
- hippocampal volume ↓: 7373 / 7249 / 6750 mm³; MMSE ↓: 29.4 / 29.4 / 27.8

Annual longitudinal slopes by ε4 dose (0 / 1 / 2):
- hippocampal: −60 / −72 / −211 mm³/yr · amyloid: +2.2 / +4.0 / +4.7 CL/yr · MMSE: +0.1 / −0.5 / −0.9 /yr

## Reproduce / customize

```bash
python generate_synthetic_adni.py     # writes synthetic_adni_style.csv + prints validation
```

Edit the `CONFIG` dict at the top to change `n_subjects`, `seed`, APOE/TREM2
frequencies, or any effect size. All effect directions are centralized there and
in `_biomarkers_from_latent()`.

## Honesty / safety note for your writeup

State plainly that the agent performs **hypothesis generation, not diagnosis or
clinical prediction**, that the demo runs on **synthetic data**, and that
observational associations are **not causal**. Rare-variant (TREM2) results are
**underpowered** at this sample size and should be flagged as illustrative only.
