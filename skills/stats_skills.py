"""
stats_engine.py — statistics + hypothesis engine (revised)
================================================================

Consumes the cleaned DataFrame + genotype groups from ingestion.py and turns them
into transparent, citation-backed hypothesis cards.

Agent-ready tools:
    compute_group_summary(df, biomarkers, scheme)          -> dict
    compute_subject_slopes(df, biomarker)                  -> DataFrame  (fallback only)
    compare_cross_sectional(df, biomarker, scheme)         -> dict
    adjusted_effect(df, biomarker, scheme)                 -> dict
    compare_biomarker_trajectories(df, biomarker, scheme)  -> dict
    analyze(df, biomarker, scheme)                         -> dict   (bundle)
    make_hypothesis_card(bundle)                           -> dict
    run_hypothesis(df, biomarker, scheme)                  -> dict
    rank_biomarkers(df, scheme)                            -> dict

Methodological notes (what changed vs. the first version, and why):
  * BASELINE cross-section for group level (avoids pseudo-replication) — unchanged.
  * LONGITUDINAL change is now estimated with a LINEAR MIXED-EFFECTS MODEL
        y ~ time * gene + baseline_age + sex + (time | subject)
    The time:gene interaction is the differential-trajectory test. This uses every
    visit, weights subjects by their information, adjusts for age/sex, and is valid
    under MAR dropout — replacing the old equal-weight per-subject-OLS-slope stage.
    If the random-slope model does not converge it falls back to random-intercept,
    and if statsmodels is unavailable / the fit fails entirely it falls back to the
    two-stage per-subject-slope contrast (flagged in the card cautions).
  * For the ε4-dose scheme we also run a LACK-OF-FIT check (allele coded as a factor
    vs. linear), because APOE ε4 dose is often non-linear (see Corder 1993).
  * rank_biomarkers now reports Benjamini-Hochberg FDR q-values across biomarkers.
Citations are drawn from a curated real set — nothing is LLM-generated.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Make the project root importable so `skills.*` resolves even when this file is
# run directly (python skills/stats_skills.py).
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from skills.ingestion_skills import group_by_genotype, AMYLOID_POSITIVITY_CENTILOID

# statsmodels is required for the mixed-effects trajectory model. The module still
# imports and runs (with a documented fallback) if it is missing.
try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


# --------------------------------------------------------------------------- #
# Metadata & curated citations (REAL references; attach by relevance)
# --------------------------------------------------------------------------- #
BIOMARKER_META = {
    "amyloid_centiloid":    {"higher_is_worse": True,  "unit": "CL",   "label": "amyloid burden"},
    "tau_suvr":             {"higher_is_worse": True,  "unit": "SUVR", "label": "tau burden"},
    "fdg_suvr":             {"higher_is_worse": False, "unit": "SUVR", "label": "FDG metabolism"},
    "hippocampal_volume":   {"higher_is_worse": False, "unit": "mm³",  "label": "hippocampal volume"},
    "entorhinal_thickness": {"higher_is_worse": False, "unit": "mm",   "label": "entorhinal thickness"},
    "mmse":                 {"higher_is_worse": False, "unit": "pts",  "label": "MMSE"},
    "cdrsb":                {"higher_is_worse": True,  "unit": "pts",  "label": "CDR-SB"},
}

CITATIONS = {
    "bateman2012": "Bateman RJ et al. Clinical and biomarker changes in dominantly inherited Alzheimer's disease. N Engl J Med. 2012;367(9):795-804.",
    "jack2010": "Jack CR Jr et al. Hypothetical model of dynamic biomarkers of the Alzheimer's pathological cascade. Lancet Neurol. 2010;9(1):119-128.",
    "jack2024": "Jack CR Jr et al. Revised criteria for diagnosis and staging of Alzheimer's disease (Alzheimer's Association Workgroup). Alzheimers Dement. 2024;20(8):5143-5169.",
    "corder1993": "Corder EH et al. Gene dose of apolipoprotein E type 4 allele and the risk of Alzheimer's disease in late onset families. Science. 1993;261(5123):921-923.",
    "jonsson2013": "Jonsson T et al. Variant of TREM2 associated with the risk of Alzheimer's disease. N Engl J Med. 2013;368(2):107-116.",
    "oasis3_2019": "LaMontagne PJ et al. OASIS-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease. medRxiv 2019.12.13.19014902.",
}


# --------------------------------------------------------------------------- #
# Small OLS with confidence intervals (numpy + scipy; no statsmodels needed)
# --------------------------------------------------------------------------- #
def _ols(X: np.ndarray, y: np.ndarray, names: List[str]) -> dict:
    """Ordinary least squares with 95% CIs and two-sided p-values.

    X must already include an intercept column. Returns per-term estimates.
    Flags rank deficiency (near-collinear design) rather than silently absorbing
    it into the pseudo-inverse.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    dof = n - p
    rank = int(np.linalg.matrix_rank(X))
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    tcrit = stats.t.ppf(0.975, dof)
    out = {}
    for i, name in enumerate(names):
        out[name] = {
            "coef": float(beta[i]),
            "se": float(se[i]),
            "t": float(tvals[i]),
            "p_value": float(pvals[i]),
            "ci95": [float(beta[i] - tcrit * se[i]), float(beta[i] + tcrit * se[i])],
        }
    out["_n"] = int(n)
    out["_dof"] = int(dof)
    out["_rank_deficient"] = bool(rank < p)
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test + mean difference with 95% CI."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"insufficient_n": True, "n_a": len(a), "n_b": len(b)}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se**4 / ((a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    tcrit = stats.t.ppf(0.975, dof)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "n_a": int(len(a)), "n_b": int(len(b)),
        "mean_diff": float(diff), "diff_ci95": [float(diff - tcrit*se), float(diff + tcrit*se)],
        "t": float(t), "p_value": float(p), "cohens_d": _cohens_d(a, b),
    }


def _bh_fdr(pvals: List[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (no external dependency)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# --------------------------------------------------------------------------- #
# 1. compute_group_summary
# --------------------------------------------------------------------------- #
def compute_group_summary(df: pd.DataFrame, biomarkers: Union[List[str], None] = None,
                          scheme: str = "e4_dose", baseline_only: bool = True) -> dict:
    """Descriptive statistics (n, mean, median, sd) per genotype group.

    For amyloid_centiloid it also reports the % above the positivity threshold.
    Uses baseline visits by default to avoid counting repeat visits as independent.
    """
    biomarkers = biomarkers or list(BIOMARKER_META)
    data = df[df["visit_month"] == 0] if baseline_only else df
    groups = group_by_genotype(data, scheme)
    summary = {"scheme": scheme, "baseline_only": baseline_only, "groups": {}}
    for label, g in groups.items():
        entry = {"n_subjects": int(g["subject_id"].nunique()), "biomarkers": {}}
        for bm in biomarkers:
            if bm not in g:
                continue
            s = pd.to_numeric(g[bm], errors="coerce").dropna()
            if len(s) == 0:
                continue
            stat = {"n": int(len(s)), "mean": round(float(s.mean()), 3),
                    "median": round(float(s.median()), 3), "sd": round(float(s.std(ddof=1)), 3)}
            if bm == "amyloid_centiloid":
                stat["pct_amyloid_positive"] = round(float((s > AMYLOID_POSITIVITY_CENTILOID).mean()*100), 1)
            entry["biomarkers"][bm] = stat
        summary["groups"][label] = entry
    return summary


# --------------------------------------------------------------------------- #
# 2. compute_subject_slopes  (retained for the mixed-model FALLBACK path)
# --------------------------------------------------------------------------- #
def compute_subject_slopes(df: pd.DataFrame, biomarker: str, min_visits: int = 2) -> pd.DataFrame:
    """Per-subject OLS slope of a biomarker over time (units per YEAR).

    Used only when the mixed-effects model cannot be fit. Subjects with fewer than
    `min_visits` non-missing measurements are skipped.
    """
    rows = []
    for sid, sub in df.groupby("subject_id"):
        s = sub.dropna(subset=[biomarker])
        if s["visit_month"].nunique() < min_visits:
            continue
        x = s["visit_month"].to_numpy(float) / 12.0
        y = pd.to_numeric(s[biomarker], errors="coerce").to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append({
            "subject_id": sid,
            "apoe_e4_count": int(sub["apoe_e4_count"].iloc[0]),
            "trem2_variant_status": sub["trem2_variant_status"].iloc[0],
            "slope": slope, "n_visits": int(s["visit_month"].nunique()),
            "span_years": round(float(x.max() - x.min()), 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. cross-sectional (baseline) + dose-linearity check + adjusted level effect
# --------------------------------------------------------------------------- #
def _dose_linearity(base: pd.DataFrame, biomarker: str) -> dict:
    """Lack-of-fit check for the linear ε4-dose assumption.

    Compares a linear-in-allele-count model to a saturated factor model
    (allele coded categorically) via an extra-sum-of-squares F-test. A small
    nonlinearity_p means the equal-increment-per-allele assumption is suspect.
    Also returns the raw per-dose group means so any deviation is visible.
    """
    d = base.dropna(subset=[biomarker, "apoe_e4_count"]).copy()
    d["_a"] = pd.to_numeric(d["apoe_e4_count"], errors="coerce")
    d["_y"] = pd.to_numeric(d[biomarker], errors="coerce")
    d = d.dropna(subset=["_a", "_y"])
    levels = sorted(int(v) for v in d["_a"].unique())
    group_means = {lv: round(float(d.loc[d["_a"] == lv, "_y"].mean()), 3) for lv in levels}
    result = {"levels": levels, "group_means": group_means}
    n = len(d)
    if len(levels) < 3 or n <= len(levels) + 1:
        result["note"] = "Insufficient dose levels/observations to test nonlinearity."
        return result
    y = d["_y"].to_numpy(float)
    # linear-in-dose model
    Xlin = np.column_stack([np.ones(n), d["_a"].to_numpy(float)])
    beta_lin = np.linalg.pinv(Xlin.T @ Xlin) @ Xlin.T @ y
    rss_lin = float(np.sum((y - Xlin @ beta_lin) ** 2))
    p_lin = Xlin.shape[1]
    # saturated factor model (one mean per allele level)
    dummies = pd.get_dummies(d["_a"].astype(int), prefix="a")
    Xfac = np.column_stack([np.ones(n), dummies.to_numpy(float)[:, 1:]])
    beta_fac = np.linalg.pinv(Xfac.T @ Xfac) @ Xfac.T @ y
    rss_fac = float(np.sum((y - Xfac @ beta_fac) ** 2))
    p_fac = Xfac.shape[1]
    df1, df2 = p_fac - p_lin, n - p_fac
    if df1 <= 0 or df2 <= 0 or rss_fac <= 0:
        result["note"] = "Nonlinearity test not estimable."
        return result
    f_stat = ((rss_lin - rss_fac) / df1) / (rss_fac / df2)
    result["nonlinearity_F"] = round(float(f_stat), 3)
    result["nonlinearity_p"] = round(float(stats.f.sf(f_stat, df1, df2)), 4)
    result["nonlinear_flag"] = bool(result["nonlinearity_p"] < 0.05)
    return result


def compare_cross_sectional(df: pd.DataFrame, biomarker: str, scheme: str = "e4_carrier") -> dict:
    """Baseline group comparison.

    For e4_carrier/trem2: Welch's test + Cohen's d.
    For e4_dose: linear trend (biomarker ~ ε4 allele count) PLUS a lack-of-fit check.
    """
    base = df[df["visit_month"] == 0]
    if scheme == "e4_dose":
        d = base.dropna(subset=[biomarker, "apoe_e4_count"])
        X = np.column_stack([np.ones(len(d)), d["apoe_e4_count"].to_numpy(float)])
        res = _ols(X, pd.to_numeric(d[biomarker]).to_numpy(float), ["intercept", "per_e4_allele"])
        return {"type": "linear_trend", "biomarker": biomarker,
                "per_e4_allele": res["per_e4_allele"], "n": res["_n"],
                "linearity_check": _dose_linearity(base, biomarker)}
    if scheme == "e4_carrier":
        a_label, b_label = "e4_carrier", "e4_noncarrier"
        grp = (pd.to_numeric(base["apoe_e4_count"], errors="coerce") >= 1)
        a = pd.to_numeric(base.loc[grp, biomarker], errors="coerce")
        b = pd.to_numeric(base.loc[~grp, biomarker], errors="coerce")
    else:  # trem2
        a_label, b_label = "R47H_carrier", "noncarrier"
        a = pd.to_numeric(base.loc[base["trem2_variant_status"] == "R47H_carrier", biomarker], errors="coerce")
        b = pd.to_numeric(base.loc[base["trem2_variant_status"] == "noncarrier", biomarker], errors="coerce")
    return {"type": "group_contrast", "biomarker": biomarker,
            "group_a": a_label, "group_b": b_label, **_welch(a, b)}


def adjusted_effect(df: pd.DataFrame, biomarker: str, scheme: str = "e4_carrier") -> dict:
    """Age/sex-adjusted BASELINE LEVEL effect via multiple regression.

    biomarker ~ genetic_term + age + sex(male=1) at baseline. Distinct from the
    trajectory model, which adjusts the RATE OF CHANGE for the same covariates.
    """
    base = df[df["visit_month"] == 0].copy()
    base["_male"] = (base["sex"].astype(str).str.upper() == "M").astype(float)
    if scheme == "trem2":
        base["_gene"] = (base["trem2_variant_status"] == "R47H_carrier").astype(float)
        gene_name = "trem2_R47H"
    elif scheme == "e4_carrier":
        base["_gene"] = (base["apoe_e4_count"] >= 1).astype(float)
        gene_name = "e4_carrier"
    else:
        base["_gene"] = base["apoe_e4_count"].astype(float)
        gene_name = "per_e4_allele"
    d = base.dropna(subset=[biomarker, "_gene", "age", "_male"])
    if len(d) < 10:
        return {"insufficient_n": True, "n": int(len(d))}
    X = np.column_stack([np.ones(len(d)), d["_gene"], d["age"], d["_male"]])
    y = pd.to_numeric(d[biomarker], errors="coerce").to_numpy(float)
    res = _ols(X, y, ["intercept", gene_name, "age", "sex_male"])
    return {"biomarker": biomarker, "gene_term": gene_name,
            "adjusted_effect": res[gene_name], "age": res["age"], "sex_male": res["sex_male"],
            "n": res["_n"], "rank_deficient": res["_rank_deficient"]}


# --------------------------------------------------------------------------- #
# 4. trajectory: linear mixed-effects model (with graceful fallback)
# --------------------------------------------------------------------------- #
def _prep_longitudinal(df: pd.DataFrame, biomarker: str, scheme: str):
    """Build the modelling frame and return (frame, reference_label, focal_label).

    focal_label is None for the continuous e4_dose scheme.
    """
    d = df.copy()
    d["_y"] = pd.to_numeric(d[biomarker], errors="coerce")
    d["_time_yr"] = pd.to_numeric(d["visit_month"], errors="coerce").astype(float) / 12.0
    d["_male"] = (d["sex"].astype(str).str.upper() == "M").astype(float)

    # baseline age per subject (constant covariate; avoids collinearity with time)
    base_age = d[d["visit_month"] == 0].groupby("subject_id")["age"].first()
    d["_base_age"] = d["subject_id"].map(base_age)
    missing = d["_base_age"].isna()
    if missing.any():  # subject with no month-0 visit -> use earliest recorded age
        first_age = d.sort_values("visit_month").groupby("subject_id")["age"].first()
        d.loc[missing, "_base_age"] = d.loc[missing, "subject_id"].map(first_age)

    if scheme == "e4_dose":
        d["_gene"] = pd.to_numeric(d["apoe_e4_count"], errors="coerce")
        ref, focal = "e4=0", None
    elif scheme == "e4_carrier":
        d["_gene"] = (pd.to_numeric(d["apoe_e4_count"], errors="coerce") >= 1).astype(float)
        ref, focal = "e4_noncarrier", "e4_carrier"
    else:
        d["_gene"] = (d["trem2_variant_status"] == "R47H_carrier").astype(float)
        ref, focal = "noncarrier", "R47H_carrier"

    d = d.dropna(subset=["_y", "_time_yr", "_gene", "_base_age", "_male"])
    return d, ref, focal


def _mixed_trajectory(df: pd.DataFrame, biomarker: str, scheme: str) -> Optional[dict]:
    """Fit y ~ time * gene + baseline_age + sex + (time | subject).

    Returns a result dict, or None if a mixed model could not be fit (caller then
    falls back to the two-stage per-subject-slope contrast).
    """
    if not _HAS_SM:
        return None
    d, ref, focal = _prep_longitudinal(df, biomarker, scheme)

    visit_counts = d.groupby("subject_id")["_time_yr"].nunique()
    n_long = int((visit_counts >= 2).sum())
    if n_long < 5 or d["subject_id"].nunique() < 5:
        return None  # not enough longitudinal information for a random-slope model

    formula = "_y ~ _time_yr * _gene + _base_age + _male"
    inter = "_time_yr:_gene"
    mdf = None
    model_kind = None
    for re_formula, kind in (("~_time_yr", "random_slope"), ("~1", "random_intercept")):
        try:
            fit = smf.mixedlm(formula, d, groups=d["subject_id"],
                              re_formula=re_formula).fit(reml=True, method="lbfgs", maxiter=200)
        except Exception:
            continue
        if getattr(fit, "converged", False) and inter in fit.params.index:
            mdf, model_kind = fit, kind
            break
        if mdf is None and inter in fit.params.index:  # keep a non-converged fit as last resort
            mdf, model_kind = fit, kind + "_noconv"
    if mdf is None:
        return None

    ci = mdf.conf_int()
    slope_ref = float(mdf.params["_time_yr"])
    diff_coef = float(mdf.params[inter])
    diff_ci = [float(ci.loc[inter, 0]), float(ci.loc[inter, 1])]
    diff_p = float(mdf.pvalues[inter])

    # per-group subject counts (for small-n cautions)
    if focal is None:  # dose scheme
        n_ref = n_focal = None
    else:
        n_focal = int(d.loc[d["_gene"] == 1, "subject_id"].nunique())
        n_ref = int(d.loc[d["_gene"] == 0, "subject_id"].nunique())

    return {
        "type": "mixedlm_slope",
        "biomarker": biomarker,
        "scheme": scheme,
        "model": model_kind,
        "converged": bool(getattr(mdf, "converged", False)),
        "reference_group": ref,
        "focal_group": focal,
        "slope_reference": round(slope_ref, 4),
        "slope_focal": round(slope_ref + diff_coef, 4) if focal else None,
        "differential_slope": {
            "coef": diff_coef,
            "ci95": diff_ci,
            "p_value": diff_p,
        },
        "n_subjects": int(d["subject_id"].nunique()),
        "n_subjects_ref": n_ref,
        "n_subjects_focal": n_focal,
        "n_obs": int(len(d)),
    }


def _twostage_trajectory(df: pd.DataFrame, biomarker: str, scheme: str) -> dict:
    """Fallback: per-subject OLS slopes then a between-group contrast/trend."""
    slopes = compute_subject_slopes(df, biomarker)
    if scheme == "e4_dose":
        by = slopes.groupby("apoe_e4_count")["slope"].agg(["count", "mean", "std"]).round(3)
        d = slopes.dropna(subset=["slope"])
        X = np.column_stack([np.ones(len(d)), d["apoe_e4_count"].to_numpy(float)])
        res = _ols(X, d["slope"].to_numpy(float), ["intercept", "per_e4_allele"])
        return {"type": "slope_trend", "biomarker": biomarker, "model": "twostage_fallback",
                "group_slopes": by.reset_index().to_dict("records"),
                "per_e4_allele_slope": res["per_e4_allele"], "n_subjects": res["_n"]}
    if scheme == "e4_carrier":
        a = slopes.loc[slopes["apoe_e4_count"] >= 1, "slope"]
        b = slopes.loc[slopes["apoe_e4_count"] == 0, "slope"]
        a_label, b_label = "e4_carrier", "e4_noncarrier"
    else:
        a = slopes.loc[slopes["trem2_variant_status"] == "R47H_carrier", "slope"]
        b = slopes.loc[slopes["trem2_variant_status"] == "noncarrier", "slope"]
        a_label, b_label = "R47H_carrier", "noncarrier"
    return {"type": "slope_contrast", "biomarker": biomarker, "model": "twostage_fallback",
            "group_a": a_label, "group_b": b_label,
            "mean_slope_a": round(float(a.mean()), 3) if len(a) else None,
            "mean_slope_b": round(float(b.mean()), 3) if len(b) else None,
            **_welch(a, b)}


def compare_biomarker_trajectories(df: pd.DataFrame, biomarker: str,
                                   scheme: str = "e4_carrier") -> dict:
    """Compare longitudinal change between genotype groups.

    Preferred estimator: linear mixed-effects model (all visits, age/sex-adjusted,
    valid under MAR). Falls back to a two-stage per-subject-slope contrast when a
    mixed model cannot be fit.
    """
    result = _mixed_trajectory(df, biomarker, scheme)
    if result is not None:
        return result
    return _twostage_trajectory(df, biomarker, scheme)


# --------------------------------------------------------------------------- #
# 5. analyze (bundle) + make_hypothesis_card
# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame, biomarker: str, scheme: str = "e4_carrier") -> dict:
    """Run cross-sectional, adjusted-level, and trajectory analyses for one biomarker."""
    return {
        "biomarker": biomarker, "scheme": scheme,
        "n_subjects": int(df["subject_id"].nunique()),
        "is_synthetic": bool(
            ("data_source" in df.columns and (df["data_source"] == "synthetic").any())
            or df["subject_id"].astype(str).str.startswith("SYN").all()),
        "cross_sectional": compare_cross_sectional(df, biomarker, scheme),
        "adjusted": adjusted_effect(df, biomarker, scheme),
        "trajectory": compare_biomarker_trajectories(df, biomarker, scheme),
    }


def _fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _cautions(bundle: dict, group_ns: List[int]) -> List[str]:
    c = ["Observational association, not causal evidence.",
         "Hypothesis-generating only; not for diagnosis or clinical prediction."]
    if bundle.get("is_synthetic"):
        c.insert(0, "Demonstrated on SYNTHETIC data — the effect direction is built "
                    "into the data generator by design and is not evidence about real biology.")
    group_ns = [n for n in group_ns if n is not None]
    if group_ns and min(group_ns) < 20:
        c.append(f"Small group size (min n={min(group_ns)} subjects); effect is exploratory "
                 "and confidence intervals are wide.")
    if bundle["scheme"] == "trem2":
        c.append("TREM2 R47H is rare, so this comparison is typically underpowered.")

    # dose-linearity caution
    lin = bundle.get("cross_sectional", {}).get("linearity_check", {})
    if lin.get("nonlinear_flag"):
        c.append("The ε4 dose–response departs from linearity (lack-of-fit p="
                 f"{lin.get('nonlinearity_p')}); the per-allele slope is an approximation — "
                 "inspect the per-dose group means.")

    # trajectory model caution reflects what was actually fit
    tr = bundle.get("trajectory", {})
    model = tr.get("model", "")
    if model == "random_slope":
        c.append("Longitudinal change estimated with a linear mixed-effects model "
                 "(random intercept + random slope per subject; age/sex-adjusted).")
    elif model == "random_intercept":
        c.append("Random-slope model did not converge; a random-intercept mixed model was "
                 "used, which assumes a common rate of change across subjects.")
    elif model and model.endswith("_noconv"):
        c.append("The mixed-effects model did not fully converge; interpret the longitudinal "
                 "estimate with extra caution.")
    elif model == "twostage_fallback":
        c.append("A mixed-effects model could not be fit; longitudinal change fell back to "
                 "per-subject OLS slopes (equal weighting), which understates uncertainty when "
                 "follow-up is unbalanced.")
    return c


def _pick_citations(bundle: dict) -> List[str]:
    ids = ["jack2024"]
    if bundle["scheme"] in ("e4_dose", "e4_carrier"):
        ids += ["corder1993", "jack2010", "bateman2012"]
    if bundle["scheme"] == "trem2":
        ids += ["jonsson2013"]
    if bundle["biomarker"] == "amyloid_centiloid":
        ids += ["oasis3_2019"]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(CITATIONS[i])
    return out


def make_hypothesis_card(bundle: dict) -> dict:
    """Assemble a Hypothesis / Evidence / Caution / Citations card from an analyze() bundle."""
    bm = bundle["biomarker"]
    meta = BIOMARKER_META.get(bm, {"label": bm, "unit": "", "higher_is_worse": True})
    cs, adj, tr = bundle["cross_sectional"], bundle["adjusted"], bundle["trajectory"]
    evidence: List[str] = []
    group_ns: List[int] = []

    # --- cross-sectional (baseline level) ---
    if cs.get("type") == "group_contrast" and not cs.get("insufficient_n"):
        group_ns += [cs["n_a"], cs["n_b"]]
        direction = "higher" if cs["mean_diff"] > 0 else "lower"
        evidence.append(
            f"At baseline, {cs['group_a']} had {direction} {meta['label']} than {cs['group_b']} "
            f"({cs['mean_a']:.2f} vs {cs['mean_b']:.2f} {meta['unit']}; "
            f"Δ={cs['mean_diff']:.2f}, 95% CI [{cs['diff_ci95'][0]:.2f}, {cs['diff_ci95'][1]:.2f}], "
            f"{_fmt_p(cs['p_value'])}, Cohen's d={cs['cohens_d']:.2f}).")
    elif cs.get("type") == "linear_trend":
        e = cs["per_e4_allele"]
        line = (f"Baseline {meta['label']} changed by {e['coef']:.2f} {meta['unit']} per ε4 allele "
                f"(95% CI [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}], {_fmt_p(e['p_value'])}).")
        lin = cs.get("linearity_check", {})
        if lin.get("group_means"):
            means = ", ".join(f"{k}:{v}" for k, v in lin["group_means"].items())
            line += f" Per-dose means ({meta['unit']}): {means}."
        evidence.append(line)

    # --- adjusted baseline level ---
    if adj and not adj.get("insufficient_n"):
        ae = adj["adjusted_effect"]
        evidence.append(
            f"After adjusting for age and sex, the {adj['gene_term']} effect on baseline "
            f"{meta['label']} was {ae['coef']:.2f} {meta['unit']} "
            f"(95% CI [{ae['ci95'][0]:.2f}, {ae['ci95'][1]:.2f}], {_fmt_p(ae['p_value'])}).")

    # --- trajectory (rate of change) ---
    if tr.get("type") == "mixedlm_slope":
        ds = tr["differential_slope"]
        group_ns += [tr.get("n_subjects_ref"), tr.get("n_subjects_focal")]
        if tr.get("focal_group"):
            evidence.append(
                f"In a mixed-effects model (adjusted for baseline age and sex), {meta['label']} "
                f"changed by {tr['slope_focal']} {meta['unit']}/yr in {tr['focal_group']} vs "
                f"{tr['slope_reference']} in {tr['reference_group']}; differential rate "
                f"Δ={ds['coef']:.3f} {meta['unit']}/yr "
                f"(95% CI [{ds['ci95'][0]:.3f}, {ds['ci95'][1]:.3f}], {_fmt_p(ds['p_value'])}).")
        else:
            evidence.append(
                f"In a mixed-effects model (adjusted for baseline age and sex), the annual change "
                f"in {meta['label']} shifted by {ds['coef']:.3f} {meta['unit']}/yr per ε4 allele "
                f"(95% CI [{ds['ci95'][0]:.3f}, {ds['ci95'][1]:.3f}], {_fmt_p(ds['p_value'])}).")
    elif tr.get("type") == "slope_contrast" and not tr.get("insufficient_n"):
        group_ns += [tr.get("n_a", 0), tr.get("n_b", 0)]
        evidence.append(
            f"Annual change ({meta['unit']}/yr) was {tr['mean_slope_a']} in {tr['group_a']} "
            f"vs {tr['mean_slope_b']} in {tr['group_b']} "
            f"(Δ={tr['mean_diff']:.3f}, 95% CI [{tr['diff_ci95'][0]:.3f}, {tr['diff_ci95'][1]:.3f}], "
            f"{_fmt_p(tr['p_value'])}, Cohen's d={tr['cohens_d']:.2f}).")
    elif tr.get("type") == "slope_trend":
        e = tr["per_e4_allele_slope"]
        evidence.append(
            f"Annual change in {meta['label']} shifted by {e['coef']:.3f} {meta['unit']}/yr per ε4 "
            f"allele (95% CI [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}], {_fmt_p(e['p_value'])}).")

    gene_label = {"e4_dose": "APOE ε4 dose", "e4_carrier": "APOE ε4 carriers",
                  "trem2": "TREM2 R47H carriers"}[bundle["scheme"]]
    worse = "greater/worse" if meta["higher_is_worse"] else "lower/greater-decline in"
    hypothesis = (f"{gene_label} show {worse} {meta['label']} and/or faster longitudinal change "
                  f"than the comparison group.")

    return {
        "hypothesis": hypothesis,
        "evidence": evidence or ["Insufficient data to compute evidence for this comparison."],
        "caution": _cautions(bundle, group_ns),
        "citations": _pick_citations(bundle),
        "provenance": {"biomarker": bm, "scheme": bundle["scheme"],
                       "n_subjects": bundle["n_subjects"], "synthetic": bundle["is_synthetic"],
                       "trajectory_model": bundle["trajectory"].get("model")},
    }


def run_hypothesis(df: pd.DataFrame, biomarker: str, scheme: str = "e4_carrier") -> dict:
    """One-call: run the analyses and return a formatted hypothesis card."""
    return make_hypothesis_card(analyze(df, biomarker, scheme))


def render_card_text(card: dict) -> str:
    """Human-readable rendering matching the card format."""
    lines = ["Hypothesis:", "  " + card["hypothesis"], "", "Evidence:"]
    lines += [f"  - {e}" for e in card["evidence"]]
    lines += ["", "Caution:"] + [f"  - {c}" for c in card["caution"]]
    lines += ["", "References:"] + [f"  - {r}" for r in card["citations"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. rank_biomarkers  (now with Benjamini-Hochberg FDR)
# --------------------------------------------------------------------------- #
def rank_biomarkers(df: pd.DataFrame, scheme: str = "e4_carrier") -> dict:
    """Rank analyzable biomarkers by the magnitude of their genotype effect.

    For carrier/TREM2 schemes the metric is |Cohen's d| of the baseline group contrast;
    for e4_dose it is the standardized per-allele slope (|beta| / baseline SD). Because
    several biomarkers are tested at once, Benjamini-Hochberg FDR q-values are reported
    alongside the raw p-values.
    """
    base = df[df["visit_month"] == 0]
    ranked = []
    for bm in BIOMARKER_META:
        if bm not in df.columns or pd.to_numeric(df[bm], errors="coerce").notna().sum() == 0:
            continue
        cs = compare_cross_sectional(df, bm, scheme)
        abs_effect = direction = p = None
        if cs.get("type") == "group_contrast" and not cs.get("insufficient_n"):
            d = cs.get("cohens_d")
            if d is not None and not np.isnan(d):
                abs_effect = abs(d)
                direction = "higher_in_carrier" if cs["mean_diff"] > 0 else "lower_in_carrier"
                p = cs["p_value"]
        elif cs.get("type") == "linear_trend":
            beta = cs["per_e4_allele"]["coef"]
            sd = pd.to_numeric(base[bm], errors="coerce").std(ddof=1)
            if sd and sd > 0:
                abs_effect = abs(beta / sd)
                direction = "increases_with_dose" if beta > 0 else "decreases_with_dose"
                p = cs["per_e4_allele"]["p_value"]
        if abs_effect is not None:
            ranked.append({"biomarker": bm, "label": BIOMARKER_META[bm]["label"],
                           "abs_effect": round(float(abs_effect), 3),
                           "direction": direction, "p_value": round(float(p), 4),
                           "_p_raw": float(p)})

    # FDR across the biomarkers tested
    if ranked:
        qvals = _bh_fdr([r["_p_raw"] for r in ranked])
        for r, q in zip(ranked, qvals):
            r["q_value_bh"] = round(float(q), 4)
            r["significant_fdr05"] = bool(q < 0.05)
            del r["_p_raw"]

    ranked.sort(key=lambda r: r["abs_effect"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    metric = "abs_std_beta_per_allele" if scheme == "e4_dose" else "abs_cohens_d"
    return {"scheme": scheme, "metric": metric if ranked else None,
            "correction": "benjamini_hochberg", "ranking": ranked}


if __name__ == "__main__":
    from skills.ingestion_skills import load_dataset, clean_biomarkers
    _csv = _os.path.join(_ROOT, "data", "synthetic_adni_style.csv")
    df, _ = clean_biomarkers(load_dataset(_csv))
    for bm, sch in [("amyloid_centiloid", "e4_dose"),
                    ("hippocampal_volume", "e4_carrier"),
                    ("mmse", "e4_carrier"),
                    ("hippocampal_volume", "trem2")]:
        print("=" * 78)
        print(f"[{sch}]  {bm}")
        print(render_card_text(run_hypothesis(df, bm, sch)))
