"""
ingestion.py — Day 2 ingestion layer for the Alzheimer's Biomarker Hypothesis Agent
====================================================================================

Five agent-ready tools (clean signatures, typed, JSON-friendly returns so they can
be registered directly as Google ADK FunctionTools, but also usable standalone in a
notebook):

    load_dataset(path)            -> pandas.DataFrame
    validate_schema(df)           -> dict   (structured report the agent reasons over)
    clean_biomarkers(df)          -> (DataFrame, dict)   cleaned data + transparent log
    group_by_genotype(df, scheme) -> dict of {label: DataFrame}  (+ describe_groups)
    annotate_variant(query)       -> dict   (curated KB + cached/live API enrichment)

Design notes
------------
* annotate_variant ALWAYS returns a correct curated baseline from VARIANT_KB, then
  *optionally* enriches it from MyVariant.info / Ensembl / GWAS Catalog. Live calls
  are cached to disk, so a flaky network never breaks the demo (Day 2 requirement).
* APOE ε-alleles are a HAPLOTYPE of two SNPs (rs429358 + rs7412), not a single
  variant. `apoe_e4_count` is the derived ε4 dosage. The KB encodes this correctly.

To register as ADK tools (in your agent notebook):
    from google.adk.tools import FunctionTool
    tools = [FunctionTool(load_dataset), FunctionTool(validate_schema), ...]
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Schema specification
# --------------------------------------------------------------------------- #
# required column -> (pandas kind, optional (min, max) plausibility range)
SCHEMA: Dict[str, Tuple[str, Union[Tuple[float, float], None]]] = {
    "subject_id":            ("str",   None),
    "visit_month":           ("int",   (0, 240)),
    "diagnosis":             ("cat",   None),
    "age":                   ("float", (40, 110)),
    "sex":                   ("cat",   None),
    "apoe_e4_count":         ("int",   (0, 2)),
    "trem2_variant_status":  ("str",   None),
    "hippocampal_volume":    ("float", (1500, 12000)),   # mm^3, bilateral
    "entorhinal_thickness":  ("float", (1.0, 6.0)),       # mm
    "amyloid_centiloid":     ("float", (-50, 250)),       # Centiloid
    "tau_suvr":              ("float", (0.5, 4.0)),        # SUVR
    "fdg_suvr":              ("float", (0.5, 2.5)),        # SUVR
    "mmse":                  ("float", (0, 30)),
    "cdrsb":                 ("float", (0, 18)),
}
NUMERIC_COLS = [c for c, (k, _) in SCHEMA.items() if k in ("int", "float")]
CATEGORICAL = {"diagnosis": {"CN", "MCI", "AD"}, "sex": {"M", "F"}}
AMYLOID_POSITIVITY_CENTILOID = 16.4   # OASIS-3 (LaMontagne et al., 2019)


# --------------------------------------------------------------------------- #
# 1. load_dataset
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    """Load a subject-level biomarker CSV/TSV into a DataFrame.

    Args:
        path: Path to a .csv or .tsv file (long format, one row per subject-visit).

    Returns:
        The loaded DataFrame with whitespace-stripped column names.
    """
    sep = "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# 2. validate_schema
# --------------------------------------------------------------------------- #
def validate_schema(df: pd.DataFrame) -> dict:
    """Check a DataFrame against the expected biomarker schema.

    Reports missing/unexpected columns, type problems, out-of-range values, and
    per-column missingness. Returns a structured dict the agent can branch on
    (e.g. refuse an analysis whose required column is absent).

    Args:
        df: The DataFrame to validate.

    Returns:
        dict with keys: ok, n_rows, n_subjects, missing_required,
        unexpected_columns, dtype_issues, range_violations, missingness, messages.
    """
    report: dict = {
        "ok": True,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else 0,
        "missing_required": [],
        "unexpected_columns": [],
        "dtype_issues": [],
        "range_violations": {},
        "missingness": {},
        "messages": [],
    }

    known = set(SCHEMA) | {"apoe_genotype", "data_source"}
    report["missing_required"] = [c for c in SCHEMA if c not in df.columns]
    report["unexpected_columns"] = [c for c in df.columns if c not in known]

    for col, (kind, rng) in SCHEMA.items():
        if col not in df.columns:
            continue
        s = df[col]
        report["missingness"][col] = round(float(s.isna().mean()) * 100, 2)
        if kind in ("int", "float"):
            coerced = pd.to_numeric(s, errors="coerce")
            n_bad = int(coerced.isna().sum() - s.isna().sum())
            if n_bad > 0:
                report["dtype_issues"].append(
                    f"{col}: {n_bad} non-numeric value(s) where numeric expected")
            if rng is not None:
                lo, hi = rng
                viol = int(((coerced < lo) | (coerced > hi)).sum())
                if viol:
                    report["range_violations"][col] = viol
        if col in CATEGORICAL:
            allowed = CATEGORICAL[col]
            bad = set(s.dropna().astype(str).str.upper().unique()) - allowed
            if bad:
                report["dtype_issues"].append(
                    f"{col}: unexpected categories {sorted(bad)} (allowed {sorted(allowed)})")

    report["ok"] = not (report["missing_required"]
                        or report["dtype_issues"]
                        or report["range_violations"])
    if report["missing_required"]:
        report["messages"].append(
            "Missing required columns: " + ", ".join(report["missing_required"]))
    if report["range_violations"]:
        report["messages"].append(
            "Out-of-range values detected; run clean_biomarkers() to quarantine them.")
    if report["ok"]:
        report["messages"].append("Schema OK.")
    return report


# --------------------------------------------------------------------------- #
# 3. clean_biomarkers
# --------------------------------------------------------------------------- #
_TREM2_CARRIER = {"r47h", "carrier", "positive", "pos", "rs75932628", "1", "true"}
_TREM2_NONCARRIER = {"wt", "noncarrier", "non-carrier", "negative", "neg", "0",
                     "false", "wildtype", "wild-type"}


def clean_biomarkers(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Coerce types, normalize categories, quarantine impossible values, de-duplicate.

    Cleaning is transparent: nothing is silently fabricated. Out-of-range numeric
    values are set to NaN (quarantined, not clipped) and counted in the log so the
    downstream stats stay honest.

    Args:
        df: Raw DataFrame (typically straight from load_dataset).

    Returns:
        (cleaned_df, log) where log records every transformation applied.
    """
    out = df.copy()
    log: dict = {"coerced_numeric": {}, "quarantined_out_of_range": {},
                 "normalized_categoricals": {}, "dropped_duplicate_visits": 0,
                 "messages": []}

    # numeric coercion
    for col in NUMERIC_COLS:
        if col in out.columns:
            before_na = out[col].isna().sum()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            made_na = int(out[col].isna().sum() - before_na)
            if made_na:
                log["coerced_numeric"][col] = made_na

    # categorical normalization
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip().str.upper().replace(
            {"MALE": "M", "FEMALE": "F"})
        log["normalized_categoricals"]["sex"] = sorted(out["sex"].dropna().unique())
    if "diagnosis" in out.columns:
        out["diagnosis"] = out["diagnosis"].astype(str).str.strip().str.upper()
        log["normalized_categoricals"]["diagnosis"] = sorted(out["diagnosis"].dropna().unique())
    if "trem2_variant_status" in out.columns:
        def _norm_trem2(v):
            t = str(v).strip().lower()
            if t in _TREM2_CARRIER:
                return "R47H_carrier"
            if t in _TREM2_NONCARRIER:
                return "noncarrier"
            return "unknown"
        out["trem2_variant_status"] = out["trem2_variant_status"].map(_norm_trem2)
        log["normalized_categoricals"]["trem2_variant_status"] = \
            sorted(out["trem2_variant_status"].dropna().unique())

    # quarantine out-of-range numeric values -> NaN
    for col, (kind, rng) in SCHEMA.items():
        if kind in ("int", "float") and rng is not None and col in out.columns:
            lo, hi = rng
            mask = (out[col] < lo) | (out[col] > hi)
            n = int(mask.sum())
            if n:
                out.loc[mask, col] = np.nan
                log["quarantined_out_of_range"][col] = n

    # de-duplicate subject-visit rows
    if {"subject_id", "visit_month"} <= set(out.columns):
        before = len(out)
        out = out.drop_duplicates(subset=["subject_id", "visit_month"], keep="first")
        log["dropped_duplicate_visits"] = int(before - len(out))
        out = out.sort_values(["subject_id", "visit_month"]).reset_index(drop=True)

    if not any([log["coerced_numeric"], log["quarantined_out_of_range"],
                log["dropped_duplicate_visits"]]):
        log["messages"].append("No corrections needed; data already clean.")
    else:
        log["messages"].append("Cleaning applied; see log fields for details.")
    return out, log


# --------------------------------------------------------------------------- #
# 4. group_by_genotype
# --------------------------------------------------------------------------- #
_SCHEMES = ("e4_dose", "e4_carrier", "trem2")


def group_by_genotype(df: pd.DataFrame, scheme: str = "e4_carrier"
                      ) -> Dict[str, pd.DataFrame]:
    """Split the cohort into genotype groups for comparison.

    Args:
        df: Cleaned DataFrame.
        scheme: 'e4_dose' (0/1/2 alleles), 'e4_carrier' (carrier vs non-carrier),
                or 'trem2' (R47H carrier vs non-carrier).

    Returns:
        Ordered dict {group_label: sub_DataFrame}.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}")
    groups: Dict[str, pd.DataFrame] = {}
    if scheme == "e4_dose":
        for k in (0, 1, 2):
            groups[f"e4={k}"] = df[df["apoe_e4_count"] == k]
    elif scheme == "e4_carrier":
        groups["e4_carrier"] = df[df["apoe_e4_count"] >= 1]
        groups["e4_noncarrier"] = df[df["apoe_e4_count"] == 0]
    else:  # trem2
        groups["TREM2_R47H_carrier"] = df[df["trem2_variant_status"] == "R47H_carrier"]
        groups["TREM2_noncarrier"] = df[df["trem2_variant_status"] == "noncarrier"]
    return groups


def describe_groups(df: pd.DataFrame, scheme: str = "e4_carrier") -> dict:
    """JSON-serializable summary of genotype groups (sizes + power flags).

    Use this as the agent-facing tool; the LLM reads sizes and decides whether a
    comparison is adequately powered before requesting statistics.
    """
    groups = group_by_genotype(df, scheme)
    summary = {"scheme": scheme, "groups": {}}
    for label, g in groups.items():
        n_subj = int(g["subject_id"].nunique())
        summary["groups"][label] = {
            "n_subjects": n_subj,
            "n_visits": int(len(g)),
            "underpowered": n_subj < 20,
        }
    summary["warning"] = (
        "One or more groups has <20 subjects; treat effects as exploratory."
        if any(v["underpowered"] for v in summary["groups"].values()) else None)
    return summary


# --------------------------------------------------------------------------- #
# 5. annotate_variant  (curated KB + cached/live API enrichment)
# --------------------------------------------------------------------------- #
# Curated, always-available baseline. Encodes the APOE haplotype nuance correctly.
VARIANT_KB: Dict[str, dict] = {
    "rs429358": {
        "rsid": "rs429358", "gene": "APOE", "chrom": "19",
        "role": "One of the two SNPs defining APOE ε2/ε3/ε4 haplotypes",
        "note": ("ε4 corresponds to the rs429358-C allele (with rs7412-C). "
                 "ε4 is the strongest common genetic risk factor for late-onset AD; "
                 "dose-dependent increase in risk and earlier age at onset."),
        "ad_relevance": "risk_modifier",
    },
    "rs7412": {
        "rsid": "rs7412", "gene": "APOE", "chrom": "19",
        "role": "Second SNP defining APOE ε2/ε3/ε4 haplotypes",
        "note": ("ε2 = rs429358-T + rs7412-T (protective); ε3 = rs429358-T + rs7412-C; "
                 "ε4 = rs429358-C + rs7412-C."),
        "ad_relevance": "risk_modifier",
    },
    "rs75932628": {
        "rsid": "rs75932628", "gene": "TREM2", "chrom": "6",
        "protein_change": "p.Arg47His (R47H)",
        "role": "Rare missense variant in TREM2",
        "note": ("R47H increases late-onset AD risk with an effect size on the order "
                 "of a single APOE ε4 allele; affects microglial function. Rare, so "
                 "cohort carrier counts are usually small and underpowered."),
        "ad_relevance": "risk_modifier",
    },
}
# friendly aliases -> canonical rsid(s)
_ALIASES = {
    "apoe": ["rs429358", "rs7412"], "apoe e4": ["rs429358", "rs7412"],
    "apoe4": ["rs429358", "rs7412"], "e4": ["rs429358", "rs7412"],
    "trem2": ["rs75932628"], "r47h": ["rs75932628"], "trem2 r47h": ["rs75932628"],
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".variant_cache")


def _cache_path(source: str, rsid: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{source}__{rsid}.json")


def _http_get_json(url: str, timeout: float = 6.0):
    """GET JSON with a short timeout. Returns parsed JSON or None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "biomarker-agent/1.0",
                                                    "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_source(source: str, rsid: str, use_network: bool) -> dict:
    """Return cached enrichment for (source, rsid); fetch live if allowed and uncached."""
    cp = _cache_path(source, rsid)
    if os.path.exists(cp):
        with open(cp) as f:
            return {"status": "cache", "data": json.load(f)}
    if not use_network:
        return {"status": "unavailable_offline", "data": None}

    urls = {
        "myvariant": f"https://myvariant.info/v1/query?q={rsid}&size=1",
        "ensembl": f"https://rest.ensembl.org/variation/human/{rsid}?content-type=application/json",
        "gwas_catalog": ("https://www.ebi.ac.uk/gwas/rest/api/"
                         f"singleNucleotidePolymorphisms/{rsid}/associations"),
    }
    data = _http_get_json(urls[source])
    if data is not None:
        try:
            with open(cp, "w") as f:
                json.dump(data, f)
        except OSError:
            pass
        return {"status": "live", "data": data}
    return {"status": "fetch_failed", "data": None}


def annotate_variant(query: str, use_network: bool = True,
                     sources: Union[List[str], None] = None) -> dict:
    """Annotate a variant/gene with curated facts plus optional live API enrichment.

    Always returns a correct curated baseline (VARIANT_KB). If `use_network` and the
    annotation is not cached, it queries MyVariant.info / Ensembl / GWAS Catalog and
    caches the result so subsequent (and offline) calls succeed.

    Args:
        query: rsID ('rs429358'), gene ('APOE', 'TREM2'), or alias ('APOE e4', 'R47H').
        use_network: If False, return only curated + previously-cached data.
        sources: Subset of ['myvariant', 'ensembl', 'gwas_catalog']; default all.

    Returns:
        dict: {query, resolved_rsids, variants:[{...curated..., enrichment:{source:status}}],
               disclaimer}.
    """
    sources = sources or ["myvariant", "ensembl", "gwas_catalog"]
    q = query.strip().lower()
    if q in _ALIASES:
        rsids = _ALIASES[q]
    elif q.startswith("rs"):
        rsids = [q]
    else:
        rsids = [r for r, v in VARIANT_KB.items() if v["gene"].lower() == q] or [query]

    variants = []
    for rsid in rsids:
        entry = dict(VARIANT_KB.get(rsid, {"rsid": rsid, "note": "Not in curated KB."}))
        enrichment = {}
        for src in sources:
            res = _fetch_source(src, rsid, use_network)
            enrichment[src] = res["status"]
        entry["enrichment_status"] = enrichment
        variants.append(entry)

    return {
        "query": query,
        "resolved_rsids": rsids,
        "variants": variants,
        "disclaimer": ("APOE ε-alleles are a two-SNP haplotype (rs429358 + rs7412); "
                       "apoe_e4_count is the derived ε4 dosage, not a single genotype. "
                       "Annotations are informational, not clinical advice."),
    }


# --------------------------------------------------------------------------- #
# Convenience: full ingestion pass (load -> validate -> clean -> describe)
# --------------------------------------------------------------------------- #
def run_ingestion(path: str, scheme: str = "e4_dose") -> dict:
    """Run the whole ingestion layer end to end and return a compact report dict."""
    df = load_dataset(path)
    pre = validate_schema(df)
    clean, log = clean_biomarkers(df)
    post = validate_schema(clean)
    return {
        "loaded_rows": int(len(df)),
        "validation_before": pre,
        "cleaning_log": log,
        "validation_after": post,
        "group_summary": describe_groups(clean, scheme),
        "clean_df": clean,   # kept for chaining; drop before JSON-serializing for the LLM
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic_adni_style.csv")
    args = ap.parse_args()
    rep = run_ingestion(args.csv)
    rep.pop("clean_df")
    print(json.dumps(rep, indent=2, default=str))
    print("\nannotate_variant('APOE e4') [offline]:")
    print(json.dumps(annotate_variant("APOE e4", use_network=False), indent=2))
