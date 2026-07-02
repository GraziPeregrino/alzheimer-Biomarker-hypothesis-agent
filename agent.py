"""
agent.py — Day 4 agent layer for the Alzheimer's Biomarker Hypothesis Agent
===========================================================================

Wires the tested ingestion + stats tools to a Google ADK / Gemini agent. The LLM
parses a free-text research question, inspects the dataset, chooses a biomarker and
a genotype scheme, calls the tools, and returns a citation-backed hypothesis card.

Two layers, same tools:
  * ADK agent (root_agent)  — the LLM does the planning; runs in the Kaggle notebook
                              via `adk run` / `adk web` or InMemoryRunner (needs a
                              Gemini API key).
  * HypothesisAgentCore     — a deterministic orchestrator using the SAME tools, with
                              a keyword intent parser instead of the LLM. Runs offline,
                              serves as the demo fallback, and is what the eval harness
                              (Day 4) scores.

SAFETY: this is a research / hypothesis-generation tool. `check_query_safety` refuses
individual-level clinical questions (diagnosis, prognosis, treatment, personal risk).
The agent is instructed to call it first; the core calls it automatically.

Follow-up context (IMPORTANT): the Gemini `root_agent` resolves references like
"now show it by dose" from ADK's own session history, NOT from this module's tracker.
Drive it through a session-aware runner with a STABLE session_id so prior turns are in
context, e.g.:

    from google.adk.runners import InMemoryRunner
    from google.genai import types
    runner = InMemoryRunner(agent=root_agent, app_name="alz")
    session = await runner.session_service.create_session(app_name="alz", user_id="u1")
    async for ev in runner.run_async(
            user_id="u1", session_id=session.id,
            new_message=types.Content(role="user",
                parts=[types.Part(text="Do e4 carriers show more amyloid?")])):
        ...  # send the follow-up "now by dose" on the SAME session.id
`adk web` maintains sessions for you. The `self._last` tracker below is ONLY for the
deterministic core; do not rely on it for the LLM agent.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import pandas as pd

from ingestion import load_dataset, clean_biomarkers, validate_schema, describe_groups
from stats_engine import (BIOMARKER_META, analyze, make_hypothesis_card,
                          render_card_text, compute_group_summary, rank_biomarkers)


# --------------------------------------------------------------------------- #
# In-memory dataset store (keeps big DataFrames OUT of the LLM context;
# the agent passes a small dataset_id string instead of the data itself).
# --------------------------------------------------------------------------- #
_STORE: Dict[str, pd.DataFrame] = {}


def _persist(dataset_id: str, df: pd.DataFrame) -> None:
    """Write the dataset to disk so a restarted kernel or out-of-order eval can restore it."""
    try:
        df.to_parquet(f"{dataset_id}.parquet")
    except Exception:  # pyarrow/fastparquet unavailable -> pickle fallback
        try:
            df.to_pickle(f"{dataset_id}.pkl")
        except Exception:
            pass


def _get_df(dataset_id: str) -> Optional[pd.DataFrame]:
    """Return the dataset from memory, transparently rehydrating from disk on a miss."""
    if dataset_id in _STORE:
        return _STORE[dataset_id]
    for path, reader in ((f"{dataset_id}.parquet", pd.read_parquet),
                         (f"{dataset_id}.pkl", pd.read_pickle)):
        if os.path.exists(path):
            try:
                _STORE[dataset_id] = reader(path)
                return _STORE[dataset_id]
            except Exception:
                continue
    return None


def _analyzable_biomarkers(df: pd.DataFrame) -> list:
    return [b for b in BIOMARKER_META if b in df.columns
            and pd.to_numeric(df[b], errors="coerce").notna().any()]


# --------------------------------------------------------------------------- #
# TOOLS (ADK auto-wraps these; each returns a dict with a 'status' key)
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = "synthetic_adni_style.csv"


def _resolve_csv(csv_path: str) -> str:
    """Return a usable path to the CSV. If it isn't found as given, fall back to
    the bundled dataset next to this module — so the tool works regardless of the
    process working directory or a made-up filename from the model."""
    if os.path.isfile(csv_path):
        return csv_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, os.path.basename(csv_path))
    if os.path.isfile(candidate):
        return candidate
    bundled = os.path.join(here, DEFAULT_DATASET)
    if os.path.isfile(bundled):
        return bundled
    return csv_path  # let load_dataset raise a clear error


def load_biomarker_dataset(csv_path: str = DEFAULT_DATASET, dataset_id: str = "current") -> dict:
    """Load, clean, and validate a subject-level biomarker CSV, then register it.

    Args:
        csv_path: Path to the CSV (long format, one row per subject-visit).
            Defaults to the bundled synthetic dataset; call with no path to use it.
        dataset_id: Handle to reference this dataset in later tool calls.

    Returns:
        dict with status, dataset_id, n_subjects, n_rows, schema_ok,
        available_biomarkers, and issues (if any).
    """
    try:
        raw = load_dataset(_resolve_csv(csv_path))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"Could not load '{csv_path}': {e}"}
    clean, log = clean_biomarkers(raw)
    rep = validate_schema(clean)
    _STORE[dataset_id] = clean
    _persist(dataset_id, clean)
    return {
        "status": "success",
        "dataset_id": dataset_id,
        "n_subjects": rep["n_subjects"],
        "n_rows": rep["n_rows"],
        "schema_ok": rep["ok"],
        "available_biomarkers": _analyzable_biomarkers(clean),
        "issues": rep["messages"] + log["messages"],
    }


def inspect_dataset(dataset_id: str = "current") -> dict:
    """Report analyzable biomarkers and genotype-group sizes so the agent can pick a
    valid, adequately-powered comparison before requesting statistics.

    Args:
        dataset_id: Handle returned by load_biomarker_dataset.
    """
    df = _get_df(dataset_id)
    if df is None:
        return {"status": "error", "message": "Unknown dataset_id; load a dataset first."}
    return {
        "status": "success",
        "biomarkers": _analyzable_biomarkers(df),
        "schemes": {s: describe_groups(df, s) for s in ("e4_dose", "e4_carrier", "trem2")},
    }


def summarize_groups(dataset_id: str = "current", scheme: str = "e4_dose") -> dict:
    """Descriptive statistics per genotype group (n, mean, median, sd; % amyloid-positive).

    Args:
        dataset_id: Handle returned by load_biomarker_dataset.
        scheme: 'e4_dose', 'e4_carrier', or 'trem2'.
    """
    df = _get_df(dataset_id)
    if df is None:
        return {"status": "error", "message": "Unknown dataset_id; load a dataset first."}
    return {"status": "success", **compute_group_summary(df, scheme=scheme)}


def generate_hypothesis(dataset_id: str = "current", biomarker: str = "amyloid_centiloid",
                        scheme: str = "e4_carrier") -> dict:
    """Run cross-sectional, age/sex-adjusted, and longitudinal analyses for one biomarker
    across a genotype scheme, and return a Hypothesis / Evidence / Caution / Citations card.

    Args:
        dataset_id: Handle returned by load_biomarker_dataset.
        biomarker: One of the analyzable biomarker column names.
        scheme: 'e4_dose', 'e4_carrier', or 'trem2'.
    """
    df = _get_df(dataset_id)
    if df is None:
        return {"status": "error", "message": "Unknown dataset_id; load a dataset first."}
    if biomarker not in BIOMARKER_META:
        return {"status": "error", "message": f"Unknown biomarker '{biomarker}'. "
                f"Choose from {list(BIOMARKER_META)}."}
    card = make_hypothesis_card(analyze(df, biomarker, scheme))
    return {"status": "success", "card": card, "rendered": render_card_text(card)}


def check_query_safety(question: str) -> dict:
    """Screen a user question for out-of-scope INDIVIDUAL clinical intent.

    Allows group-level research questions (e.g. 'do ε4 carriers show more amyloid?').
    Blocks personal diagnosis, prognosis, treatment, or risk questions. Call this FIRST.

    Args:
        question: The user's raw question.

    Returns:
        dict with allowed (bool), and if blocked, a reason and a suggested response.
    """
    q = question.lower()
    # Clinical intent is AUTHORITATIVE: these are always checked, and a match here is
    # never overridden by methodological-sounding phrasing. This prevents a bypass such
    # as "Should I take this medication and adjust my dose?" (a methodological clause must
    # not disable the safety block).
    clinical_patterns = [
        r"\b(do|will|am|could|might|would|should)\s+i\b.*\b(have|get|develop|die|need|take)\b",
        r"\bmy (risk|chance|odds|prognosis|diagnosis|result|scan|genotype|treatment|medication)\b",
        r"\bdiagnos(e|is|ing) me\b", r"\bhow long (do|have) i\b",
        r"\b(what|which) (drug|medication|treatment|therapy|dose)\b.*\b(should|for me|i take)\b",
        r"\b(prescrib|cure|treat)\w*\b.*\b(me|my|patient)\b",
        r"\b(this|my) patient('?s)?\b.*\b(treat|diagnos|prognos|medicat|manage)\w*",
        r"\bshould (this|my|the) patient\b", r"\btell me if i\b.*\b(have|will)\b",
        r"\bshould i take\b",
    ]
    for pat in clinical_patterns:
        if re.search(pat, q):
            return {
                "allowed": False,
                "reason": "individual_clinical_query",
                "suggested_response": (
                    "I can't help with individual diagnosis, prognosis, treatment, or "
                    "personal risk — this is a research tool for exploring group-level "
                    "biomarker patterns, not clinical advice. For anything about a specific "
                    "person, please consult a qualified clinician. I can, however, explore "
                    "questions like how APOE ε4 carriers differ from non-carriers on amyloid "
                    "or hippocampal measures in the dataset."),
            }
    return {"allowed": True, "reason": None}


def rank_biomarkers_by_effect(dataset_id: str = "current", scheme: str = "e4_carrier") -> dict:
    """Rank all analyzable biomarkers by the magnitude of their genotype effect, largest
    first. Use this for questions like 'which biomarker shows the strongest genotype effect?'.

    Args:
        dataset_id: Handle returned by load_biomarker_dataset.
        scheme: 'e4_dose', 'e4_carrier', or 'trem2'.
    """
    df = _get_df(dataset_id)
    if df is None:
        return {"status": "error", "message": "Unknown dataset_id; load a dataset first."}
    return {"status": "success", **rank_biomarkers(df, scheme)}


TOOLS = [load_biomarker_dataset, inspect_dataset, summarize_groups,
         generate_hypothesis, rank_biomarkers_by_effect, check_query_safety]


# --------------------------------------------------------------------------- #
# System instruction for the LLM agent
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = """
You are the Alzheimer's Biomarker Hypothesis Agent — a RESEARCH assistant that turns
subject-level biomarker data into transparent, citation-backed, hypothesis-generating
cards. You are NOT a clinical tool.

Workflow for every research question:
1. Call check_query_safety FIRST. If not allowed, reply with its suggested_response and stop.
2. If no dataset is loaded yet, call load_biomarker_dataset with NO arguments — it
   loads the bundled synthetic dataset by default. Never invent a CSV filename.
3. Call inspect_dataset to see which biomarkers are analyzable and whether the genotype
   groups are adequately powered.
4. Map the question to ONE biomarker column and ONE scheme
   ('e4_dose' | 'e4_carrier' | 'trem2'). If the question is too vague to pick a biomarker,
   ask a brief clarifying question instead of guessing.
5. Call generate_hypothesis and present the returned card faithfully.

Hard rules:
- Never invent statistics or citations. Report ONLY what the tools return; every number and
  reference must come from a tool result.
- Always preserve the card's Caution items, especially the synthetic-data and
  not-for-diagnosis notes.
- Prefer ε4-based schemes; treat TREM2 comparisons as exploratory and flag small groups.
- Keep responses concise and neutral.
"""


# --------------------------------------------------------------------------- #
# Deterministic orchestrator core (offline demo + eval target)
# --------------------------------------------------------------------------- #
_BIOMARKER_KEYWORDS = [
    (("amyloid", "centiloid", "plaque", "abeta", "a-beta", "aβ", "ab42"), "amyloid_centiloid"),
    (("tau",), "tau_suvr"),
    (("fdg", "glucose", "metabolism", "hypometabol"), "fdg_suvr"),
    (("hippocamp",), "hippocampal_volume"),
    (("entorhinal",), "entorhinal_thickness"),
    (("mmse", "mini-mental"), "mmse"),
    (("cdr", "dementia rating", "sum of boxes"), "cdrsb"),
    (("cognit", "memory decline"), "mmse"),
]


def parse_intent(question: str) -> dict:
    """Keyword intent parser: map free text -> (biomarker, scheme). Mirrors what the LLM
    does; used by the deterministic core and as an eval baseline."""
    q = question.lower()
    biomarker = None
    for keys, col in _BIOMARKER_KEYWORDS:
        if any(k in q for k in keys):
            biomarker = col
            break
    if any(k in q for k in ("trem2", "r47h")):
        scheme = "trem2"
    elif any(k in q for k in ("dose", "copies", "allele count", "per allele", "gene dose",
                              "number of e4", "homozyg", "0/1/2")):
        scheme = "e4_dose"
    else:
        scheme = "e4_carrier"
    return {"biomarker": biomarker, "scheme": scheme, "biomarker_found": biomarker is not None}


class HypothesisAgentCore:
    """Deterministic orchestrator: safety -> intent (+memory) -> tools -> card.

    The ADK agent performs these same steps via the LLM; this class makes the pipeline
    runnable and testable without a model, and remembers the last analysis so follow-ups
    like 'what about hippocampus?' or 'now by dose' resolve."""

    def __init__(self, csv_path: str, dataset_id: str = "current"):
        self.dataset_id = dataset_id
        self.load_result = load_biomarker_dataset(csv_path, dataset_id)
        self._last: Dict[str, Optional[str]] = {"biomarker": None, "scheme": None}

    def ask(self, question: str) -> dict:
        safety = check_query_safety(question)
        if not safety["allowed"]:
            return {"mode": "refused", "text": safety["suggested_response"]}

        intent = parse_intent(question)
        ql = question.lower()
        is_ranking = (intent["biomarker"] is None and
                      any(k in ql for k in ("which biomarker", "which biomarkers", "strongest",
                                            "rank", "biggest effect", "most affected",
                                            "largest effect", "compare biomarkers")))
        if is_ranking:
            scheme = intent["scheme"]
            res = rank_biomarkers_by_effect(self.dataset_id, scheme)
            lines = [f"Biomarkers ranked by |effect| for {scheme}:"]
            for r in res.get("ranking", []):
                lines.append(f"  {r['rank']}. {r['label']} ({r['biomarker']}): "
                             f"effect={r['abs_effect']}, {r['direction']}, p={r['p_value']}")
            self._last = {"biomarker": res["ranking"][0]["biomarker"] if res.get("ranking") else None,
                          "scheme": scheme}
            return {"mode": "ranking", "scheme": scheme, "ranking": res.get("ranking", []),
                    "text": "\n".join(lines)}

        is_followup = bool(re.match(r"^\s*(now|then|and|also|what about|how about|same|instead)\b", ql)
                           or "what about" in ql or "how about" in ql)

        biomarker = intent["biomarker"] or (self._last["biomarker"] if is_followup else None)
        explicit_scheme = intent["scheme"] if any(
            k in ql for k in ("trem2", "r47h", "dose", "copies", "carrier", "allele")) else None
        if explicit_scheme:
            scheme = explicit_scheme
        elif is_followup and self._last["scheme"]:
            scheme = self._last["scheme"]
        else:
            scheme = intent["scheme"]

        if biomarker is None:
            df = _get_df(self.dataset_id)
            opts = ", ".join(_analyzable_biomarkers(df)) if df is not None else ""
            return {"mode": "clarify",
                    "text": f"Which biomarker would you like to examine? Options: {opts}."}

        self._last = {"biomarker": biomarker, "scheme": scheme}
        result = generate_hypothesis(self.dataset_id, biomarker, scheme)
        return {"mode": "answer", "biomarker": biomarker, "scheme": scheme,
                "card": result["card"], "text": result["rendered"]}


# --------------------------------------------------------------------------- #
# ADK agent definition (built only if google-adk is installed).
# `adk run` / `adk web` look for `root_agent`.
# --------------------------------------------------------------------------- #
try:
    from google.adk.agents import Agent  # type: ignore

    root_agent = Agent(
        model="gemini-2.5-flash",
        name="alz_biomarker_hypothesis_agent",
        description=("Research assistant that generates citation-backed hypothesis cards "
                     "from subject-level Alzheimer's biomarker data."),
        instruction=SYSTEM_INSTRUCTION,
        tools=TOOLS,
    )
except Exception:  # ADK not installed in this environment
    root_agent = None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic_adni_style.csv")
    args = ap.parse_args()

    core = HypothesisAgentCore(args.csv)
    print("LOADED:", {k: core.load_result[k] for k in
                      ("n_subjects", "n_rows", "schema_ok", "available_biomarkers")})
    demo = [
        "Do APOE e4 carriers show higher amyloid burden?",     # -> amyloid, e4_carrier
        "What about hippocampal volume?",                       # follow-up reuses scheme
        "Now show it by e4 dose",                               # reuse biomarker, dose
        "Does TREM2 R47H affect hippocampal decline?",          # trem2
        "Should I have the APOE test to know if I'll get Alzheimer's?",  # REFUSED
        "How does tau relate to genotype?",                     # tau
    ]
    for q in demo:
        print("\n" + "#" * 78 + f"\nQ: {q}")
        r = core.ask(q)
        print(f"[mode={r['mode']}"
              + (f" | {r.get('biomarker')} × {r.get('scheme')}]" if r["mode"] == "answer" else "]"))
        print(r["text"] if r["mode"] != "answer" else r["text"][:600] + "\n...")
