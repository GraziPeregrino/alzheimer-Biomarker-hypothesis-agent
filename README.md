# Alzheimer's Biomarker Hypothesis Agent

An agentic research workflow that turns subject level Alzheimer's biomarker data into
transparent, citation backed, **hypothesis-generating** cards. Built on Google ADK +
Gemini for the *AI Agents: Intensive Vibe Coding* capstone.

> **Research / education tool, not for diagnosis or clinical use.** Every result is
> observational and hypothesis generating. The included dataset is **synthetic** (see
> below); reported effects are built into the data generator by design, not evidence
> about real biology.

![Architecture](docs/Biomarker Agent Workflow (1).pdf)

## What it does

A researcher asks a plain language question (e.g. *"Do APOE ε4 carriers show higher
amyloid burden and faster hippocampal decline?"*). The agent screens the question for
safety, inspects the dataset, selects one biomarker and one genotype scheme, runs the
statistics, and returns a Hypothesis / Evidence / Caution / Citations card whose numbers
and references are all traceable to tool output.

## Repository layout

```
.
├── main.py                          # entry point — runs the agent core on a question
├── agents/
│   ├── coordinator_agent.py         # ADK/Gemini agent + routing, guardrail & deterministic core
│   └── alz_agent/                   # thin wrapper so `adk web` can discover the agent
├── skills/
│   ├── ingestion_skills.py          # load / validate / clean / group / annotate
│   └── stats_skills.py              # summaries, slopes, trajectories, adjusted effects, ranking, cards
├── mcp_servers/
│   └── annotation_server.py         # MCP server wrapping variant annotation as tools
├── data/
│   ├── generate_synthetic_adni.py   # synthetic ADNI/OASIS style data generator
│   ├── synthetic_adni_style.csv     # demo dataset (synthetic, ADNI ready schema)
│   └── DATA_DICTIONARY.md           # schema, units, calibration anchors, citations
├── evals/
│   ├── evals.py                     # routing / safety / grounding / ranking harness
│   └── eval_results.json            # latest eval scorecard (17/17)
├── docs/
│   └── architecture_diagram.png     # architecture diagram
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. (re)generate the synthetic dataset (written to data/)
python data/generate_synthetic_adni.py --n 300 --seed 42

# 2. run the evaluation harness (no API key needed — scores the deterministic core)
python evals/evals.py

# 3. ask the deterministic agent core a question (offline, no API key)
python main.py "Do APOE e4 carriers show more amyloid?"
```

To run the **Gemini** agent, install `google-adk`, put your Gemini credentials in a
`.env` file at the project root, then launch `adk web agents` and pick **alz_agent** in
the browser. The agent is defined in `agents/coordinator_agent.py` (`root_agent`); the
`agents/alz_agent/` package is a thin wrapper that lets ADK discover it.

The agent launches the annotation **MCP server** automatically as a subprocess, so
you don't run it yourself. To exercise it directly (no API key needed):

```bash
python mcp_servers/annotation_server.py    # starts the MCP server over stdio
```

## How it maps to the course

- **Agents:** ADK + Gemini orchestrator that plans and calls tools.
- **Skills & interoperability:** Python-function skills (in `skills/`); variant
  annotation via Ensembl / MyVariant / GWAS Catalog with an offline-safe cache, also
  exposed through an MCP server (`mcp_servers/annotation_server.py`) and wired into
  the agent as an MCP toolset.
- **Context & memory:** in-memory dataset store with disk rehydration; session
  follow-ups.
- **Quality & security:** eval harness (routing, safety, grounding, ranking) and
  a defense-in-depth clinical-question guardrail.
- **Packaging:** this repo, the notebook, and the architecture diagram.

## Data note

The demo data is synthetic but matches an ADNI/OASIS style schema and is calibrated to
literature consistent effect directions (APOE ε4 → higher amyloid, faster hippocampal
decline; TREM2 R47H ≈ one ε4 allele). The pipeline is **ADNI/OASIS-3 ready**: point it at
approved, access controlled real data with the same columns to run the real analysis.
See `data/DATA_DICTIONARY.md`.

## Limitations & ethics

Observational associations only, no causal claims. Per-subject OLS slopes approximate a
linear mixed effects model (the confirmatory upgrade). Rare variants (TREM2) are
underpowered. Citations come from a curated set so they are always verifiable. The agent
refuses individual diagnosis, prognosis, treatment, and personal-risk questions.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
