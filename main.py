"""
main.py — entry point for the Alzheimer's Biomarker Hypothesis Agent.

Runs the deterministic agent core on a research question. This path is fully
offline (no Gemini key required) and uses the same tools as the LLM agent.

    python main.py                                  # runs a demo question
    python main.py "Do APOE e4 carriers show more amyloid?"

To drive the Gemini/ADK agent in a browser instead, run:  adk web agents
"""
from __future__ import annotations

import argparse

from agents.coordinator_agent import HypothesisAgentCore

DEMO_QUESTION = "How does APOE e4 relate to amyloid burden?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION,
                        help="a plain-language research question")
    args = parser.parse_args()

    core = HypothesisAgentCore("synthetic_adni_style.csv")
    if core.load_result.get("status") != "success":
        raise SystemExit("Could not load dataset: "
                         + core.load_result.get("message", "unknown error"))

    result = core.ask(args.question)
    print(result["text"])


if __name__ == "__main__":
    main()
