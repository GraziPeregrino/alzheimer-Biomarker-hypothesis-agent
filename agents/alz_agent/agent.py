"""Import-safe wrapper so `adk web` / `adk run` can discover the agent.

The project folder name contains hyphens (not a valid Python identifier), so
ADK cannot use it as an agent name. This thin package has a valid name and
re-exports the `root_agent` defined in agents/coordinator_agent.py.
"""
import os
import sys

# Walk up from this file to find the project root (the dir holding
# requirements.txt), then put it on sys.path so `agents.*` and `tools.*` import
# cleanly no matter where this package lives.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root):  # stop at filesystem root
    if os.path.exists(os.path.join(_root, "requirements.txt")):
        break
    _root = os.path.dirname(_root)

if _root not in sys.path:
    sys.path.insert(0, _root)

# Load the project-root .env (where the Gemini key lives) so credentials are
# present regardless of ADK's own .env search path.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_root, ".env"))
except Exception:
    pass

from agents.coordinator_agent import root_agent  # noqa: E402

__all__ = ["root_agent"]
