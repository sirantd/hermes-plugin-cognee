"""Make the Hermes MemoryProvider ABC importable for provider tests.

Looks for a NousResearch/hermes-agent checkout via $HERMES_AGENT_PATH or a
sibling ../hermes-agent directory and puts it on sys.path. Provider tests
`pytest.importorskip("agent.memory_provider")`, so they skip cleanly if no
checkout is present (client tests still run).
"""
import os
import sys
from pathlib import Path

_candidates = [
    os.environ.get("HERMES_AGENT_PATH"),
    str(Path(__file__).resolve().parent / "hermes-agent"),
    str(Path(__file__).resolve().parent.parent / "hermes-agent"),
]
for _c in _candidates:
    if _c and Path(_c).is_dir() and _c not in sys.path:
        sys.path.insert(0, _c)
        break
