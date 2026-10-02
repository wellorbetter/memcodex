"""Tests for ProgramEvolve (Algorithm 2 + Eq.4 acceptance)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcodex.evolve import acceptance, program_evolve
from memcodex.records import load_episodes
from .fake_llm import FakeLLM


def test_acceptance_picks_higher_accuracy():
    parent = {"acc": 0.5, "tokens": 100}
    child = {"acc": 0.6, "tokens": 100}
    assert acceptance(child, parent)
    assert not acceptance(parent, child)


def test_acceptance_tie_lower_tokens_wins():
    parent = {"acc": 0.6, "tokens": 100}
    child = {"acc": 0.6, "tokens": 80}
    assert acceptance(child, parent)
    assert not acceptance(parent, child)


def test_full_evolve_loop_with_fake_llm():
    episodes = load_episodes(
        Path(__file__).resolve().parents[1] / "examples" / "demo_dataset" / "episodes.json"
    )
    base = {
        "summary": "Write 1-3 bullet facts.",
        "relational": "List relations as A <relation> B lines.",
        "episode": "Extract key entities.",
    }
    llm = FakeLLM()
    archive = Path(__file__).resolve().parents[1] / "tests" / "archive_test.jsonl"
    if archive.exists():
        archive.unlink()
    try:
        out = program_evolve(base, llm, episodes, rounds=1, children=1, archive_path=str(archive))
        assert isinstance(out, list)
        assert llm.calls >= 1
        assert archive.exists()  # baseline always archived
    finally:
        if archive.exists():
            archive.unlink()
