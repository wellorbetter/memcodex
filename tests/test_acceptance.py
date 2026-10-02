"""Acceptance: full pipeline with FakeLLM, no network."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcodex.arena import DirectBaseline, FlatRetrievalBaseline, MemCodexSystem, run_system
from memcodex.records import load_episodes

from .fake_llm import FakeLLM


def test_demo_pipeline_runs():
    llm = FakeLLM()
    episodes = load_episodes(Path(__file__).resolve().parents[1] / "examples" / "demo_dataset" / "episodes.json")
    assert episodes
    for system in (
        DirectBaseline(llm),
        FlatRetrievalBaseline(llm),
        MemCodexSystem(llm),
    ):
        stats = run_system(system, episodes)
        assert stats["n"] > 0
        assert 0.0 <= stats["acc"] <= 1.0
        assert stats["tokens"] > 0