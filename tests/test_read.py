from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcodex.layers import make_layer
from memcodex.read import query_adaptive_read
from memcodex.records import Record

from .fake_llm import FakeLLM


def _build(records, llm):
    layers = []
    for name in ("summary", "relational", "episode"):
        lp = make_layer(name, llm)
        items = lp.construct(records)
        lp.index(items)
        layers.append(lp)
    return layers


def test_read_respects_budget():
    llm = FakeLLM()
    records = [Record(text=f"record {i} about the search workstream") for i in range(20)]
    layers = _build(records, llm)
    rr = query_adaptive_read("search workstream", layers, llm, total_budget=64)
    assert rr.tokens_used <= 64


def test_read_collects_evidence_when_budget_rich():
    llm = FakeLLM()
    records = [Record(text="Alice assigned Bob to the search workstream")]
    layers = _build(records, llm)
    rr = query_adaptive_read("Who did Alice assign?", layers, llm, total_budget=256)
    assert rr.evidence
    assert rr.tokens_used <= 256
    assert rr.stopped_at == -1  # all layers traversed


def test_read_stops_when_budget_thin():
    llm = FakeLLM()
    records = [Record(text=f"record {i} about the search workstream") for i in range(20)]
    layers = _build(records, llm)
    rr = query_adaptive_read("search workstream", layers, llm, total_budget=64)
    assert rr.stopped_at >= 0