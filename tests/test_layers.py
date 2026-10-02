from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcodex.layers import Item, _tokenize, bm25_index, bm25_search, make_layer
from memcodex.records import Record

from .fake_llm import FakeLLM


def test_tokenize_mixed():
    toks = _tokenize("Alice assigned Bob, 2026-07-01")
    assert "alice" in toks and "bob" in toks and "2026" in toks


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "Alice assigned Bob to the search workstream",
        "The weather was rainy in Seattle",
        "Bob reported metrics on Friday",
    ]
    idx = bm25_index(docs)
    hits = bm25_search(idx, "Who did Alice assign to the search workstream?", k=3)
    assert hits and hits[0][0] == 0


def test_item_tokens_heuristic():
    it = Item(text="abcd")
    assert it.tokens == 2


def test_make_layer_registry():
    llm = FakeLLM()
    for name in ("summary", "relational", "episode"):
        lp = make_layer(name, llm)
        assert lp.name == name


def test_layer_construct_and_retrieve():
    llm = FakeLLM()
    records = [
        Record(text="Alice assigned Bob to the search workstream"),
        Record(text="The weather was rainy in Seattle"),
    ]
    lp = make_layer("summary", llm)
    items = lp.construct(records)
    assert len(items) == 1
    lp.index(items)
    rr = lp.retrieve("who did Alice assign", budget=128)
    assert rr is not None