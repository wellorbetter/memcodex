"""Arena: run systems on episodes; score answers with token F1."""
from __future__ import annotations

import time

from .layers import _tokenize, bm25_index, bm25_search, make_layer
from .llm import LLM
from .read import query_adaptive_read
from .records import Episode, Question

NEGATION_MARKERS = ("no", "not", "none", "cannot", "did not")
NEGATION_MARKERS_ZH = ("\u65e0", "\u6ca1\u6709", "\u65e0\u6cd5")


def _is_negated(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in NEGATION_MARKERS) or any(
        m in text for m in NEGATION_MARKERS_ZH
    )


def tokenize_answer(text: str) -> set[str]:
    return set(_tokenize(text))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 with a negation shortcut (paper: binary correctness)."""
    if _is_negated(gold) != _is_negated(prediction):
        return 0.0
    pred = tokenize_answer(prediction)
    goldw = tokenize_answer(gold)
    if not pred or not goldw:
        return 1.0 if pred == goldw else 0.0
    common = len(pred & goldw)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(goldw)
    return 2 * precision * recall / (precision + recall)


ANSWER_SYSTEM = (
    "Answer the question using only the provided evidence. "
    "If the evidence does not contain the answer, say exactly: NO INFORMATION"
)


class DirectBaseline:
    """No memory: answer each question directly from the raw records."""

    name = "direct"

    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, episode: Episode, question: Question) -> str:
        user = (
            "Evidence (raw records):\n"
            + "\n".join(r.text for r in episode.records)
            + "\n\nQuestion: "
            + question.text
        )
        return self.llm.chat(ANSWER_SYSTEM, user)


DEFAULT_LAYER_SPECS: dict[str, str] = {
    "summary": (
        "Write 3-5 bullet facts covering this episode: who, what, when, "
        "numbers, decisions. Output bullets only, no preamble."
    ),
    "relational": (
        "List relations as `A <relation> B` lines, one per line, "
        "covering every explicit fact in the records."
    ),
    "episode": (
        "Keep concrete details verbatim: names, dates, quantities, "
        "places, actions. One item per line."
    ),
}


class MemCodexSystem:
    """Hierarchical memory: build layers once per episode, read adaptively per question."""

    name = "memcodex"

    def __init__(
        self,
        llm: LLM,
        layer_specs: dict | None = None,
        total_budget: int = 1024,
    ):
        self.llm = llm
        self.layer_specs = layer_specs or dict(DEFAULT_LAYER_SPECS)
        self.total_budget = total_budget
        self._layers: dict = {}

    def build(self, episode: Episode) -> None:
        self._layers = {}
        for name, spec in self.layer_specs.items():
            lp = make_layer(name, self.llm, spec)
            items = lp.construct(episode.records)
            lp.index(items)
            self._layers[name] = lp

    def answer(self, episode: Episode, question: Question) -> str:
        if not self._layers:
            self.build(episode)
        layers = [self._layers[n] for n in self.layer_specs]
        rr = query_adaptive_read(question.text, layers, self.llm, self.total_budget)
        evidence = "\n".join(it.text for it in rr.evidence)
        user = "Evidence (layered memory):\n" + evidence + "\n\nQuestion: " + question.text
        return self.llm.chat(ANSWER_SYSTEM, user)


def run_system(system, episodes: list) -> dict:
    """Run a system over all episodes/questions; return per-question metrics."""
    total_tokens = 0
    latency_ms = 0.0
    correct = 0.0
    total_q = 0
    for ep in episodes:
        for q in ep.questions:
            llm = system.llm
            prev_prompt = llm.total_prompt_tokens
            prev_completion = llm.total_completion_tokens
            t0 = time.perf_counter()
            ans = system.answer(ep, q)
            latency_ms += (time.perf_counter() - t0) * 1000
            total_tokens += (
                llm.total_prompt_tokens - prev_prompt
                + llm.total_completion_tokens - prev_completion
            )
            correct += f1_score(ans, q.answer)
            total_q += 1
    return {
        "acc": correct / max(1, total_q),
        "tokens": total_tokens / max(1, total_q),
        "latency_ms": latency_ms / max(1, total_q),
        "n": total_q,
    }

class FlatRetrievalBaseline:
    """Flat retrieval baseline: no hierarchy; BM25 over all raw records."""

    def __init__(self, llm: LLM, k: int = 5):
        self.llm = llm
        self.k = k
        self._index = None
        self._texts: list[str] = []

    def _ensure(self, episode: Episode) -> None:
        if self._index is None:
            self._texts = [r.text for r in episode.records]
            self._index = bm25_index(self._texts)

    def answer(self, episode: Episode, question: Question) -> str:
        self._ensure(episode)
        hits = bm25_search(self._index, question.text, k=self.k)
        evidence = "\n".join(self._texts[i] for i, _ in hits) or "(no matching records)"
        user = "Evidence (retrieved records):\n" + evidence + "\n\nQuestion: " + question.text
        return self.llm.chat(ANSWER_SYSTEM, user)