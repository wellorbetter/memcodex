"""Layer programs: the pluggable units of MemCodex.

Each LayerProgram is the paper's "layer" - an executable piece of memory
program (construct / index / retrieve). Self-programming in this
implementation means the meta-agent rewrites `prompt_template` and the
layer composition (layer_specs), not the structural methods.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .llm import LLM
from .records import Record, count_tokens

CHUNK_TOKENS = 256


@dataclass
class Item:
    text: str
    tokens: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = count_tokens(self.text)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())
    return tokens


def _chunk_texts(texts: list[str], chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for t in texts:
        if cur and count_tokens(cur) + count_tokens(t) > chunk_tokens:
            chunks.append(cur)
            cur = t
        else:
            cur = (cur + "\n" + t) if cur else t
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------- BM25

def bm25_index(docs: list[str], k1: float = 1.5, b: float = 0.75) -> dict:
    tokenized = [_tokenize(d) for d in docs]
    df: Counter = Counter()
    for terms in tokenized:
        df.update(set(terms))
    doc_len = [len(t) for t in tokenized]
    avgdl = sum(doc_len) / max(1, len(doc_len))
    return {"tokenized": tokenized, "df": df, "doc_len": doc_len,
            "avgdl": avgdl, "k1": k1, "b": b, "n": len(docs)}


def bm25_search(index: dict, query: str, k: int = 5) -> list[tuple[int, float]]:
    q_terms = _tokenize(query)
    if not q_terms or index["n"] == 0:
        return []
    n, df, avgdl, k1, b = index["n"], index["df"], index["avgdl"], index["k1"], index["b"]
    scores = [0.0] * n
    for term in set(q_terms):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, terms in enumerate(index["tokenized"]):
            tf = terms.count(term)
            if tf == 0:
                continue
            dl = index["doc_len"][i]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * tf * (k1 + 1) / denom
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in ranked[:k] if s > 0]


# ---------------------------------------------------------- LayerProgram

class LayerProgram:
    name: str = "layer"

    def __init__(self, llm: LLM, prompt_template: str = "", budget_share: float = 1.0):
        self.llm = llm
        self.prompt_template = prompt_template
        self.budget_share = budget_share
        self._index: dict = {}
        self._items: list[Item] = []

    # -- write side -------------------------------------------------------
    def construct(self, records: list[Record]) -> list[Item]:
        raise NotImplementedError

    def index(self, items: list[Item]) -> None:
        self._items = items
        self._index = bm25_index([it.text for it in items])

    # -- read side ---------------------------------------------------------
    def retrieve(self, query: str, budget: int) -> tuple[list[Item], int]:
        raise NotImplementedError

    def _fit_budget(self, items: list[Item], budget: int) -> tuple[list[Item], int]:
        kept: list[Item] = []
        used = 0
        for it in items:
            if used + it.tokens > budget and kept:
                break
            kept.append(it)
            used += it.tokens
        return kept, used


# ---------------------------------------------------------------- layers

DEFAULT_SUMMARY_PROMPT = (
    "You are a memory compiler. Summarize the given conversation records "
    "into 3-6 concise bullet facts (who, what, where, when). Keep names and "
    "numbers. Output bullets only, no preamble."
)

DEFAULT_RELATIONAL_PROMPT = (
    "Extract explicit facts from the records as lines of the form "
    "subject | relation | object. One fact per line. No preamble."
)

DEFAULT_EPISODE_PROMPT = (
    "You keep raw records. Return the records most relevant to the query, "
    "verbatim, one per line, no preamble."
)


class SummaryLayer(LayerProgram):
    name = "summary"

    def __init__(self, llm: LLM, prompt_template: str = "", budget_share: float = 0.4):
        super().__init__(llm, prompt_template or DEFAULT_SUMMARY_PROMPT, budget_share)

    def construct(self, records: list[Record]) -> list[Item]:
        items: list[Item] = []
        for chunk in _chunk_texts([r.text for r in records]):
            summary = self.llm.chat(self.prompt_template, chunk).strip()
            items.append(Item(text=summary, meta={"layer": self.name}))
        return items

    def retrieve(self, query: str, budget: int) -> tuple[list[Item], int]:
        hits = bm25_search(self._index, query, k=4)
        items = [self._items[i] for i, _ in hits]
        return self._fit_budget(items, budget)


class RelationalLayer(LayerProgram):
    name = "relational"

    def __init__(self, llm: LLM, prompt_template: str = "", budget_share: float = 0.3):
        super().__init__(llm, prompt_template or DEFAULT_RELATIONAL_PROMPT, budget_share)

    def construct(self, records: list[Record]) -> list[Item]:
        items: list[Item] = []
        for chunk in _chunk_texts([r.text for r in records]):
            out = self.llm.chat(self.prompt_template, chunk).strip()
            for line in out.splitlines():
                line = line.strip(" -\t")
                if line:
                    items.append(Item(text=line, meta={"layer": self.name}))
        return items

    def retrieve(self, query: str, budget: int) -> tuple[list[Item], int]:
        hits = bm25_search(self._index, query, k=5)
        items = [self._items[i] for i, _ in hits]
        return self._fit_budget(items, budget)


class EpisodeLayer(LayerProgram):
    name = "episode"

    def __init__(self, llm: LLM, prompt_template: str = "", budget_share: float = 0.3):
        super().__init__(llm, prompt_template or DEFAULT_EPISODE_PROMPT, budget_share)

    def construct(self, records: list[Record]) -> list[Item]:
        return [Item(text=r.text, meta={"layer": self.name}) for r in records]

    def retrieve(self, query: str, budget: int) -> tuple[list[Item], int]:
        hits = bm25_search(self._index, query, k=4)
        items = [self._items[i] for i, _ in hits]
        return self._fit_budget(items, budget)


LAYER_REGISTRY: dict[str, type[LayerProgram]] = {
    "summary": SummaryLayer,
    "relational": RelationalLayer,
    "episode": EpisodeLayer,
}


def make_layer(name: str, llm: LLM, prompt_template: str = "", budget_share: float = 1.0) -> LayerProgram:
    cls = LAYER_REGISTRY[name]
    return cls(llm, prompt_template, budget_share)