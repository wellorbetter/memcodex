"""QueryAdaptiveRead - the paper's Algorithm 1.

Traverse layers from coarse to fine. Stop early when the marginal new
evidence from a layer is below eps * budget AND the remaining budget can
no longer afford the cheapest layer (min_layer_cost).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM
from .layers import Item, LayerProgram


@dataclass
class ReadResult:
    evidence: list[Item] = field(default_factory=list)
    tokens_used: int = 0
    stopped_at: int = -1  # index of the layer where stop condition fired


def query_adaptive_read(
    query: str,
    layers: list[LayerProgram],
    llm: LLM,
    total_budget: int,
    min_layer_cost: int = 64,
    eps: float = 0.05,
) -> ReadResult:
    result = ReadResult()
    seen_texts: set[str] = set()
    remaining = total_budget
    for i, layer in enumerate(layers):
        if remaining < min_layer_cost:
            result.stopped_at = i
            break
        layer_budget = max(min_layer_cost, int(layer.budget_share * total_budget))
        items, used = layer.retrieve(query, min(layer_budget, remaining))
        new_items = [it for it in items if it.text not in seen_texts]
        result.evidence.extend(new_items)
        result.tokens_used += sum(it.tokens for it in new_items)
        seen_texts.update(it.text for it in new_items)
        remaining -= sum(it.tokens for it in new_items)
        marginal = len(new_items) / max(1, len(items))
        if marginal < eps and remaining < min_layer_cost:
            result.stopped_at = i
            break
        # stop when this layer added nothing new and budget is thin
        if not new_items and remaining < min_layer_cost:
            result.stopped_at = i
            break
    return result