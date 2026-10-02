"""ProgramEvolve - the paper's Algorithm 2 with Eq.4 acceptance.

A "program" is a JSON mapping of layer name -> prompt text (the executable
memory program). The meta-program rewrites prompts; children are accepted
when the acceptance test (Eq.4) says they beat their parent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .arena import MemCodexSystem, run_system
from .llm import LLM, LLMError
from .records import Episode

META_SYSTEM = "You are a meta-program that designs memory systems for a language agent."

META_PROMPT = """Design the prompts for a hierarchical memory.
Layers (names are fixed, order is coarse-to-fine): {layer_names}
Current program:
{current}
Last evaluation: acc={acc:.3f}, tokens={tokens:.0f}
Rewrite each layer's prompt to improve answer accuracy and lower token usage.
The agent builds each layer from episode records, then answers questions
using a coarse-to-fine budgeted read over the layers.
Return ONLY a JSON object mapping each layer name to its new prompt text.
"""


@dataclass
class Program:
    id: str
    layer_specs: dict
    parent: str | None = None
    metrics: dict | None = None


def _extract_json(raw: str) -> dict:
    """Parse meta-program output into a JSON object (raises ValueError)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("meta output is not a JSON object")
    return obj


def _parse_program(raw: str, layer_names: list) -> dict:
    obj = _extract_json(raw)
    specs = {}
    for name in layer_names:
        v = obj.get(name)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"layer {name} missing or not a string")
        specs[name] = v.strip()
    return specs


def evaluate_program(layer_specs: dict, llm: LLM, episodes: list, budget: int) -> dict:
    system = MemCodexSystem(llm, layer_specs=layer_specs, total_budget=budget)
    return run_system(system, episodes)


def acceptance(child_m: dict, parent_m: dict) -> bool:
    """Eq.4 (simplified): accept if accuracy improves, or accuracy equal and tokens lower."""
    if child_m["acc"] > parent_m["acc"] + 1e-9:
        return True
    if abs(child_m["acc"] - parent_m["acc"]) <= 1e-9:
        return child_m["tokens"] < parent_m["tokens"] - 1e-9
    return False


def _load_archive(path: Path) -> list:
    if not path.exists():
        return []
    progs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        progs.append(
            Program(
                id=obj["id"],
                layer_specs=obj["layer_specs"],
                parent=obj.get("parent"),
                metrics=obj.get("metrics"),
            )
        )
    return progs


def program_evolve(
    base_specs: dict,
    llm: LLM,
    episodes: list,
    rounds: int = 1,
    parents: int = 1,
    children: int = 2,
    budget: int = 1024,
    archive_path: str = "archive.jsonl",
) -> list:
    """Run the evolution loop; return the list of accepted programs (appends to archive)."""
    path = Path(archive_path)
    archive = _load_archive(path)
    layer_names = list(base_specs.keys())
    accepted: list = list(archive)
    counter = len(accepted)

    # baseline evaluation if we have nothing yet
    candidates: list = []
    if not archive:
        counter += 1
        base = Program(id=f"p{counter:03d}", layer_specs=dict(base_specs), parent=None)
        base.metrics = evaluate_program(base_specs, llm, episodes, budget)
        print(f"[base] {base.id} acc={base.metrics['acc']:.3f} tokens={base.metrics['tokens']:.0f}")
        candidates.append(base)
        _append(archive_path, base)

    for rnd in range(1, rounds + 1):
        # select parents: top-`parents` by accuracy among candidates/accepted
        pool = accepted if accepted else candidates
        ranked = sorted(
            pool, key=lambda p: (p.metrics or {}).get("acc", 0.0), reverse=True
        )
        chosen = ranked[: max(1, parents)]
        for parent in chosen:
            cur_prompt = META_PROMPT.format(
                layer_names=", ".join(layer_names),
                current=json.dumps(parent.layer_specs, ensure_ascii=False, indent=1),
                acc=(parent.metrics or {}).get("acc", 0.0),
                tokens=(parent.metrics or {}).get("tokens", 0.0),
            )
            for _ in range(children):
                counter += 1
                cid = f"p{counter:03d}"
                spec = None
                for attempt in range(3):
                    try:
                        raw = llm.chat(META_SYSTEM, cur_prompt, temperature=0.8)
                        spec = _parse_program(raw, layer_names)
                        break
                    except (ValueError, LLMError) as err:
                        print(f"[warn] child {cid} mutation parse failed (attempt {attempt + 1}): {err}")
                if spec is None:
                    print(f"[skip] child {cid}: unparseable mutation after 3 attempts")
                    continue
                child = Program(id=cid, layer_specs=spec, parent=parent.id)
                child.metrics = evaluate_program(spec, llm, episodes, budget)
                ok = acceptance(child.metrics, parent.metrics or {"acc": 0.0, "tokens": float("inf")})
                verdict = "accept" if ok else "reject"
                print(
                    f"[round {rnd}] {cid} (parent {parent.id}) "
                    f"acc={child.metrics['acc']:.3f} tokens={child.metrics['tokens']:.0f} "
                    f"-> {verdict}"
                )
                if ok:
                    accepted.append(child)
                    _append(archive_path, child)
                    candidates.append(child)
    return accepted


def _append(path: str, prog: Program) -> None:
    obj = {
        "id": prog.id,
        "layer_specs": prog.layer_specs,
        "parent": prog.parent,
        "metrics": prog.metrics,
        "ts": time.time(),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")