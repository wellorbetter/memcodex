"""Command-line interface for MemCodex.

Usage:
  python -m memcodex demo
  python -m memcodex evolve --rounds 2 --children 2
"""
from __future__ import annotations

import argparse
import sys

from .arena import (
    DEFAULT_LAYER_SPECS,
    DirectBaseline,
    FlatRetrievalBaseline,
    MemCodexSystem,
    run_system,
)
from .config import load_config
from .evolve import program_evolve
from .llm import LLM
from .records import load_episodes

SYSTEMS = ("direct", "flat", "memcodex")


def build_systems(cfg) -> dict:
    llm = LLM(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    return {
        "direct": DirectBaseline(llm),
        "flat": FlatRetrievalBaseline(llm),
        "memcodex": MemCodexSystem(llm, total_budget=cfg.total_budget),
    }


def cmd_demo(cfg) -> int:
    episodes = load_episodes(cfg.dataset_path)
    systems = build_systems(cfg)
    rows = {name: run_system(sys, episodes) for name, sys in systems.items()}
    print(f"{'system':<12}{'acc':>7}{'tokens/q':>11}{'ms/q':>9}{'n':>5}")
    for name in SYSTEMS:
        s = rows[name]
        print(f"{name:<12}{s['acc']:>7.3f}{s['tokens']:>11.1f}{s['latency_ms']:>9.1f}{s['n']:>5}")
    return 0


def cmd_evolve(cfg, args) -> int:
    episodes = load_episodes(cfg.dataset_path)
    llm = LLM(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    accepted = program_evolve(
        base_specs=DEFAULT_LAYER_SPECS,
        llm=llm,
        episodes=episodes,
        rounds=args.rounds,
        parents=1,
        children=args.children,
        budget=cfg.total_budget,
        archive_path=args.archive,
    )
    if not accepted:
        print("no programs accepted")
        return 0
    best = max(accepted, key=lambda p: (p.metrics or {}).get("acc", 0.0))
    m = best.metrics or {}
    print(f"best={best.id} acc={m.get('acc', 0):.3f} tokens/q={m.get('tokens', 0):.1f} parent={best.parent}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memcodex")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run all systems on the dataset").set_defaults(
        func=lambda cfg, args: cmd_demo(cfg)
    )
    e = sub.add_parser("evolve", help="run the self-programming loop")
    e.add_argument("--rounds", type=int, default=1)
    e.add_argument("--children", type=int, default=2)
    e.add_argument("--archive", default="archive.jsonl")
    e.set_defaults(func=cmd_evolve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())