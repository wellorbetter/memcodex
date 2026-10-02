# MemCodex

A runnable implementation of the **MemCodex** system from
[arXiv:2609.39765](https://arxiv.org/abs/2609.39765) — a 3-layer memory
(Summary / Relational / Episode), query-adaptive reading with an explicit
token budget, and a self-programming loop in which a meta-program rewrites
the memory program's layer prompts and children are accepted only when they
beat their parent (Eq.4).

## Paper

> MemCodex, arXiv:2609.39765.
> <https://arxiv.org/abs/2609.39765>

BibTeX:

```bibtex
@misc{memcodex2026,
  title  = {MemCodex},
  author = {Wang and others},
  year   = {2026},
  eprint = {2609.39765},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2609.39765}
}
```

What this repo implements:

| Paper component | Where |
|---|---|
| 3-layer memory program (Summary / Relational / Episode) | `memcodex/layers.py` |
| Query-adaptive read under a token budget | `memcodex/read.py` |
| Arena: direct / flat-retrieval / MemCodex baselines, token-F1 scoring | `memcodex/arena.py` |
| Self-programming: Algorithm 2 (mutate -> evaluate -> accept) + Eq.4 | `memcodex/evolve.py` |
| CLI: `demo` and `evolve` | `memcodex/cli.py` |

## Design notes

- **No heavyweight deps**: `requests` + Python stdlib only. No LangChain /
  LlamaIndex / vector DB. BM25 stands in for the paper's retriever so the
  token-budget semantics stay exact and testable offline.
- **Self-programing = prompt rewriting, not structural mutation.** The
  meta-program rewrites the layer prompt texts; children are accepted when
  they improve accuracy over the parent (ties broken by lower token usage).
  This keeps the loop auditable and the whole system a small, readable
  codebase.
- **Lazy-loading, thin context**: every module is self-contained; the LLM
  client is a plain OpenAI-compatible HTTP call.
- **Deterministic tests** run against a fake LLM (`tests/fake_llm.py`), so
  `python tests/run.py` needs no network.

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

## Quick start (local LLM)

Point it at any OpenAI-compatible endpoint (verified on this machine with
the local Qwen server):

```bash
export MEMCODEX_LLM_BASE_URL=http://127.0.7.0.1:1820/v1
export MEMCODEX_LLM_API_KEY=local
export MEMCODEX_LLM_MODEL=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python -m memcodex demo
```

Note: the demo on the bundled 6-question sample is a smoke test, not a
superiority claim — on that small sample the direct baseline can be
competitive. Run `evolve` with a real dataset and more rounds for the
intended comparison.

## Self-programming (evolve)

```bash
python -m memcodex evolve --rounds 4 --children 3 --archive archive.jsonl
```

Each round: candidate programs (rewritten layer prompts) are generated,
evaluated on the training split in the arena, and candidates that beat their
parent under the Eq.4 acceptance rule are appended to the archive
(`archive.jsonl`).

## Configuration

| Env var | Default |
|---|---|
| `MEMCODEX_LLM_BASE_URL` | `http://127.0.0.1:18200/v1` |
| `MEMCODEX_LLM_API_KEY` | `local` |
| `MEMCODEX_LLM_MODEL` | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| `MEMCODEX_DATASET_PATH` | `examples/demo_dataset/episodes.json` |
| `MEMCODEX_TOTAL_BUDGET` | `1024` (tokens per question read) |

## Tests

```bash
python tests/run.py
```

12 tests (layers / read / acceptance / evolve), all offline, no pytest
dependency.

## License

MIT.
