# MemCodex

A thin, runnable reference implementation of **MemCodex: Self-Programming
Hierarchical Memory for Language Agents** (arXiv:2609.39765v1, Wang & Liu,
University of Montreal).

The paper's claim in one sentence: *memory is a program*. Instead of fixed
retrieval pipelines, MemCodex lets a language agent **compile memory into
hierarchical "layer programs" (construct / index / retrieve) and then
rewrite those programs itself** to maximize answer accuracy under a token
budget.

## What this repo implements

| Paper object | Where |
|---|---|
| Layer interface (construct / index / retrieve) | `memcodex/layers.py` |
| Content channel + routing channel | `memcodex/layers.py` (BM25 over records and over layer-routed query rewrites) |
| QueryAdaptiveRead (Algorithm 1, token-budgeted adaptive read) | `memcodex/read.py` |
| ProgramEvolve (Algorithm 2) with Eq.4 acceptance | `memcodex/evolve.py` |
| Arena + baselines (direct, flat-retrieval) | `memcodex/arena.py` |
| Token heuristics, episode records | `memcodex/records.py` |

Self-programming in this implementation = the meta-agent rewrites each
layer's prompt (and layer composition), not the structural Python methods.
That keeps the loop auditable and the whole system a small, readable codebase.

## Design notes

- **No heavyweight deps**: `requests` + Python stdlib only. No LangChain /
  LlamaIndex / vector DB. BM25 stands in for the paper's retriever so the
  token-budget semantics stay exact and testable offline.
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

Point it at any OpenAI-compatible endpoint (e.g. the local Qwen server on
this machine):

```bash
export MEMCODEX_LLM_BASE_URL=http://127.0.0.1:18200/v1
export MEMCODEX_LLM_API_KEY=local
export MEMCODEX_LLM_MODEL=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python -m memcodex demo
```

Expected output (demo dataset, 9 questions):

```
system         acc  tokens/q     ms/q
direct      ...
flat        ...
memcodex    ...
```

## Evolution loop

```bash
python -m memcodex evolve --rounds 2 --children 2
```

Runs ProgramEvolve: for each generation the meta-agent produces `children`
candidate programs (rewritten layer prompts), each is evaluated on the
training split in the arena, and candidates that beat their parent under the
Eq.4 acceptance rule are appended to the archive (`archive.jsonl`).

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

9/9 offline (fake LLM, no network required).

## Dataset format

`examples/demo_dataset/episodes.json`:

```json
[
  {
    "name": "episode_1",
    "records": ["...one fact per record..."],
    "questions": [{"text": "...", "answer": "..."}]
  }
]
```

## Scope & limitations

- Faithful to the paper's *structure* (layer programs, adaptive read,
  evolution loop, arena); the retrieval kernel is BM25, not the paper's
  full retriever.
- Eq.4 acceptance is implemented as "beat parent accuracy, tie-break by
  lower tokens".
- Single-process, single-endpoint; no distributed evolution.

## License

MIT.