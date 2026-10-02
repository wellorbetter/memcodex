# Memory（memcodex 包）

2026-10-03 | 检索内核=BM25（bm25_index/bm25_search），无 sklearn/whoosh 等外部依赖 | memcodex/layers.py | 本包 | active
2026-10-03 | query_adaptive_read 预算循环：浅层到深层逐层扩读，ReadResult 携带各层读到的内容 | memcodex/read.py | 本包 | active
%s | 评分=token 级 F1（f1_score）非精确匹配；arena.run_system 与 cli 共享 DEFAULT_LAYER_SPECS（键 summary/relational/episode） | memcodex/arena.py | 本包 | active
2026-10-03 | records.load_episodes 用 utf-8-sig 容错 BOM；episodes.json 必须无 BOM 落盘 | memcodex/records.py | 本包 | active
2026-10-03 | NEGATION_MARKERS 中文标记必须 \u 转义，否则源码 mojibake | memcodex/arena.py | 本包 | active
2026-10-03 | FakeLLM 继承 LLM 接口带 token 计数，可离线跑通 arena/evolve 全循环 | tests/fake_llm.py | 本包 | active
