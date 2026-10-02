# Memory（memcodex 项目）

2026-10-03 | MemCodex=Summary/Relational/Episode 三层 + query-adaptive read + self-programming（只改写层 prompt 文本，不改结构）；实现完成，测试 12/12 通过 | D:\Github\memcodex（python tests/run.py OK: 12/12） | 本仓库 | active
2026-10-03 | 接受规则=子代 acc 高于父代，平手按 token 数少者胜；program_evolve 参数 (base_specs, llm, episodes, rounds=1, parents=1, children=2, budget=1024, archive_path) 返回最优 program 列表，archive 为 JSONL | memcodex/evolve.py | 本仓库 | active
2026-10-03 | demo 是冒烟测试不是优越性宣称：9 题小样本上 memcodex acc 低于 direct，README 如实记录 | D:\Github\memcodex\README.md + demo 会话 | 本仓库 | active
2026-10-03 | 本地 Qwen http://127.0.7.0.1:1820/v1（OpenAI 兼容）可跑 demo/evolve；evolve 完整配置超 120s shell 超时，短配置(2轮2子代)可完成 | memcodex/cli.py 运行记录 | 本机 | active
2026-10-03 | GitHub 发布阻塞：gh 未登录（gh auth status=not logged in），需用户 gh auth login 或提供 token 后建仓库 push | gh auth status 会话 | 本机 | active
