# RouterXBench Experiment Log

Newest entries first. Format: date · stage · goal → outcome.

---

## 2026-05-17 · Literature recon · 2025-10 to 2026-05 routing landscape

**Goal**: 系统调研 routing/LLM cascade 最新文献，验证 RouterXBench v2 的 niche 是否仍 defensible (跨 query 分布 / 模型 swap / cost regime 三轴 + training-free adaptation)。

**Run**: general-purpose agent multi-step web research, 60+ min.

**Output**: `related_work/related_work.md` (21 KB, 62 verified papers).

**Key findings**:
- 62 篇 paper 收录 (54 in-window 2025-10..2026-05 + 4 已知 competitors + 4 pre-window canonical)
- 4 个 dominant theme: probe-based routing / training-free model swap / online bandit cost regime / UQ-based abstention
- 5 个 Top Threat (必须显式区分): ZeroRouter (2601.06220), PORT (2509.02718, NeurIPS 2025), ICL-Router (2510.09719, AAAI 2026), Zero-Shot Confidence (2605.02241), Illusion of Certainty (2511.04418)
- **Niche verdict**: 仍 defensible。No paper covers all 3 axes jointly under {static deploy ∩ no online reward ∩ no federation ∩ no profiling examples ∩ no preference data}.
- **新发现的 risk**:
  - 2603.20895 Prefill Activations 用 prefill activations，跟 RouterXBench v1 重叠
  - 2510.07429 BaRP "one policy many tradeoffs" 直接挑战 axis-3 framing
  - 2605.07180 RouteBench (2026-05-08) 已提"3 维 routing generalization" — 必须 fetch 全文对比
- **"Life-long routing" terminology 未被认领**, 可以 claim
- Recommended P0 baseline 复现: ZeroRouter, PORT, EquiRouter, ICL-Router

**Files**: `related_work/related_work.md`

---

## 2026-05-16 · Stage 0 · raw-data audit

**Goal**: 摸清现有 jsonl schema、样本数、字段分布，判断是否能离线产生 router 训练标签。

**Run**: `python experiment/00_audit_data.py` → `experiment/MAIN_RESULTS.md` Section "Stage 0".

**Key findings**:
- 所有 jsonl 只含 `instruction` + `response`（GT answer）。**无 weak/strong correctness label** → 必须 forward 一次 weak model 才能产生 router 训练标签。
- `big_math_*.jsonl` 额外带 `solve_rate` (float)、`difficulty_tier` (5-bucket)、`source` (8 个来源)、`domain`。`solve_rate` 可直接作 IRT-style 难度软标签。
- `math.jsonl` 带 MATH 标准 `level` (1-5)。
- `mmlu_pro` 14 个 domain 各自一份 jsonl，做 leave-one-domain-out OOD 直接可用。
- 训练集普遍 4000 样本，测试集 1000；`magpie` response 极长 (p95 ≈ 4.3k char)，要警惕 prompt 截断。
- `alpaca` `response` 是长文本指令回答，**判对错必须 LLM-judge**；`big_math` / `math` / `mmlu` response 是短答案（数字 / 选项字母 / 数字），**可纯规则匹配**绕开 xVerify。

**Environment**:
- 8× A100-80GB, all idle.
- `/home` 6.2 T free; `/mnt/zhuhe` 2.7 T mounted (rslex datastore).
- `/mnt/zhuhe/models/` 包含 Llama-3.1-8B-Instruct、Llama-3.1-70B-FP8、Qwen2.5 全家、Qwen3 全家、QwQ-32B 等，**无 xVerify**。

**Decisions** (user-confirmed):
- Weak = Llama-3.1-8B-Instruct
- Judge = pure rule-based for big_math/math/mmlu（绕开 xVerify）；alpaca 暂缓
- Smoke test: 200 × 3 dataset (big_math/mmlu + 备用 alpaca)

**Open items**:
- 是否真的需要安装 xVerify？若仅用 rule-based judge 走 smoke test 可完全绕开。
- Strong model 是否本次先不跑（只算 weak correctness 训 router）。
