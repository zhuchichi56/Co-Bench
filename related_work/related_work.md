# RouterXBench v2 — Life-long LLM Routing Literature Review

**Time window**: 2025-10-01 → 2026-05-17
**Date compiled**: 2026-05-17
**Total verified papers**: 62 (54 newly identified + 4 already-known competitors + 4 pre-window references kept for context)

---

## Executive Summary

The LLM-routing literature has exploded between 2025-10 and 2026-05. The dominant trend is a shift away from monolithic supervised routers (RouteLLM 2024, RouterDC 2024) toward (a) **training-free / zero-shot** routers that target the "new model" problem, (b) **online / bandit** routers that target non-stationary cost-quality regimes, and (c) **probe-based** routers that reuse the deployment LLM's own hidden states. The community has begun to call out *router collapse* (EquiRouter, 2602.03478) and *the routing-benchmark gap* (LLMRouterBench, 2601.07206; RouterArena, 2510.00202; RouterXBench v1, 2602.11877). What is still missing — and where RouterXBench v2 sits — is a **joint study** of the three generalization axes (query distribution, model swap, cost regime) under a *training-free, single-node, static-deploy* assumption. No paper found in the time window owns all three axes simultaneously.

---

## Paper Table (verified arxiv IDs)

Legend: A=Routing core, B=Methodology (verifier/judge/uncertainty), C=Adaptation/lifelong, D=Eval/benchmark.
Axis1 = query distribution; Axis2 = model swap; Axis3 = cost regime. n=none, p=partial, f=full.

| # | arXiv ID | Date | Title (short) | Cat | Axis1 | Axis2 | Axis3 | Adaptation | Deploy | Code | Main Metric | Relation |
|---|---------|------|---------------|-----|-------|-------|-------|------------|--------|------|-------------|----------|
| 1 | 2510.00202 | 25-09-30 | RouterArena | A/D | f | n | p | none | static | yes | 8400 queries, 23 datasets | **competitor** |
| 2 | 2601.07206 | 26-01-12 | LLMRouterBench | A/D | p | p | f | none | static | yes | 400K instances, 33 models | **competitor** |
| 3 | 2602.11877 | 26-02-12 | RouterXBench v1 | A/D | p | n | p | none | static | yes | +16.68% router ability | self |
| 4 | 2602.03478 | 26-02-03 | EquiRouter / Routing Collapses | A | n | n | f | none | static | yes | -17% cost at GPT-4 level | known competitor |
| 5 | 2601.22318 | 26-01-29 | Federate-the-Router | A/C | p | n | p | federated | federated | unclear | cost/accuracy gap | known competitor |
| 6 | 2604.00136 | 26-03-31 | ParetoBandit | A/C | n | p | f | online | online_reward | yes | 0.4% budget compliance | known competitor |
| 7 | 2601.06220 | 26-01-09 | ZeroRouter / Universal Latent Space | A/C | p | f | f | zero-shot | static | unclear | 50 unseen LLMs | **Top Threat** |
| 8 | 2509.02718 | 25-09-02 | PORT (Training-Free Online Routing) | A/C | p | n | f | training-free | online | yes (fzwark/PORT) | 3.55x perf, NeurIPS 2025 | **Top Threat** |
| 9 | 2510.09719 | 25-10-10 | ICL-Router | A/C | p | f | p | few-shot | static | yes | AAAI 2026 | **Top Threat** |
| 10 | 2603.20895 | 26-03-21 | LLM Router: Prefill Activations | A | p | p | f | none | static | unclear | 45.58% gap closed, 74% cost save | reproduce |
| 11 | 2602.09924 | 26-02-10 | LLMs Encode Their Failures (linear probes) | A/B | p | p | f | none | static | yes (KabakaWilliam/llms_know_difficulty) | 37% cost reduction @ AIME | reproduce |
| 12 | 2601.03511 | 26-01-07 | IntroLM | A/B | p | p | p | none | static | unclear | AUROC 90% (Qwen3-8B); ACL 2026 Findings | reproduce |
| 13 | 2605.02241 | 26-05-04 | Zero-Shot Confidence for Small LLMs | B/A | f | p | p | zero-shot | static | yes | AUROC 0.717-0.833 OOD | **Top Threat** |
| 14 | 2510.07429 | 25-10-08 | BaRP (Bandit Feedback Routing) | A/C | p | n | f | online | online_reward | unclear | +12.46% over offline | reproduce |
| 15 | 2510.08439 | 25-10-09 | xRouter (RL routing) | A/C | n | n | f | RL/online | online_reward | yes | Pareto frontier improvement | reproduce |
| 16 | 2602.02823 | 26-02-02 | R2-Router (length-budget) | A | p | p | f | none | static | unclear | SOTA at 4-5x lower cost | reproduce |
| 17 | 2604.22520 | 26-04-30 | RouteLMT (translation router) | A | p | p | p | LoRA-trained | static | unclear | translation BLEU/gain | cite_only |
| 18 | 2604.23530 | 26-04-26 | MTRouter (multi-turn) | A | p | p | f | trained | static | yes | -58.7% cost vs GPT-5; ACL 2026 | reproduce |
| 19 | 2603.04445 | 26-02-23 | Dynamic Routing Survey | A | n | n | n | n/a | n/a | n/a | survey | cite_only |
| 20 | 2511.10233 | 25-11-13 | EvoReal (synthetic→real routing) | A | f | n | n | progressive adapt | static | unclear | combinatorial routing | orthogonal |
| 21 | 2502.08773 | 25-02-13 | UniRoute (Universal Model Routing) | A/C | p | f | p | training-free | static | unclear | 30+ unseen LLMs | pre-window cite |
| 22 | 2506.16655 | 25-06-19 | Arch-Router (preference) | A | n | f | n | few-shot | static | yes (HF) | 93.17% acc | cite_only |
| 23 | 2509.22984 | 25-09-26 | Inter-Cascade (deferral→learning) | A/C | p | n | p | online_KD | online | unclear | +33% weak, -49.6% cost | reproduce |
| 24 | 2602.21227 | 26-02-04 | Budget-Aware Agentic Routing | A | p | n | f | trained | static | unclear | sequential cost-success | reproduce |
| 25 | 2603.30035 | 26-03-31 | Reward-Based Online Routing (NeuralUCB) | A/C | n | n | f | online | online_reward | unclear | utility reward | cite_only |
| 26 | 2601.01330 | 26-01-04 | JiSi / Beyond Gemini-3-Pro | A | n | f | p | none | static | unclear | 47% of Gemini-3-Pro cost | reproduce |
| 27 | 2604.17650 | 26-04-19 | LENS (prompt drift evaluation) | D | f | n | n | none | static | unclear | 73% avg loss under shift | **Top Threat** |
| 28 | 2502.16268 | 25-02-23 | ThinkBench (OOD reasoning eval) | D | f | n | n | none | static | unclear | OOD reasoning robustness | cite_only |
| 29 | 2605.07180 | 26-05-08 | BoundaryRouter / RouteBench | A/D | f | n | p | trained | static | unclear | -60.6% latency vs agent | reproduce |
| 30 | 2604.07036 | 26-04-08 | ReDAct (uncertainty-aware deferral) | A/B | p | p | f | none | static | unclear | 15% defer matches large | reproduce |
| 31 | 2603.21172 | 26-03-22 | Entropy Insufficient (selective pred) | B | p | n | p | none | static | unclear | TriviaQA, BioASQ, MedicalQA | reproduce |
| 32 | 2603.08907 | 26-03-09 | TIB Cross-Domain UQ | B | f | n | n | conformal | static | unclear | MASSIVE, NyayaBench, CLINC, Banking77 | orthogonal |
| 33 | 2604.27914 | 26-04-30 | Geometry-Calibrated Conformal Abstention | B | p | n | p | none | static | unclear | 75% conditional correctness | cite_only |
| 34 | 2604.13991 | 26-04-13 | Adaptive Conformal for Factuality | B | p | n | p | conformal | static | unclear | factuality span filtering | cite_only |
| 35 | 2603.24704 | 26-03-31 | Conformal Selective with Risk Control | B | p | n | n | conformal | static | unclear | abstention guarantees | cite_only |
| 36 | 2602.13110 | 26-02-19 | SCOPE conformal pairwise judging | B | p | n | n | none | static | unclear | calibrated abstention | cite_only |
| 37 | 2602.05073 | 26-03 | UQ in LLM Agents (survey) | B | n | n | n | n/a | n/a | n/a | survey | cite_only |
| 38 | 2512.22245 | 25-12-23 | Linear Probes for Judge Calibration | B | p | p | n | none | static | unclear | 10x compute saving, Brier | reproduce-as-judge |
| 39 | 2508.06225 | 25-08 | Overconfidence in LLM-as-Judge | B | n | n | n | none | static | unclear | TH-Score | cite_only |
| 40 | 2511.21140 | 25-11-26 | How to Correctly Report LLM-as-Judge | B/D | n | n | n | n/a | n/a | unclear | bias correction | cite_only |
| 41 | 2510.06265 | 25-10-07 | LLM Hallucination Survey | B | n | n | n | n/a | n/a | n/a | survey | cite_only |
| 42 | 2512.19920 | 25-12 | Behaviorally Calibrated RL | B | n | n | n | RL | static | unclear | accuracy/halluc ratio | cite_only |
| 43 | 2603.12658 | 26-03-13 | Continual Learning in LLMs survey | C | n | n | n | n/a | n/a | n/a | survey | cite_only |
| 44 | 2604.14375 | 26-04-15 | Modular Continual Learning (Zero-Leakage Routing) | C/A | p | n | n | continual | static | unclear | task discovery | orthogonal |
| 45 | 2601.18510 | 26-01-26 | JitRL (Just-in-Time RL for agents) | C | n | n | n | online/RL | online | unclear | dynamic memory | orthogonal |
| 46 | 2511.04847 | 25-11-06 | TTA for LLM Agents via Env | C | p | n | n | online/TTA | online | unclear | environment-interaction | cite_only |
| 47 | 2510.10223 | 25-10-11 | SyTTA (4 extra tokens) | C | f | n | n | TTA | static | upon-accept | +120% agri-QA | reproduce |
| 48 | 2505.20633 | 25-05-26 | Test-Time Learning for LLMs (TLM) | C | f | n | n | TTA | static | unclear | perplexity minimization | pre-window cite |
| 49 | 2602.11167 | 26-02-15 | Visualizing Factual Hallucination via Internal States | B | n | n | n | none | static | unclear | clustering hallucinations | cite_only |
| 50 | 2603.01326 | 26-03-03 | Truth as a Trajectory (TaT) | B | p | n | n | none | static | unclear | geometry of correctness | orthogonal |
| 51 | 2604.19974 | 26-04-29 | Sparse Autoencoders for Correctness | B | p | n | n | none | static | unclear | 62→81% accuracy at 53% cov | orthogonal |
| 52 | 2511.04418 | 25-11-06 | Illusion of Certainty (UQ failure under shift) | B/D | f | n | p | none | static | unclear | UQ fails | **Top Threat** |
| 53 | 2604.17112 | 26-04-25 | Cross-Model Disagreement UQ | B | p | f | n | none | static | unclear | self-consistency + disagreement | reproduce |
| 54 | 2605.09195 | 26-05-13 | Geometry of Forgetting (drift axis) | B/D | f | n | n | none | static | unclear | drift orthogonal to correctness | cite_only |
| 55 | 2605.07180 supplement | — | — | — | — | — | — | — | — | — | — | — |
| 56 | 2603.23848 | 26-03-30 | BeliefShift (temporal drift bench) | D | f | n | n | none | static | unclear | 2400 trajectories | orthogonal |
| 57 | 2604.05096 | 26-04-08 | RAG-or-Learning (continuous knowledge drift) | C/D | f | n | n | RAG | static | unclear | drift adaptation | orthogonal |
| 58 | 2603.08999 | 26-03-12 | Confidence-Aware Self-Consistency | B | n | n | p | none | static | unclear | -80% tokens | cite_only |
| 59 | 2602.08948 | 26-02-13 | CoRefine (confidence-guided refine) | B | n | n | p | none | static | unclear | test-time compute | cite_only |
| 60 | 2601.05905 | 26-01-12 | Diagnosing Truthfulness via Neighborhood | B | n | n | n | none | static | unclear | truthfulness probe | cite_only |
| 61 | 2605.14241 | 26-05-21 (border) | Latency-Quality Routing (LQM-ContextRoute) | A | n | p | f | online_bandit | online | unclear | latency-quality | cite_only |
| 62 | 2602.01240 | 26-02-02 | Prototype-Based Routing (text detection) | A | f | p | n | zero-shot | static | unclear | detector routing | orthogonal |

(IDs 21, 22, 48 are pre-window but kept as required cites; 55 was a placeholder removed.)

---

## Top 5 Threats — papers that overlap our niche most

These are works that another reviewer will hold up as "this is essentially what RouterXBench v2 already does." We must explicitly differentiate.

| # | arXiv | Title | Overlap (why scary) | How we differentiate |
|---|-------|-------|---------------------|----------------------|
| **T1** | 2601.06220 | ZeroRouter (Breaking Model Lock-in) | Universal latent space → zero-shot onboard 50 unseen LLMs. Hits axis-2 (model swap) full and axis-3 (cost regime) full. Frames itself as "training-free adaptation." | They still need offline preference data to *learn* the universal latent space; they do not test cross-domain query shift (axis-1). Our axis-1 results + training-free axis-2 mechanism (no offline data) is the wedge. |
| **T2** | 2509.02718 | PORT (Training-Free Online Routing) | Explicitly named "first training-free online routing." NeurIPS 2025. Full axis-3, partial axis-1. Open-sourced. | Pure online setting with one-time optimization over initial queries. *Requires* on-stream queries to refine. We do *static* deploy with no online observation budget — this is a strictly harder regime. |
| **T3** | 2510.09719 | ICL-Router | AAAI 2026. Few-shot capability vectors → "seamless integration of new models without retraining." Full axis-2. | Their model representation needs profiling on a representative query set ahead of inference. We do *zero* profiling examples. |
| **T4** | 2605.02241 | Zero-Shot Confidence for Small LLMs | Demonstrates that token log-prob *zero-shot* matches supervised baselines (AUROC 0.83 OOD). Full axis-1 (cross-domain robustness as primary contribution). | They only consider 2-model local-to-cloud; we do K-model + abstain. They do not jointly study model swap. |
| **T5** | 2511.04418 | Illusion of Certainty | Shows UQ-based routers / abstainers *fail under distribution shift*. Directly threatens any uncertainty-based router on our axis-1 stress test. | Use this as a *motivating citation*, then position our training-free framework as a robustness intervention. Risk: if their paper already proposes a fix, we're scooped. |

---

## Reproduce list (8–12 baselines) — strongly recommended to run

| Priority | arXiv | Why reproduce |
|----------|-------|---------------|
| P0 | 2601.06220 (ZeroRouter) | Most direct competitor on axis-2 + axis-3. Must beat or differentiate. |
| P0 | 2509.02718 (PORT) | Training-free online competitor; only fair comparison if we cripple it to static. |
| P0 | 2602.03478 (EquiRouter) | Rank-based router; known to mitigate collapse. Strong cost-regime baseline. |
| P0 | 2510.09719 (ICL-Router) | Few-shot model swap. Hits axis-2. |
| P1 | 2603.20895 (Prefill Activations) | Probe-based, training-free flavor; tests our hidden-state premise. |
| P1 | 2602.09924 (LLMs Encode Failures) | Linear-probe routing; closest in spirit to our ProbeDirichlet. |
| P1 | 2601.03511 (IntroLM) | Pre-generation self-evaluation; ACL 2026 Findings. |
| P1 | 2510.07429 (BaRP) | Bandit-feedback router; needs to be shown weak under static deploy. |
| P2 | 2602.02823 (R2-Router) | Length-budget axis variant of axis-3. |
| P2 | 2604.07036 (ReDAct) | Uncertainty-based deferral; strong abstain baseline. |
| P2 | 2509.22984 (Inter-Cascade) | Online knowledge distillation; positions our static no-distillation choice. |
| P2 | 2502.08773 (UniRoute) | Pre-window canonical universal routing baseline. |

---

## Cite-only list — must reference, no implementation needed

- 2603.04445 (Moslem & Kelleher) Dynamic Model Routing Survey — primary survey reference
- 2603.12658 Continual Learning in LLMs (survey)
- 2602.05073 UQ in LLM Agents (survey)
- 2510.06265 LLM Hallucination Survey
- 2506.16655 Arch-Router (preference routing, pre-window)
- 2406.18665 RouteLLM (canonical 2024, pre-window)
- 2403.12031 RouterBench (original benchmark, pre-window)
- 2410.10347 Cascade Routing (Dekoninck) — unified theoretical framing
- 2604.27914, 2604.13991, 2603.24704, 2602.13110 (conformal abstention family)
- 2511.21140 How to Correctly Report LLM-as-Judge — methodology guide
- 2512.22245 Linear Probes for Judge Calibration — methodology guide
- 2604.17650 LENS (prompt drift) — motivating distribution-shift work
- 2605.09195 Geometry of Forgetting — knowledge drift orthogonal axis
- 2603.23848 BeliefShift — temporal drift benchmark
- 2601.01330 JiSi — model-aggregation alternative paradigm
- 2603.08999, 2602.08948 — confidence-aware self-consistency (test-time compute)
- 2604.00136 ParetoBandit — known competitor (already on our list)
- 2601.22318 Federate-the-Router — known competitor (already on our list)

---

## Orthogonal but useful (methods we may borrow)

- 2604.19974 Sparse Autoencoder correctness features — could be a feature for our training-free router
- 2603.01326 Truth as a Trajectory (TaT) — alternative geometric signal vs. our Dirichlet aggregation
- 2603.08907 Transfer-Informed Betting — formal cross-domain UQ; possible theoretical scaffold for axis-1 guarantees
- 2604.17112 Cross-Model Disagreement — gives us a *training-free* axis-2 signal (multi-model ensemble disagreement)
- 2604.14375 Modular Continual Learning Routing — zero-leakage reconstruction; nice analogy
- 2603.13426 OATS (Outcome-Aware Tool Selection) — offline embedding adjustment, no train-time cost
- 2511.10233 EvoReal — synthetic→real generalization framework

---

## Field landscape — 4 dominant themes (2025-10 → 2026-05)

1. **Probe-based / activation routing** (2603.20895, 2602.09924, 2601.03511, 2601.13288). Reusing the deployed LLM's own hidden states. This is RouterXBench v1's home territory and is now crowded — we must clearly state what is new in v2.
2. **Training-free / zero-shot routing for model swap** (2601.06220, 2502.08773, 2510.09719, 2509.02718). The "new model arrives" problem is recognized as canonical. Most solutions still require *some* profiling data; pure axis-2 zero-shot with *zero examples* is open.
3. **Online / bandit routing for cost regime** (2604.00136 ParetoBandit, 2510.07429 BaRP, 2603.30035 NeuralUCB, 2510.08439 xRouter, 2509.02718 PORT, 2605.14241 LQM-ContextRoute). All assume online reward feedback. Our static deploy without online reward sits *outside* this entire cluster.
4. **Uncertainty / abstention as routing primitive** (2604.07036 ReDAct, 2603.21172 Entropy Insufficient, 2604.27914 Conformal Abstention, 2511.04418 Illusion of Certainty). The community is converging on "UQ-driven abstain = the strong-model deferral signal," but 2511.04418 is a red flag that UQ fails under shift — which is exactly axis-1.

Evaluation: three major benchmarks in-window — RouterArena (2510.00202), LLMRouterBench (2601.07206), and our RouterXBench v1 (2602.11877). RouteBench (2605.07180) is a router-for-agent-or-not variant. None of them jointly test *all three* of our axes.

---

## Final Verdict (≈220 words)

**Is the niche still defensible? — Yes, but narrower than before.**

The three-axis framing (query distribution × model swap × cost regime) is *not* claimed by any single paper in the time window. ZeroRouter (2601.06220) owns axes 2+3 with training-free flavor; PORT (2509.02718) owns axes 1+3 but is online-only; Zero-Shot Confidence for Small LLMs (2605.02241) owns axis-1 only. **The unique wedge for RouterXBench v2 is jointly hitting all three under a static-deploy, no-online-reward, no-federation assumption — and crucially proving that training-free adaptation can match training-based competitors on each.**

**Has "life-long routing" been claimed?** Direct search returns *no* paper using "life-long routing" or "lifelong router" as primary terminology in the window. "Continual learning in LLMs" (2603.12658) is the closest concept but does not target routers. The term is open.

**Risks I had not flagged:** (1) **2511.04418 "Illusion of Certainty"** shows UQ-based selection fails under shift — if our training-free method is UQ-based, our axis-1 claim could be challenged. (2) **2603.20895 "Prefill is All You Need"** uses prefill activations exactly like RouterXBench v1 — we need to clearly state how v2 advances beyond v1. (3) **2510.07429 BaRP "One Policy, Many Trade-offs"** claims one router covers many cost regimes — directly threatens axis-3 framing. (4) **2605.07180 RouteBench** (May 8, 2026) introduces three-dimensional routing generalization (in-domain / paraphrase / OOD); we must either subsume or cite — preferably both.

---

## Sources

- [LLMRouterBench](https://arxiv.org/abs/2601.07206)
- [RouterArena](https://arxiv.org/abs/2510.00202)
- [Dynamic Model Routing Survey](https://arxiv.org/abs/2603.04445)
- [ZeroRouter](https://arxiv.org/abs/2601.06220)
- [PORT (Training-Free Online Routing)](https://arxiv.org/abs/2509.02718)
- [ICL-Router](https://arxiv.org/abs/2510.09719)
- [EquiRouter](https://arxiv.org/abs/2602.03478)
- [Federate-the-Router](https://arxiv.org/abs/2601.22318)
- [ParetoBandit](https://arxiv.org/abs/2604.00136)
- [RouterXBench v1](https://arxiv.org/abs/2602.11877)
- [Zero-Shot Confidence Small LLMs](https://arxiv.org/abs/2605.02241)
- [Prefill Activations Routing](https://arxiv.org/abs/2603.20895)
- [LLMs Encode Their Failures](https://arxiv.org/abs/2602.09924)
- [IntroLM](https://arxiv.org/abs/2601.03511)
- [BaRP Bandit Routing](https://arxiv.org/abs/2510.07429)
- [xRouter RL](https://arxiv.org/abs/2510.08439)
- [R2-Router](https://arxiv.org/abs/2602.02823)
- [MTRouter](https://arxiv.org/abs/2604.23530)
- [BoundaryRouter / RouteBench](https://arxiv.org/abs/2605.07180)
- [ReDAct](https://arxiv.org/abs/2604.07036)
- [Inter-Cascade](https://arxiv.org/abs/2509.22984)
- [Budget-Aware Agentic Routing](https://arxiv.org/abs/2602.21227)
- [Reward-Based Online (NeuralUCB)](https://arxiv.org/abs/2603.30035)
- [JiSi Beyond Gemini-3-Pro](https://arxiv.org/abs/2601.01330)
- [LENS Distribution Shift](https://arxiv.org/abs/2604.17650)
- [Illusion of Certainty](https://arxiv.org/abs/2511.04418)
- [Entropy Insufficient](https://arxiv.org/abs/2603.21172)
- [Geometry-Calibrated Conformal Abstention](https://arxiv.org/abs/2604.27914)
- [TIB Cross-Domain UQ](https://arxiv.org/abs/2603.08907)
- [SyTTA](https://arxiv.org/abs/2510.10223)
- [Continual Learning Survey](https://arxiv.org/abs/2603.12658)
- [Multilingual Routing in MoE](https://arxiv.org/abs/2510.04694)
- [Linear Probes for Judge Calibration](https://arxiv.org/abs/2512.22245)
- [Overconfidence in LLM-as-Judge](https://arxiv.org/abs/2508.06225)
- [BeliefShift](https://arxiv.org/abs/2603.23848)
- [Geometry of Forgetting](https://arxiv.org/abs/2605.09195)
- [Sparse Autoencoder Correctness](https://arxiv.org/abs/2604.19974)
- [Truth as a Trajectory](https://arxiv.org/abs/2603.01326)
- [Arch-Router](https://arxiv.org/abs/2506.16655)
- [UniRoute](https://arxiv.org/abs/2502.08773)
