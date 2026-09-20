# Progress Report: Foundation of Robotics

## 1. Project Title

**Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents**

*Report date: 2026-09-20. Stage: literature review (preliminary). No implementation or experiments yet.*

## 2. Team

Members listed in the team literature workbook: Atharva Narkhede, Ranjeet Kulkarni, Pratik Pansare, Agraj Singh. Repository owner: Aditya Kishore.
*[Team list, student IDs and roles to be confirmed before submission.]*

## 3. Introduction

Long-running robots with egocentric cameras accumulate far more visual data than persistent memory can hold. Which observations to keep, and in what form, determines what the robot can later recall or act upon. This report summarizes the project's problem formulation, a **preliminary** literature review of **8 papers** (half of the planned 16), the research gap identified so far, and the plan for the work. The project is at the literature-review stage; **the proposed method has not been implemented and no experimental results exist.**

## 4. Problem Statement

Given a continuous egocentric visual stream and a fixed persistent-memory budget `B`, decide which observations or events to retain, which to discard, and at what fidelity, so that future task performance degrades as little as possible. The future task is unknown when memory is written.

Conceptual formulation (to be investigated, not a final algorithm):

```
maximize   E_{tau ~ p(tau)} [ U(tau, M_T) ]   subject to   cost(M_t) <= B
```

where `M_t` is the memory at time `t`, `tau` is a future task drawn from an unknown distribution, and `U` is downstream performance of a fixed reader.

## 5. Motivation

Storing every observation is costly in storage, context length and latency; aggressive deletion may remove evidence needed later (e.g., "take me where the spill happened"). The reviewed papers, as summarized in the team workbook, each cite these pressures: storage limits (P01, P02), context-window and latency limits (P03, P08), and token and latency cost of dense history (P05, P06).

## 6. Literature Review

**Scope and limitations of this review.** Only **8 papers** (P01-P08) have been reviewed, out of a planned 16. Rows P09-P16 (Pratik Pansare, Agraj Singh) are empty. **The papers were not independently re-read for this preliminary report:** all paper details below are taken from the team workbook and are unverified against the original papers. Conclusions are therefore preliminary.

| ID | Paper (preprint date) | Memory / method (as recorded in workbook) | Task known at write time? | Budget | Online / offline formation |
|---|---|---|---|---|---|
| P01 | Worth Remembering: Surprise-Gated Robot Episodic Memory (arXiv 2026-06) | Bayesian-surprise gate in V-JEPA-2 latent space; event layer on a 4D scene graph | No | Implicit (threshold); compared at equal budget with uniform/random | Online, causal (as recorded) |
| P02 | Learning to Forget: H²-EMV (arXiv 2026-04) | Hierarchical episodic memory; LLM relevance with rules learned from user feedback | No (learned from feedback) | 45% memory reduction reported (relative) | Not verified |
| P03 | ReMEmbR (arXiv 2024-09; workbook lists 2025†) | VLM captions + embeddings in vector DB with pose/time; query-time retrieval | No | None | Not verified |
| P04 | RoboMemory (arXiv 2025-08) | Episodic/semantic/spatial/procedural memories with consolidation | No | None unified | Not verified |
| P05 | Scaling Up Memory for Robot Control via Experience Retrieval (MemER; arXiv 2025-10-23; ICLR 2026) | Task-conditioned VLM keyframe selection; bounded keyframe set | **Yes** | ≤8 keyframes + 8 recent frames; no eviction | Not verified |
| P06 | NativeMEM (arXiv 2026-07) | One native token per frame per view; learned with action-prediction loss | No (compression is generic) | 1 token/frame-view; ~5,000 frames in 32 GB | Not verified |
| P07 | AMEGO (arXiv 2024-09) | Hand-object interaction tracklets and location segments | No | No explicit budget | Online (as recorded) |
| P08 | EgoLife (arXiv 2025-03) | Hierarchical summaries (clip, hour, day) + retrieval | Partially | No fixed budget | Not verified |

Dates in the table are the arXiv preprint month, read from each arXiv ID, except P05 whose submission date and venue were supplied by the team. Venue/publication years are not recorded in the workbook other than for P05 (ICLR 2026). †The workbook's "Year" column appears to mix preprint and venue years (for P03 it lists 2025 although the arXiv ID indicates September 2024); to be corrected by the team.

**Reported results (as recorded in the workbook; unverified against the papers and not comparable across papers).** P01: ≥12% improvement over a scene-graph memory baseline (DAAAM) on QA, and better than uniform/random at equal budget. P02: 45% smaller memory and 35% lower query compute with QA accuracy maintained. P05: >90% task progress on three real-robot tasks; long-history baseline ~34% worse on average. P06: 84.0% simulated average success vs. 32.4% for Mem-0; 98.7% real-robot average. P07: 12.7% above common video-QA baselines on AMB. P08: 35.4% vs. 25.0% on >24 h QA. P03 and P04 report qualitative improvements over their baselines (no numbers recorded for P04).

## 7. Existing Approaches

Our taxonomy (an interpretation of the workbook) groups the work as:

- **Retrieval-based memory:** P03, P08 (retrieval stage); keep a large store, select at query time.
- **Frame/keyframe selection:** P05 (task-conditioned), P07 (interaction-based).
- **Token-level compression:** P06; every frame is compressed to one token.
- **Hierarchical summarization:** P08, P02.
- **Surprise/novelty-based memory:** P01.
- **Learned forgetting:** P02.
- **Multi-memory and consolidation:** P04.

## 8. Research Gap

**Our novelty is not established.** In particular, *Worth Remembering* (P01) already addresses unknown future tasks using surprise-gated memory formation, so **our possible contribution is not simply unknown-future-task memory selection.** P05 addresses task-conditioned bounded memory (task known) and P02 addresses learned forgetting (user feedback).

The preliminary review suggests a potential research gap. The reviewed methods appear to differ in *how* they allocate memory: write-time surprise gating (P01), interaction-based storage (P07), task-conditioned keyframe selection (P05), learned forgetting (P02), fixed per-frame compression (P06), hierarchical summarization (P08, P02), and query-time retrieval (P03). However, the current review has not established a method that **jointly** performs:

1. explicit utility estimation,
2. hard memory-budget allocation,
3. variable-fidelity representation,
4. online eviction/reallocation, and
5. evaluation across multiple future task types.

**This is a hypothesis, not a finding. It requires validation against a broader literature review.** The workbook records partial coverage only: proxy signals rather than utility estimates (P01, P02, P05, P07); an implicit budget (P01) or a bounded set without eviction (P05); fixed rather than variable fidelity (P06, P08); and evaluations that are mostly QA-style or control-style separately. Several reviewed papers do report multiple question types (P01, P03, P07, P08), so the last point is one of degree (the same memory under varying budgets and task types), not absence. Unreviewed work (P09-P16 and adjacent areas such as video-LLM frame/token selection) could already cover parts of this.

## 9. Proposed Direction

Rather than proposing a complex architecture now, the project takes a staged, falsifiable approach:

- **H0 (gap validation):** single proxies (surprise, interaction, recency) are inconsistent predictors of retrospective downstream utility across task types and fall short of an oracle selection.
- **H1 (allocation):** if H0 holds, an explicit budget-constrained allocation using a multi-component utility estimate and per-item fidelity outperforms single-signal baselines at equal budget across task types.
- **Pivot rule:** if H0 fails, the project shifts to an analysis/benchmark contribution or to another gap identified by the complete review.

**Potential contribution (contingent on validation).** A budget-controlled comparison of existing memory-allocation signals (surprise, interaction, recency, task-conditioned) against an oracle retrospective-utility reference across multiple future task types; and, only if that comparison shows a measurable gap, a simple allocation policy combining explicit utility estimation, a hard memory budget, variable-fidelity storage and online eviction/reallocation. No improvement over P01 or any other method is claimed.

The eventual method may change after deeper literature analysis.

## 10. Methodology / Initial Framework

Conceptual pipeline (not implemented):

```
stream -> candidate items -> utility estimator (pluggable) -> budgeted allocator -> memory M_t -> frozen reader -> tasks
                                                     (keep / evict / re-compress, fidelity tiers)
```

Candidate utility components (hypotheses only): novelty, interaction/state change, semantic importance, temporal context, object/entity importance, uncertainty, estimated future task utility. The framework expresses existing approaches as special cases (e.g., P01 as a surprise-scored write gate), which enables equal-budget comparison. This is a modeling convenience, not evidence of novelty.

## 11. Experimental Plan

1. **Common harness:** frozen downstream reader; single budget-accounting scheme; interchangeable memory policies.
2. **Baselines at equal budget:** full memory, uniform, random, recency, surprise gating (approximation of P01), interaction-based (approximation of P07), task-conditioned with known task (approximation of P05, informed reference), retrieval over a large store (P03 style, different regime).
3. **Gap validation (E1-E3):** oracle retrospective utility versus proxy scores, per task type, across budgets.
4. **If supported (E4-E7):** utility-aware allocation, budget sweep, multi-fidelity, eviction.
5. **Ablations:** utility components, fidelity tiers, eviction, budget, reader sensitivity.
6. **Candidate data** (from the workbook): NaVQA (P03), EPIC-KITCHENS / AMB (P07), EgoLife / EgoLifeQA (P08); availability and compute to be checked.
7. **Metrics:** accuracy versus budget curves per task type; average and worst case; oracle gap; multiple seeds.

Details: [docs/project_proposal.md](../docs/project_proposal.md), [docs/research_roadmap.md](../docs/research_roadmap.md).

## 12. Expected Outcomes

Depending on the gap-validation result:

- A budget-versus-utility **evaluation protocol** and an **empirical analysis** of when single proxies fail or suffice.
- If supported, a simple **utility-aware allocation policy** with ablations showing which components matter.
- A negative or mixed result is also an acceptable outcome and will be reported as such.

No performance improvement is claimed or expected as a foregone conclusion.

## 13. Current Progress

| Item | Status |
|---|---|
| Literature workbook | 8 of 16 papers populated (P01-P08); P09-P16 empty; papers not independently re-read |
| Literature analysis and gap assessment | Drafted: [docs/literature_analysis.md](../docs/literature_analysis.md) |
| Research proposal | Drafted: [docs/project_proposal.md](../docs/project_proposal.md) |
| Research roadmap | Drafted: [docs/research_roadmap.md](../docs/research_roadmap.md) |
| Baseline reproduction | Not started |
| Proposed method | **Not implemented** |
| Experiments and results | **None; no experimental results exist** |

**Next steps:** complete P09-P16 and adjacent areas; verify workbook entries against the original papers; add an online/offline column to the workbook; resolve data-quality items (e.g., workbook year column, P04 details); set up the evaluation harness; then run the gap-validation experiments.

## 14. References

Links as recorded in the team workbook (not independently verified). Years are arXiv preprint years unless a venue is stated.

1. P01. *Worth Remembering: Surprise-Gated Robot Episodic Memory.* arXiv preprint, 2026. https://arxiv.org/abs/2606.03787
2. P02. *Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment (H²-EMV).* arXiv preprint, 2026. https://arxiv.org/abs/2604.11306
3. P03. *ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation.* arXiv preprint, 2024 (workbook lists 2025; venue not recorded). https://arxiv.org/abs/2409.13682 (code: https://github.com/NVIDIA-AI-IOT/remembr)
4. P04. *RoboMemory: A Brain-Inspired Multi-Memory Agentic Framework for Lifelong Learning in Physical Embodied Systems.* arXiv preprint, 2025. https://arxiv.org/abs/2508.19875 (code: https://github.com/1737536149/RoboMemory)
5. P05. *Scaling Up Memory for Robot Control via Experience Retrieval* (MemER). arXiv preprint, 23 October 2025; ICLR 2026. https://arxiv.org/abs/2510.20328 (project page: https://jen-pan.github.io/memer/)
6. P06. *NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation.* arXiv preprint, 2026. https://arxiv.org/abs/2607.06678 (code: https://github.com/OpenDriveLab/NativeMEM)
7. P07. *AMEGO: Active Memory from long EGOcentric videos.* arXiv preprint, 2024. https://arxiv.org/abs/2409.10917 (code: https://github.com/gabrielegoletto/AMEGO)
8. P08. *EgoLife: Towards Egocentric Life Assistant.* arXiv preprint, 2025. https://arxiv.org/abs/2503.03803 (code: https://github.com/EvolvingLMMs-Lab/EgoLife)
