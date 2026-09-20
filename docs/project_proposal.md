# Project Proposal

## Title

**Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents**

**Status:** Proposal at the literature-review stage. Nothing described below has been implemented or evaluated. Literature statements are taken from the team workbook (8 papers; P09-P16 empty) and were not independently re-read. The framing may change after the full literature review (currently 8 of 16 planned papers are populated). See [literature_analysis.md](literature_analysis.md) for the evidence base.

---

## Abstract

A long-running egocentric robot continuously observes its environment, yet its persistent memory is bounded. Storing every observation is expensive and redundant, whereas aggressive discarding may remove evidence that later becomes useful for a task unknown at observation time. We study how an embodied agent should decide which observations or events to retain, which to discard, and at what fidelity, so that future task performance degrades as little as possible under a fixed memory budget.

A preliminary review of eight recent works (surprise-gated episodic memory, learned forgetting, retrieval-based memory, multi-memory architectures, task-conditioned keyframe selection, native token compression, egocentric structured memory, and hierarchical summarization) suggests that existing methods address parts of this problem with single proxies for utility (e.g., surprise, interaction, LLM-judged relevance, or the known task). Notably, Worth Remembering (P01) already addresses unknown future tasks with surprise-gated memory formation, so our possible contribution is not simply unknown-future-task memory selection. The preliminary review suggests a potential research gap: the current review has not established a method that jointly performs explicit utility estimation, hard memory-budget allocation, variable-fidelity representation, online eviction/reallocation, and evaluation across multiple future task types. This hypothesis requires validation against a broader literature review; only 8 papers were reviewed and they were not independently re-read.

The proposal is staged: first reproduce and compare existing selection strategies at equal budgets under a common, frozen downstream reader; then test whether single proxies track retrospective ("oracle") downstream utility; and only then design a utility-aware allocation policy. If existing proxies already track oracle utility, the project pivots toward an analysis/benchmark contribution.

---

## Motivation

- Embodied agents that operate for hours or days accumulate visual histories larger than storage, context windows, or latency budgets allow (reported in P01, P02, P03, P05, P06, P08).
- The relevance of an observation depends on future tasks that are unknown at write time (e.g., "take me where the spill happened", P01; "where did you put my keys?", P02).
- The reviewed works each address a slice: what to write (P01, P07), what to forget (P02), what to retrieve (P03, P08), how to compress (P06), how to organize (P04), what to select when the task is known (P05). Per the workbook, no single reviewed work is recorded as treating the whole problem as budget-constrained allocation over unknown future tasks; this is a hypothesis to be validated, not an established fact.

---

## Problem statement

Given a continuous egocentric visual stream `o_1, ..., o_T` and a persistent memory budget `B`, design a policy that maintains a memory `M_t` (deciding what to write, evict and at what fidelity) such that downstream performance of a fixed reader on future tasks drawn from an unknown distribution `p(tau)` is maximized:

```
maximize   E_{tau ~ p(tau)} [ U(tau, M_T) ]
subject to cost(M_t) <= B   for all t
```

Items may be stored at different fidelities (e.g., raw keyframe, embedding/token, caption, summary), each with a cost. The policy operates online and does not observe `tau` at write time.

This is a conceptual formulation for comparison and analysis; it is not committed as the final algorithm.

---

## Research question

> Given a continuous egocentric visual stream and a fixed memory budget, how should an embodied agent decide which observations/events to retain, which to discard, and at what fidelity, so that future task performance degrades as little as possible?

**Sub-questions.**

1. Do generic strategies (uniform, random, recency) and single-proxy strategies (surprise, interaction) leave a measurable gap to oracle retrospective utility at equal budget?
2. Does the best single proxy depend on the type of future task?
3. Does explicit utility-aware allocation, with per-item fidelity and eviction, provide a meaningful advantage over surprise-based, retrieval-based, task-conditioned, and learned-forgetting approaches, or is it subsumed by them?

---

## Hypothesis

Stated so that it can be falsified.

- **H0 (gap validation):** Under a fixed budget, single-signal proxies (surprise, recency, interaction) are inconsistent predictors of retrospective downstream utility across task types, and each falls short of an oracle-utility selection.
- **H1 (allocation):** A policy that allocates the budget using an explicit multi-component utility estimate (candidates: novelty, interaction/state change, semantic importance, temporal context, object/entity importance, uncertainty, estimated future task utility) and selects fidelity per item achieves better average downstream performance across a set of task types than any single-signal baseline at the same budget.
- **H2 (robustness):** The advantage holds across budget levels and degrades more gracefully as the budget shrinks.

The utility components are hypotheses only. Whether any of them contributes is an empirical question, addressed by ablations.

**Decision rule.** If H0 is not supported, pivot (Direction A in the analysis: benchmark/analysis contribution, or a different gap identified by the complete review). If H1 fails with simple estimators, report a negative result and analyze the failure instead of adding architectural complexity.

---

## Objectives

1. Complete the literature review to 16 papers, including adjacent areas not yet covered, and re-assess the research gap.
2. Build a common evaluation harness: a frozen downstream reader, a budget-accounting scheme, and a set of interchangeable memory policies.
3. Reproduce or approximate baseline behaviors of existing approaches at equal budget.
4. Validate (or refute) the research gap with an oracle-utility analysis (H0).
5. If supported, design and evaluate the simplest utility-aware allocation policy that tests H1 and H2.
6. Report honestly, including negative results.

---

## Proposed conceptual framework

The framework is deliberately simple and modular. It is a *structure for experiments*, not a neural architecture.

```
egocentric stream --> [event/segment proposal] --> candidate items
                                                        |
                          utility estimator  <----------+   (signals: novelty, interaction,
                          (pluggable)                        semantics, temporal context,
                                |                            entity importance, uncertainty,
                                v                            estimated future task utility)
                       budgeted allocator  --> keep / evict / re-compress
                       (fixed budget B,           (choose fidelity per item)
                        fidelity tiers)
                                |
                                v
                     persistent memory M_t  --> frozen reader --> downstream tasks
```

**Components (conceptual).**

- **Item proposal.** Split the stream into candidate items (frames, segments, or events). Choice depends on the dataset.
- **Utility estimator.** A pluggable module producing a score (or scores) per item. Initial versions use hand-crafted or off-the-shelf signals so that each component can be ablated; learned estimators are considered only after the simple versions are understood.
- **Budgeted allocator.** A greedy/threshold or knapsack-style rule that maintains `cost(M_t) <= B`, with three actions: keep, evict, or re-compress to a cheaper fidelity tier.
- **Fidelity tiers.** For example raw keyframe, embedding/token, caption, summary. The set of tiers depends on the reader and is to be decided in Phase 3/4.
- **Frozen reader.** The downstream model answering questions or supporting decisions from memory, held fixed across all policies.

**Interpretation of relation to prior work.** P01 corresponds to a write-time gate with surprise as the score; P05 to a task-conditioned score with a small bounded set; P02 to a learned relevance score with forgetting; P06 and P08 to fixed fidelity schemes. The framework can express these as special cases, which is what enables an equal-budget comparison. This expressiveness is a modeling convenience, not evidence of novelty.

---

## Expected contribution

**Potential contribution wording:** a budget-controlled comparison of existing memory-allocation signals (surprise, interaction, recency, task-conditioned) against an oracle retrospective-utility reference across multiple future task types; and, only if that comparison shows a measurable gap, a simple allocation policy combining explicit utility estimation, a hard memory budget, variable-fidelity storage and online eviction/reallocation. Which of the following is realized depends on Phase 3 results. No claim is made yet.

1. **A budget-vs-utility evaluation protocol** for memory policies over multiple unknown future task types, including an oracle-utility yardstick.
2. **An empirical analysis** of when single proxies (surprise, interaction, recency) succeed or fail relative to oracle utility.
3. **If supported:** a simple utility-aware, multi-fidelity allocation policy and evidence about which utility components matter.

**Novelty caveat.** P01 already addresses task-agnostic, write-time selection under limited memory with equal-budget comparisons (per the workbook). P01 already addresses unknown future tasks, so the project does not claim that unknown-future-task memory selection, or "using a utility signal instead of surprise", is by itself novel. Differentiation, if any, is a hypothesis: the reviewed methods appear to differ in how they allocate memory, but the current review has not established a method that jointly performs explicit utility estimation, hard budget allocation, variable-fidelity representation, online eviction/reallocation and multi-task evaluation. This requires validation against a broader literature review.

---

## Baseline methods

All baselines run at equal budget under the same frozen reader. Approximations of published methods will be labeled as approximations, since P01's code is not recorded and P02's is not released (per the workbook).

| Baseline | Role | Source / analog |
|---|---|---|
| Full memory | Unconstrained reference | - |
| Uniform sampling | Generic baseline | Used in P01 comparison |
| Random retention | Generic baseline | Used in P01 comparison |
| Recency (FIFO) | Generic baseline | Common practice |
| Surprise gating | Novelty-based | Approximation of P01 |
| Interaction-based selection | Event-based | Approximation of P07 |
| Task-conditioned selection (task known) | Informed reference, not a fair competitor | Approximation of P05 |
| Retrieval over a large store | Different-regime reference (no budget) | P03, P08 (retrieval stage) |
| Learned/LLM-based forgetting | Learned-forgetting analog, if feasible | Approximation of P02 (no code; user-feedback part likely out of scope) |
| Fixed per-frame compression | Fidelity-fixed reference, if feasible | P06 |
| Oracle retrospective selection | Upper bound for a given budget | Defined by us |

---

## Evaluation methodology

- **Frozen downstream reader.** One VLM/LLM reader for all policies, so that differences are attributable to memory selection.
- **Budget accounting.** A single cost unit (e.g., stored items and bytes/tokens) applied to every policy; budgets swept over several levels chosen after inspecting dataset statistics. No specific values are fixed at this stage.
- **Multi-task evaluation.** Memory is formed once per stream and then evaluated on several question/task types (e.g., spatial, temporal, object/state, counting) to approximate unknown future tasks.
- **Metrics.** Downstream accuracy versus budget per task type; average and worst-case across task types; write/read cost; oracle gap (policy vs. oracle at same budget).
- **Statistics.** Multiple seeds, confidence intervals, and reporting of negative results.
- **Datasets (candidates from the workbook).** NaVQA (P03), EPIC-KITCHENS / Active Memories Benchmark (P07), EgoLife / EgoLifeQA (P08). Availability, licenses and compute are to be checked. Manipulation settings (P05, P06) are a later, optional extension.

---

## Expected experiments

| # | Experiment | Purpose | Related hypothesis |
|---|---|---|---|
| E1 | Equal-budget baseline comparison (uniform, random, recency, surprise, interaction) on a QA benchmark | Establish what existing strategies already achieve | H0 |
| E2 | Oracle retrospective-utility analysis: correlation between proxy scores (surprise, interaction, recency, ...) and item contribution to downstream answers | Validate or refute the research gap | H0 |
| E3 | Task-type dependence: rank baselines separately per task type | Test whether the best proxy varies by task | H0 |
| E4 | Utility-aware allocation vs. best single-proxy baseline at equal budget | Test the main hypothesis (only if E2/E3 support the gap) | H1 |
| E5 | Budget sweep | Test robustness and graceful degradation | H2 |
| E6 | Multi-fidelity allocation vs. single-fidelity | Test whether fidelity choice adds value | H1 |
| E7 | Eviction/re-scoring as the stream grows vs. write-once | Test whether revisiting decisions matters | H1 |

E4-E7 are contingent on E1-E3. No experiment has been run.

---

## Ablation studies

To be run only once a method exists.

1. **Utility components:** remove or isolate novelty, interaction/state change, semantic importance, temporal context, object/entity importance, uncertainty, estimated future task utility.
2. **Fidelity tiers:** single tier vs. multiple tiers.
3. **Eviction / re-compression:** on vs. off.
4. **Budget level:** sensitivity across the sweep.
5. **Utility estimator:** hand-crafted vs. learned (if learned is attempted).
6. **Reader sensitivity:** result stability across a second reader (if compute allows).
7. **Item granularity:** frame vs. segment vs. event.

---

## Risks and limitations

| Risk | Consequence | Mitigation |
|---|---|---|
| Novelty is pre-empted by P01, P02, P05 or unreviewed work | Weak contribution | Complete the 16-paper review first; use an explicit decision rule (H0) and be ready to pivot to an analysis/benchmark contribution |
| The literature review is incomplete (8 of 16 papers; P09-P16 empty) and the papers were not independently re-read | Gap assessment may be wrong | Fill P09-P16; add missing areas (video-LLM frame/token selection, streaming-video memory, budgeted selection) |
| Surprise already tracks oracle utility | H1 has little room | Treat as a valid finding; pivot |
| "Future task utility" is unobservable at write time; proxy queries may not match real tasks | Circular or misleading utility estimates | Use held-out task types; report per-task-type results |
| Oracle utility (leave-one-out or greedy gain) is expensive or noisy | Weak yardstick | Restrict to modest-size evaluation sets; report variance |
| Baselines are approximations of unreleased or unavailable code | Unfair or unfaithful comparisons | Label approximations; prefer codebases recorded in the workbook |
| Compute/data constraints | Reduced scope | Start with QA on existing benchmarks; manipulation is optional |
| Workbook entries may contain summarization errors | Wrong premises | Verify key claims against original papers |
| Reader-dependent results | Findings may not transfer | Fix one reader; test a second if possible |

**Limitations of scope.** Initially QA-style evaluation; closed-loop robot execution is not planned for the first phase. Evaluation is offline on recorded streams rather than on a physical robot.

---

## Future work

- Extension to closed-loop manipulation and navigation (memory-for-control, cf. P05, P06).
- Learned utility estimators trained from proxy or self-supervised future queries.
- Interaction with retrieval (joint write-time allocation and read-time selection).
- Longer horizons (hours to days) and real-robot deployment.
- Personalization or user feedback as an additional utility signal (cf. P02).

The eventual method may change substantially after deeper literature analysis and the results of the gap-validation phase.
