# Literature Analysis and Research-Gap Assessment

**Project:** Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents
**Status:** Preliminary. Based on the 8 papers currently in `literature_review.xlsx` (P01-P08). No implementation exists yet.
**Last updated:** 2026-09-20

---

## 0. Scope, sources, and reading conventions

**Source of evidence.** This document is derived *only* from the team workbook [`literature_review.xlsx`](../literature_review.xlsx) (sheet "Literature Review"). **The source papers were not independently re-read for this preliminary analysis**, so every "reported" statement is *as recorded by the team member who filled the row*, is **unverified against the original paper**, and inherits any summarization error in the workbook. Numbers are quoted from the workbook and are **not comparable across papers** (different benchmarks, metrics, baselines).

**Coverage.**

| Rows | Assigned to | Status |
|---|---|---|
| P01-P04 | Atharva Narkhede | Populated (4 papers) |
| P05-P08 | Ranjeet Kulkarni | Populated (4 papers) |
| P09-P12 | Pratik Pansare | **Empty** - not yet reviewed |
| P13-P16 | Agraj Singh | **Empty** - not yet reviewed |

**Limitation:** only 8 papers were reviewed, P09-P16 are empty, and the papers were not independently re-read. Conclusions below are therefore provisional. Half of the planned review is missing, and topics such as video-LLM frame/token selection, streaming-video memory, map/experience-buffer pruning, and budgeted-selection theory have not been covered at all (see Section 12, gap G8).

**Provenance labels used throughout.**

- **Reported** - what the workbook records for the paper (not independently verified).
- **Interpretation** - our reading or inference; may be wrong.
- **Potential gap** - a hypothesis about what is missing; must be validated before being claimed.

**Data-quality notes on the workbook (to be resolved by the team).**

1. **Preprint vs. venue year.** P05 (MemER) was submitted to arXiv on 23 October 2025 and is an ICLR 2026 paper (dates supplied by the team, not recorded in the workbook); the workbook's 2026 appears to be the venue year. The "Year" column seems to mix preprint and venue years: P03 lists 2025 although its arXiv ID (2409) indicates a September 2024 preprint. Venue years are otherwise not recorded. Team to correct the workbook column (preprint year and venue/year separately).
2. P01: no code link is recorded. P02: recorded as "Not publicly released at time of review". P05: the "code link" is a project page (`jen-pan.github.io/memer/`), not a confirmed repository.
3. P04: dataset ("EmbodiedBench and custom ... environments"), downstream task ("e.g. EmbodiedBench-style ...") and main result ("Improves success rate ...") are hedged and contain **no numbers**. Verify against the paper before relying on them.
4. The workbook has no column for **online vs. offline** operation. Section 9 lists it only where the workbook text supports it (P01, P07); everything else is marked "Not verified" rather than inferred.
5. Rows dated 2026 (P01, P02, P06) were taken from the workbook as given; their arXiv IDs (2606, 2604, 2607) are consistent with 2026 preprints.

---

## 1. Project motivation

A long-running egocentric robot observes a continuous visual stream, but persistent memory is bounded. Storing everything is costly (storage, token/context cost, retrieval latency) and largely redundant. Discarding aggressively risks deleting evidence that becomes valuable for a task that is unknown at the time of observation (e.g., "take me where the spill happened", P01; "where did you put my keys?", P02).

**Reported (from the workbook):** the reviewed works consistently identify the same pressure from different angles - storage limits (P01, P02), context-window/latency limits (P03, P08), and token/latency cost of dense history (P05, P06).

---

## 2. Problem definition

**Informal.** Given a continuous egocentric stream and a fixed persistent-memory budget, decide *which* observations/events to retain, *which* to discard, and *at what fidelity*, so that future task performance degrades as little as possible.

**Candidate formalization (hypothesis, not a final algorithm).**

- Stream: observations `o_1, o_2, ..., o_T`, arriving online.
- Memory state `M_t`, updated by a policy `pi` that may write, evict, or re-compress items.
- Each item `i` may be stored at fidelity `f` (e.g., raw keyframe, single token/embedding, caption, summary) with cost `c(i, f)`.
- Budget: `sum_i c(i, f_i) <= B` at all times (persistent memory).
- Future tasks `tau ~ p(tau)`, unknown when memory is written. `U(tau, M)` is downstream performance of a fixed reader given memory `M`.

```
maximize_pi   E_{tau ~ p(tau)} [ U(tau, M_T) ]
subject to    cost(M_t) <= B   for all t
```

**Interpretation.** If item values were known and static this resembles a multiple-choice knapsack. The difficulty is that values are unknown, task-dependent, and may change with later observations, so the problem is an *online* allocation with a *proxy* for value. The formalization is a lens for comparing papers, not a claim about the right method.

Utility components under consideration (all hypotheses): novelty, interaction/state change, semantic importance, temporal context, object/entity importance, uncertainty, and estimated future task utility.

---

## 3. Existing approaches (as recorded in the workbook)

| ID | Paper | Year (workbook; see note) | Paper | Code |
|---|---|---|---|---|
| P01 | Worth Remembering: Surprise-Gated Robot Episodic Memory | 2026 | [arXiv 2606.03787](https://arxiv.org/abs/2606.03787) | not recorded |
| P02 | Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment (H²-EMV) | 2026 | [arXiv 2604.11306](https://arxiv.org/abs/2604.11306) | not publicly released (per workbook) |
| P03 | ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation | 2025 | [arXiv 2409.13682](https://arxiv.org/abs/2409.13682) | [NVIDIA-AI-IOT/remembr](https://github.com/NVIDIA-AI-IOT/remembr) |
| P04 | RoboMemory: A Brain-Inspired Multi-Memory Agentic Framework for Lifelong Learning in Physical Embodied Systems | 2025 | [arXiv 2508.19875](https://arxiv.org/abs/2508.19875) | [1737536149/RoboMemory](https://github.com/1737536149/RoboMemory) |
| P05 | Scaling Up Memory for Robot Control via Experience Retrieval (MemER) | arXiv 2025-10-23; ICLR 2026 | [arXiv 2510.20328](https://arxiv.org/abs/2510.20328) | [project page](https://jen-pan.github.io/memer/) |
| P06 | NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation | 2026 | [arXiv 2607.06678](https://arxiv.org/abs/2607.06678) | [OpenDriveLab/NativeMEM](https://github.com/OpenDriveLab/NativeMEM) |
| P07 | AMEGO: Active Memory from long EGOcentric videos | 2024 | [arXiv 2409.10917](https://arxiv.org/abs/2409.10917) | [gabrielegoletto/AMEGO](https://github.com/gabrielegoletto/AMEGO) |
| P08 | EgoLife: Towards Egocentric Life Assistant | 2025 | [arXiv 2503.03803](https://arxiv.org/abs/2503.03803) | [EvolvingLMMs-Lab/EgoLife](https://github.com/EvolvingLMMs-Lab/EgoLife) |

Years are as recorded in the workbook, except P05 (preprint and venue given explicitly by the team). See data-quality note 1: P03's arXiv ID indicates a 2024 preprint, so its recorded 2025 may be a venue year (not verified).

**One-line summaries (Reported).**

- **P01** - Online, causal Bayesian-surprise gate in V-JEPA-2 latent space decides which observations are written as episodic events into a 4D scene graph (DAAAM). Task-agnostic.
- **P02** - Hierarchical episodic memory with LLM-based relevance estimation; natural-language forgetting "rules" are updated from user feedback about wrongly forgotten items.
- **P03** - VLM-captioned navigation segments stored as embeddings with pose/timestamp in a vector DB; retrieval at query time. No active forgetting.
- **P04** - Separate episodic, semantic, spatial and procedural memories coordinated by a planner-critic loop, with per-module consolidation.
- **P05** - Task-conditioned VLM (Qwen2.5-VL-7B) nominates keyframes; temporal clustering keeps the median frame per cluster; small bounded keyframe set plus recent frames.
- **P06** - Learned tokenizer compresses every historical frame to one native visual token per view, trained with action-prediction loss.
- **P07** - Online detection of hand-object interactions and location segments; stores structured interaction tracklets instead of uniformly sampled frames.
- **P08** - Hierarchical summarization (30 s clips -> captions -> hourly/daily summaries) plus query-time retrieval of top-3 relevant clips.

---

## 4. Taxonomy of existing memory approaches

Categories are not mutually exclusive; the mapping below is **interpretation** based on the workbook's "Selection / Compression Method" column.

| Category | What it means here | Papers (primary) | Also partially |
|---|---|---|---|
| Retrieval-based memory | Keep a large store; select at *query time* | P03, P08 (retrieval stage) | P05 (task-conditioned selection), P02 (query-time compute reduction) |
| Frame / keyframe selection | Choose which raw frames/segments to keep | P05, P07 | P01 (selects events) |
| Token-level compression | Compress *every* frame into a few tokens | P06 | - |
| Hierarchical summarization | Multi-level summaries (clip -> hour -> day) | P08, P02 | P04 (consolidation) |
| Surprise / novelty-based memory | Write only when prediction error is high | P01 | P07 (interaction-triggered, not novelty-triggered) |
| Learned forgetting | Learned policy decides what to prune | P02 | P06 (learned compression, but no forgetting) |
| Multi-memory / consolidation | Functionally separate stores with consolidation | P04 | P02 (hierarchy), P01 (event layer on scene graph) |

**Interpretation.** Most papers combine mechanisms. A useful orthogonal view is *when* the decision is made: **write-time** (P01, P07), **compression-time** (P06), **consolidation/forgetting-time** (P02, P04, P08), or **read-time** (P03, P08, P05's task conditioning).

---

## 5. Comparison of the 8 reviewed papers

All cells are **Reported** (as recorded in the workbook; unverified against the papers). Results are not comparable across rows.

| ID | Memory representation | Selection / compression method | Datasets / benchmarks | Main reported result |
|---|---|---|---|---|
| P01 | 4D scene graph (DAAAM) + discrete episodic events embedded in V-JEPA-2 latent | Bayesian surprise gating, online, causal, unsupervised, write-time | Custom long-horizon robot deployment videos; Kinetics-400 (event boundaries) | ≥12% improvement over prior scene-graph memory baseline (DAAAM) on QA; outperforms uniform/random baselines at equal budget; beats supervised/offline methods on event segmentation |
| P02 | Hierarchical multi-level summarized events | LLM relevance estimation with learned NL rules updated from user feedback | Simulated household recordings; 20.5 h real ARMAR-7 recordings | 45% smaller memory and 35% lower query compute vs. non-forgetting baseline while maintaining QA accuracy; second-round query accuracy +70% |
| P03 | VLM captions + embeddings in vector DB with pose/time metadata | Query-time retrieval (semantic + spatial + temporal); no forgetting | NaVQA (introduced by paper) | Outperforms LLM- and VLM-only baselines on NaVQA with lower latency; real-robot deployment |
| P04 | Episodic / semantic / spatial / procedural modules, planner-critic loop | Per-module consolidation (e.g., episodic -> semantic) | EmbodiedBench and custom lifelong-task simulations (hedged in workbook) | Qualitative: improved success rate and planning consistency over single-store baselines (no numbers recorded) |
| P05 | Selected raw keyframes + recent frames | Task-conditioned VLM nominates keyframes; single-linkage clustering (d=5); median frame kept | Custom real-robot tasks (Object Search, Counting Scoops, Dust & Replace); DROID-based setup | >90% task progress on all three tasks; Long History baseline ~34% worse on average |
| P06 | One native visual token per historical frame per camera view | Learned tokenizer using the VLA's vision encoder, trained with action-prediction loss | RMBench + button-pressing sims; real: Click Buttons, Put Back Block, Grocery Checkout Scanning | 84.0% avg. sim success vs. 32.4% for Mem-0; 98.7% avg. real-robot success |
| P07 | Hand-object interaction tracklets + location segments with visual features | Online interaction start/end detection; match to previously seen objects/locations | EPIC-KITCHENS; Active Memories Benchmark (20K+ QA) | Surpasses common video-QA baselines by 12.7% on AMB |
| P08 | Clip features + captions + timestamped hourly/daily summaries | Hierarchical summarization; query-time retrieval of relevant clips | EgoLife (300 h collected, 266 h retained); EgoLifeQA (3,000 QA) | EgoGPT+EgoRAG 35.4% on >24 h QA vs. 25.0% for EgoGPT alone |

---

## 6. What each approach optimizes

| ID | Reported optimization target | Interpretation: implicit notion of "utility" |
|---|---|---|
| P01 | Write-time selection of observations with high Bayesian surprise; QA accuracy and event-boundary quality at equal memory budget | Utility ≈ unpredictability of the observation (the workbook's own limitation note calls surprise "a proxy for utility, not utility itself") |
| P02 | Memory size and query compute while preserving QA accuracy; personalization via feedback | Utility ≈ relevance to the user as judged by an LLM under learned rules |
| P03 | Retrieval quality/latency over a growing history | No write-time utility; utility is decided at query time by similarity |
| P04 | Planning consistency / task success via functional separation of memory types | No explicit utility signal (the workbook notes the consolidation policy is not optimized for downstream utility) |
| P05 | Task progress with a small bounded keyframe set | Utility ≈ relevance to the *known* task instruction as judged by a VLM |
| P06 | Token, memory and latency cost of history while retaining action-relevant information | Utility ≈ whatever helps action prediction; all frames are kept in compressed form |
| P07 | Query efficiency by storing interaction/location structure instead of the full video | Utility ≈ activity-relevant (hand-object interaction) segments |
| P08 | Recall over week-scale video via multi-level summaries and retrieval | Compression is task-agnostic; relevance is decided at retrieval |

---

## 7. Is the future task/query known during memory formation?

| ID | Reported answer | Note |
|---|---|---|
| P01 | **No** - task-agnostic, surprise-driven | |
| P02 | **No** - relevance learned online from user feedback | Adapts to user priorities over time (workbook: "learns an implicit utility function") |
| P03 | **No** - memory built independent of any query | |
| P04 | **No** - open-ended lifelong operation | |
| P05 | **Yes** - task instruction provided when selecting keyframes | Only reviewed paper where the task conditions the *write* decision |
| P06 | **No** for compression; task-specific VLA fine-tuning done later | |
| P07 | **No** - task-agnostic, reused for later queries | |
| P08 | **Partially** - summarization task-agnostic; retrieval query-aware | |

**Interpretation.** 7 of 8 papers form memory without knowing the task, but none of them (per the workbook) *models* a distribution over future tasks when deciding what to keep; they use proxies (surprise, interaction, LLM relevance, generic learned compression).

---

## 8. Is a fixed memory budget explicitly considered?

| ID | Reported budget treatment | Hard global cap on an unbounded stream? |
|---|---|---|
| P01 | Implicit, via surprise threshold; compared to uniform/random at *equal* budgets | Not stated as a hard cap |
| P02 | Relative reduction reported (45% memory, 35% compute vs. non-forgetting baseline) | Not stated |
| P03 | None; memory grows, scalability via retrieval | No |
| P04 | None unified; per-module consolidation | No |
| P05 | 8 recent frames/camera + ≤8 selected keyframes; **no eviction mechanism** | Bounded set size, but no eviction as the stream continues |
| P06 | 1 token per frame-view; supports up to 5,000 frames within 32 GB; ~200 frames under 100 ms | Cost is fixed *per frame* but grows linearly with history |
| P07 | No numeric budget; implicit compression via structure | No |
| P08 | No fixed budget; compression from hierarchical summaries; top-3 clips retrieved | No |

**Interpretation.** Per the workbook, only P01 evaluates against baselines at an explicitly equal budget, and no reviewed paper enforces a *hard* persistent budget with eviction over an unbounded stream.

---

## 9. Online or offline memory formation?

The workbook has no dedicated field for this. Only what the workbook text states is recorded; everything else is **Not verified** and is not inferred.

| ID | Online / offline | Basis |
|---|---|---|
| P01 | **Online, causal** | Reported explicitly ("online, causal, unsupervised") |
| P02 | Not verified | Workbook says memory is "built incrementally" and relevance is "learned online from user feedback"; when forgetting is executed is not stated |
| P03 | Not verified | Query-time retrieval is reported; memory-building mode is not stated |
| P04 | Not verified | "Periodic consolidation" is reported; online/offline mode is not stated |
| P05 | Not verified | Not stated whether keyframe selection runs streaming or over a completed history |
| P06 | Not verified | Per-frame compression is reported; online deployment is not stated |
| P07 | **Online** | Reported ("online detection of interaction start/end") |
| P08 | Not verified | Not stated whether summarization runs over completed recordings |

**Action item:** add an "Online/Offline" column to the workbook and verify each paper against its source.

---

## 10. Downstream evaluation tasks

| ID | Downstream task (Reported) | Type |
|---|---|---|
| P01 | Robot QA (temporal, spatial, binary); generic event-boundary segmentation (Kinetics-400) | QA / segmentation |
| P02 | NL QA about the robot's past (simulated household + real ARMAR-7) | QA |
| P03 | Spatial, temporal, descriptive QA for navigation; real-robot goal-directed navigation | QA / navigation |
| P04 | Long-horizon embodied task execution and planning (hedged) | Planning / execution |
| P05 | Long-horizon real-world manipulation | Manipulation (closed-loop) |
| P06 | Long-horizon manipulation (sim and real) | Manipulation (closed-loop) |
| P07 | Very-long egocentric video QA: sequencing, concurrency, temporal grounding | QA |
| P08 | Long-context egocentric QA: event recall, habits, relations, recommendations | QA |

**Interpretation.** Two largely separate evaluation traditions appear: **memory-for-QA** (P01, P02, P03, P07, P08) and **memory-for-control** (P05, P06; P04 is closer to this side, but its evaluation is hedged in the workbook). No reviewed paper evaluates one memory on *both* kinds of task, and no common benchmark is shared across papers beyond partial overlap in style.

---

## 11. Limitations of existing approaches

**Reported (workbook "Limitation / Research Gap" column), condensed:**

- **P01** - Surprise is a proxy for utility; may miss boring-but-useful events and over-store novel-but-irrelevant ones; evaluated mainly on QA/navigation, not manipulation; no explicit downstream task-conditioning.
- **P02** - Needs explicit user feedback (may be unavailable in autonomous settings); LLM relevance scoring costs latency and may hallucinate; evaluated on QA, not closed-loop execution.
- **P03** - No active compression/forgetting; memory grows unboundedly; quality depends on captioning/embedding; no task-utility-driven pruning.
- **P04** - Multi-module complexity and overhead; consolidation not optimized for downstream utility; hard to attribute gains to a module.
- **P05** - No eviction as memory grows; tested mainly on minute-scale horizons.
- **P06** - Not designed for hours/days persistent memory; action-loss supervision may miss higher-level semantic needs; compresses every frame rather than deciding what to keep.
- **P07** - No explicit task-utility scoring or hard memory budget; human egocentric setting, not robotic control.
- **P08** - Text summaries can lose fine-grained visual detail; single-pass retrieval; no embodied/action component; no task-utility-based compression.

**Interpretation (cross-cutting).**

1. Utility is always *proxied* (surprise, interaction, LLM judgment, action loss), never directly estimated or validated against downstream value.
2. Budgets are mostly implicit or relative.
3. Fidelity is fixed by architecture (one token, a caption, a summary) rather than allocated per item.
4. Evaluations are heterogeneous, so it is unclear which proxy is better under which task.

---

## 12. Research gaps

Each gap lists the evidence from the workbook and a confidence level. All are **Potential gaps** pending Phase 3 validation and completion of the literature review.

| # | Potential gap | Evidence in workbook | Confidence |
|---|---|---|---|
| **G1** | **Future-task utility is proxied, not modeled or measured.** The workbook does not record any reviewed method that estimates how useful an item would be across a distribution of possible future tasks, or that checks whether its proxy correlates with actual downstream utility. | P01 (surprise as proxy, own limitation); P07, P08 (no task-utility scoring); P04 (no utility signal) | Medium |
| **G2** | **Hard persistent budget with eviction on an unbounded stream.** | P05 no eviction; P03 unbounded growth; P07/P08 no budget; P01 implicit threshold; P06 linear growth | Medium |
| **G3** | **Fidelity allocation.** Which items to keep *and at what fidelity* (raw, token, caption, summary) is not recorded in the workbook as decided per item by any reviewed method; fidelity appears to be set by the architecture. | P06 (fixed 1 token/frame); P08, P02 (fixed hierarchy) | Medium (interpretation of the workbook; verify) |
| **G4** | **Delayed / retrospective utility.** Whether an observation's value can change as later observations arrive, and whether memory can be re-scored and re-compressed. | Indirect only: P01 limitation on "boring but useful"; P05 no eviction | Low |
| **G5** | **No shared budget-vs-utility evaluation protocol across task types.** Only P01 is recorded as comparing to uniform/random at equal budget; papers use different benchmarks. Several papers do report multiple question types (P01, P03, P07, P08), so the gap concerns evaluating one memory across budgets and task types, not the absence of multiple question types. | Sections 8 and 10 | Medium |
| **G6** | **QA-memory and control-memory are evaluated separately.** Task-agnostic methods are mostly evaluated on QA; control-oriented methods either know the task (P05) or compress every frame (P06). | P01 limitation ("not manipulation"); P05, P06 | Medium |
| **G7** | **Autonomous utility signal.** P02's utility comes from user feedback and P05's from the known task instruction; neither is available in fully autonomous, task-unknown operation. | P02, P05 limitations | Medium |
| **G8** | **The review itself is incomplete.** 8 of 16 planned papers; adjacent areas not covered (video-LLM frame/token selection, streaming-video memory, map / experience-buffer pruning, budgeted selection theory). Any novelty claim could be pre-empted by unreviewed work. | Workbook rows P09-P16 empty | High |

### Is the proposed novelty already established?

**No. Novelty is not established, and parts of the idea are already covered.**

- **Worth Remembering (P01) already addresses unknown future tasks using surprise-gated memory formation**, task-agnostically, with equal-budget comparison against uniform/random baselines (as recorded). **Our possible contribution is therefore not simply unknown-future-task memory selection**, and a project that only proposes "another gating signal" would be weakly differentiated from P01.
- **P05** already uses a task-conditioned signal for bounded keyframe memory (task known). **P02** already frames "what to forget under a storage budget" with a learned policy (user feedback).
- **Candidate gap (hypothesis requiring further validation).** The reviewed methods appear to differ in how they allocate memory, but the current review has not established a method that **jointly** performs (i) explicit utility estimation, (ii) hard memory-budget allocation, (iii) variable-fidelity representation, (iv) online eviction/reallocation, and (v) evaluation across multiple future task types. This is a statement about the contents of an 8-paper, unverified workbook, not about the field. It requires validation against a broader literature review.
- **If the Phase 3 validation shows that surprise (or another single proxy) already tracks oracle utility closely at equal budget, the utility-aware direction has little room and the project should pivot** (see Section 14).

---

## 13. Candidate novelty directions

Ordered by our current view of *defensibility*, not by ambition. All are hypotheses.

| Dir. | Idea | Differentiates from | Main risk |
|---|---|---|---|
| **A** | **Analysis/benchmark contribution:** a budget-vs-utility protocol that evaluates memory policies (surprise, interaction, uniform, random, recency, task-conditioned oracle) on several *future task types* with the same stored memory. Includes an *oracle retrospective utility* to test whether proxies track real value. | Papers that each use a single benchmark/proxy (G1, G5) | Lower "method" novelty; depends on a fair, reproducible harness |
| **B** | **Multi-fidelity budgeted allocation:** decide keep/discard *and* fidelity per event under a global budget. | P06 (fixed 1 token/frame), P08/P02 (fixed hierarchy) (G3) | Needs a cost model and a fixed reader; may reduce to a known scheduling/knapsack heuristic |
| **C** | **Eviction and re-scoring as memory fills:** age-aware retention that revisits earlier decisions. | P05 (no eviction), P01 (write-once gating) (G2, G4) | Delayed-utility signal may be weak; eviction can compound errors |
| **D** | **Multi-signal utility (novelty + interaction + semantics + temporal context + uncertainty)** combined and validated against oracle utility. | P01 (single signal), P07 (interaction only) | Easily reads as incremental over P01 unless paired with A or B |
| **E** | **Self-supervised future-utility estimator:** learn a scorer from synthetic/proxy future queries, so no user feedback or known task is required. | P02 (user feedback), P05 (task known) (G7) | Proxy queries may not match real tasks; risk of circularity |
| **F** | **Unified evaluation across QA and manipulation** for the same memory. | Separate traditions (G6) | Large scope; likely beyond a single course project |

**Interpretation.** A + B (+ C if time allows) is the most defensible combination for a course-scale project: A is low-risk and useful even if the method fails; B is the clearest gap relative to the reviewed papers. D and E are candidate *components*, not standalone contributions.

---

## 14. Recommended research hypothesis

The hypothesis is staged so that it can be **falsified before any complex method is built.**

- **H0 (gap validation).** Under a fixed budget, single-signal proxies (surprise, recency, interaction detection) are *inconsistent* predictors of retrospective downstream utility across task types (e.g., spatial vs. temporal vs. object-state queries): the best proxy differs by task, and each falls measurably short of an oracle-utility selection.
- **H1 (budget-aware allocation).** Given H0, a memory policy that allocates a fixed budget using an explicit, multi-component utility estimate, and chooses fidelity per item, yields better *average* downstream performance across a *set* of task types than any single-signal baseline at the same budget.
- **H2 (robustness).** The advantage of H1 holds across budgets, and degrades more gracefully as the budget shrinks.

**Decision rule (pre-registered intent).**

- If H0 is *not* supported (a single proxy such as surprise tracks oracle utility closely across tasks), the utility-aware allocation direction is weak. Pivot to an analysis/benchmark contribution (Direction A) or to an unexplored gap found in the completed review.
- If H0 is supported but H1 fails with simple estimators, report it as a negative result and analyze why, rather than adding architectural complexity.

The eventual method may change after the full 16-paper review.

---

## 15. Recommended experimental setup

**Principle:** first establish what existing methods already do; keep the memory policy modular and the downstream reader *fixed and frozen* so that differences are attributable to memory selection.

1. **Reproduce/approximate baselines first (Phase 2).** Prefer codebases recorded in the workbook: ReMEmbR/NaVQA (P03), AMEGO/AMB (P07), EgoLife/EgoLifeQA (P08), NativeMEM (P06), RoboMemory (P04). P01 has no recorded code link and P02 is not released, so their gating/forgetting ideas would need to be re-implemented from the papers (as approximations, to be labeled as such).
2. **Common harness.** One frozen reader (VLM/LLM) answering questions from whatever memory a policy produces; identical inputs, identical cost accounting. Start with a QA setting (cheapest and closest to P01-P03, P07, P08) before considering manipulation (P05, P06).
3. **Budget definition.** Define budget in a unit that is comparable across policies (e.g., stored items and bytes/tokens), and sweep several budget levels (proposed values are to be chosen once the data are known; no values are fixed here).
4. **Baselines at equal budget.**
   - Full memory (upper bound / unconstrained reference)
   - Uniform sampling, random retention, recency (FIFO)
   - Surprise gating (approximation of P01)
   - Interaction-based selection (approximation of P07)
   - Task-conditioned selection with known task (approximation of P05) as an *informed reference*, not a fair competitor
   - Retrieval over a large store (P03-style) as a *different-regime reference*
5. **Oracle retrospective utility.** For a labeled evaluation set, measure each item's contribution (e.g., leave-one-out or greedy-by-gain) to get an oracle selection at each budget. This is the yardstick for H0.
6. **Multiple future tasks per stored memory.** Form memory once, then evaluate on several question/task types (spatial, temporal, object/state, counting, etc.) to test robustness to *unknown* future tasks.
7. **Metrics.** Downstream accuracy vs. budget curves per task type; average over task types; worst-case task type; write/read cost. Multiple seeds and confidence intervals.
8. **Candidate data.** Start from datasets already in the workbook: NaVQA (P03), EPIC-KITCHENS/AMB (P07), EgoLife/EgoLifeQA (P08). Availability, licensing and compute must be checked. Other egocentric datasets not yet reviewed may be considered once the review is complete.
9. **Ablations** (once a method exists): utility components; fidelity tiers on/off; eviction on/off; budget sensitivity; reader sensitivity.

**No results exist yet.** Nothing in this document is an experimental finding of this project.
