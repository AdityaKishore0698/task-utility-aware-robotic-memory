# Research Roadmap

**Project:** Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents
**Last updated:** 2026-09-20
**Current phase:** Phase 1 (Literature review), in progress: 8 of 16 planned papers populated.

Phases are gated: a phase starts when the prior phase's success criterion is met, or when the team decides to proceed with a documented reason. Timelines are intentionally not fixed here; dates should be added by the team once course deadlines and compute access are known. The method may change after deeper literature analysis or after Phase 3.

| Phase | Name | Status |
|---|---|---|
| 1 | Literature review | In progress (8/16 papers) |
| 2 | Baseline reproduction | Not started |
| 3 | Research gap validation | Not started |
| 4 | Proposed method | Not started (contingent on Phase 3) |
| 5 | Evaluation | Not started |
| 6 | Ablations | Not started |
| 7 | Paper / report preparation | Not started (progress report drafted) |

---

## Phase 1: Literature review

- **Objective:** Establish what existing methods already do and determine the actual research gap, without assuming our novelty.
- **Inputs:**
  - [literature_review.xlsx](../literature_review.xlsx) (P01-P08 populated; P09-P12 assigned to Pratik Pansare and P13-P16 to Agraj Singh, currently empty)
  - Original papers, to be checked against the workbook summaries (not yet independently re-read)
- **Expected output:**
  - Completed workbook (16 papers) with an added "Online/Offline" column
  - Resolved data-quality items (e.g., mixed preprint/venue years in the workbook, P04 dataset and results, missing code links)
  - Updated [literature_analysis.md](literature_analysis.md) with the gap assessment revised after the full review
  - Coverage of adjacent areas not yet reviewed: video-LLM frame/token selection, streaming-video memory, map/experience-buffer pruning, budgeted selection
- **Success criterion:** All 16 rows are populated and verified against the source papers. The gap analysis explicitly states whether each candidate gap (G1-G8) survives, and identifies any paper that pre-empts it.

---

## Phase 2: Baseline reproduction

- **Objective:** Reproduce or approximate what existing memory strategies do, at equal budget, in a common harness.
- **Inputs:**
  - Codebases recorded in the workbook: ReMEmbR/NaVQA (P03), AMEGO/AMB (P07), EgoLife/EgoLifeQA (P08), NativeMEM (P06), RoboMemory (P04)
  - P01 and P02 method descriptions (no public code recorded; approximations required)
  - Datasets and compute access (to be confirmed)
- **Expected output:**
  - A frozen-reader evaluation harness with a single budget-accounting scheme
  - Simple baselines: full memory, uniform, random, recency
  - Approximations of surprise gating (P01) and interaction-based selection (P07), labeled as approximations
  - A reproducibility log (versions, seeds, configuration) in `experiments/`
- **Success criterion:** Baselines run end to end on at least one QA benchmark. A sanity check reproduces the *qualitative* trends of the original papers where their code or protocol allows. Discrepancies are documented rather than tuned away.

---

## Phase 3: Research gap validation

- **Objective:** Test H0: whether single proxies fall short of oracle retrospective utility and vary by task type. This decides whether to proceed with a utility-aware method.
- **Inputs:**
  - Phase 2 harness and baselines
  - Evaluation sets with multiple task/question types
  - A definition of oracle retrospective utility (leave-one-out or greedy-gain)
- **Expected output:**
  - Oracle-utility selections at several budgets
  - Correlation and gap analysis of surprise, interaction, recency and other signals vs. oracle utility, per task type
  - A written go/no-go decision with the evidence
- **Success criterion:** A clear, documented answer to H0 with uncertainty estimates.
  - *Go:* proxies are inconsistent across task types and leave a measurable oracle gap.
  - *Pivot:* a single proxy tracks oracle utility closely. The project then shifts to an analysis/benchmark contribution or to another gap.

---

## Phase 4: Proposed method

- **Objective:** Build the simplest utility-aware, budget-constrained allocation policy that can test H1, using hand-crafted or off-the-shelf signals first.
- **Inputs:**
  - Phase 3 decision and evidence about which signals matter
  - Harness and baselines from Phase 2
  - The conceptual framework in [project_proposal.md](project_proposal.md) (item proposal, utility estimator, budgeted allocator, fidelity tiers)
- **Expected output:**
  - A pluggable utility estimator and budgeted allocator in `src/`
  - Optional extensions considered only if justified by Phase 3: multiple fidelity tiers, eviction/re-compression, a learned estimator
  - Design notes explaining each choice and how it differs from P01, P02 and P05
- **Success criterion:** The method runs end to end under a hard budget in the common harness, with each component individually switchable. No neural architecture is added without evidence from Phase 3 that it is needed.

---

## Phase 5: Evaluation

- **Objective:** Test H1 and H2 against the baselines at equal budget across task types.
- **Inputs:**
  - Phase 4 method and Phase 2 baselines
  - Budget sweep and multi-task evaluation protocol
- **Expected output:**
  - Budget-vs-utility curves per task type, average and worst-case performance, oracle gap, cost measurements
  - Multiple seeds with confidence intervals
  - Tables and figures in `experiments/` and `reports/`
- **Success criterion:** A pre-declared comparison against the strongest baseline at equal budget, reported with uncertainty. The result is stated honestly whether it is positive, mixed or negative.

---

## Phase 6: Ablations

- **Objective:** Identify which components account for any observed effect.
- **Inputs:**
  - Phase 5 method and results
  - Ablation list: utility components, fidelity tiers, eviction on/off, budget sensitivity, estimator type, reader sensitivity, item granularity
- **Expected output:**
  - Ablation tables that attribute effects to specific components
  - A clear statement of which components are necessary and which are not
- **Success criterion:** Each claimed contribution is backed by an ablation that supports or refutes it. Components that do not help are reported as such.

---

## Phase 7: Paper / report preparation

- **Objective:** Communicate the findings accurately and reproducibly.
- **Inputs:**
  - Results and analyses from Phases 3-6
  - Literature analysis and proposal documents
  - Progress report [project_progress_report.md](../reports/project_progress_report.md)
- **Expected output:**
  - Final report (and, if warranted, a paper draft) with related work, method, experiments and limitations
  - Reproducibility package: code, configurations, seeds and instructions
  - README updated to the final status
- **Success criterion:** Every claim is traceable to a documented experiment or cited source. Novelty claims are consistent with the completed literature review. The repository reproduces the reported tables.
