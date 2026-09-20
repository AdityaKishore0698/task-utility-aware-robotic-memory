# Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents

A Foundation of Robotics research project on bounded episodic memory for long-running egocentric robots.

> **Status: literature-review stage.** No method has been implemented and no experiments have been run. Any claim of novelty is provisional until the literature review is complete.

## Motivation

A long-running egocentric robot continuously observes its environment, but persistent memory is bounded. Storing every observation is expensive and redundant. Discarding aggressively can remove evidence that later becomes useful for a task that is unknown at observation time.

## Research problem

> Given a continuous egocentric visual stream and a fixed memory budget, how should an embodied agent decide which observations/events to retain, which to discard, and at what fidelity, so that future task performance degrades as little as possible?

Conceptual formulation (a lens for comparison, not a final algorithm):

```
maximize   future downstream utility
subject to persistent memory <= B
```

Candidate utility signals under investigation (hypotheses only): novelty, interaction/state change, semantic importance, temporal context, object/entity importance, uncertainty, estimated future task utility.

**Initial hypothesis.** Allocating memory by information utility may preserve more useful information per unit of memory than generic compression, uniform sampling, random retention or recency. This is *not* assumed to be novel: the literature review must determine the actual gap, and the project pivots if existing surprise-based, retrieval-based, task-conditioned or learned-forgetting approaches already cover it.

## Current status

| Item | Status |
|---|---|
| Literature review | **Preliminary: 8 of 16 planned papers** (P01-P08). P09-P16 are assigned but empty |
| Research-gap analysis | Drafted, provisional |
| Project proposal | Drafted |
| Roadmap | Drafted |
| Progress report | Drafted |
| Baseline reproduction | Not started |
| Proposed method | Not implemented |
| Experimental results | None |

Key caveat: our novelty is **not** established. Worth Remembering (P01) already addresses unknown future tasks with surprise-gated memory formation, so our possible contribution is not simply unknown-future-task memory selection. The preliminary review suggests a potential research gap: it has not established a method that jointly performs explicit utility estimation, hard memory-budget allocation, variable-fidelity representation, online eviction/reallocation, and evaluation across multiple future task types. This hypothesis requires validation against a broader literature review. Only 8 papers were reviewed, P09-P16 are empty, and the papers were not independently re-read for this preliminary work. See [docs/literature_analysis.md](docs/literature_analysis.md).

## Repository structure

```
.
├── README.md                     This file
├── literature_review.xlsx        Team literature workbook (primary source, P01-P16)
├── docs/
│   ├── literature_analysis.md    Structured literature and research-gap analysis
│   ├── project_proposal.md       Research proposal
│   └── research_roadmap.md       Seven-phase roadmap with success criteria
├── reports/
│   └── project_progress_report.md  Progress report for the course
├── literature/                   (reserved) notes and extracted paper summaries
├── src/                          (reserved) implementation, empty for now
└── experiments/                  (reserved) experiment configs, logs, results, empty for now
```

## Literature review

The workbook `literature_review.xlsx` is the primary source. It currently holds 8 papers:

| ID | Paper | Link |
|---|---|---|
| P01 | Worth Remembering: Surprise-Gated Robot Episodic Memory | [arXiv](https://arxiv.org/abs/2606.03787) |
| P02 | Learning to Forget: Hierarchical Episodic Memory for Lifelong Robot Deployment (H²-EMV) | [arXiv](https://arxiv.org/abs/2604.11306) |
| P03 | ReMEmbR | [arXiv](https://arxiv.org/abs/2409.13682) · [code](https://github.com/NVIDIA-AI-IOT/remembr) |
| P04 | RoboMemory | [arXiv](https://arxiv.org/abs/2508.19875) · [code](https://github.com/1737536149/RoboMemory) |
| P05 | Scaling Up Memory for Robot Control via Experience Retrieval (MemER; arXiv Oct 2025, ICLR 2026) | [arXiv](https://arxiv.org/abs/2510.20328) · [project](https://jen-pan.github.io/memer/) |
| P06 | NativeMEM | [arXiv](https://arxiv.org/abs/2607.06678) · [code](https://github.com/OpenDriveLab/NativeMEM) |
| P07 | AMEGO | [arXiv](https://arxiv.org/abs/2409.10917) · [code](https://github.com/gabrielegoletto/AMEGO) |
| P08 | EgoLife | [arXiv](https://arxiv.org/abs/2503.03803) · [code](https://github.com/EvolvingLMMs-Lab/EgoLife) |

Rows P09-P12 (Pratik Pansare) and P13-P16 (Agraj Singh) are assigned but not yet filled. The analysis documents state which conclusions depend on the missing rows. Workbook entries are unverified against the original papers. Data-quality items to resolve (e.g., the workbook's mixed preprint/venue years, P04 details) are listed in the analysis.

## Planned implementation

Not started. The plan is deliberately simple and modular, with no complex neural architecture until existing methods and the research gap are understood:

1. A common evaluation harness with a **frozen downstream reader** and a single budget-accounting scheme.
2. Interchangeable memory policies: uniform, random, recency, surprise gating, interaction-based, and (later) utility-aware allocation.
3. A pluggable utility estimator and a budgeted allocator (keep / evict / re-compress, with fidelity tiers) only if the gap-validation phase supports it.

Code will live in `src/`.

## Experimental plan

Summarized here; details in [docs/project_proposal.md](docs/project_proposal.md) and [docs/research_roadmap.md](docs/research_roadmap.md).

- **Baselines at equal budget:** full memory, uniform, random, recency, surprise gating (approximation of P01), interaction-based (approximation of P07), task-conditioned with known task (approximation of P05), retrieval over a large store (P03 style).
- **Gap validation first:** test whether single proxies track an oracle retrospective utility, and whether the best proxy varies by task type.
- **If supported:** budget-constrained utility-aware allocation, multi-fidelity, eviction; budget sweeps; ablations.
- **Candidate data:** NaVQA (P03), EPIC-KITCHENS / Active Memories Benchmark (P07), EgoLife / EgoLifeQA (P08). Availability and compute to be checked.
- **Metrics:** accuracy versus budget curves per task type, average and worst case, oracle gap, multiple seeds.

## Team

Atharva Narkhede, Ranjeet Kulkarni, Pratik Pansare, Agraj Singh (per the literature workbook). Repository owner: Aditya Kishore. *[Confirm the team list.]*
