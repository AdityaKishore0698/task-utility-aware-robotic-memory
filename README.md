# Task-Utility-Aware Episodic Memory Compression for Long-Horizon Egocentric Robotic Agents

A Foundation of Robotics research project on bounded episodic memory for long-running egocentric robots.

> **Status: early experiments (gap validation).** An evaluation harness on the FindingDory benchmark and three exploratory milestones are done (see [Experiments so far](#experiments-so-far)). The proposed method itself is not implemented yet, there is no real downstream reader in the loop, and any claim of novelty is provisional until the literature review is complete.

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
| Evaluation harness | Done: FindingDory loader + selection-hit scoring (`src/`) |
| Baselines | Generic baselines (uniform, random, recency, oracle_greedy) done; crude zero-model proxies for P01 (surprise) and P07 (interaction) done. Faithful reproductions of the papers' methods not started |
| Experimental results | Milestones 1-3 (exploratory, 100 validation + 100 train episodes). First positive evidence for H1; held-out confirmation pending |
| Proposed method | Not implemented |
| Downstream reader (VLM) | Not started |

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
├── src/
│   ├── data/findingdory.py       FindingDory metadata loader (episodes, queries, oracle frame indices)
│   ├── eval/selection_hit.py     Scores retained frames against each query's oracle frames
│   ├── policies/baselines.py     full_memory, uniform, random, recency, oracle_greedy
│   ├── policies/visual_signals.py  top-K, stratified and NMS selection over per-frame scores
│   └── scripts/                  Data download, signal extraction, milestone runners, bootstrap CIs
├── experiments/                  One folder per milestone: README (design, results, caveats) + small CSV/TXT outputs
└── data/                         (gitignored) downloaded FindingDory metadata and per-frame signal CSVs
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

## Implementation

The code is deliberately simple and modular, with no neural model yet. Everything runs on a laptop CPU, except the video decoding for the signals (see below).

**Benchmark.** [FindingDory](https://huggingface.co/datasets/yali30/findingdory) (`yali30/findingdory`): long egocentric robot episodes (roughly 400-3,500 frames), each with many queries annotated with the frame indices that answer them (the "oracle" frames).

**Metric: selection hit.** A policy keeps at most B frames per episode. A query counts as a hit if at least one kept frame is among its oracle frames. Hit rate is pooled over all solvable queries. This needs no downstream model, but it is lenient (see caveats). Details in `src/eval/selection_hit.py`.

**Policies** (all under the same hard frame budget B):

| Policy | Rule |
|---|---|
| `full_memory` | Keep everything (unconstrained reference) |
| `uniform` | B evenly spaced frames |
| `random` / `stratified_random` | B random frames / one random frame per equal time segment |
| `recency` | The last B frames (FIFO) |
| `{signal}` (top-K) | The B highest-scoring frames |
| `stratified_{signal}` | Highest-scoring frame in each of B equal time segments |
| `nms_{signal}` | Frames in score order, skipping any too close in time to one already kept |
| `oracle_greedy` | Retrospective upper bound: greedy max-coverage over the oracle labels |

**Signals** (crude, zero-model stand-ins for literature proxies):
- `pixel_novelty`: mean absolute difference from the previous frame (64×64 grayscale). Approximates P01's surprise signal.
- `motion_interaction`: spatial concentration of frame-to-frame change over an 8×8 grid. Approximates P07's interaction detection. Not validated against real interaction labels.

**Not built yet:** a frozen downstream reader (VLM), the utility estimator and budgeted allocator, fidelity tiers, and eviction.

## Experiments so far

Each milestone folder in `experiments/` has a README with its design, full results, interpretation and caveats.

| Milestone | Question | Main finding |
|---|---|---|
| [phase2_milestone1](experiments/phase2_milestone1/) | How far are generic baselines from a query-aware oracle? | Recency is far worse than everything else; uniform is strong. Large oracle gap at small budgets (B=4: uniform 0.29 vs oracle_greedy 0.72), nearly closed by B≈128 |
| [phase2_milestone2](experiments/phase2_milestone2/) | Do crude surprise/interaction signals beat the baselines? | Top-K by signal beats uniform only at B≤8-16 and loses from B≈32, because picked frames bunch in time |
| [milestone3_coverage_pilot](experiments/milestone3_coverage_pilot/) | Does signal + temporal coverage beat either alone? (pilot of H1, predictions written before running) | Yes: `nms_motion_interaction` reaches 0.59 vs uniform 0.28 at B=4 (validation) and closes about 70-80% of the uniform-to-oracle gap at B=4-16 on both splits. It also beats the `stratified_random` control, so the gain is credited to the signal |

**Caveats that limit these results:**
- The coverage idea came from looking at Milestone 2 results on the same splits. 1,378 train episodes are still untouched and are reserved for one confirmatory run.
- The selection-hit metric is lenient and saturates: from B≈64 every well-spread policy scores 0.97-0.99. A real reader that must *use* the frame is likely stricter.
- Signals are crude heuristics, and there is no downstream reader yet.
- About 30 comparisons per split with no multiple-comparison correction.

### Reproducing

Requires Python 3 with `pandas`, `numpy` and `requests` (plus `opencv-python` for signal extraction). Run from the repository root.

```bash
python3 src/scripts/download_findingdory.py    # ~1.4 MB of metadata, no video
python3 src/scripts/run_phase2_milestone1.py   # index-only baselines, a few seconds
```

Milestones 2 and 3 also need the per-frame signal CSVs in `data/findingdory/` (validation for Milestone 2; validation and the 100-episode train sample for Milestone 3). These were produced by fetching the episode videos (`fetch_findingdory_videos.py`; about 2.7 GB for the 100 validation episodes) and running `extract_pixel_novelty.py` and `extract_motion_concentration.py` on a lab machine. See [experiments/phase2_milestone2/README.md](experiments/phase2_milestone2/README.md) for the exact steps.

```bash
python3 src/scripts/run_phase2_milestone2.py
python3 src/scripts/run_milestone3_coverage.py
```

## Experimental plan

Next steps: run the confirmatory test on the untouched train episodes, then add a real downstream reader and stronger signals (learned embeddings) before building the full allocator.

The original plan is summarized below; details in [docs/project_proposal.md](docs/project_proposal.md) and [docs/research_roadmap.md](docs/research_roadmap.md).

- **Baselines at equal budget:** full memory, uniform, random, recency, surprise gating (approximation of P01), interaction-based (approximation of P07), task-conditioned with known task (approximation of P05), retrieval over a large store (P03 style).
- **Gap validation first:** test whether single proxies track an oracle retrospective utility, and whether the best proxy varies by task type.
- **If supported:** budget-constrained utility-aware allocation, multi-fidelity, eviction; budget sweeps; ablations.
- **Data:** FindingDory is in use (see above). Further candidates: NaVQA (P03), EPIC-KITCHENS / Active Memories Benchmark (P07), EgoLife / EgoLifeQA (P08). Availability and compute to be checked.
- **Metrics:** accuracy versus budget curves per task type, average and worst case, oracle gap, multiple seeds.

## Team

Atharva Narkhede, Ranjeet Kulkarni, Pratik Pansare, Agraj Singh (per the literature workbook). Repository owner: Aditya Kishore. *[Confirm the team list.]*
