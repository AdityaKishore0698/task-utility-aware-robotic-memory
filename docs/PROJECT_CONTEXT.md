# Project Context: Work Done So Far

**Purpose of this file.** A single onboarding document for anyone joining this repository. Read it first. It explains what the project is, what has been done, where every piece lives, what the results mean (and do not mean), and what to do next. Other documents go deeper; this one ties them together.

**Last updated:** 2026-11-16 (covers work up to commit `9b32ce5`).

---

## 1. The project in one paragraph

A robot that runs for a long time sees a continuous first-person ("egocentric") video stream, but it can only store a limited amount of it. Later, it may be asked questions or given tasks about what it saw ("where did I last see the mug?", "which room did I visit second?"), and these questions are **not known when the frames are being stored**. This project studies **how to choose which observations to keep under a fixed memory budget B** so that future questions can still be answered. The working hypothesis is that keeping frames based on an estimate of their *usefulness* (utility) beats generic strategies such as uniform sampling, random sampling or keeping the most recent frames.

```
maximize   future downstream utility
subject to persistent memory <= B
```

This is a course research project (Foundation of Robotics). The repository owner is Aditya Kishore; the team listed in the literature workbook is Atharva Narkhede, Ranjeet Kulkarni, Pratik Pansare and Agraj Singh (still marked "confirm" in the README).

---

## 2. Timeline of what has happened

| Date | What was done | Where |
|---|---|---|
| 2026-09-20 | Preliminary literature review (8 of 16 papers), research-gap analysis, project proposal, 7-phase roadmap, course progress report | `docs/`, `reports/`, `literature_review.xlsx` (commit `b2955a3`) |
| 2026-10-14 | FindingDory benchmark chosen and downloaded (metadata only). Evaluation harness and generic baselines built. **Milestone 1** run | `src/`, `experiments/phase2_milestone1/` |
| 2026-10-15 | 100 validation episode videos fetched to the lab machine `cir`; two crude visual signals extracted. **Milestone 2** run, plus a check on a seeded 100-episode train sample | `experiments/phase2_milestone2/` |
| 2026-10-15 | **Milestone 3** (coverage pilot, predictions written before running) | `experiments/milestone3_coverage_pilot/` |
| 2026-11-16 | Code and experiments committed (`8283148`), README brought up to date (`9b32ce5`), this file written | |

---

## 3. Where the project stands against the roadmap

The roadmap in [research_roadmap.md](research_roadmap.md) defines seven gated phases. **That file has not been updated since 2026-09-20 and still says Phase 2 is "not started".** The real state is:

| Phase | Name | Real status |
|---|---|---|
| 1 | Literature review | **Partly done.** 8 of 16 papers (P01–P08). P09–P12 (Pratik Pansare) and P13–P16 (Agraj Singh) are empty. Entries not yet verified against the original papers |
| 2 | Baseline reproduction | **Partly done.** Harness, generic baselines, and crude approximations of P01 (surprise) and P07 (interaction) exist on one benchmark (FindingDory). No faithful reproduction of any paper's method, and no real downstream reader (VLM) |
| 3 | Research-gap validation (H0) | **Started informally.** Milestones 1–3 give early evidence (large oracle gap at small budgets; best signal varies by category), but there is no written go/no-go decision yet, and the oracle used is a simplified one |
| 4 | Proposed method | Not started. Milestone 3's "signal + coverage" policies are a first, very simple instance of the H1 idea, not the proposed method |
| 5–7 | Evaluation, ablations, write-up | Not started |

---

## 4. Hypotheses being tested

From [project_proposal.md](project_proposal.md):

- **H0 (gap validation).** Under a fixed budget, single-signal proxies (surprise, recency, interaction) are inconsistent predictors of downstream utility across task types, and each falls short of an oracle selection.
- **H1 (allocation).** A policy that allocates the budget using an explicit, multi-component utility estimate (and chooses fidelity per item) beats any single-signal baseline at the same budget, averaged across task types.
- **H2 (robustness).** The advantage holds across budgets and degrades more gracefully as the budget shrinks.

**Decision rule.** If H0 fails, pivot (to a benchmark/analysis contribution or another gap). If H1 fails with simple estimators, report a negative result rather than adding architectural complexity.

**Novelty is not established.** Paper P01 (Worth Remembering) already does surprise-gated memory for unknown future tasks. The possible gap is a method that *jointly* does utility estimation, hard budget allocation, variable fidelity, online eviction, and multi-task evaluation. This must be checked against the full 16-paper review. See [literature_analysis.md](literature_analysis.md), sections 12–14.

---

## 5. Key concepts and vocabulary

| Term | Meaning in this repo |
|---|---|
| **Budget B** | Maximum number of frames a policy may keep per episode. Swept over 4, 8, 16, 32, 64, 128, 256 (and 512 in Milestone 1) |
| **Policy** | A rule that, given an episode, returns the set of kept frame indices (size ≤ B) |
| **Oracle frames** | For each FindingDory query, the frame indices that answer it (from the dataset's `answer` column). Multi-goal queries have one set per subgoal |
| **Selection hit** | A query is a hit if the kept frames include at least one oracle frame **for every subgoal**. Unsolvable queries (`[[-1]]`) are excluded |
| **Hit rate** | Hits ÷ solvable queries, pooled across all episodes in a split |
| **oracle_greedy** | Upper-bound reference: greedily picks the frame that hits the most not-yet-covered (query, subgoal) oracle sets, using the ground-truth labels (standard greedy max-coverage, so near-optimal rather than optimal). Not achievable by a real policy |
| **Signal** | A per-frame score computed from the video (here: `pixel_novelty`, `motion_interaction`) |
| **Top-K** | Keep the B frames with the highest signal score |
| **Stratified** | Split the episode into B equal time segments and keep one frame per segment (best-scoring, or random for the control) |
| **NMS** | Temporal non-maximum suppression: take frames in score order, skipping any within `L // (2B)` frames of one already kept |
| **Segment coverage** | Fraction of the B equal time segments containing at least one kept frame. Measures how "bunched" a selection is |
| **Gap closed** | `(policy − uniform) / (oracle − uniform)`: how much of the distance from uniform to the oracle a policy recovers |
| **P01…P16** | Paper IDs in `literature_review.xlsx` |
| **cir** | The lab GPU machine where videos are stored and signals were extracted. Not part of this repo |

---

## 6. The benchmark: FindingDory

- Hugging Face dataset [`yali30/findingdory`](https://huggingface.co/datasets/yali30/findingdory).
- Long egocentric robot episodes in simulated homes, roughly 400–3,500 frames each.
- **Validation split:** 100 episodes, 5,870 queries, 60 task types. **Train split:** 1,478 episodes, 55 task types (the 5 "Object Attributes" tasks are held out of train).
- Each query has a `high_level_category` (Single-Goal Spatial, Single-Goal Temporal, Multi-Goal) and a finer `low_level_category` (e.g. Room Visitation, Interaction Order).
- Metadata (parquet files + `episode_id_mappings.json`) is about 1.4 MB and is downloaded by `src/scripts/download_findingdory.py` into `data/findingdory/`.
- Videos live in a 32.5 GB `videos.zip`. `src/scripts/fetch_findingdory_videos.py` fetches only the episodes needed, using HTTP range requests (about 2.7 GB for the 100 validation videos).
- **Episode length:** not in the metadata. The loader approximates it as (max oracle frame index + 1). Milestone 2 checked this against the real videos: exact for 56/100 validation episodes, undercounted the rest by about 26 frames on average (max 87). Milestones 2–3 use the true frame counts from the signal CSVs.

`data/` is gitignored. To work on Milestones 2–3 you need the four signal CSVs in `data/findingdory/` (ask the repo owner, or regenerate them on a machine with the videos).

---

## 7. Code map

All code is plain Python (no neural models). Run everything from the repository root, because scripts import `src.*` as a package.

| File | What it does |
|---|---|
| `src/data/findingdory.py` | `load_episodes(split)` → list of `Episode(ep_id, video_path, episode_length, queries)`. Each `Query` holds its categories and `subgoal_frame_sets`. The docstring documents the schema and caveats |
| `src/eval/selection_hit.py` | `score_episode(episode, retained)` and `score_episode_by_category(...)`. Implements the selection-hit metric and lists its simplifications |
| `src/policies/baselines.py` | Index-only policies (no pixels): `full_memory`, `uniform`, `random_retention`, `stratified_random`, `recency`, `oracle_greedy` |
| `src/policies/visual_signals.py` | Signal-driven policies: `top_k_by_score`, `stratified_top1`, `temporal_nms`, plus `segment_coverage` diagnostic |
| `src/scripts/download_findingdory.py` | Downloads the metadata |
| `src/scripts/fetch_findingdory_videos.py` | Range-fetches selected videos from `videos.zip` |
| `src/scripts/extract_pixel_novelty.py` | Per-frame `pixel_novelty` scores → CSV (needs OpenCV + videos) |
| `src/scripts/extract_motion_concentration.py` | Per-frame `motion_interaction` scores → CSV (needs OpenCV + videos) |
| `src/scripts/run_phase2_milestone1.py` | Milestone 1 runner |
| `src/scripts/run_phase2_milestone2.py` | Milestone 2 runner (`--split train` for the train sample) |
| `src/scripts/bootstrap_signal_vs_baseline.py` | Episode-level bootstrap confidence intervals for Milestone 2 |
| `src/scripts/run_milestone3_coverage.py` | Milestone 3 runner (includes its own bootstraps) |

**Dependencies:** Python 3, `pandas`, `pyarrow` (for parquet), `numpy`, `requests`; `opencv-python` only for signal extraction. There is no `requirements.txt` yet.

**Policy interface.** Index-only policies take `(episode_length, budget)`; signal policies take `(scores, budget)`; `oracle_greedy` takes `(episode, budget)`. All return a `frozenset[int]` of kept frame indices, which is passed to `score_episode`. To add a new policy, write a function with the same shape and add it to a runner.

---

## 8. Experiments and results

Each milestone folder has a README with full design, tables, interpretation and caveats. Read those before building on a result.

### Milestone 1: generic baselines vs oracle (validation, index-only)

Pooled hit rate:

| Policy | B=4 | B=32 | B=128 | B=512 |
|---|---|---|---|---|
| recency | 0.057 | 0.121 | 0.332 | 0.632 |
| uniform | 0.289 | 0.926 | 0.994 | 0.999 |
| random (5 seeds) | 0.332 | 0.793 | 0.976 | 0.997 |
| oracle_greedy | 0.719 | 1.000 | 1.000 | 1.000 |

**Takeaways.** Recency is by far the worst (FindingDory asks about events from early in the episode). Uniform is a strong baseline. There is a large oracle gap at small budgets that nearly closes by B≈128. The best generic baseline varies somewhat by category.

### Milestone 2: crude visual signals, top-K selection

- `pixel_novelty` (stand-in for P01's surprise) and `motion_interaction` (stand-in for P07's interaction detection).
- Both beat uniform only at B ≤ 8–16 and **lose to it from B≈32**, on validation and on the train sample. `pixel_novelty` is even significantly worse than random at B=32–64.
- The purpose-built `motion_interaction` signal does **not** win on the "Interaction" category; uniform does at B=16/32, and `pixel_novelty` does at B=4/8.
- Diagnosis carried into Milestone 3: top-K picks are **bunched in time**.

### Milestone 3: signal + temporal coverage (pilot of H1)

Predictions were saved before running and left unedited. Pooled hit rate, validation (train sample in brackets):

| Policy | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| uniform | 0.280 (0.294) | 0.494 (0.560) | 0.743 (0.796) | 0.922 (0.948) |
| stratified_random (control) | 0.331 (0.355) | 0.513 (0.556) | 0.703 (0.765) | 0.877 (0.909) |
| motion_interaction (top-K) | 0.420 (0.446) | 0.602 (0.622) | 0.780 (0.779) | 0.869 (0.862) |
| stratified_motion_interaction | 0.580 (0.639) | 0.787 (0.841) | 0.925 (0.948) | 0.978 (0.981) |
| **nms_motion_interaction** | **0.593 (0.653)** | **0.826 (0.868)** | **0.953 (0.963)** | **0.983 (0.981)** |
| oracle_greedy | 0.719 (0.767) | 0.906 (0.949) | 0.997 (1.000) | 1.000 (1.000) |

**Takeaways.**
- Top-K picks cover only 34–60% of time segments (uniform: 100%), so bunching is real.
- Adding a coverage constraint to either signal beats uniform significantly at B=4–16 on both splits and closes about **70–80% of the uniform→oracle gap**.
- It also beats `stratified_random`, so the gain comes from the signal, not just from spreading frames out.
- Neither ingredient works alone. This is the first direct positive evidence for H1's core idea.
- Predictions P1 (partly), P2, P3 held; P4 and P5 failed as stated. The README reports this honestly.

---

## 9. Caveats and known limitations (read before claiming anything)

1. **Not a held-out result.** The coverage idea came from looking at Milestone 2 on the same splits. **1,378 train episodes are untouched** and reserved for one confirmatory run. Do not look at them for anything else first.
2. **The metric is lenient and saturates.** Oracle frame sets are large (median about 90 frames, ≈6–7% of an episode), and one frame anywhere in the set counts. From B≈64 every well-spread policy scores 0.97–0.99. A real reader that must *use* the frame will be stricter.
3. **No downstream reader.** Selection hit measures whether a sufficient frame was kept, not whether a VLM could answer from it.
4. **Ordered multi-goal queries** are scored without checking order, which likely over-counts hits there.
5. **Crude signals.** Raw pixel differences, no learned embeddings, no object/hand detection, not validated against interaction labels.
6. **Simplified oracle.** `oracle_greedy` is max-coverage over binary labels, not the proposal's leave-one-out / greedy-gain utility.
7. **Statistics.** About 30 comparisons per split with no multiple-comparison correction (though most CIs are far from zero).
8. **One benchmark only.** Nothing has been run on NaVQA, EPIC-KITCHENS / AMB, or EgoLifeQA.
9. **Fixed-budget, frame-level only.** No fidelity tiers, no online eviction, no streaming setting: policies see the whole episode's scores at once (offline), which a real robot cannot do.

---

## 10. Working conventions in this repo

These are how the existing work was done; please keep to them.

- **Simple before complex.** Try the cheapest version of an idea first; add models only when evidence says it is needed.
- **Write predictions before running.** Each milestone README has a design and predictions section saved before the run and left unedited; results and interpretation go in separate sections afterwards.
- **Do not tune on the data you report.** Policy parameters are fixed in advance. A failed prediction is reported, not fixed by redesigning on the same data.
- **Report uncertainty.** Episode-level bootstrap 95% CIs (2,000 resamples, seeded); random policies averaged over seeds 0–4.
- **One folder per experiment** under `experiments/<milestone>/` with a README plus small CSV/TXT outputs (tracked). Large or scratch outputs go under `experiments/**/scratch/` (gitignored). Downloaded data goes in the top-level `data/` (gitignored).
- **Label approximations as approximations.** E.g. "crude stand-in for P01", never "P01".
- **Be careful on shared machines.** On `cir`, signal extraction was CPU-only and avoided environments used by other people's jobs.

---

## 11. Stale or incomplete documents

| Document | Issue |
|---|---|
| [research_roadmap.md](research_roadmap.md) | Last updated 2026-09-20; phase statuses are out of date (see section 3 above) |
| [project_proposal.md](project_proposal.md), [literature_analysis.md](literature_analysis.md) | Written before any experiments; candidate datasets do not mention FindingDory |
| [../reports/project_progress_report.md](../reports/project_progress_report.md) and the `.docx` | Describe the literature-review stage only |
| `literature_review.xlsx` | Rows P09–P16 empty; P01–P08 not verified against the papers |
| `literature/` | Reserved, empty |
| README team list | Marked "*[Confirm the team list.]*" |

---

## 12. Suggested next steps (in priority order)

1. **Confirmatory run on the untouched train episodes.** Run the Milestone 3 policies, unchanged, on the 1,378 remaining train episodes (needs their videos and signal CSVs). Write the predictions down first. This turns the pilot into a real result or refutes it.
2. **Add a real downstream reader.** A frozen VLM that answers FindingDory queries from the kept frames. This addresses the metric's leniency and saturation, and is the roadmap's intended evaluation.
3. **Finish the literature review** (P09–P16), verify P01–P08 against the papers, and check adjacent areas (video-LLM frame/token selection, streaming-video memory, budgeted selection). Specifically look for prior work that already combines a saliency signal with temporal-coverage constraints, since that is what Milestone 3 found useful.
4. **Write the Phase 3 go/no-go decision** for H0 with the evidence so far, and update the roadmap, proposal and progress report.
5. **Stronger signals.** Learned embeddings (e.g. CLIP- or V-JEPA-style) for surprise/novelty; an actual interaction detector.
6. **Move toward the actual method:** a streaming/online setting, a multi-component utility estimator, fidelity tiers and eviction, each switchable for ablations.
7. **Housekeeping:** add `requirements.txt`, a second benchmark, and per-category analysis of the Milestone 3 policies.

---

## 13. Getting started (first hour)

1. Read this file, then the [README](../README.md), then the three `experiments/*/README.md` files in order.
2. Skim [project_proposal.md](project_proposal.md) (hypotheses, baselines, evaluation) and [literature_analysis.md](literature_analysis.md) sections 12–14 (gaps and novelty).
3. Set up and run Milestone 1, which needs no video:

```bash
pip install pandas pyarrow numpy requests
python3 src/scripts/download_findingdory.py
python3 src/scripts/run_phase2_milestone1.py
```

4. Compare your output to `experiments/phase2_milestone1/overall_hit_rate.csv`.
5. For Milestones 2–3, get the signal CSVs into `data/findingdory/` and run `run_phase2_milestone2.py` and `run_milestone3_coverage.py`.
