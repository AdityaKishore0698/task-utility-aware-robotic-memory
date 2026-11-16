# Phase 2 / Milestone 1 — index-space selection-hit sweep (FindingDory validation)

**What this is.** The first real (non-literature) experiment in this project. It
compares budget-constrained memory policies that need no model and no video
frames — they only see frame *indices* — scored against FindingDory's own
oracle frame-index answers. See `src/eval/selection_hit.py` for exactly what
"hit" means and its simplifications, and `src/policies/baselines.py` for the
policies. This is a narrow, cheap first instantiation of experiments E1–E3 in
[docs/project_proposal.md](../../docs/project_proposal.md), not a full test of
H0: the policies here (uniform/recency/random) are generic baselines, not the
literature's actual proxies (surprise, interaction), and `oracle_greedy` is a
maximum-coverage approximation over FindingDory's binary oracle labels, not the
proposal's leave-one-out / greedy-gain retrospective utility.

**How to reproduce.**
```bash
python3 src/scripts/download_findingdory.py   # fetches ~1.4 MB of metadata, no video
python3 src/scripts/run_phase2_milestone1.py  # ~4s on a laptop CPU, no GPU/API needed
```
- Data: FindingDory validation split (`yali30/findingdory` on Hugging Face), 100
  episodes, 5,870 queries, downloaded 2026-10-14.
- Episode length is approximated as (max oracle frame index referenced in that
  episode) + 1 — a lower bound on true video length; see
  `src/data/findingdory.py` docstring.
- Budgets swept: 4, 8, 16, 32, 64, 128, 256, 512 frames (capped at episode length).
- Random policy: seeds 0–4, mean and population stddev reported.
- No video frames were downloaded or used for this milestone.

**Results.** See `overall_hit_rate.csv` (pooled across all 100 episodes) and
`per_category_hit_rate.csv` (broken down by `high_level_category` /
`low_level_category`). Pooled hit-rate at a few budgets:

| policy | B=4 | B=32 | B=128 | B=512 |
|---|---|---|---|---|
| full_memory (unconstrained) | 1.000 | 1.000 | 1.000 | 1.000 |
| recency (FIFO) | 0.057 | 0.121 | 0.332 | 0.632 |
| uniform | 0.289 | 0.926 | 0.994 | 0.999 |
| random (mean of 5 seeds) | 0.332 | 0.793 | 0.976 | 0.997 |
| oracle_greedy (retrospective upper bound) | 0.719 | 1.000 | 1.000 | 1.000 |

**Reading, as interpretation (not a finished result).**
- At small budgets (≤32 frames out of 400–3,500), `oracle_greedy` is far above
  every budget-agnostic baseline (0.72 vs. 0.29 at B=4), i.e. there is a sizeable
  gap between what a query-aware retrospective selection could achieve and what
  uniform/random/recency actually achieve. This gap narrows and nearly closes by
  B≈128–256 for this benchmark.
- `recency` is the weakest policy at every budget by a wide margin, and is
  especially bad on Multi-Goal Tasks (hit-rate 0.012 at B=32, vs. 0.786 for
  uniform) — consistent with FindingDory's queries routinely referencing events
  from well before the end of the episode, which a FIFO/recency memory would
  have already discarded.
- The best generic baseline differs by task category (e.g. uniform's
  Multi-Goal/Spatial/Temporal hit-rates diverge at B=32: 0.786 / 0.951 / 0.955),
  which is the kind of task-type dependence E3 asks about — though with only
  two non-oracle baselines compared so far, this is weak evidence, not a
  conclusion.

**What this does NOT show.** It does not test the literature's actual proxies
(surprise, interaction) — those need per-frame visual embeddings, i.e. the
video. It does not involve a real downstream reader (VLM) — `oracle_greedy`
uses the dataset's own ground-truth labels, which is a much easier signal than
anything a real policy could compute online. Treat these numbers as a
plausibility check on the project's framing, not as evidence for or against H1.

**Next (Milestone 2, not started).** Add surprise-based and interaction-based
policies, which require per-frame visual features — i.e. a decision on how to
access the video (download the full 32.5 GB `videos.zip`, or range-fetch only
the ~100 validation episode `.mp4` files from inside the zip) and which
lightweight embedding model to use (e.g. a small open-weights CLIP, which can
run on CPU for this scale).
