# Phase 2 / Milestone 2 — two crude visual-signal proxies

**What this is.** Extends Milestone 1 ([experiments/phase2_milestone1/](../phase2_milestone1/))
with two new policies, both zero-model, both scored under the exact same
`selection_hit` harness as Milestone 1's baselines:

- `pixel_novelty` — approximates P01's Bayesian-surprise signal. The
  "predictor" is just the previous frame; the score is the mean absolute
  pixel difference (on a 64x64 downsampled grayscale frame) from it —
  frame-to-frame prediction error in raw pixel space, not a learned latent
  space. See `src/scripts/extract_pixel_novelty.py`.
- `motion_interaction` — approximates P07's interaction-detection signal. The
  score is the *spatial concentration* of frame-to-frame change (standard
  deviation across an 8x8 grid of block-mean differences), on the theory that
  a hand/object manipulation changes a small region against an otherwise
  static background, while the robot walking or turning changes the whole
  frame roughly uniformly. This is a hand-picked heuristic with no
  object/hand detection and no validation against real interaction labels.
  See `src/scripts/extract_motion_concentration.py`.

Both use `src/policies/visual_signals.py`'s `top_k_by_score`: under a budget,
keep the frames with the highest score. See `src/eval/selection_hit.py` for
what "hit" means (unchanged from Milestone 1).

**Why this and not a real embedding model yet.** Per the project's "simple
before complex" discipline (docs/research_roadmap.md), this tests the cheapest
possible version of "does a surprise-like signal help at all" before spending
effort on CLIP/V-JEPA-style embeddings. If `pixel_novelty` already looks
informative, that's a reason to invest in a real embedding model next; if it
doesn't, that's informative too.

**Where the compute happened.** Video decoding needed the actual FindingDory
video files, which live on `cir` (a lab GPU machine), not in this repo (see
[docs/research_roadmap.md](../../docs/research_roadmap.md) Phase 2 notes on
data access). Concretely:
1. `src/scripts/fetch_findingdory_videos.py` range-fetched the 100 validation
   episode videos from the dataset's zip archive directly to `cir` (2.74 GB;
   the full archive is 32.5 GB and wasn't needed).
2. `src/scripts/extract_pixel_novelty.py` was copied to `cir` and run there
   (CPU only, `opencv` + `numpy` from an existing conda env -- no GPU, and
   deliberately avoiding the conda env with another user's live training job
   in it) against all 100 videos, producing one CSV of per-frame scores.
3. Same for `src/scripts/extract_motion_concentration.py`.
4. Both resulting CSVs (small -- one float per frame, ~179k rows each) were
   copied back to `data/findingdory/signals_{pixel_novelty,motion_concentration}_validation.csv`
   in this repo (gitignored like the rest of `data/`, but small enough to
   regenerate or hand around if needed).
5. `src/scripts/run_phase2_milestone2.py` runs entirely locally from there,
   and includes whichever of the two signal CSVs it finds.

**How to reproduce**, given access to the video files on `cir`:
```bash
# on cir:
python3 extract_pixel_novelty.py data/findingdory/videos/validation signals_pixel_novelty_validation.csv
python3 extract_motion_concentration.py data/findingdory/videos/validation signals_motion_concentration_validation.csv
# copy both resulting CSVs to data/findingdory/ locally, then:
python3 src/scripts/run_phase2_milestone2.py
```

**Side finding: Milestone 1's episode-length approximation.** Milestone 1
approximated each episode's length as (max oracle frame index referenced) + 1,
since the metadata parquet alone doesn't give true video length (documented as
a caveat in `src/data/findingdory.py`). Now that the actual frame counts are
known (from decoding the videos), `run_phase2_milestone2.py` reports exactly
how often that approximation was exact vs. an undercount, and by how much, as
part of its output -- see the run log / results below, rather than leaving it
as an unverified caveat.

**Results.** See `overall_hit_rate.csv` and `per_category_hit_rate.csv` for
the full budget x policy x category breakdown. Pooled hit-rate at a few
budgets (100/100 validation episodes, run on the TRUE frame counts from the
actual videos, not Milestone 1's approximation):

| policy | B=4 | B=8 | B=16 | B=32 | B=64 | B=128 | B=512 |
|---|---|---|---|---|---|---|---|
| recency | 0.032 | 0.039 | 0.053 | 0.090 | 0.192 | 0.316 | 0.635 |
| uniform | 0.280 | 0.494 | 0.743 | 0.922 | 0.984 | 0.993 | 0.998 |
| random | 0.327 | 0.486 | 0.640 | 0.792 | 0.913 | 0.976 | 0.997 |
| pixel_novelty | 0.396 | 0.563 | 0.692 | 0.817 | 0.881 | 0.942 | 0.995 |
| motion_interaction | **0.420** | **0.602** | **0.780** | 0.869 | 0.932 | 0.982 | 0.994 |
| oracle_greedy | 0.719 | 0.906 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 |

**Reading, as interpretation (not a finished result).** Both signals show the
**same qualitative shape**, despite being designed around different cues
(overall change magnitude vs. spatial concentration of change): each beats
`uniform` at the tightest budgets, then is overtaken by it, with the
crossover happening at a different point per signal.
- `pixel_novelty` is best at B=4, B=8; `uniform` overtakes it from B=16 on,
  and the gap widens with budget (0.82 vs. 0.92 at B=32; 0.88 vs. 0.98 at B=64).
- `motion_interaction` is best at B=4, B=8, *and still B=16* (0.780 vs.
  uniform's 0.743); `uniform` overtakes it from B=32 on, with a narrower gap
  than pixel_novelty's (0.93 vs. 0.98 at B=64, vs. pixel_novelty's 0.88 vs. 0.98).
  `motion_interaction` is the stronger of the two signals at every budget
  tested except the very largest, where all non-oracle policies are close to 1.

Two independently-designed crude signals showing the same "helps only under
severe scarcity, then net-negative" shape is a more general piece of evidence
for H0 (docs/project_proposal.md) than either alone: a single proxy's
usefulness is inconsistent not just across task category (Milestone 1) but
across *budget level*, and this now holds for two different kinds of proxy.
It does not by itself establish H0 for the literature's real proxies
(surprise in a learned latent space, as in P01; learned interaction detection,
as in P07) -- both signals here are much cruder and could behave differently,
and the FindingDory oracle labels being scored against are a dataset-specific,
binary notion of "solves the query," not a graded utility measure.

**Side finding: the heuristic doesn't even win on its own target category.**
Checking `per_category_hit_rate.csv` specifically for `low_level_category ==
"Interaction"` (594 solvable queries): `motion_interaction` does not show a
targeted advantage there. At B=16/32, `uniform` beats both signals on this
category (0.96/0.98 vs. motion_interaction's 0.90/0.94); at B=4/8, the
*simpler, generic* `pixel_novelty` actually beats the purpose-built
`motion_interaction` (0.77 vs. 0.73 at B=4; 0.87 vs. 0.83 at B=8). A signal's
conceptual match to a task category is not a reliable guide to which signal
performs best on it, at least for these crude heuristics -- a useful, concrete
caution for any future hand-designed utility component.

**Side finding confirmed.** Milestone 1's episode-length approximation (max
oracle frame index + 1) was exact for 56/100 episodes and undercounted the
other 44 by 25.7 frames on average (max 87) -- small relative to episode
lengths of hundreds to thousands of frames, so Milestone 1's qualitative
conclusions likely still hold, but this is now a verified statement rather
than an assumption.

## Generalization check: seeded 100-episode train sample

**Setup.** To check whether the validation pattern is specific to those 100
episodes, the same pipeline was run on 100 train-split episodes sampled with
`random.Random(0)` from the 1,478 train episodes (list:
`train_sample_episodes.txt`). Same signals, same budgets, same scoring; results
in `train_sample/`. Two differences to keep in mind: the train split has 55 of
the 60 tasks (the 5 "Object Attributes" tasks are held out of train by the
dataset), and the sampled train episodes are **shorter** on average (~1,429
frames vs. ~1,782 for validation), so any given budget covers a larger
fraction of each episode. Before seeing the train results we predicted that
shorter episodes would move the crossover to a *smaller* budget.

| policy (train sample) | B=4 | B=8 | B=16 | B=32 | B=64 | B=512 |
|---|---|---|---|---|---|---|
| recency | 0.025 | 0.029 | 0.038 | 0.070 | 0.173 | 0.673 |
| uniform | 0.294 | 0.560 | 0.796 | 0.948 | 0.986 | 0.998 |
| random | 0.350 | 0.522 | 0.681 | 0.833 | 0.939 | 0.998 |
| pixel_novelty | 0.435 | 0.576 | 0.703 | 0.807 | 0.871 | 0.997 |
| motion_interaction | 0.446 | 0.622 | 0.779 | 0.862 | 0.948 | 0.994 |
| oracle_greedy | 0.767 | 0.949 | 1.000 | 1.000 | 1.000 | 1.000 |

**Statistical check.** `src/scripts/bootstrap_signal_vs_baseline.py`
computes 95% episode-level bootstrap CIs on each signal-minus-comparator
difference (full output: `bootstrap_ci.txt`). Summary vs. `uniform`
(+ = signal significantly better, − = significantly worse, ~ = not
distinguishable):

| | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|
| pixel_novelty, validation | + | + | − | − | − |
| pixel_novelty, train | + | ~ | − | − | − |
| motion_interaction, validation | + | + | + | − | − |
| motion_interaction, train | + | + | ~ | − | − |

**Reading, as interpretation.**
1. **The crossover replicates.** On both splits, both signals are
   significantly better than `uniform` at B=4 and significantly worse at
   B=32 and B=64. The "helps only under severe scarcity" shape is not a quirk
   of the validation episodes.
2. **The crossover comes earlier on the train sample, for both signals,**
   by about one budget step. That matches the direction we predicted from the
   shorter train episodes. It is consistent with the crossover depending on
   budget *relative to episode length*, but it does not establish that: the
   splits also differ in tasks and scenes, and we tested only one sample.
3. **`pixel_novelty` becomes significantly worse than random selection** at
   B=64 on validation and at B=32 and B=64 on train. A surprise-style signal
   that does worse than picking frames at random is a strong result in
   itself. *Hypothesis, not yet tested:* high-change frames come in temporal
   bursts (e.g. during one fast turn), so the top-K frames bunch into a few
   time spans and leave the rest of the episode uncovered. `motion_interaction`
   is never significantly worse than random on either split, which suggests
   it suffers less from this.
4. **The oracle gap at small budgets is still large** on train (B=4:
   oracle_greedy 0.767 vs. best non-oracle 0.446), as on validation.

**Caveats specific to this check.** The bootstrap's `random` comparator uses
one seed, so its CIs cover episode variance but not random-selection
variance. There are 20 comparisons per split with no multiple-comparison
correction. The train result rests on one 100-episode sample. Everything in
"What this still does NOT show" below still applies.

**What this still does NOT show** (unchanged from Milestone 1, plus two more):
no real learned embedding model is involved; no real downstream reader is
involved (`oracle_greedy` still uses the dataset's own ground-truth labels);
`pixel_novelty` is sensitive to camera motion and lighting changes, not just
semantically meaningful events, so a high score can come from the robot
simply turning quickly rather than anything task-relevant; and
`motion_interaction`'s "spatial concentration" heuristic will also fire on
things that aren't interactions (e.g. a small independently-moving object
elsewhere in frame). Treat both as lower bounds on what their real literature
counterparts (prediction error in a learned embedding space for P01; learned
interaction detection for P07) could achieve, not upper bounds.
