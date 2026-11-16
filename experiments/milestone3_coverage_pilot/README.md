# Milestone 3 — pilot test of H1: signal + temporal coverage

**Status of this file.** The *Design* and *Predictions* sections below were
written and saved **before** the experiment was run, and are left unedited
afterwards. Results and interpretation are added in separate sections below
them.

**Where this sits in the roadmap.** This is an exploratory pilot of H1
(docs/project_proposal.md), run before the formal Phase 3 go/no-go write-up.
It uses the same data (FindingDory validation + the seeded 100-episode train
sample), the same two crude signals, and the same selection-hit scoring as
[Milestone 2](../phase2_milestone2/). Nothing here is tuned on these splits:
every policy is parameter-free or has one parameter fixed in advance.

## Motivation (from Milestone 2)

Both crude signals beat `uniform` only at very small budgets and lose to it
from B≈16–32 onward, on both splits. `pixel_novelty` is even significantly
worse than *random* selection at B=32–64. The working hypothesis: the
signals pick frames that are individually informative but **bunched in
time**, so above a small budget, how evenly frames cover the episode matters
more than which frames score highest.

## Design

Policies, all under the same hard frame budget B:

| policy | rule | parameters |
|---|---|---|
| `uniform` | B evenly spaced frames (Milestone 1) | none |
| `stratified_random` | **control**: split the episode into B equal time segments, one random frame per segment | seed (5 seeds averaged) |
| `{signal}` (top-K) | B highest-scoring frames (Milestone 2) | none |
| `stratified_{signal}` | split into B equal segments, highest-scoring frame per segment | none |
| `nms_{signal}` | frames in score order, skipping any within `gap = L // (2B)` frames of one already picked; fill to B by score if suppression runs out | `gap` fixed a priori at half the uniform spacing |
| `oracle_greedy` | retrospective upper bound (Milestone 1) | none |

for `{signal}` in `pixel_novelty`, `motion_interaction`.

The `stratified_random` control matters: `stratified_{signal}` and
`uniform` have identical temporal coverage, so if `stratified_{signal}` beats
`uniform`, that could be due to the signal, *or* merely due to not sitting on
uniform's fixed grid positions. `stratified_{signal}` must also beat
`stratified_random` for the gain to be credited to the signal.

**Diagnostic for the "bunching" hypothesis.** For each top-K signal policy,
report *segment coverage*: the fraction of the B equal time segments that
contain at least one picked frame (`uniform`/stratified policies are 1.0 by
construction).

**Statistics.** 95% episode-level bootstrap CIs (2,000 resamples, seeded)
on pooled hit-rate differences, for: `stratified_{signal}` vs `uniform`,
vs `stratified_random`, vs top-K `{signal}`; and `nms_{signal}` vs `uniform`
and vs top-K `{signal}`. Random policies are averaged over 5 seeds *within*
each episode before bootstrapping (fixing Milestone 2's single-seed caveat).

## Predictions (written before running)

- **P1 (bunching).** Top-K segment coverage is well below 1.0 at B ≥ 16, and
  lower for `pixel_novelty` than for `motion_interaction` (consistent with
  pixel_novelty being the one that falls below random).
- **P2 (coverage repairs the large-budget loss).** `stratified_{signal}` is
  not significantly worse than `uniform` at any budget, and is significantly
  better than top-K `{signal}` at B ≥ 32.
- **P3 (the signal adds value on top of coverage) — the real H1 test.**
  `stratified_{signal}` is significantly better than `stratified_random` at
  some budgets. If it is not, the signals carry no usable information beyond
  coverage, which would count against H1 for these signals.
- **P4 (cost at tiny budgets).** At B=4, `stratified_{signal}` may do *worse*
  than top-K `{signal}`, because the coverage constraint forbids spending
  several frames on the single most informative stretch.
- **P5.** `nms_{signal}` falls between top-K and stratified: better than
  top-K at B ≥ 32, closer to top-K at B=4.

Any of these failing is a result to report, not to fix by redesigning the
policies on the same data.

---

## Results (added after running)

Reproduce: `python3 src/scripts/run_milestone3_coverage.py` (~1 min, CPU;
needs the four signal CSVs in `data/findingdory/`). Full output:
`results.txt`; per-split tables: `{validation,train}_overall_hit_rate.csv`.

Pooled hit-rate, 100 episodes per split:

| policy | val B=4 | val B=8 | val B=16 | val B=32 | train B=4 | train B=8 | train B=16 | train B=32 |
|---|---|---|---|---|---|---|---|---|
| uniform | 0.280 | 0.494 | 0.743 | 0.922 | 0.294 | 0.560 | 0.796 | 0.948 |
| stratified_random (control) | 0.331 | 0.513 | 0.703 | 0.877 | 0.355 | 0.556 | 0.765 | 0.909 |
| pixel_novelty (top-K) | 0.396 | 0.563 | 0.692 | 0.817 | 0.435 | 0.576 | 0.703 | 0.807 |
| motion_interaction (top-K) | 0.420 | 0.602 | 0.780 | 0.869 | 0.446 | 0.622 | 0.779 | 0.862 |
| stratified_pixel_novelty | 0.535 | 0.739 | 0.881 | 0.947 | 0.582 | 0.770 | 0.896 | 0.955 |
| nms_pixel_novelty | 0.537 | 0.768 | 0.897 | 0.948 | 0.592 | 0.787 | 0.894 | 0.958 |
| stratified_motion_interaction | 0.580 | 0.787 | 0.925 | 0.978 | 0.639 | 0.841 | 0.948 | 0.981 |
| nms_motion_interaction | **0.593** | **0.826** | **0.953** | **0.983** | **0.653** | **0.868** | **0.963** | **0.981** |
| oracle_greedy (upper bound) | 0.719 | 0.906 | 0.997 | 1.000 | 0.767 | 0.949 | 1.000 | 1.000 |

From B=64 upward every coverage-based policy (uniform, stratified_*,
nms_*) lands at 0.97–0.99 on both splits (see the CSVs): the metric saturates (see
caveat 2 below).

**Share of the uniform→oracle gap closed** (`(policy − uniform) /
(oracle − uniform)`; B=32 omitted from the claim because the gap being
divided is tiny there):

| | val B=4 | val B=8 | val B=16 | train B=4 | train B=8 | train B=16 |
|---|---|---|---|---|---|---|
| motion_interaction (top-K) | 32% | 26% | 15% | 32% | 16% | −8% |
| stratified_motion_interaction | 68% | 71% | 72% | 73% | 72% | 74% |
| nms_motion_interaction | 71% | 81% | 83% | 76% | 79% | 82% |

### Predictions checked

- **P1 (bunching) — partly supported.** Top-K picks cover only 34–60% of
  the time segments at every budget on both splits (uniform: 100%), so
  bunching is real and large. But the predicted ordering (pixel_novelty
  bunches more than motion_interaction) holds only at B ≤ 32 and reverses
  at B ≥ 64. Coverage alone therefore does not explain why pixel_novelty,
  but not motion_interaction, fell below random at B=32–64 in Milestone 2.
- **P2 (coverage repairs the large-budget loss) — supported.**
  `stratified_{signal}` is never significantly worse than `uniform` at any
  tested budget (4–128), on either split, for either signal; and it is significantly better
  than its top-K version at every budget from 4 to 128.
- **P3 (the signal adds value beyond coverage; the H1 test) — supported.**
  `stratified_{signal}` beats the `stratified_random` control significantly
  at B=4–64 on both splits for both signals, by +0.20 to +0.29 at B=4–8.
  Within each time segment, the high-change frame is much more likely to be
  an oracle frame than a random frame is. So the gain over `uniform` is
  credited to the signal, not just to moving off uniform's grid.
- **P4 (coverage costs something at tiny budgets) — failed.** We predicted
  `stratified_{signal}` might lose to top-K at B=4. It wins by +0.14 to
  +0.19, even at B=4, on both splits. Bunching hurts even with only four
  frames to spend.
- **P5 (nms falls between top-K and stratified) — failed in its specific
  form.** `nms_{signal}` beats top-K everywhere as expected, but it is not
  "between": its point estimates are equal to or *above* `stratified` at
  B ≤ 16, making it the best non-oracle policy there. (nms vs. stratified was
  not a pre-declared comparison and was not significance-tested.)

### Reading, as interpretation

The headline is that **a crude signal plus a coverage constraint beats
`uniform` by large, significant margins at B=4–16 on both splits for both
signals** (and at B=32 for motion_interaction; for pixel_novelty at B=32
only on validation)
(e.g. validation B=4: 0.59 vs. 0.28), and closes roughly 70–80% of the
uniform→oracle gap at B=4–16. Neither ingredient does this alone: coverage
without the signal (`stratified_random`) only slightly beats uniform at B=4
and is below it by B=16 (point estimates); the signal without coverage (top-K) loses to uniform by
B=16–32. This is the first direct, positive evidence for H1's core idea
(combining a utility estimate with a budget/coverage constraint beats any
single criterion), at the scale of this pilot.

### Caveats that limit this result

1. **The hypothesis was formed on the same data.** The coverage idea came
   from looking at Milestone 2's results on these same two splits. The
   policies were not tuned on them, but this is still not a held-out
   confirmation. **1,378 train episodes remain untouched** and should be
   used for one confirmatory run before treating this as established.
2. **The metric is lenient and saturates.** Oracle frame sets are large
   (median ~90 frames, ≈6–7% of an episode; only ~30% are a single
   contiguous span). A "hit" needs just one retained frame anywhere in the
   set, so any well-spread policy hits nearly everything from B≈64 on, and
   coverage is rewarded heavily even below that. A real downstream reader
   (a VLM that must *use* the frame) is likely stricter, so the coverage
   benefit measured here may be an upper bound.
3. **Crude signals, no real reader**, exactly as in Milestones 1–2.
4. About 30 comparisons per split with no multiple-comparison correction,
   though almost all the claimed differences have CIs far from zero.
