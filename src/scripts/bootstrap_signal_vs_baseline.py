"""Episode-level bootstrap confidence intervals for the Milestone 2 comparison:
is each visual-signal policy's pooled hit-rate significantly different from
`uniform` and from `random`, at each budget, on each split?

For each (signal, comparator, budget) we report the pooled hit-rate difference
(signal minus comparator) and a 95% percentile-bootstrap CI from resampling
EPISODES with replacement (2,000 resamples, seeded). Episodes, not queries,
are resampled because the ~60 queries within an episode share one memory and
are not independent.

Caveats:
  - The `random` comparator here is a single seed (0), so the CI reflects
    episode-sampling variance only, not random-selection variance. The
    milestone tables average 5 seeds; small differences vs. random should be
    read with that in mind.
  - No multiple-comparison correction (20 comparisons per split).
  - Same selection-hit scoring and its simplifications as everywhere else
    (see src/eval/selection_hit.py).

Usage:
    python3 src/scripts/bootstrap_signal_vs_baseline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.findingdory import Episode, load_episodes  # noqa: E402
from src.eval.selection_hit import score_episode  # noqa: E402
from src.policies import baselines  # noqa: E402
from src.policies.visual_signals import top_k_by_score  # noqa: E402
from src.scripts.run_phase2_milestone2 import load_scores, signal_csvs  # noqa: E402

BUDGETS = [4, 8, 16, 32, 64]
N_BOOT = 2000
SIGNALS = ("pixel_novelty", "motion_interaction")
COMPARATORS = ("uniform", "random")


def per_episode_counts(split: str) -> list[dict[str, dict[int, tuple[int, int]]]]:
    sigs = {k: load_scores(p) for k, p in signal_csvs(split).items() if p.exists()}
    rows = []
    for ep in load_episodes(split):
        if not all(ep.ep_id in s for s in sigs.values()):
            continue
        length = len(sigs["pixel_novelty"][ep.ep_id])
        e = Episode(ep.ep_id, ep.video_path, length, ep.queries)
        row: dict[str, dict[int, tuple[int, int]]] = {}
        for b in BUDGETS:
            for name, frames in (
                ("uniform", baselines.uniform(length, b)),
                ("random", baselines.random_retention(length, b, seed=0)),
                ("pixel_novelty", top_k_by_score(sigs["pixel_novelty"][ep.ep_id], b)),
                ("motion_interaction", top_k_by_score(sigs["motion_interaction"][ep.ep_id], b)),
            ):
                s = score_episode(e, frames)
                row.setdefault(name, {})[b] = (s.num_hit, s.num_solvable)
        rows.append(row)
    return rows


def main() -> int:
    for split in ("validation", "train"):
        per_ep = per_episode_counts(split)
        n = len(per_ep)
        rng = random.Random(0)
        print(f"\n=== {split}: {n} episodes, pooled hit-rate difference (signal - comparator), "
              f"95% bootstrap CI over episodes ===")
        for sig in SIGNALS:
            for comp in COMPARATORS:
                cells = []
                for b in BUDGETS:
                    def diff(sample):
                        hs = sum(per_ep[i][sig][b][0] for i in sample)
                        ss = sum(per_ep[i][sig][b][1] for i in sample)
                        hc = sum(per_ep[i][comp][b][0] for i in sample)
                        sc = sum(per_ep[i][comp][b][1] for i in sample)
                        return hs / ss - hc / sc

                    point = diff(range(n))
                    boots = sorted(diff([rng.randrange(n) for _ in range(n)]) for _ in range(N_BOOT))
                    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
                    mark = "+" if lo > 0 else ("-" if hi < 0 else "~")
                    cells.append(f"B{b}: {point:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")
                print(f"  {sig:18s} vs {comp:7s} " + "  ".join(cells))
    print("\n+ = signal significantly better, - = significantly worse, ~ = CI includes 0 (not distinguishable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
