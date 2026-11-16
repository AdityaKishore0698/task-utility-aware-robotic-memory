"""Phase 2 / Milestone 1: budget-vs-coverage comparison of index-space memory
policies on FindingDory validation, using zero models and zero video downloads
(see docs/research_roadmap.md Phase 2, and src/eval/selection_hit.py for what
"coverage" means and does not mean here).

Policies compared, all under the SAME hard frame budget per episode:
  - full_memory     unconstrained upper reference (ignores budget)
  - uniform         evenly spaced frames
  - recency         most recent `budget` frames (FIFO)
  - random          uniformly random frames, averaged over multiple seeds
  - oracle_greedy   retrospective upper-bound reference (NOT a fair competitor;
                     built with full knowledge of the episode's queries)

This operationalizes E1 (equal-budget baseline comparison) and gives a first,
crude read on E2/E3 (oracle gap, and whether it varies by task category) from
docs/project_proposal.md -- crude because "oracle_greedy" here is a maximum-
coverage approximation over the dataset's own binary oracle-frame labels, not
the leave-one-out / greedy-gain retrospective utility defined in the proposal,
and because hits are scored without a real downstream reader (see
src/eval/selection_hit.py docstring for the full list of simplifications).

Usage:
    python3 src/scripts/run_phase2_milestone1.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.findingdory import Episode, load_episodes  # noqa: E402
from src.eval.selection_hit import score_episode, score_episode_by_category  # noqa: E402
from src.policies import baselines  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "phase2_milestone1"
BUDGETS = [4, 8, 16, 32, 64, 128, 256, 512]
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def retained_sets_for_episode(episode: Episode, budgets: list[int]) -> dict[str, dict[int, frozenset[int]]]:
    """Return {policy_name: {budget: retained_frame_set}} for one episode."""
    max_budget = max(budgets)
    oracle_order = baselines.oracle_greedy_order(episode, max_budget)

    out: dict[str, dict[int, frozenset[int]]] = {
        "full_memory": {},
        "uniform": {},
        "recency": {},
        "oracle_greedy": {},
    }
    for b in budgets:
        out["full_memory"][b] = baselines.full_memory(episode.episode_length, b)
        out["uniform"][b] = baselines.uniform(episode.episode_length, b)
        out["recency"][b] = baselines.recency(episode.episode_length, b)
        out["oracle_greedy"][b] = frozenset(oracle_order[:b])

    for seed in RANDOM_SEEDS:
        name = f"random_seed{seed}"
        out[name] = {
            b: baselines.random_retention(episode.episode_length, b, seed=seed) for b in budgets
        }
    return out


def main() -> int:
    t0 = time.time()
    episodes = load_episodes("validation")
    print(f"loaded {len(episodes)} validation episodes, "
          f"{sum(len(e.queries) for e in episodes)} queries total")

    # overall[(policy, budget)] -> (num_hit, num_solvable) pooled across all episodes
    overall: dict[tuple[str, int], list[int]] = {}
    # per_category[(policy, budget, hi_cat, lo_cat)] -> [num_hit, num_solvable]
    per_category: dict[tuple[str, int, str, str], list[int]] = {}

    for i, episode in enumerate(episodes):
        budgets = [b for b in BUDGETS if b <= episode.episode_length] + (
            [episode.episode_length] if episode.episode_length < BUDGETS[-1] else []
        )
        budgets = sorted(set(budgets))
        retained = retained_sets_for_episode(episode, budgets)

        for policy_name, per_budget in retained.items():
            for b, frames in per_budget.items():
                s = score_episode(episode, frames)
                key = (policy_name, b)
                overall.setdefault(key, [0, 0])
                overall[key][0] += s.num_hit
                overall[key][1] += s.num_solvable

                by_cat = score_episode_by_category(episode, frames)
                for (hi, lo), cs in by_cat.items():
                    ckey = (policy_name, b, hi, lo)
                    per_category.setdefault(ckey, [0, 0])
                    per_category[ckey][0] += cs.num_hit
                    per_category[ckey][1] += cs.num_solvable

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(episodes)} episodes ({time.time() - t0:.1f}s elapsed)")

    # Collapse the per-seed random policies into a mean +/- population-stddev "random" row.
    collapsed: dict[tuple[str, int], tuple[int, int, float | None]] = {}
    seed_rates: dict[tuple[int], dict[int, list[float]]] = {}
    for (policy_name, b), (hit, solv) in overall.items():
        if policy_name.startswith("random_seed"):
            seed_rates.setdefault(b, []).append(hit / solv if solv else float("nan"))
        else:
            collapsed[(policy_name, b)] = (hit, solv, None)
    for b, rates in seed_rates.items():
        collapsed[("random", b)] = (
            -1, -1, mean(rates)
        )
        collapsed[("random_std", b)] = (-1, -1, pstdev(rates) if len(rates) > 1 else 0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_csv = OUT_DIR / "overall_hit_rate.csv"
    with overall_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "budget", "num_hit", "num_solvable", "hit_rate"])
        for (policy_name, b), (hit, solv) in sorted(overall.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if policy_name.startswith("random_seed"):
                continue
            rate = hit / solv if solv else float("nan")
            w.writerow([policy_name, b, hit, solv, f"{rate:.4f}"])
        for b in sorted(seed_rates):
            m = mean(seed_rates[b])
            sd = pstdev(seed_rates[b]) if len(seed_rates[b]) > 1 else 0.0
            w.writerow(["random_mean", b, "", "", f"{m:.4f}"])
            w.writerow(["random_std", b, "", "", f"{sd:.4f}"])
    print(f"wrote {overall_csv}")

    per_cat_csv = OUT_DIR / "per_category_hit_rate.csv"
    with per_cat_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "budget", "high_level_category", "low_level_category",
                    "num_hit", "num_solvable", "hit_rate"])
        for (policy_name, b, hi, lo), (hit, solv) in sorted(
            per_category.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
        ):
            if policy_name.startswith("random_seed"):
                continue
            rate = hit / solv if solv else float("nan")
            w.writerow([policy_name, b, hi, lo, hit, solv, f"{rate:.4f}"])
    print(f"wrote {per_cat_csv}")

    # Readable summary to stdout: pooled hit-rate, policy x budget.
    policies_order = ["full_memory", "recency", "uniform", "random", "oracle_greedy"]
    print("\nPooled hit-rate (fraction of solvable queries whose retained memory "
          "covers every required subgoal), by policy x budget:\n")
    header = "policy".ljust(14) + "".join(str(b).rjust(8) for b in BUDGETS)
    print(header)
    for name in policies_order:
        row = [name.ljust(14)]
        for b in BUDGETS:
            if name == "random":
                val = collapsed.get(("random", b))
                cell = f"{val[2]:.3f}" if val else "   -"
            else:
                val = overall.get((name, b))
                cell = f"{val[0] / val[1]:.3f}" if val and val[1] else "   -"
            row.append(cell.rjust(8))
        print("".join(row))

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
