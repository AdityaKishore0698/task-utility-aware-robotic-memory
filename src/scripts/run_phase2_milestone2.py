"""Phase 2 / Milestone 2: add crude, zero-model visual-signal policies to the
Milestone 1 budget-vs-coverage comparison, on the same FindingDory validation
episodes and the same selection-hit scoring:
  - `pixel_novelty`      approximates P01's surprise signal (see
                          src/scripts/extract_pixel_novelty.py)
  - `motion_interaction` approximates P07's interaction signal (see
                          src/scripts/extract_motion_concentration.py)
Either signal's CSV may be present or absent; whichever are found under
data/findingdory/ are included (an episode is only scored if every loaded
signal has data for it, so the comparison stays apples-to-apples).

This also cross-checks Milestone 1's episode-length approximation: Milestone 1
approximated each episode's length as (max oracle frame index referenced) + 1,
since the parquet alone doesn't give true video length. Now that we have the
real per-episode frame counts (from decoding the actual videos on `cir`), we
report how often that approximation was exact vs. an undercount, instead of
silently assuming it was fine.

Prerequisite: src/scripts/extract_pixel_novelty.py and/or
extract_motion_concentration.py have been run against the validation videos
(on a machine that has them -- see experiments/phase2_milestone2/README.md for
how this was actually run) and their output CSVs copied to
data/findingdory/signals_{pixel_novelty,motion_concentration}_validation.csv.

Usage:
    python3 src/scripts/run_phase2_milestone2.py                 # validation (default)
    python3 src/scripts/run_phase2_milestone2.py --split train   # sampled train episodes
For --split train, only episodes that have signal data are scored, i.e. the
seeded 100-episode sample in experiments/phase2_milestone2/train_sample_episodes.txt
whose videos were actually processed.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.findingdory import Episode, load_episodes  # noqa: E402
from src.eval.selection_hit import score_episode, score_episode_by_category  # noqa: E402
from src.policies import baselines  # noqa: E402
from src.policies.visual_signals import top_k_by_score  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "findingdory"
EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "phase2_milestone2"


def signal_csvs(split: str) -> dict[str, Path]:
    return {
        "pixel_novelty": DATA_DIR / f"signals_pixel_novelty_{split}.csv",
        "motion_interaction": DATA_DIR / f"signals_motion_concentration_{split}.csv",
    }


def out_dir(split: str) -> Path:
    # Validation results stay at the top level (where they were first written);
    # the train generalization check goes in its own subdirectory.
    return EXP_DIR if split == "validation" else EXP_DIR / "train_sample"
BUDGETS = [4, 8, 16, 32, 64, 128, 256, 512]
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def load_scores(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/scripts/extract_pixel_novelty.py against the "
            "videos first (see experiments/phase2_milestone2/README.md)."
        )
    by_ep: dict[str, list[tuple[int, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            by_ep.setdefault(row["ep_id"], []).append((int(row["frame_idx"]), float(row["score"])))
    out: dict[str, list[float]] = {}
    for ep_id, pairs in by_ep.items():
        pairs.sort(key=lambda p: p[0])
        out[ep_id] = [s for _, s in pairs]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    args = ap.parse_args()
    OUT_DIR = out_dir(args.split)

    t0 = time.time()
    episodes = load_episodes(args.split)
    signal_sets: dict[str, dict[str, list[float]]] = {}
    for name, path in signal_csvs(args.split).items():
        if path.exists():
            signal_sets[name] = load_scores(path)
            print(f"loaded signal '{name}' from {path.name}")
        else:
            print(f"signal '{name}' not found at {path} -- skipping it this run")
    if not signal_sets:
        raise FileNotFoundError(
            "no signal CSVs found under data/findingdory/; run at least one of "
            "src/scripts/extract_pixel_novelty.py / extract_motion_concentration.py first"
        )
    # Use whichever signal has the most coverage to drive the episode-length
    # cross-check (they should all agree on frame counts; this just needs one).
    primary_scores = next(iter(signal_sets.values()))

    # Cross-check Milestone 1's episode-length approximation against the real
    # frame counts we now have from decoding video.
    exact, undercount, overcount, missing = 0, 0, 0, 0
    length_deltas = []
    for ep in episodes:
        true_len = len(primary_scores.get(ep.ep_id, []))
        if true_len == 0:
            # For the train split most episodes have no video (only the sample
            # was processed) -- that's expected, not a data problem.
            missing += 1
            continue
        delta = true_len - ep.episode_length
        length_deltas.append(delta)
        if delta == 0:
            exact += 1
        elif delta > 0:
            undercount += 1
        else:
            overcount += 1
    print(f"episode-length cross-check: {exact} exact, {undercount} undercounted by "
          f"Milestone 1's approximation, {overcount} overcounted, {missing} missing signal data")
    if length_deltas:
        print(f"  undercount size when wrong: mean={mean([d for d in length_deltas if d > 0] or [0]):.1f} "
              f"frames, max={max(length_deltas)}")

    overall: dict[tuple[str, int], list[int]] = {}
    per_category: dict[tuple[str, int, str, str], list[int]] = {}

    n_used = 0
    for episode in episodes:
        if not all(episode.ep_id in sc for sc in signal_sets.values()):
            continue  # need every loaded signal present for this episode to compare fairly
        n_used += 1
        # Use the TRUE frame count (from the video) as episode_length for every
        # policy in this run, so the visual-signal policies and the Milestone 1
        # baselines are compared on an identical, now-verified footing rather
        # than Milestone 1's approximation.
        true_len = len(primary_scores[episode.ep_id])
        ep_for_eval = Episode(
            ep_id=episode.ep_id, video_path=episode.video_path,
            episode_length=true_len, queries=episode.queries,
        )
        budgets = [b for b in BUDGETS if b <= true_len] + ([true_len] if true_len < BUDGETS[-1] else [])
        budgets = sorted(set(budgets))
        oracle_order = baselines.oracle_greedy_order(ep_for_eval, max(budgets))

        policies: dict[str, dict[int, frozenset[int]]] = {
            "full_memory": {b: baselines.full_memory(true_len, b) for b in budgets},
            "uniform": {b: baselines.uniform(true_len, b) for b in budgets},
            "recency": {b: baselines.recency(true_len, b) for b in budgets},
            "oracle_greedy": {b: frozenset(oracle_order[:b]) for b in budgets},
        }
        for sig_name, sc in signal_sets.items():
            policies[sig_name] = {b: top_k_by_score(sc[episode.ep_id], b) for b in budgets}
        for seed in RANDOM_SEEDS:
            policies[f"random_seed{seed}"] = {
                b: baselines.random_retention(true_len, b, seed=seed) for b in budgets
            }

        for policy_name, per_budget in policies.items():
            for b, frames in per_budget.items():
                s = score_episode(ep_for_eval, frames)
                key = (policy_name, b)
                overall.setdefault(key, [0, 0])
                overall[key][0] += s.num_hit
                overall[key][1] += s.num_solvable

                by_cat = score_episode_by_category(ep_for_eval, frames)
                for (hi, lo), cs in by_cat.items():
                    ckey = (policy_name, b, hi, lo)
                    per_category.setdefault(ckey, [0, 0])
                    per_category[ckey][0] += cs.num_hit
                    per_category[ckey][1] += cs.num_solvable

    print(f"scored {n_used}/{len(episodes)} episodes (those with every loaded signal available: "
          f"{', '.join(signal_sets)})")

    seed_rates: dict[int, list[float]] = {}
    for (policy_name, b), (hit, solv) in list(overall.items()):
        if policy_name.startswith("random_seed"):
            seed_rates.setdefault(b, []).append(hit / solv if solv else float("nan"))

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

    policies_order = (
        ["full_memory", "recency", "uniform", "random"]
        + list(signal_sets.keys())
        + ["oracle_greedy"]
    )
    print("\nPooled hit-rate by policy x budget:\n")
    header = "policy".ljust(16) + "".join(str(b).rjust(8) for b in BUDGETS)
    print(header)
    for name in policies_order:
        row = [name.ljust(16)]
        for b in BUDGETS:
            if name == "random":
                rates = seed_rates.get(b)
                cell = f"{mean(rates):.3f}" if rates else "   -"
            else:
                val = overall.get((name, b))
                cell = f"{val[0] / val[1]:.3f}" if val and val[1] else "   -"
            row.append(cell.rjust(8))
        print("".join(row))

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
