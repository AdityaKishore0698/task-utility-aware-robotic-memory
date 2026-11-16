"""Milestone 3: pilot test of H1 -- does combining a visual signal with a
temporal-coverage constraint beat both `uniform` and the plain top-K signal?
Design and pre-registered predictions: experiments/milestone3_coverage_pilot/README.md.

Runs on both splits that have signal data (FindingDory validation, and the
seeded 100-episode train sample), writes per-split pooled hit-rate tables,
the segment-coverage diagnostic, and episode-level bootstrap CIs for the
pre-declared comparisons.

Usage:
    python3 src/scripts/run_milestone3_coverage.py
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.findingdory import Episode, load_episodes  # noqa: E402
from src.eval.selection_hit import score_episode  # noqa: E402
from src.policies import baselines  # noqa: E402
from src.policies.visual_signals import (  # noqa: E402
    segment_coverage,
    stratified_top1,
    temporal_nms,
    top_k_by_score,
)
from src.scripts.run_phase2_milestone2 import load_scores, signal_csvs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "milestone3_coverage_pilot"
BUDGETS = [4, 8, 16, 32, 64, 128, 256]
BOOT_BUDGETS = [4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 2000
SIGNALS = ("pixel_novelty", "motion_interaction")

# (policy, comparator) pairs to bootstrap, per signal; pre-declared in the README.
COMPARISONS = [
    ("stratified_{s}", "uniform"),
    ("stratified_{s}", "stratified_random"),
    ("stratified_{s}", "{s}"),
    ("nms_{s}", "uniform"),
    ("nms_{s}", "{s}"),
]


def episode_rows(split: str):
    """Per episode: {policy: {budget: (hits, solvable)}} with random policies
    averaged over SEEDS (hits may be fractional), plus segment coverage of
    each top-K signal policy."""
    sigs = {k: load_scores(p) for k, p in signal_csvs(split).items()}
    rows, coverage = [], {s: {b: [] for b in BUDGETS} for s in SIGNALS}
    for ep in load_episodes(split):
        if not all(ep.ep_id in sc for sc in sigs.values()):
            continue
        length = len(sigs["pixel_novelty"][ep.ep_id])
        e = Episode(ep.ep_id, ep.video_path, length, ep.queries)
        budgets = [b for b in BUDGETS if b <= length]
        oracle_order = baselines.oracle_greedy_order(e, max(budgets))
        row: dict[str, dict[int, tuple[float, int]]] = {}

        def put(name, b, frames):
            s = score_episode(e, frames)
            row.setdefault(name, {})[b] = (s.num_hit, s.num_solvable)

        for b in budgets:
            put("uniform", b, baselines.uniform(length, b))
            put("oracle_greedy", b, frozenset(oracle_order[:b]))
            for rname, fn in (("random", baselines.random_retention),
                              ("stratified_random", baselines.stratified_random)):
                hits, solv = 0.0, 0
                for seed in SEEDS:
                    s = score_episode(e, fn(length, b, seed=seed))
                    hits += s.num_hit / len(SEEDS)
                    solv = s.num_solvable
                row.setdefault(rname, {})[b] = (hits, solv)
            for sig in SIGNALS:
                sc = sigs[sig][ep.ep_id]
                topk = top_k_by_score(sc, b)
                put(sig, b, topk)
                put(f"stratified_{sig}", b, stratified_top1(sc, b))
                put(f"nms_{sig}", b, temporal_nms(sc, b))
                coverage[sig][b].append(segment_coverage(topk, length, b))
        rows.append(row)
    return rows, coverage


def pooled(rows, policy, b):
    h = sum(r[policy][b][0] for r in rows if b in r.get(policy, {}))
    s = sum(r[policy][b][1] for r in rows if b in r.get(policy, {}))
    return h / s if s else float("nan")


def bootstrap(rows, a, c, b, rng):
    usable = [r for r in rows if b in r[a] and b in r[c]]
    n = len(usable)

    def diff(sample):
        ha = sum(usable[i][a][b][0] for i in sample)
        sa = sum(usable[i][a][b][1] for i in sample)
        hc = sum(usable[i][c][b][0] for i in sample)
        sc = sum(usable[i][c][b][1] for i in sample)
        return ha / sa - hc / sc

    point = diff(range(n))
    boots = sorted(diff([rng.randrange(n) for _ in range(n)]) for _ in range(N_BOOT))
    return point, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policies = (["uniform", "random", "stratified_random"]
                + [p for s in SIGNALS for p in (s, f"nms_{s}", f"stratified_{s}")]
                + ["oracle_greedy"])
    report = []
    for split in ("validation", "train"):
        t0 = time.time()
        rows, coverage = episode_rows(split)
        report.append(f"\n##### {split}: {len(rows)} episodes ({time.time() - t0:.0f}s)")

        with (OUT_DIR / f"{split}_overall_hit_rate.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy"] + BUDGETS)
            for p in policies:
                w.writerow([p] + [f"{pooled(rows, p, b):.4f}" for b in BUDGETS])

        report.append("\npooled hit-rate:")
        report.append("policy".ljust(30) + "".join(str(b).rjust(8) for b in BUDGETS))
        for p in policies:
            report.append(p.ljust(30) + "".join(f"{pooled(rows, p, b):8.3f}" for b in BUDGETS))

        report.append("\nsegment coverage of top-K picks (fraction of B equal segments hit; uniform = 1.0):")
        report.append("signal".ljust(30) + "".join(str(b).rjust(8) for b in BUDGETS))
        for sig in SIGNALS:
            cells = []
            for b in BUDGETS:
                v = coverage[sig][b]
                cells.append(f"{sum(v) / len(v):8.3f}" if v else "       -")
            report.append(sig.ljust(30) + "".join(cells))

        report.append("\nbootstrap 95% CI of pooled difference (policy - comparator); "
                      "+ better, - worse, ~ indistinguishable:")
        rng = random.Random(0)
        for sig in SIGNALS:
            for a_t, c_t in COMPARISONS:
                a, c = a_t.format(s=sig), c_t.format(s=sig)
                cells = []
                for b in BOOT_BUDGETS:
                    pt, lo, hi = bootstrap(rows, a, c, b, rng)
                    mark = "+" if lo > 0 else ("-" if hi < 0 else "~")
                    cells.append(f"B{b}:{pt:+.3f}[{lo:+.3f},{hi:+.3f}]{mark}")
                report.append(f"  {a:29s} vs {c:26s} " + "  ".join(cells))

    text = "\n".join(report)
    print(text)
    (OUT_DIR / "results.txt").write_text(text.lstrip() + "\n")
    print(f"\nwrote {OUT_DIR}/results.txt and per-split CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
