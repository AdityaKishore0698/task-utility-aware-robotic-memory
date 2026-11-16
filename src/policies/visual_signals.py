"""Memory policies driven by a precomputed per-frame visual signal (e.g. the
pixel_novelty score from src/scripts/extract_pixel_novelty.py), as opposed to
the index-only policies in baselines.py which see no pixel data at all.
"""
from __future__ import annotations


def top_k_by_score(scores: list[float], budget: int) -> frozenset[int]:
    """Retain the `budget` frames with the highest score. Ties broken by
    frame index (earlier frame wins) for determinism.

    This is the generic "write-time gate" shape used by every literature
    policy we're approximating (P01's surprise gate, P07's interaction
    detector): score every frame by some signal, keep the top ones under a
    budget. What differs between approximations is only how `scores` was
    computed -- see the caller for which signal this is and its caveats.
    """
    budget = min(budget, len(scores))
    if budget <= 0:
        return frozenset()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return frozenset(order[:budget])


def segment_edges(length: int, budget: int) -> list[int]:
    """Boundaries of `budget` equal-width time segments over [0, length).
    Segment i is [edges[i], edges[i+1]). Segment starts coincide with the
    frames `baselines.uniform` picks, so stratified policies differ from
    uniform only in where inside each segment they pick."""
    return [(i * length) // budget for i in range(budget + 1)]


def stratified_top1(scores: list[float], budget: int) -> frozenset[int]:
    """Coverage-constrained signal policy: split the episode into `budget`
    equal time segments and keep the highest-scoring frame in each (ties ->
    earliest). Same temporal coverage as `uniform` by construction; the
    signal only chooses the position within each segment. Parameter-free."""
    n = len(scores)
    budget = min(budget, n)
    if budget <= 0:
        return frozenset()
    edges = segment_edges(n, budget)
    picks = set()
    for a, b in zip(edges[:-1], edges[1:]):
        picks.add(max(range(a, b), key=lambda i: (scores[i], -i)))
    return frozenset(picks)


def temporal_nms(scores: list[float], budget: int) -> frozenset[int]:
    """Signal-first policy with anti-clustering: take frames in descending
    score order, skipping any frame within `gap` frames of an already-picked
    one, where gap = max(1, length // (2 * budget)) -- half the uniform
    spacing, fixed a priori (not tuned). If suppression leaves fewer than
    `budget` frames, the remainder is filled with the highest-scoring
    unpicked frames, so the budget is always used in full."""
    n = len(scores)
    budget = min(budget, n)
    if budget <= 0:
        return frozenset()
    gap = max(1, n // (2 * budget))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    suppressed = [False] * n
    picks: list[int] = []
    for i in order:
        if len(picks) == budget:
            break
        if suppressed[i]:
            continue
        picks.append(i)
        for j in range(max(0, i - gap + 1), min(n, i + gap)):
            suppressed[j] = True
    if len(picks) < budget:
        chosen = set(picks)
        for i in order:
            if len(picks) == budget:
                break
            if i not in chosen:
                picks.append(i)
                chosen.add(i)
    return frozenset(picks)


def segment_coverage(frames: frozenset[int], length: int, budget: int) -> float:
    """Diagnostic: fraction of the `budget` equal time segments containing at
    least one retained frame. `uniform` and `stratified_*` score 1.0 by
    construction; a policy whose picks bunch together in time scores low."""
    budget = min(budget, length)
    if budget <= 0:
        return float("nan")
    edges = segment_edges(length, budget)
    covered = 0
    for a, b in zip(edges[:-1], edges[1:]):
        if any(a <= f < b for f in frames):
            covered += 1
    return covered / budget
