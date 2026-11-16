"""Budget-constrained memory-selection policies that operate purely on frame-index
space (no pixel/embedding access required).

Each policy has the signature `policy(episode_length: int, budget: int, **kwargs) ->
frozenset[int]` and returns the set of frame indices the policy would retain in
persistent memory, given a hard budget of at most `budget` frames, formed WITHOUT
knowledge of the episode's queries (queries are only used afterward, at evaluation
time, to score the retained set -- this mirrors the project's "future task unknown
at write time" framing; see docs/project_proposal.md).

`oracle_greedy` is the one exception: it is an upper-bound reference computed WITH
knowledge of the queries (retrospectively), using a greedy maximum-coverage
approximation. It is not a fair competitor and is labeled as such everywhere it is
used; see docs/project_proposal.md's "Baseline methods" table ("Oracle retrospective
selection ... Upper bound for a given budget").
"""
from __future__ import annotations

import random

from src.data.findingdory import Episode


def full_memory(episode_length: int, budget: int) -> frozenset[int]:
    """Retain everything (ignores budget). Unconstrained upper reference."""
    return frozenset(range(episode_length))


def uniform(episode_length: int, budget: int) -> frozenset[int]:
    """Retain `budget` evenly spaced frames across the episode."""
    budget = min(budget, episode_length)
    if budget <= 0 or episode_length <= 0:
        return frozenset()
    if budget >= episode_length:
        return frozenset(range(episode_length))
    step = episode_length / budget
    idx = {int(i * step) for i in range(budget)}
    return frozenset(idx)


def random_retention(episode_length: int, budget: int, seed: int = 0) -> frozenset[int]:
    """Retain `budget` uniformly-random frames."""
    budget = min(budget, episode_length)
    rng = random.Random(seed)
    return frozenset(rng.sample(range(episode_length), budget)) if budget > 0 else frozenset()


def stratified_random(episode_length: int, budget: int, seed: int = 0) -> frozenset[int]:
    """Control for the stratified signal policies: split the episode into
    `budget` equal time segments (same boundaries as uniform /
    visual_signals.stratified_top1) and keep one uniformly-random frame per
    segment. Separates "using the signal to choose within a segment" from
    "merely not sitting on uniform's fixed grid positions"."""
    budget = min(budget, episode_length)
    if budget <= 0:
        return frozenset()
    rng = random.Random(seed)
    edges = [(i * episode_length) // budget for i in range(budget + 1)]
    return frozenset(rng.randrange(a, b) for a, b in zip(edges[:-1], edges[1:]))


def recency(episode_length: int, budget: int) -> frozenset[int]:
    """Retain only the most recent `budget` frames (FIFO memory)."""
    budget = min(budget, episode_length)
    if budget <= 0:
        return frozenset()
    return frozenset(range(episode_length - budget, episode_length))


def oracle_greedy_order(episode: Episode, max_budget: int) -> list[int]:
    """Retrospective upper-bound reference: greedily order frames by how many
    still-uncovered (query, subgoal) oracle sets they hit, using the standard
    greedy maximum-coverage approximation (guarantees (1 - 1/e) of the optimal
    coverage at every prefix length, not the true optimum).

    Returns an ORDERED list of up to `max_budget` frames: because greedy coverage
    is monotonic, the first K elements of this list are exactly what a from-scratch
    greedy run with budget=K would select, for every K <= max_budget. Callers
    sweeping multiple budgets should call this once per episode with the largest
    budget needed and take prefixes, rather than re-running greedy per budget.

    This is NOT a fair baseline: it is computed with full knowledge of every query
    in the episode, which no realistic write-time policy would have. It exists only
    to give an upper-bound reference curve for the budget-vs-coverage plots.
    """
    max_budget = min(max_budget, episode.episode_length)
    if max_budget <= 0:
        return []

    # Each "unit to cover" is one subgoal's frame set (excluding unsolvable queries).
    uncovered = [s for q in episode.queries for s in q.subgoal_frame_sets if s]
    order: list[int] = []
    for _ in range(max_budget):
        if not uncovered:
            break
        counts: dict[int, int] = {}
        for s in uncovered:
            for f in s:
                counts[f] = counts.get(f, 0) + 1
        if not counts:
            break
        best_frame = max(counts, key=counts.get)
        order.append(best_frame)
        uncovered = [s for s in uncovered if best_frame not in s]
    return order


def oracle_greedy(episode: Episode, budget: int) -> frozenset[int]:
    """Convenience single-budget wrapper around oracle_greedy_order. For a sweep
    over multiple budgets on the same episode, call oracle_greedy_order once with
    the largest budget and take prefixes instead -- this recomputes from scratch
    and is O(budget) times more expensive than necessary for a sweep.
    """
    return frozenset(oracle_greedy_order(episode, budget))


# Registry of policies that need only (episode_length, budget) -- i.e. everything
# except oracle_greedy, which additionally needs the episode's queries.
INDEX_ONLY_POLICIES = {
    "full_memory": full_memory,
    "uniform": uniform,
    "recency": recency,
    # random_retention takes a `seed`; callers sweep seeds explicitly (see
    # src/eval/selection_hit.py) rather than registering it here with one fixed seed.
}
