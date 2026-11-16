"""Selection-hit scoring: given a budget-constrained retained frame set, does it
"solve" a query?

This is a deliberately cheap, zero-model stand-in for the project's "frozen
downstream reader" (see docs/project_proposal.md, "Evaluation methodology"). It
reuses FindingDory's own notion of correctness ("choosing any frame from the
oracle list solves the task" -- FindingDory paper) rather than calling an actual
VLM: a query counts as solved iff the retained memory contains, for EVERY
subgoal, at least one frame from that subgoal's oracle set.

Known simplifications (flag honestly, do not treat as a finished evaluation):
  - Multi-goal "Ordered Revisitation" queries require visiting subgoals in a
    specific order; this scorer only checks coverage of each subgoal's frame set,
    not the order in which they would be reached. It therefore likely OVER-counts
    hits on ordered multi-goal queries relative to a real closed-loop agent.
  - This measures whether a *sufficient* frame was retained, not whether a real
    VLM reader could actually find and use it. It is a proxy for an upper bound on
    what any reader could do with that memory, not a measured reader accuracy.
  - Unsolvable queries (empty subgoal_frame_sets, i.e. the original [[-1]] marker)
    are excluded from scoring entirely, for every policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data.findingdory import Episode, Query


def query_is_hit(query: Query, retained: frozenset[int]) -> bool | None:
    """Return True/False if the query is solvable, or None if it is marked
    unsolvable in the dataset (excluded from scoring)."""
    if query.is_unsolvable:
        return None
    return all(bool(retained & s) for s in query.subgoal_frame_sets)


@dataclass(frozen=True)
class EpisodeScore:
    ep_id: str
    num_solvable: int
    num_hit: int

    @property
    def hit_rate(self) -> float:
        return self.num_hit / self.num_solvable if self.num_solvable else float("nan")


def score_episode(episode: Episode, retained: frozenset[int]) -> EpisodeScore:
    num_solvable = 0
    num_hit = 0
    for q in episode.queries:
        hit = query_is_hit(q, retained)
        if hit is None:
            continue
        num_solvable += 1
        num_hit += int(hit)
    return EpisodeScore(ep_id=episode.ep_id, num_solvable=num_solvable, num_hit=num_hit)


def score_episode_by_category(
    episode: Episode, retained: frozenset[int]
) -> dict[tuple[str, str], EpisodeScore]:
    """Same as score_episode but broken down by (high_level_category, low_level_category)."""
    buckets: dict[tuple[str, str], list[Query]] = {}
    for q in episode.queries:
        buckets.setdefault((q.high_level_category, q.low_level_category), []).append(q)

    out: dict[tuple[str, str], EpisodeScore] = {}
    for key, qs in buckets.items():
        num_solvable = 0
        num_hit = 0
        for q in qs:
            hit = query_is_hit(q, retained)
            if hit is None:
                continue
            num_solvable += 1
            num_hit += int(hit)
        out[key] = EpisodeScore(ep_id=episode.ep_id, num_solvable=num_solvable, num_hit=num_hit)
    return out
