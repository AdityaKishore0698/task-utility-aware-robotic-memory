"""Load and parse the FindingDory metadata parquet into a clean, typed structure.

Schema confirmed by direct inspection of the validation parquet on 2026-10-14
(5,870 rows / 100 episodes / 60 task_ids):

    ep_id                  str   e.g. "ep_1"
    video                  str   relative path inside videos.zip, e.g. "videos/val/ep_1.mp4"
    question               str   natural-language query
    answer                 str   repr() of a list-of-lists of frame indices, one sublist
                                 per subgoal (always nested, even for single-goal tasks);
                                 [[-1]] marks an unsolvable query
    task_id                str   e.g. "task_26" (60 distinct task definitions)
    high_level_category    str   "Single-Goal Spatial Tasks" | "Single-Goal Temporal Tasks"
                                  | "Multi-Goal Tasks"
    low_level_category     str   finer category, e.g. "Room Visitation", "Interaction Order"
    num_interactions       int   number of pick/place interactions sampled in the episode

Caveats (do not overclaim beyond this):
  - Episode length (frame count) is NOT given directly anywhere in the parquet or in
    episode_id_mappings.json. We approximate it as (max frame index referenced by any
    answer in that episode) + 1. This likely slightly UNDER-counts the true video length
    if the tail of the video is never an oracle answer for any of its ~60 queries; we have
    not verified this against the actual video files. Treat `episode_length` as a lower
    bound, not a verified frame count.
  - "Answer solves the query" here means: the query's oracle frame-index set(s) as given
    in the `answer` column. Multi-goal queries carry one sublist per subgoal; we do not
    enforce the *order* subgoals must be visited in (relevant to the "Ordered Revisitation"
    category) -- this loader exposes the raw per-subgoal sets and leaves ordering logic to
    the evaluator, which currently ignores it (see src/eval/selection_hit.py docstring).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "findingdory"


@dataclass(frozen=True)
class Query:
    task_id: str
    question: str
    high_level_category: str
    low_level_category: str
    num_interactions: int
    # One frozenset of acceptable frame indices per subgoal, in the order given by the
    # dataset. A single-goal query has exactly one subgoal. Empty tuple => unsolvable
    # ([[-1]] in the raw data).
    subgoal_frame_sets: tuple[frozenset[int], ...]

    @property
    def is_unsolvable(self) -> bool:
        return len(self.subgoal_frame_sets) == 0


@dataclass(frozen=True)
class Episode:
    ep_id: str
    video_path: str
    # Approximate episode length; see module docstring caveat.
    episode_length: int
    queries: tuple[Query, ...] = field(default_factory=tuple)


def _parse_answer(raw: str) -> tuple[frozenset[int], ...]:
    """Parse the `answer` column into a tuple of frame-index sets, one per subgoal.

    [[-1]] (unsolvable) parses to an empty tuple.
    """
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if parsed == [[-1]] or parsed == [-1]:
        return ()
    # Defensive: handle a stray non-nested list (not observed, but keep the loader
    # correct if the dataset schema varies between splits).
    if parsed and not isinstance(parsed[0], list):
        parsed = [parsed]
    return tuple(frozenset(sub) for sub in parsed)


def load_episodes(split: str = "validation") -> list[Episode]:
    """Load one FindingDory split into a list of Episode objects.

    Requires the metadata parquet to already be present under data/findingdory/
    (see src/scripts/download_findingdory.py).
    """
    fname = {
        "validation": "validation-00000-of-00001.parquet",
        "train": "train-00000-of-00001.parquet",
    }.get(split)
    if fname is None:
        raise ValueError(f"unknown split {split!r}; expected 'validation' or 'train'")

    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python3 src/scripts/download_findingdory.py"
            + (" --train" if split == "train" else "")
        )

    df = pd.read_parquet(path)
    df["subgoal_frame_sets"] = df["answer"].apply(_parse_answer)

    episodes: list[Episode] = []
    for ep_id, group in df.groupby("ep_id", sort=False):
        queries = tuple(
            Query(
                task_id=row.task_id,
                question=row.question,
                high_level_category=row.high_level_category,
                low_level_category=row.low_level_category,
                num_interactions=int(row.num_interactions),
                subgoal_frame_sets=row.subgoal_frame_sets,
            )
            for row in group.itertuples(index=False)
        )
        max_idx = -1
        for q in queries:
            for s in q.subgoal_frame_sets:
                if s:
                    max_idx = max(max_idx, max(s))
        episodes.append(
            Episode(
                ep_id=ep_id,
                video_path=group["video"].iloc[0],
                episode_length=max_idx + 1,
                queries=queries,
            )
        )
    # groupby does not guarantee a stable, human-friendly order across pandas versions
    episodes.sort(key=lambda e: e.ep_id)
    return episodes


if __name__ == "__main__":
    eps = load_episodes("validation")
    print(f"loaded {len(eps)} episodes, {sum(len(e.queries) for e in eps)} queries total")
    print(f"episode_length stats: min={min(e.episode_length for e in eps)} "
          f"max={max(e.episode_length for e in eps)}")
