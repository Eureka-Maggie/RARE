"""Reward hook compatibility for the released rubric reward managers.

The managers in this repository compute rewards through OpenAI rubric judges
and do not call a dataset-specific rule-based scorer.  The hook remains because
the upstream trainer passes one to every reward-manager constructor.
"""


def default_compute_score(*_args, **_kwargs):
    raise NotImplementedError("this release supports rubric-based rewards only")


def get_default_compute_score(_reward_name: str | None):
    return default_compute_score


__all__ = ["default_compute_score", "get_default_compute_score"]
