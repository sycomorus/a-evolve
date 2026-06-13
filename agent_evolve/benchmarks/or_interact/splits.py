"""Shared OR-Interact split configuration helpers."""

from __future__ import annotations


DEFAULT_TRAIN_SIZE = 50
DEFAULT_TEST_LIMIT = 50


def train_size_from_limit(limit_train: int | None) -> int:
    return limit_train if limit_train is not None else DEFAULT_TRAIN_SIZE


def evaluation_limit_for_split(
    split: str,
    *,
    limit_train: int | None,
    limit_test: int | None,
    legacy_limit: int | None = None,
) -> int:
    split_key = split.lower()
    if split_key == "train":
        limit = limit_train
    elif split_key in {"holdout", "test"}:
        limit = limit_test
    else:
        raise ValueError(f"unknown split {split!r}; expected train, holdout, or test")

    if limit is not None:
        return limit
    if legacy_limit is not None:
        return legacy_limit
    return DEFAULT_TEST_LIMIT
