from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapDifferenceCI:
    mean_a: float
    mean_b: float
    mean_difference: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_items: int

    @property
    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper


def paired_bootstrap_difference_ci(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> BootstrapDifferenceCI:

    if not scores_a or not scores_b:
        raise ValueError(
            "Paired score mappings must not be empty."
        )

    ids_a = set(scores_a)
    ids_b = set(scores_b)

    if ids_a != ids_b:
        raise ValueError(
            "Paired samples must contain exactly "
            "the same item IDs."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if n_resamples <= 0:
        raise ValueError(
            "n_resamples must be greater than 0."
        )

    item_ids = sorted(ids_a)

    values_a = np.asarray(
        [
            scores_a[item_id]
            for item_id in item_ids
        ],
        dtype=np.float64,
    )

    values_b = np.asarray(
        [
            scores_b[item_id]
            for item_id in item_ids
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values_a)):
        raise ValueError(
            "scores_a contains non-finite values."
        )

    if not np.all(np.isfinite(values_b)):
        raise ValueError(
            "scores_b contains non-finite values."
        )

    differences = values_b - values_a

    rng = np.random.default_rng(seed)

    n_items = len(differences)

    bootstrap_means = np.empty(
        n_resamples,
        dtype=np.float64,
    )

    for index in range(n_resamples):
        sample_indices = rng.integers(
            low=0,
            high=n_items,
            size=n_items,
        )

        bootstrap_means[index] = np.mean(
            differences[sample_indices]
        )

    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return BootstrapDifferenceCI(
        mean_a=float(np.mean(values_a)),
        mean_b=float(np.mean(values_b)),
        mean_difference=float(np.mean(differences)),
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_items=n_items,
    )