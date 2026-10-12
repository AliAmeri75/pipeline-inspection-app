"""Pure helpers for comparing prior excavation and ILI-informed schedules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_optimal_result(
    results: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Return the lowest-cost safe result, or the lowest-cost result if none is safe."""

    if not results:
        raise ValueError("At least one scheduling result is required.")
    safe_results = [result for result in results if bool(result.get("is_safe"))]
    candidates = safe_results or list(results)
    return min(candidates, key=lambda result: float(result["total_cpp"])), bool(
        safe_results
    )


def calculate_value_of_information(
    prior_result: dict[str, Any],
    ili_result: dict[str, Any],
) -> dict[str, float | str | bool]:
    """Calculate net VoI using each method's independently optimized lifecycle cost.

    Positive VoI means the ILI-informed strategy has the lower expected cost. The
    ILI result's total already contains its discounted inspection costs.
    """

    prior_cost = float(prior_result["total_cpp"])
    ili_cost = float(ili_result["total_cpp"])
    value = prior_cost - ili_cost
    if value > 0:
        interpretation = "ILI-informed scheduling has the lower expected lifecycle cost."
    elif value < 0:
        interpretation = "Prior excavation scheduling has the lower expected lifecycle cost."
    else:
        interpretation = "The two optimized strategies have the same expected lifecycle cost."
    return {
        "prior_cost": prior_cost,
        "ili_cost": ili_cost,
        "voi": value,
        "ili_justified": value > 0,
        "interpretation": interpretation,
    }
