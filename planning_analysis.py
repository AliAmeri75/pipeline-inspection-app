"""Pure helpers for comparing prior excavation and ILI-informed schedules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_lowest_cost_result(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return the lowest-cost tested interval, regardless of safety status."""

    if not results:
        raise ValueError("At least one scheduling result is required.")
    return min(results, key=lambda result: float(result["total_cpp"]))


def select_optimal_result(
    results: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    """Return the lowest-cost result that satisfies the safety criteria.

    An unsafe schedule is never returned as an optimum.  ``None`` tells the UI
    that the tested interval range contains no valid recommendation.
    """

    if not results:
        raise ValueError("At least one scheduling result is required.")
    safe_results = [result for result in results if bool(result.get("is_safe"))]
    if not safe_results:
        return None, False
    return min(safe_results, key=lambda result: float(result["total_cpp"])), True


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


def compare_ili_intervals_to_prior_safe(
    prior_safe_result: dict[str, Any] | None,
    ili_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare each ILI candidate with one minimum-cost safe prior baseline.

    An ILI interval is economically justified only when it satisfies the safety
    criteria and costs less than the safe prior excavation alternative.  If no
    safe prior solution exists, a valid economic comparison cannot be made.
    """

    rows: list[dict[str, Any]] = []
    for ili_result in sorted(
        ili_results, key=lambda result: float(result["interval"])
    ):
        ili_is_safe = bool(ili_result.get("is_safe"))
        ili_cost = float(ili_result["total_cpp"])

        if prior_safe_result is None:
            prior_interval: int | float | str = "Not found"
            prior_cost = None
            value = None
            economically_justified = "N/A"
        else:
            prior_interval = prior_safe_result["interval"]
            prior_cost = float(prior_safe_result["total_cpp"])
            value = prior_cost - ili_cost
            economically_justified = "Yes" if ili_is_safe and value > 0 else "No"

        rows.append(
            {
                "Prior safe interval (years)": prior_interval,
                "Prior safe cost ($M)": prior_cost,
                "ILI interval (years)": ili_result["interval"],
                "ILI cost ($M)": ili_cost,
                "ILI safe": "Yes" if ili_is_safe else "No",
                "ILI economically justified?": economically_justified,
                "Cost difference / VoI ($M)": value,
            }
        )
    return rows
