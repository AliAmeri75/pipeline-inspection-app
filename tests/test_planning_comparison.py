"""Regression tests for prior excavation and ILI-informed scheduling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from planning_analysis import (
    calculate_value_of_information,
    compare_ili_intervals_to_prior_safe,
    format_optional_decimal,
    select_lowest_cost_result,
    select_optimal_result,
)
from Simp_Sys_Thresh_J3_optimized import (
    DEFAULT_MAX_ANNUAL_BURST_POF,
    DEFAULT_MAX_ANNUAL_LEAK_POF,
    EXCAVATION_DECISION_MODE,
    ILI_DECISION_MODE,
    exceeds_safety_criteria,
    run_flexible_simulation,
    safety_limits,
)


class PlanningComparisonTests(unittest.TestCase):
    def test_cost_optimum_is_returned_even_when_it_is_unsafe(self):
        results = [
            {"interval": 5, "total_cpp": 1.0, "is_safe": False},
            {"interval": 10, "total_cpp": 1.4, "is_safe": True},
        ]
        optimum = select_lowest_cost_result(results)
        self.assertEqual(optimum["interval"], 5)
        self.assertFalse(optimum["is_safe"])

    def test_class_one_defaults_preserve_original_safety_limits(self):
        self.assertEqual(
            safety_limits({}),
            (DEFAULT_MAX_ANNUAL_LEAK_POF, DEFAULT_MAX_ANNUAL_BURST_POF),
        )
        self.assertFalse(exceeds_safety_criteria(1e-2, 1e-3, {}))
        self.assertTrue(exceeds_safety_criteria(1.01e-2, 1e-3, {}))
        self.assertTrue(exceeds_safety_criteria(1e-2, 1.01e-3, {}))

    def test_class_specific_safety_limits_are_used(self):
        criteria = {
            "max_annual_leak_pof": 2e-2,
            "max_annual_burst_pof": 2e-3,
        }
        self.assertFalse(exceeds_safety_criteria(1.5e-2, 1.5e-3, criteria))
        self.assertTrue(exceeds_safety_criteria(2.1e-2, 1.5e-3, criteria))

    def test_invalid_safety_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            safety_limits({"max_annual_leak_pof": 1.1})

    def test_optimization_prefers_safe_candidates(self):
        results = [
            {"interval": 5, "total_cpp": 1.0, "is_safe": False},
            {"interval": 10, "total_cpp": 1.4, "is_safe": True},
            {"interval": 15, "total_cpp": 1.8, "is_safe": True},
        ]
        optimal, has_safe = select_optimal_result(results)
        self.assertTrue(has_safe)
        self.assertEqual(optimal["interval"], 10)

    def test_optimization_does_not_recommend_an_unsafe_candidate(self):
        results = [
            {"interval": 5, "total_cpp": 1.0, "is_safe": False},
            {"interval": 10, "total_cpp": 0.8, "is_safe": False},
        ]
        optimal, has_safe = select_optimal_result(results)
        self.assertFalse(has_safe)
        self.assertIsNone(optimal)

    def test_voi_is_prior_cost_minus_ili_cost(self):
        comparison = calculate_value_of_information(
            {"total_cpp": 1.25}, {"total_cpp": 0.90}
        )
        self.assertAlmostEqual(comparison["voi"], 0.35)
        self.assertTrue(comparison["ili_justified"])

    def test_each_ili_interval_is_compared_with_one_safe_prior_baseline(self):
        prior_safe = {"interval": 15, "total_cpp": 0.25, "is_safe": True}
        ili_results = [
            {"interval": 5, "total_cpp": 0.30, "is_safe": True},
            {"interval": 10, "total_cpp": 0.20, "is_safe": True},
            {"interval": 20, "total_cpp": 0.15, "is_safe": False},
        ]

        rows = compare_ili_intervals_to_prior_safe(prior_safe, ili_results)

        self.assertEqual(
            [row["Prior safe interval (years)"] for row in rows],
            [15, 15, 15],
        )
        self.assertEqual(
            [row["ILI economically justified?"] for row in rows],
            ["No", "Yes", "No"],
        )
        self.assertAlmostEqual(rows[1]["Cost difference / VoI ($M)"], 0.05)

    def test_ili_economic_comparison_is_unavailable_without_safe_prior(self):
        rows = compare_ili_intervals_to_prior_safe(
            None,
            [{"interval": 10, "total_cpp": 0.20, "is_safe": True}],
        )

        self.assertEqual(rows[0]["Prior safe interval (years)"], "Not found")
        self.assertIsNone(rows[0]["Prior safe cost ($M)"])
        self.assertIsNone(rows[0]["Cost difference / VoI ($M)"])
        self.assertEqual(rows[0]["ILI economically justified?"], "N/A")

        table = pd.DataFrame(rows)
        rendered = table.style.format(
            {
                "Prior safe cost ($M)": format_optional_decimal,
                "ILI cost ($M)": "{:.6f}",
                "Cost difference / VoI ($M)": format_optional_decimal,
            }
        ).to_html()
        self.assertIn("Not found", rendered)

    def test_prior_excavation_does_not_use_pod_or_charge_ili_cost(self):
        joints = [
            {
                "name": "Test joint",
                "count": 1,
                "cracks": [{"mean_length": 40.0, "mean_depth": 1.0}],
                "properties": {},
                "birth_rate": 0.0,
            }
        ]
        costs = {
            "Cr1": 0.2,
            "Cr2": 0.05,
            "Cf_leak": 2.0,
            "Cf_burst": 20.0,
            "inspection_cost_per_km": 0.015,
            "pipe_length_km": 1.0,
            "rate": 0.04,
        }
        simulation = {
            "dS": 2.0,
            "m": 3.0,
            "Di": 20 * 25.4,
            "FPR_rep1": 1.25,
            "D_rep1": 3.175,
            "FPR_rep2": 1.10,
            "D_rep2": 4.445,
            "lifetime_arr": [0, 5, 5],
            "df_arr": [1, 0, 0],
            "threshold": 1e-3,
            "inspection_interval": 2,
            "resample_growth_each_step": False,
        }
        common = (
            "equidistant",
            5,
            1,
            30,
            joints,
            costs,
            simulation,
            [0.74, 15.6],
            0.0,
        )

        np.random.seed(0)
        ili = run_flexible_simulation(
            *common,
            default_birth_rate=0.0,
            property_seed=0,
            track_debug=False,
            decision_mode=ILI_DECISION_MODE,
        )
        np.random.seed(0)
        prior = run_flexible_simulation(
            *common,
            default_birth_rate=0.0,
            property_seed=0,
            track_debug=False,
            decision_mode=EXCAVATION_DECISION_MODE,
        )

        self.assertEqual(float(np.sum(ili[14])), 0.0)
        self.assertGreater(float(np.sum(ili[9])), 0.0)
        self.assertGreater(float(np.sum(prior[14])), 0.0)
        self.assertEqual(float(np.sum(prior[9])), 0.0)


if __name__ == "__main__":
    unittest.main()
