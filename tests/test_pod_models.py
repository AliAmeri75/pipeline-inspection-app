"""Tests for selectable ILI probability-of-detection models."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import Simp_Sys_Thresh_J3_optimized as simulator
from Simp_Sys_Thresh_J3_optimized import (
    POD,
    POD_DEPTH_LENGTH_MODEL,
    POD_DEPTH_ONLY_MODEL,
)


class ProbabilityOfDetectionTests(unittest.TestCase):
    def test_depth_only_model_preserves_original_equation(self):
        depth = np.array([0.0, 0.5, 1.0, 2.0])
        expected = 1 - np.exp(-2.0 * depth)

        actual = POD(depth, 2.0)

        np.testing.assert_allclose(actual, expected)

    def test_depth_and_length_model_uses_both_dimensions(self):
        shallow_short = POD(
            0.5,
            1.0,
            length=10.0,
            pod_model=POD_DEPTH_LENGTH_MODEL,
            q_length=0.01,
        )
        deep_short = POD(
            1.0,
            1.0,
            length=10.0,
            pod_model=POD_DEPTH_LENGTH_MODEL,
            q_length=0.01,
        )
        shallow_long = POD(
            0.5,
            1.0,
            length=40.0,
            pod_model=POD_DEPTH_LENGTH_MODEL,
            q_length=0.01,
        )

        self.assertGreater(deep_short, shallow_short)
        self.assertGreater(shallow_long, shallow_short)

    def test_zero_length_coefficient_matches_depth_only_model(self):
        depth_only = POD(1.2, 0.8, pod_model=POD_DEPTH_ONLY_MODEL)
        depth_length = POD(
            1.2,
            0.8,
            length=35.0,
            pod_model=POD_DEPTH_LENGTH_MODEL,
            q_length=0.0,
        )

        self.assertAlmostEqual(depth_only, depth_length)

    def test_invalid_model_or_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown POD model"):
            POD(1.0, 1.0, pod_model="unsupported")
        with self.assertRaisesRegex(ValueError, "coefficients"):
            POD(1.0, -1.0)
        with self.assertRaisesRegex(ValueError, "length is required"):
            POD(1.0, 1.0, pod_model=POD_DEPTH_LENGTH_MODEL)

    def test_inspection_planning_passes_length_model_to_pod(self):
        joints = [
            {
                "name": "Test joint",
                "count": 1,
                "cracks": [{"mean_length": 30.0, "mean_depth": 0.8}],
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
            "inspection_interval": 1,
            "pod_model": POD_DEPTH_LENGTH_MODEL,
            "pod_depth_coefficient": 0.7,
            "pod_length_coefficient": 0.02,
            "resample_growth_each_step": False,
        }

        with patch.object(simulator, "POD", wraps=POD) as pod_spy:
            simulator.run_flexible_simulation(
                "equidistant",
                1,
                1,
                3,
                joints,
                costs,
                simulation,
                [0.0, 0.0],
                99.0,
                default_birth_rate=0.0,
                property_seed=4,
                track_debug=False,
            )

        length_model_calls = [
            call
            for call in pod_spy.call_args_list
            if call.kwargs.get("pod_model") == POD_DEPTH_LENGTH_MODEL
        ]
        self.assertTrue(length_model_calls)
        for call in length_model_calls:
            self.assertGreater(float(call.kwargs["length"]), 0.0)
            self.assertEqual(call.args[1], 0.7)
            self.assertEqual(call.kwargs["q_length"], 0.02)


if __name__ == "__main__":
    unittest.main()
