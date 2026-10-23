"""Regression tests for the readable equidistant-only simulation."""

from __future__ import annotations

import unittest

import numpy as np

from Simp_Sys_Equidistant_streaming import (
    run_flexible_equidistant_simulation,
    run_flexible_simulation,
)


def _inputs():
    joints = [
        {
            "name": "Cracked joint",
            "count": 1,
            "cracks": [{"mean_length": 30.0, "mean_depth": 0.8}],
            "properties": {},
            "birth_rate": 0.0,
        },
        {
            "name": "Crack-free group",
            "count": 3,
            "cracks": [],
            "properties": {},
            "birth_rate": 0.0,
        },
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
        "inspection_interval": 2,
        "resample_growth_each_step": False,
    }
    return joints, costs, simulation


class EquidistantStreamingTests(unittest.TestCase):
    def test_direct_api_returns_aggregate_results(self):
        joints, costs, simulation = _inputs()
        result = run_flexible_equidistant_simulation(
            2,
            1,
            4,
            joints,
            costs,
            simulation,
            [0.74, 15.6],
            1.0,
            property_seed=7,
            simulation_seed=17,
            track_debug=False,
        )

        np.testing.assert_allclose(result[6], result[7] + result[8] + result[9])
        self.assertEqual(result[16]["scheduling_mode"], "equidistant")
        self.assertEqual(result[16]["streaming_passes"], 1)
        self.assertEqual(result[16]["joint_cpp"].shape, (2, 3))
        self.assertIn("one realization at a time", result[16]["storage_mode"])

    def test_threshold_mode_is_explicitly_rejected(self):
        joints, costs, simulation = _inputs()
        with self.assertRaisesRegex(ValueError, "only approach='equidistant'"):
            run_flexible_simulation(
                "threshold",
                2,
                1,
                4,
                joints,
                costs,
                simulation,
                [0.74, 15.6],
                1.0,
            )

    def test_inspection_cost_is_charged_per_km_for_each_ili(self):
        joints, costs, simulation = _inputs()
        costs["inspection_cost_per_km"] = 0.01
        costs["pipe_length_km"] = 10.0
        costs["rate"] = 0.0
        result = run_flexible_equidistant_simulation(
            2,
            1,
            4,
            joints,
            costs,
            simulation,
            [0.74, 15.6],
            1.0,
            property_seed=7,
            simulation_seed=17,
            track_debug=False,
        )

        self.assertAlmostEqual(float(np.sum(result[9])), 0.10)

    def test_inspection_times_must_be_on_the_simulation_grid(self):
        joints, costs, simulation = _inputs()
        simulation["inspection_times"] = [0.5]
        with self.assertRaisesRegex(ValueError, "coincide with a simulation time"):
            run_flexible_equidistant_simulation(
                2,
                1,
                4,
                joints,
                costs,
                simulation,
                [0.74, 15.6],
                1.0,
                track_debug=False,
            )


if __name__ == "__main__":
    unittest.main()
