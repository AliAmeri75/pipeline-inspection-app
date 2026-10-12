"""Tests for the one-sample-at-a-time Monte Carlo driver."""

from __future__ import annotations

import unittest

import numpy as np

from Simp_Sys_Thresh_J3_streaming import run_flexible_simulation


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
        "total_insp_cost": 0.015,
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
        "threshold": 1.0,
        "inspection_interval": 2,
        "resample_growth_each_step": False,
    }
    return joints, costs, simulation


class StreamingSimulationTests(unittest.TestCase):
    def test_accumulates_outputs_without_returning_sample_histories(self):
        joints, costs, simulation = _inputs()
        result = run_flexible_simulation(
            "equidistant",
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

        cpp, cff, crr, cii = result[6:10]
        action_probabilities = result[15]
        debug = result[16]

        np.testing.assert_allclose(cpp, cff + crr + cii)
        self.assertEqual(debug["joint_cpp"].shape, (2, 3))
        self.assertEqual(debug["joint_leak_pofs"].shape, (2, 3))
        self.assertEqual(debug["joint_action_probabilities"].shape, (2, 3, 3))
        self.assertEqual(len(action_probabilities), 2)
        for probabilities in action_probabilities:
            self.assertEqual(probabilities.shape, (3, 3))
            np.testing.assert_allclose(probabilities.sum(axis=0), 1.0)
        self.assertEqual(
            debug["retained_samples"] + debug["rejected_initial_failures"], 4
        )
        self.assertIsNone(debug["crack_depths"])
        self.assertIn("one sample at a time", debug["storage_mode"])

    def test_threshold_mode_uses_repeatable_streaming_passes(self):
        joints, costs, simulation = _inputs()
        simulation["threshold"] = -1.0
        result = run_flexible_simulation(
            "threshold",
            1,
            1,
            2,
            joints[:1],
            costs,
            simulation,
            [0.74, 15.6],
            1.0,
            property_seed=3,
            simulation_seed=5,
            track_debug=False,
        )

        self.assertEqual(result[16]["streaming_passes"], 2)
        np.testing.assert_allclose(result[5], [0.0])


if __name__ == "__main__":
    unittest.main()
