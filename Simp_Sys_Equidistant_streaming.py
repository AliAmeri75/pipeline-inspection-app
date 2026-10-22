#!/usr/bin/env python3
"""Readable, low-memory simulation for equidistant inspection planning.

Scope
-----
This file intentionally supports only a fixed/equidistant inspection schedule.
It does not contain threshold scheduling or rollback logic.

Memory strategy
---------------
The simulation processes one Monte Carlo realization at a time:

1. sample the pipe properties and initial cracks;
2. simulate that realization through the complete time horizon;
3. add its costs, failures, repairs, and actions to running totals;
4. discard its crack states before starting the next realization.

Consequently, the main state arrays do not grow with the Monte Carlo sample
count.  Only joint-by-time summary arrays are retained.  One optional sample
history may be retained for debugging and crack-growth plots.

Model equations
---------------
The crack-growth, failure, detection, and repair equations are reused from
``Simp_Sys_Thresh_J3_optimized.py``.  Keeping the physical model in one place
avoids introducing a second, slightly different implementation of the same
equations.

Public functions
----------------
``run_equidistant_simulation``
    Legacy grouped-joint input interface, without an ``approach`` argument.

``run_flexible_equidistant_simulation``
    Clearer interface for individually described joints or joint groups.

``run_simulation`` and ``run_flexible_simulation``
    Compatibility wrappers.  They accept the old leading ``approach`` argument
    but require it to be ``"equidistant"``.

All four functions return the same 17-item tuple used by the optimized model.
The action item contains probabilities with shape ``(3, number_of_times)`` for
each joint/group, rather than a raw action history for every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import Simp_Sys_Thresh_J3_optimized as model


# Re-export commonly used model definitions so this module can replace the
# original import in client code when desired.
ILI_DECISION_MODE = model.ILI_DECISION_MODE
EXCAVATION_DECISION_MODE = model.EXCAVATION_DECISION_MODE
DECISION_MODES = model.DECISION_MODES
PROPERTY_NAMES = model.PROPERTY_NAMES
DEFAULT_PROPERTY_SPECS = model.DEFAULT_PROPERTY_SPECS
MEAN_WT = model.MEAN_WT


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class InitialFailure(RuntimeError):
    """Internal signal: a sampled system has already failed at time zero."""


@dataclass
class JointLayout:
    """Joint/group information that is unchanged across Monte Carlo samples."""

    uses_joint_definitions: bool
    joint_definitions: list[dict[str, Any]]
    names: list[str]
    counts: np.ndarray
    initial_crack_counts: list[int]
    initial_crack_specs: list[list[dict[str, float]] | None]
    birth_rates: np.ndarray
    supplied_property_samples: list[dict[str, Any]]

    @property
    def number_of_groups(self) -> int:
        return len(self.names)


@dataclass
class JointState:
    """Changing crack state for one joint/group in one realization."""

    lengths: np.ndarray
    depths: np.ndarray
    growth_parameters: np.ndarray
    repair_1_ages: np.ndarray
    repair_2_ages: np.ndarray
    qsf: np.ndarray
    fsf: np.ndarray
    magnification: np.ndarray
    birth_times: list[list[float]]
    birth_pointer: list[int]
    next_unused_slot: list[int]
    active_cracks: list[list[int]]
    reusable_failed_slots: list[list[int]]
    actions: np.ndarray


@dataclass
class SampleResult:
    """Outputs from one retained Monte Carlo realization."""

    total_cost: np.ndarray
    failure_cost: np.ndarray
    repair_cost: np.ndarray
    inspection_cost: np.ndarray
    cracked_joint_cost: np.ndarray
    crack_free_joint_cost: np.ndarray
    expected_repairs: np.ndarray
    joint_total_cost: np.ndarray
    joint_failure_cost: np.ndarray
    joint_repair_cost: np.ndarray
    joint_expected_repairs: np.ndarray
    joint_cumulative_leaks: np.ndarray
    joint_cumulative_bursts: np.ndarray
    actions: list[np.ndarray]
    debug: dict[str, Any]


@dataclass
class RunningTotals:
    """Sufficient statistics; no individual sample histories are stored."""

    total_cost: np.ndarray
    failure_cost: np.ndarray
    repair_cost: np.ndarray
    inspection_cost: np.ndarray
    cracked_joint_cost: np.ndarray
    crack_free_joint_cost: np.ndarray
    expected_repairs: np.ndarray
    joint_total_cost: np.ndarray
    joint_failure_cost: np.ndarray
    joint_repair_cost: np.ndarray
    joint_expected_repairs: np.ndarray
    joint_cumulative_leaks: np.ndarray
    joint_cumulative_bursts: np.ndarray
    action_counts: np.ndarray
    retained_samples: int = 0
    rejected_initial_failures: int = 0

    @classmethod
    def empty(cls, number_of_groups: int, number_of_times: int) -> "RunningTotals":
        time_vector = np.zeros(number_of_times)
        joint_time = np.zeros((number_of_groups, number_of_times))
        return cls(
            total_cost=time_vector.copy(),
            failure_cost=time_vector.copy(),
            repair_cost=time_vector.copy(),
            inspection_cost=time_vector.copy(),
            cracked_joint_cost=time_vector.copy(),
            crack_free_joint_cost=time_vector.copy(),
            expected_repairs=time_vector.copy(),
            joint_total_cost=joint_time.copy(),
            joint_failure_cost=joint_time.copy(),
            joint_repair_cost=joint_time.copy(),
            joint_expected_repairs=joint_time.copy(),
            joint_cumulative_leaks=joint_time.copy(),
            joint_cumulative_bursts=joint_time.copy(),
            action_counts=np.zeros(
                (number_of_groups, 3, number_of_times), dtype=np.int64
            ),
        )

    def add(self, sample: SampleResult) -> None:
        """Add one realization to the running sums."""

        self.total_cost += sample.total_cost
        self.failure_cost += sample.failure_cost
        self.repair_cost += sample.repair_cost
        self.inspection_cost += sample.inspection_cost
        self.cracked_joint_cost += sample.cracked_joint_cost
        self.crack_free_joint_cost += sample.crack_free_joint_cost
        self.expected_repairs += sample.expected_repairs
        self.joint_total_cost += sample.joint_total_cost
        self.joint_failure_cost += sample.joint_failure_cost
        self.joint_repair_cost += sample.joint_repair_cost
        self.joint_expected_repairs += sample.joint_expected_repairs
        self.joint_cumulative_leaks += sample.joint_cumulative_leaks
        self.joint_cumulative_bursts += sample.joint_cumulative_bursts

        for joint_index, action_history in enumerate(sample.actions):
            for action_code in (0, 1, 2):
                self.action_counts[joint_index, action_code] += (
                    action_history[0] == action_code
                )

        self.retained_samples += 1


# ---------------------------------------------------------------------------
# Validation and fixed input preparation
# ---------------------------------------------------------------------------


def _validate_general_inputs(
    horizon: float,
    time_step: float,
    number_of_samples: int,
    costs: dict[str, float],
    decision_mode: str,
    debug_sample_index: int,
) -> None:
    if horizon < 0:
        raise ValueError("horizon/T must be non-negative.")
    if time_step <= 0:
        raise ValueError("time_step/dt must be positive.")
    if number_of_samples < 1 or int(number_of_samples) != number_of_samples:
        raise ValueError("number_of_samples/nx must be a positive integer.")
    if costs["pipe_length_km"] <= 0:
        raise ValueError("pipe_length_km must be positive.")
    if decision_mode not in DECISION_MODES:
        raise ValueError(f"decision_mode must be one of {sorted(DECISION_MODES)}.")
    if debug_sample_index < 0:
        raise ValueError("debug_sample_idx cannot be negative.")


def _prepare_layout(
    num_cracks_list,
    joint_counts,
    pipe_props_list,
    joint_definitions,
    default_birth_rate,
) -> JointLayout:
    """Normalize either supported input style into one joint layout."""

    if joint_definitions is not None:
        joints = [dict(joint) for joint in joint_definitions]
        if not joints:
            raise ValueError("joint_definitions must contain at least one joint.")

        names = [
            joint.get("name", f"Joint {index + 1}")
            for index, joint in enumerate(joints)
        ]
        if len(set(names)) != len(names):
            raise ValueError("Every explicitly modelled joint must have a unique name.")

        crack_specs = [list(joint.get("cracks", [])) for joint in joints]
        crack_counts = [len(cracks) for cracks in crack_specs]
        counts = [joint.get("count", 1) for joint in joints]
        if any(int(count) != count or count < 1 for count in counts):
            raise ValueError("Each joint/group count must be a positive integer.")

        birth_rates = np.asarray(
            [joint.get("birth_rate", default_birth_rate) for joint in joints],
            dtype=float,
        )
        if np.any(birth_rates < 0):
            raise ValueError("Joint birth rates cannot be negative.")

        return JointLayout(
            uses_joint_definitions=True,
            joint_definitions=joints,
            names=names,
            counts=np.asarray(counts, dtype=int),
            initial_crack_counts=crack_counts,
            initial_crack_specs=crack_specs,
            birth_rates=birth_rates,
            supplied_property_samples=[],
        )

    # Legacy input: one statistically representative group for each initial
    # crack count, with the number of physical joints given by joint_counts.
    original_groups = list(num_cracks_list)
    properties_by_group = dict(zip(original_groups, pipe_props_list))
    active_groups = [
        cracks for cracks in original_groups if joint_counts.get(cracks, 0) > 0
    ]
    if not active_groups:
        raise ValueError("joint_counts must contain at least one positive count.")

    missing_properties = [
        cracks for cracks in active_groups if cracks not in properties_by_group
    ]
    if missing_properties:
        raise ValueError(
            f"Missing pipe properties for crack groups: {missing_properties}"
        )

    return JointLayout(
        uses_joint_definitions=False,
        joint_definitions=[],
        names=[f"{cracks}-crack group" for cracks in active_groups],
        counts=np.asarray(
            [joint_counts[cracks] for cracks in active_groups], dtype=int
        ),
        initial_crack_counts=active_groups,
        initial_crack_specs=[None] * len(active_groups),
        birth_rates=np.full(len(active_groups), default_birth_rate, dtype=float),
        supplied_property_samples=[properties_by_group[c] for c in active_groups],
    )


def _make_inspection_schedule(
    horizon: float,
    time_step: float,
    simulation_parameters: dict[str, Any],
) -> np.ndarray:
    """Build and validate the fixed inspection schedule."""

    schedule = model._schedule_from_parameters(
        "equidistant", horizon, time_step, simulation_parameters
    )
    simulation_times = np.arange(0, horizon + time_step, time_step)
    off_grid = [
        inspection_time
        for inspection_time in schedule
        if not np.any(np.isclose(simulation_times, inspection_time))
    ]
    if off_grid:
        raise ValueError(
            "Every inspection time must coincide with a simulation time. "
            f"Off-grid values: {off_grid}"
        )
    return schedule


def _select_debug_group(
    layout: JointLayout,
    debug_joint,
    debug_crack_count,
) -> int:
    if debug_joint is not None:
        index = (
            layout.names.index(debug_joint)
            if isinstance(debug_joint, str)
            else int(debug_joint)
        )
        if not 0 <= index < layout.number_of_groups:
            raise ValueError("debug_joint is outside the available joint range.")
        return index

    if debug_crack_count is not None:
        return layout.initial_crack_counts.index(debug_crack_count)

    return next(
        (
            index
            for index, cracks in enumerate(layout.initial_crack_counts)
            if cracks > 0
        ),
        0,
    )


# ---------------------------------------------------------------------------
# Reproducible sampling for one realization
# ---------------------------------------------------------------------------


def _derived_seed(base_seed: int, sample_index: int, stage: int) -> int:
    """Return a repeatable, separate seed for each sample and sampling stage."""

    modulus = np.iinfo(np.uint32).max
    return int(
        (int(base_seed) + sample_index * 1_000_003 + stage * 10_007) % modulus
    )


def _one_supplied_value(values, sample_index: int) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return float(values[0])
    if sample_index >= values.size:
        raise ValueError(
            "A supplied pipe-property population has fewer values than the "
            "requested number of samples."
        )
    return float(values[sample_index])


def _sample_pipe_properties_for_one_realization(
    layout: JointLayout,
    sample_index: int,
    property_seed: int,
) -> list[dict[str, Any]]:
    if layout.uses_joint_definitions:
        return model.sample_individual_joint_properties(
            layout.joint_definitions,
            1,
            seed=_derived_seed(property_seed, sample_index, stage=1),
        )

    properties_for_sample = []
    for crack_count, supplied in zip(
        layout.initial_crack_counts, layout.supplied_property_samples
    ):
        joint_properties = {
            name: np.asarray([_one_supplied_value(supplied[name], sample_index)])
            for name in PROPERTY_NAMES
        }
        joint_properties["num_initial_cracks"] = crack_count
        properties_for_sample.append(joint_properties)
    return properties_for_sample


def _sample_birth_times(
    layout: JointLayout,
    horizon: float,
    sample_index: int,
    simulation_seed: int,
) -> list[list[list[float]]]:
    np.random.seed(_derived_seed(simulation_seed, sample_index, stage=2))
    return [
        [
            model.generate_initiation_times(
                model.generate_num_cracks(horizon, birth_rate),
                horizon,
                birth_rate,
            )
        ]
        for birth_rate in layout.birth_rates
    ]


def _sample_initial_cracks(
    layout: JointLayout,
    sample_index: int,
    simulation_seed: int,
    crack_length_std: float,
    crack_depth_std: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Sample initial cracks and also build the survival-check matrix."""

    total_cracks = sum(layout.initial_crack_counts)
    survival_matrix = np.zeros((1, 2 * total_cracks))
    lengths_by_group: list[np.ndarray] = []
    depths_by_group: list[np.ndarray] = []
    column = 0

    for group_index, crack_count in enumerate(layout.initial_crack_counts):
        if crack_count:
            crack_input = (
                layout.initial_crack_specs[group_index]
                if layout.uses_joint_definitions
                else crack_count
            )
            lengths, depths = model.sample_initial_cracks(
                crack_input,
                1,
                seed=_derived_seed(
                    simulation_seed, sample_index, stage=10 + group_index
                ),
                length_std=crack_length_std,
                depth_std=crack_depth_std,
            )
            lengths = lengths[:, 0].copy()
            depths = depths[:, 0].copy()
        else:
            lengths = np.zeros(0)
            depths = np.zeros(0)

        lengths_by_group.append(lengths)
        depths_by_group.append(depths)
        survival_matrix[0, column:column + crack_count] = lengths
        depth_column = total_cracks + column
        survival_matrix[0, depth_column:depth_column + crack_count] = depths
        column += crack_count

    return lengths_by_group, depths_by_group, survival_matrix


# ---------------------------------------------------------------------------
# State initialization and one-realization simulation
# ---------------------------------------------------------------------------


def _initialize_joint_states(
    layout: JointLayout,
    properties: list[dict[str, Any]],
    initial_lengths: list[np.ndarray],
    initial_depths: list[np.ndarray],
    birth_times: list[list[list[float]]],
    horizon: float,
    number_of_times: int,
) -> list[JointState]:
    """Allocate state for one realization only."""

    # This preserves the original spare-slot rule for cracks born later.
    expected_births = model.exp_num(horizon, float(np.max(layout.birth_rates)))
    spare_slots = int(np.ceil(expected_births)) + 40
    states = []

    for group_index, initial_crack_count in enumerate(layout.initial_crack_counts):
        number_of_slots = initial_crack_count + spare_slots
        lengths = np.zeros((number_of_slots, 1))
        depths = np.zeros_like(lengths)
        lengths[:initial_crack_count, 0] = initial_lengths[group_index]
        depths[:initial_crack_count, 0] = initial_depths[group_index]

        states.append(
            JointState(
                lengths=lengths,
                depths=depths,
                growth_parameters=np.full_like(
                    lengths, properties[group_index]["alpha"]
                ),
                repair_1_ages=np.zeros(1),
                repair_2_ages=np.zeros(1),
                qsf=np.zeros_like(lengths),
                fsf=np.zeros_like(lengths),
                magnification=np.zeros_like(lengths),
                birth_times=birth_times[group_index],
                birth_pointer=[0],
                next_unused_slot=[initial_crack_count],
                active_cracks=[list(range(initial_crack_count))],
                reusable_failed_slots=[[]],
                actions=np.zeros((1, number_of_times), dtype=np.int8),
            )
        )

    return states


def _apply_one_joint_step(
    *,
    state: JointState,
    time_value: float,
    time_index: int,
    properties: dict[str, Any],
    costs: dict[str, float],
    simulation_parameters: dict[str, Any],
    measurement_errors,
    q_pod: float,
    inspection_times: np.ndarray,
    birth_rate: float,
    decision_mode: str,
) -> tuple[float, float, float, int, int, float]:
    """Advance one group by one step and return its six summary values."""

    output = model.Joint_planning_step(
        time_value,
        time_index,
        state.lengths,
        state.depths,
        state.growth_parameters,
        state.repair_1_ages,
        state.repair_2_ages,
        state.birth_pointer,
        state.next_unused_slot,
        state.birth_times,
        state.active_cracks,
        properties,
        costs,
        simulation_parameters,
        measurement_errors,
        q_pod,
        state.qsf,
        state.fsf,
        state.magnification,
        inspection_times,
        state.reusable_failed_slots,
        birth_rate,
        state.actions,
        decision_mode,
    )

    # Joint_planning_step mutates these arrays/lists and also returns them.
    # Assigning the returned references keeps this function compatible if the
    # physical model later changes an in-place operation to a replacement.
    (
        total_cost,
        failure_cost,
        repair_cost,
        _,
        leaks,
        bursts,
        state.lengths,
        state.depths,
        state.growth_parameters,
        state.repair_1_ages,
        state.repair_2_ages,
        state.birth_pointer,
        state.next_unused_slot,
        state.birth_times,
        state.active_cracks,
        state.qsf,
        state.fsf,
        state.magnification,
        state.reusable_failed_slots,
        expected_repairs,
        state.actions,
    ) = output

    return total_cost, failure_cost, repair_cost, leaks, bursts, expected_repairs


def _simulate_one_realization(
    *,
    sample_index: int,
    layout: JointLayout,
    horizon: float,
    time_step: float,
    costs: dict[str, float],
    simulation_parameters: dict[str, Any],
    measurement_errors,
    q_pod: float,
    inspection_times: np.ndarray,
    property_seed: int,
    simulation_seed: int,
    crack_length_std: float,
    crack_depth_std: float,
    capture_debug: bool,
    debug_group: int,
    decision_mode: str,
) -> SampleResult:
    """Simulate one realization from year zero to the horizon."""

    times = np.arange(0, horizon + time_step, time_step)
    number_of_times = len(times)
    number_of_groups = layout.number_of_groups

    properties = _sample_pipe_properties_for_one_realization(
        layout, sample_index, property_seed
    )
    initial_lengths, initial_depths, survival_matrix = _sample_initial_cracks(
        layout,
        sample_index,
        simulation_seed,
        crack_length_std,
        crack_depth_std,
    )
    if not model.initial_system_survival_mask(
        survival_matrix, properties, layout.initial_crack_counts
    )[0]:
        raise InitialFailure

    birth_times = _sample_birth_times(
        layout, horizon, sample_index, simulation_seed
    )

    # A separate random stream is used for crack births, growth, observations,
    # and repairs after initialization.
    np.random.seed(_derived_seed(simulation_seed, sample_index, stage=100))
    states = _initialize_joint_states(
        layout,
        properties,
        initial_lengths,
        initial_depths,
        birth_times,
        horizon,
        number_of_times,
    )

    local_simulation_parameters = dict(simulation_parameters)
    local_simulation_parameters.update({"T": horizon, "dt": time_step})

    total_cost = np.zeros(number_of_times)
    failure_cost = np.zeros(number_of_times)
    repair_cost = np.zeros(number_of_times)
    inspection_cost = np.zeros(number_of_times)
    cracked_joint_cost = np.zeros(number_of_times)
    crack_free_joint_cost = np.zeros(number_of_times)
    expected_repairs = np.zeros(number_of_times)

    joint_total_cost = np.zeros((number_of_groups, number_of_times))
    joint_failure_cost = np.zeros_like(joint_total_cost)
    joint_repair_cost = np.zeros_like(joint_total_cost)
    joint_expected_repairs = np.zeros_like(joint_total_cost)
    joint_cumulative_leaks = np.zeros_like(joint_total_cost)
    joint_cumulative_bursts = np.zeros_like(joint_total_cost)

    if capture_debug:
        debug_slots = states[debug_group].lengths.shape[0]
        debug_lengths = np.zeros((debug_slots, number_of_times + 1))
        debug_depths = np.zeros_like(debug_lengths)
        debug_lengths[:, 0] = states[debug_group].lengths[:, 0]
        debug_depths[:, 0] = states[debug_group].depths[:, 0]
    else:
        debug_lengths = None
        debug_depths = None

    # The inspection schedule is already known.  Therefore one sample can be
    # advanced through every time step without waiting for the other samples.
    for time_index, time_value in enumerate(times):
        if (
            decision_mode == ILI_DECISION_MODE
            and np.any(np.isclose(inspection_times, time_value))
        ):
            inspection_cost[time_index] = (
                costs["total_insp_cost"]
                / (1 + costs["rate"]) ** time_value
            )

        for group_index, (initial_cracks, physical_count) in enumerate(
            zip(layout.initial_crack_counts, layout.counts)
        ):
            (
                group_total,
                group_failure,
                group_repair,
                leaks,
                bursts,
                group_repairs,
            ) = _apply_one_joint_step(
                state=states[group_index],
                time_value=time_value,
                time_index=time_index,
                properties=properties[group_index],
                costs=costs,
                simulation_parameters=local_simulation_parameters,
                measurement_errors=measurement_errors,
                q_pod=q_pod,
                inspection_times=inspection_times,
                birth_rate=layout.birth_rates[group_index],
                decision_mode=decision_mode,
            )

            joint_total_cost[group_index, time_index] = group_total
            joint_failure_cost[group_index, time_index] = group_failure
            joint_repair_cost[group_index, time_index] = group_repair
            joint_expected_repairs[group_index, time_index] = group_repairs

            previous_leaks = (
                joint_cumulative_leaks[group_index, time_index - 1]
                if time_index
                else 0
            )
            previous_bursts = (
                joint_cumulative_bursts[group_index, time_index - 1]
                if time_index
                else 0
            )
            joint_cumulative_leaks[group_index, time_index] = previous_leaks + leaks
            joint_cumulative_bursts[group_index, time_index] = (
                previous_bursts + bursts
            )

            total_cost[time_index] += physical_count * group_total
            failure_cost[time_index] += physical_count * group_failure
            repair_cost[time_index] += physical_count * group_repair
            expected_repairs[time_index] += physical_count * group_repairs
            if initial_cracks:
                cracked_joint_cost[time_index] += physical_count * group_total
            else:
                crack_free_joint_cost[time_index] += physical_count * group_total

            if capture_debug and group_index == debug_group:
                debug_lengths[:, time_index + 1] = states[group_index].lengths[:, 0]
                debug_depths[:, time_index + 1] = states[group_index].depths[:, 0]

        total_cost[time_index] += inspection_cost[time_index]

    debug = {
        "times": times,
        "crack_depths": debug_depths,
        "crack_lengths": debug_lengths,
        "active_cracks": (
            states[debug_group].active_cracks[0] if capture_debug else None
        ),
        "wall_thickness": (
            properties[debug_group]["WT"][0] if capture_debug else None
        ),
        "inspection_times": inspection_times.copy(),
        "attempt_idx": sample_index,
        "crack_count_group": layout.initial_crack_counts[debug_group],
        "joint_name": layout.names[debug_group],
    }

    return SampleResult(
        total_cost=total_cost,
        failure_cost=failure_cost,
        repair_cost=repair_cost,
        inspection_cost=inspection_cost,
        cracked_joint_cost=cracked_joint_cost,
        crack_free_joint_cost=crack_free_joint_cost,
        expected_repairs=expected_repairs,
        joint_total_cost=joint_total_cost,
        joint_failure_cost=joint_failure_cost,
        joint_repair_cost=joint_repair_cost,
        joint_expected_repairs=joint_expected_repairs,
        joint_cumulative_leaks=joint_cumulative_leaks,
        joint_cumulative_bursts=joint_cumulative_bursts,
        actions=[state.actions for state in states],
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Final probability, hazard, and result calculations
# ---------------------------------------------------------------------------


def _system_probability(
    joint_probabilities: np.ndarray,
    joint_counts: np.ndarray,
) -> np.ndarray:
    """Combine representative-joint probabilities into system probability."""

    return 1 - np.prod(
        (1 - joint_probabilities) ** joint_counts[:, None], axis=0
    )


def _conditional_hazard(
    cumulative_probability: np.ndarray,
    pipe_length_km: float,
) -> np.ndarray:
    """Convert cumulative probability to the conditional annual hazard."""

    hazard = np.zeros_like(cumulative_probability)
    for time_index in range(1, len(cumulative_probability)):
        previous_probability = cumulative_probability[time_index - 1]
        if previous_probability < 1:
            hazard[time_index] = (
                cumulative_probability[time_index] - previous_probability
            ) / (1 - previous_probability) / pipe_length_km
    return hazard


def _mean(values: np.ndarray, retained_samples: int) -> np.ndarray:
    return values / retained_samples


def _finalize_result(
    *,
    totals: RunningTotals,
    layout: JointLayout,
    times: np.ndarray,
    inspection_times: np.ndarray,
    costs: dict[str, float],
    sim_params: dict[str, Any],
    requested_samples: int,
    debug: dict[str, Any] | None,
    debug_group: int,
) -> tuple:
    retained = totals.retained_samples
    if retained == 0:
        raise RuntimeError("All Monte Carlo samples failed at time zero.")

    joint_leak_probabilities = _mean(totals.joint_cumulative_leaks, retained)
    joint_burst_probabilities = _mean(totals.joint_cumulative_bursts, retained)
    system_leak_probability = _system_probability(
        joint_leak_probabilities, layout.counts
    )
    system_burst_probability = _system_probability(
        joint_burst_probabilities, layout.counts
    )
    system_leak_hazard = _conditional_hazard(
        system_leak_probability, costs["pipe_length_km"]
    )
    system_burst_hazard = _conditional_hazard(
        system_burst_probability, costs["pipe_length_km"]
    )

    unsafe_times = [
        (
            time_value,
            system_burst_hazard[time_index],
            system_burst_probability[time_index],
        )
        for time_index, time_value in enumerate(times)
        if model.exceeds_safety_criteria(
            system_leak_hazard[time_index],
            system_burst_hazard[time_index],
            sim_params,
        )
    ]

    action_probabilities = totals.action_counts / retained
    if debug is None:
        debug = {
            "crack_depths": None,
            "crack_lengths": None,
            "active_cracks": None,
            "wall_thickness": None,
            "sample_idx": None,
            "attempt_idx": None,
            "crack_count_group": layout.initial_crack_counts[debug_group],
            "joint_name": layout.names[debug_group],
        }

    debug.update(
        {
            "times": times,
            "inspection_times": inspection_times.copy(),
            "joint_names": list(layout.names),
            "joint_counts": layout.counts.copy(),
            "joint_leak_pofs": joint_leak_probabilities,
            "joint_burst_pofs": joint_burst_probabilities,
            "joint_cpp": _mean(totals.joint_total_cost, retained),
            "joint_cff": _mean(totals.joint_failure_cost, retained),
            "joint_crr": _mean(totals.joint_repair_cost, retained),
            "joint_expected_repairs": _mean(
                totals.joint_expected_repairs, retained
            ),
            "joint_action_probabilities": action_probabilities,
            "action_codes": np.asarray([0, 1, 2], dtype=np.int8),
            "requested_samples": int(requested_samples),
            "retained_samples": retained,
            "rejected_initial_failures": totals.rejected_initial_failures,
            "storage_mode": (
                "one realization at a time; only joint/time aggregates retained"
            ),
            "scheduling_mode": "equidistant",
            "streaming_passes": 1,
        }
    )

    # Preserve the established return order so existing result-unpacking code
    # can use this implementation without changes.
    return (
        times,
        system_leak_probability,
        system_burst_probability,
        system_leak_hazard,
        system_burst_hazard,
        inspection_times.copy(),
        _mean(totals.total_cost, retained),
        _mean(totals.failure_cost, retained),
        _mean(totals.repair_cost, retained),
        _mean(totals.inspection_cost, retained),
        len(unsafe_times) == 0,
        unsafe_times,
        _mean(totals.cracked_joint_cost, retained),
        _mean(totals.crack_free_joint_cost, retained),
        _mean(totals.expected_repairs, retained),
        [
            action_probabilities[group_index]
            for group_index in range(layout.number_of_groups)
        ],
        debug,
    )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def run_equidistant_simulation(
    T,
    dt,
    nx,
    num_cracks_list,
    joint_counts,
    pipe_props_list,
    costs,
    sim_params,
    S_err,
    q_pod,
    birth_rate,
    *,
    track_debug=True,
    debug_sample_idx=10,
    debug_crack_count=None,
    joint_definitions=None,
    property_seed=0,
    simulation_seed=None,
    crack_length_std=15.6,
    crack_depth_std=0.74,
    debug_joint=None,
    decision_mode=ILI_DECISION_MODE,
):
    """Run a low-memory Monte Carlo analysis for a fixed inspection schedule.

    ``sim_params`` must contain ``inspection_interval`` or explicit
    ``inspection_times``.  If ``joint_definitions`` is supplied, it replaces
    the legacy grouped inputs ``num_cracks_list``, ``joint_counts``, and
    ``pipe_props_list``.
    """

    _validate_general_inputs(
        T, dt, nx, costs, decision_mode, debug_sample_idx
    )
    layout = _prepare_layout(
        num_cracks_list,
        joint_counts,
        pipe_props_list,
        joint_definitions,
        birth_rate,
    )
    inspection_times = _make_inspection_schedule(T, dt, sim_params)
    debug_group = _select_debug_group(
        layout, debug_joint, debug_crack_count
    )
    simulation_seed = property_seed if simulation_seed is None else simulation_seed

    times = np.arange(0, T + dt, dt)
    totals = RunningTotals.empty(layout.number_of_groups, len(times))
    debug = None

    for sample_index in range(int(nx)):
        # debug_sample_idx refers to a retained sample, not an attempted sample.
        # Until that sample is reached, the latest retained history is kept as
        # a fallback.  Thus a short run still returns one useful debug history.
        # Only one realization's history exists in memory at any moment.
        capture_debug = track_debug and totals.retained_samples <= debug_sample_idx
        try:
            sample = _simulate_one_realization(
                sample_index=sample_index,
                layout=layout,
                horizon=T,
                time_step=dt,
                costs=costs,
                simulation_parameters=sim_params,
                measurement_errors=S_err,
                q_pod=q_pod,
                inspection_times=inspection_times,
                property_seed=property_seed,
                simulation_seed=simulation_seed,
                crack_length_std=crack_length_std,
                crack_depth_std=crack_depth_std,
                capture_debug=capture_debug,
                debug_group=debug_group,
                decision_mode=decision_mode,
            )
        except InitialFailure:
            totals.rejected_initial_failures += 1
            continue

        if capture_debug:
            debug = sample.debug.copy()
            debug["sample_idx"] = totals.retained_samples
        totals.add(sample)

    return _finalize_result(
        totals=totals,
        layout=layout,
        times=times,
        inspection_times=inspection_times,
        costs=costs,
        sim_params=sim_params,
        requested_samples=int(nx),
        debug=debug,
        debug_group=debug_group,
    )


def run_flexible_equidistant_simulation(
    T,
    dt,
    nx,
    joints,
    costs,
    sim_params,
    S_err,
    q_pod,
    *,
    default_birth_rate=0.0,
    property_seed=0,
    simulation_seed=None,
    crack_length_std=15.6,
    crack_depth_std=0.74,
    track_debug=True,
    debug_sample_idx=10,
    debug_joint=None,
    decision_mode=ILI_DECISION_MODE,
):
    """Run the equidistant model using explicit joint/group definitions."""

    return run_equidistant_simulation(
        T,
        dt,
        nx,
        [],
        {},
        [],
        costs,
        sim_params,
        S_err,
        q_pod,
        default_birth_rate,
        track_debug=track_debug,
        debug_sample_idx=debug_sample_idx,
        joint_definitions=joints,
        property_seed=property_seed,
        simulation_seed=simulation_seed,
        crack_length_std=crack_length_std,
        crack_depth_std=crack_depth_std,
        debug_joint=debug_joint,
        decision_mode=decision_mode,
    )


def run_simulation(approach, *args, **kwargs):
    """Compatibility wrapper for the old ``run_simulation`` call style."""

    if approach != "equidistant":
        raise ValueError(
            "This module supports only approach='equidistant'. "
            "Use Simp_Sys_Thresh_J3_optimized.py for threshold scheduling."
        )
    return run_equidistant_simulation(*args, **kwargs)


def run_flexible_simulation(approach, *args, **kwargs):
    """Compatibility wrapper for the old flexible-interface call style."""

    if approach != "equidistant":
        raise ValueError(
            "This module supports only approach='equidistant'. "
            "Use Simp_Sys_Thresh_J3_optimized.py for threshold scheduling."
        )
    return run_flexible_equidistant_simulation(*args, **kwargs)


if __name__ == "__main__":
    print(
        "This is the low-memory equidistant simulation module. "
        "Import run_equidistant_simulation or "
        "run_flexible_equidistant_simulation to run an analysis."
    )
