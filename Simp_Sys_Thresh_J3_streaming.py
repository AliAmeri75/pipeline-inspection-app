#!/usr/bin/env python3
"""Low-memory Monte Carlo driver for the pipeline planning model.

This module intentionally leaves ``Simp_Sys_Thresh_J3_optimized.py`` unchanged.
It advances one Monte Carlo realization through the complete time horizon,
adds that realization to running totals, and then discards its crack states.

Memory therefore scales mainly with ``joints * time_steps`` instead of
``samples * joints * crack_slots``.  Joint-level expected costs, failure
probabilities, repair probabilities, and action probabilities are retained for
later analysis.  Individual sample histories are not retained, except for one
optional debug realization.

The public return order matches the optimized module.  The item in the old
``action_arr`` position is now a list of arrays shaped ``(3, n_times)``.  Rows
0, 1, and 2 contain the probability of each action for that joint or joint
group.  Aggregate joint results are also available in ``debug_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import Simp_Sys_Thresh_J3_optimized as base


ILI_DECISION_MODE = base.ILI_DECISION_MODE
EXCAVATION_DECISION_MODE = base.EXCAVATION_DECISION_MODE
DECISION_MODES = base.DECISION_MODES
PROPERTY_NAMES = base.PROPERTY_NAMES
DEFAULT_PROPERTY_SPECS = base.DEFAULT_PROPERTY_SPECS
MEAN_WT = base.MEAN_WT


class _InitialFailure(RuntimeError):
    """Signal that one attempted realization failed at time zero."""


@dataclass
class _Layout:
    flexible: bool
    joints: list[dict[str, Any]]
    num_cracks: list[int]
    counts: np.ndarray
    birth_rates: np.ndarray
    names: list[str]
    crack_specs: list[list[dict[str, float]] | None]
    pipe_properties: list[dict[str, Any]]


def _seed(base_seed: int, sample_index: int, stage: int) -> int:
    """Create reproducible independent seeds for sample and simulation stage."""

    modulus = np.iinfo(np.uint32).max
    return int((int(base_seed) + sample_index * 1_000_003 + stage * 10_007) % modulus)


def _prepare_layout(
    num_cracks_list,
    joint_counts,
    pipe_props_list,
    joint_definitions,
    birth_rate,
) -> _Layout:
    if joint_definitions is not None:
        joints = [dict(joint) for joint in joint_definitions]
        if not joints:
            raise ValueError("joint_definitions must contain at least one joint.")
        names = [joint.get("name", f"Joint {index + 1}") for index, joint in enumerate(joints)]
        if len(set(names)) != len(names):
            raise ValueError("Every explicitly modelled joint must have a unique name.")
        crack_specs = [list(joint.get("cracks", [])) for joint in joints]
        num_cracks = [len(cracks) for cracks in crack_specs]
        raw_counts = [joint.get("count", 1) for joint in joints]
        if any(int(count) != count or count < 1 for count in raw_counts):
            raise ValueError("Each joint/group count must be a positive integer.")
        birth_rates = np.asarray(
            [joint.get("birth_rate", birth_rate) for joint in joints], dtype=float
        )
        if np.any(birth_rates < 0):
            raise ValueError("Joint birth rates cannot be negative.")
        return _Layout(
            flexible=True,
            joints=joints,
            num_cracks=num_cracks,
            counts=np.asarray(raw_counts, dtype=int),
            birth_rates=birth_rates,
            names=names,
            crack_specs=crack_specs,
            pipe_properties=[],
        )

    original_groups = list(num_cracks_list)
    props_by_group = dict(zip(original_groups, pipe_props_list))
    active_groups = [cracks for cracks in original_groups if joint_counts.get(cracks, 0) > 0]
    if not active_groups:
        raise ValueError("joint_counts must contain at least one positive count.")
    missing = [cracks for cracks in active_groups if cracks not in props_by_group]
    if missing:
        raise ValueError(f"Missing pipe properties for crack groups: {missing}")
    return _Layout(
        flexible=False,
        joints=[],
        num_cracks=active_groups,
        counts=np.asarray([joint_counts[cracks] for cracks in active_groups], dtype=int),
        birth_rates=np.full(len(active_groups), birth_rate, dtype=float),
        names=[f"{cracks}-crack group" for cracks in active_groups],
        crack_specs=[None] * len(active_groups),
        pipe_properties=[props_by_group[cracks] for cracks in active_groups],
    )


def _property_value(values, sample_index: int) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        return float(array[0])
    if sample_index >= array.size:
        raise ValueError(
            "A supplied pipe-property population has fewer values than the requested samples."
        )
    return float(array[sample_index])


def _sample_properties(
    layout: _Layout,
    sample_index: int,
    property_seed: int,
) -> list[dict[str, Any]]:
    if layout.flexible:
        return base.sample_individual_joint_properties(
            layout.joints,
            1,
            seed=_seed(property_seed, sample_index, 1),
        )

    sampled = []
    for cracks, properties in zip(layout.num_cracks, layout.pipe_properties):
        record = {
            name: np.asarray([_property_value(properties[name], sample_index)])
            for name in PROPERTY_NAMES
        }
        record["num_initial_cracks"] = cracks
        sampled.append(record)
    return sampled


def _debug_group_index(layout: _Layout, debug_joint, debug_crack_count) -> int:
    if debug_joint is not None:
        index = layout.names.index(debug_joint) if isinstance(debug_joint, str) else int(debug_joint)
        if not 0 <= index < len(layout.names):
            raise ValueError("debug_joint is outside the available joint range.")
        return index
    if debug_crack_count is None:
        return next((index for index, cracks in enumerate(layout.num_cracks) if cracks > 0), 0)
    return layout.num_cracks.index(debug_crack_count)


def _simulate_one_sample(
    *,
    sample_index: int,
    simulation_seed: int,
    property_seed: int,
    layout: _Layout,
    T: float,
    dt: float,
    costs: dict[str, float],
    sim_params: dict[str, Any],
    S_err,
    q_pod: float,
    inspection_times: np.ndarray,
    crack_length_std: float,
    crack_depth_std: float,
    capture_debug: bool,
    debug_group: int,
    decision_mode: str,
) -> dict[str, Any]:
    """Simulate one complete realization without retaining other samples."""

    properties = _sample_properties(layout, sample_index, property_seed)
    sim = dict(sim_params)
    sim.update({"T": T, "dt": dt})
    times = np.arange(0, T + dt, dt)
    nt = len(times)

    np.random.seed(_seed(simulation_seed, sample_index, 2))
    births = [
        [
            base.generate_initiation_times(
                base.generate_num_cracks(T, joint_birth_rate), T, joint_birth_rate
            )
        ]
        for joint_birth_rate in layout.birth_rates
    ]

    total_initial_cracks = sum(layout.num_cracks)
    crack_samples = np.zeros((1, 2 * total_initial_cracks))
    start_column = 0
    for joint_index, cracks in enumerate(layout.num_cracks):
        if cracks:
            crack_input = layout.crack_specs[joint_index] if layout.flexible else cracks
            lengths, depths = base.sample_initial_cracks(
                crack_input,
                1,
                seed=_seed(simulation_seed, sample_index, 10 + joint_index),
                length_std=crack_length_std,
                depth_std=crack_depth_std,
            )
            crack_samples[:, start_column:start_column + cracks] = lengths.T
            depth_start = total_initial_cracks + start_column
            crack_samples[:, depth_start:depth_start + cracks] = depths.T
        start_column += cracks

    if not base.initial_system_survival_mask(
        crack_samples, properties, layout.num_cracks
    )[0]:
        raise _InitialFailure

    # Use a separate reproducible stream for births, growth, observations, and repairs.
    np.random.seed(_seed(simulation_seed, sample_index, 100))
    worst = int(np.ceil(base.exp_num(T, float(np.max(layout.birth_rates))))) + 40
    lengths_by_joint = []
    depths_by_joint = []
    growth_by_joint = []
    rep1 = []
    rep2 = []
    qsf_states = []
    fsf_states = []
    mv_states = []
    pointers = []
    next_slots = []
    active = []
    failed_locations = []
    actions = []

    start_column = 0
    for joint_index, cracks in enumerate(layout.num_cracks):
        total_slots = cracks + worst
        lengths = np.zeros((total_slots, 1))
        depths = np.zeros_like(lengths)
        if cracks:
            lengths[:cracks, 0] = crack_samples[0, start_column:start_column + cracks]
            depth_start = total_initial_cracks + start_column
            depths[:cracks, 0] = crack_samples[0, depth_start:depth_start + cracks]
        start_column += cracks

        lengths_by_joint.append(lengths)
        depths_by_joint.append(depths)
        growth_by_joint.append(np.full_like(lengths, properties[joint_index]["alpha"]))
        rep1.append(np.zeros(1))
        rep2.append(np.zeros(1))
        qsf_states.append(np.zeros_like(lengths))
        fsf_states.append(np.zeros_like(lengths))
        mv_states.append(np.zeros_like(lengths))
        pointers.append([0])
        next_slots.append([cracks])
        active.append([list(range(cracks))])
        failed_locations.append([[]])
        actions.append(np.zeros((1, nt), dtype=np.int8))

    if capture_debug:
        debug_slots = lengths_by_joint[debug_group].shape[0]
        debug_lengths = np.zeros((debug_slots, nt + 1))
        debug_depths = np.zeros_like(debug_lengths)
        debug_lengths[:, 0] = lengths_by_joint[debug_group][:, 0]
        debug_depths[:, 0] = depths_by_joint[debug_group][:, 0]
    else:
        debug_lengths = debug_depths = None

    joint_cpp = np.zeros((len(layout.names), nt))
    joint_cff = np.zeros_like(joint_cpp)
    joint_crr = np.zeros_like(joint_cpp)
    joint_repairs = np.zeros_like(joint_cpp)
    cumulative_leaks = np.zeros_like(joint_cpp)
    cumulative_bursts = np.zeros_like(joint_cpp)
    cpp = np.zeros(nt)
    cff = np.zeros(nt)
    crr = np.zeros(nt)
    cii = np.zeros(nt)
    cpp_cracked = np.zeros(nt)
    cpp_crack_free = np.zeros(nt)
    num_repairs = np.zeros(nt)

    for time_index, time_value in enumerate(times):
        if (
            decision_mode == ILI_DECISION_MODE
            and np.any(np.isclose(inspection_times, time_value))
        ):
            cii[time_index] = (
                costs["total_insp_cost"] / (1 + costs["rate"]) ** time_value
            )

        for joint_index, (cracks, count) in enumerate(
            zip(layout.num_cracks, layout.counts)
        ):
            result = base.Joint_planning_step(
                time_value,
                time_index,
                lengths_by_joint[joint_index],
                depths_by_joint[joint_index],
                growth_by_joint[joint_index],
                rep1[joint_index],
                rep2[joint_index],
                pointers[joint_index],
                next_slots[joint_index],
                births[joint_index],
                active[joint_index],
                properties[joint_index],
                costs,
                sim,
                S_err,
                q_pod,
                qsf_states[joint_index],
                fsf_states[joint_index],
                mv_states[joint_index],
                inspection_times,
                failed_locations[joint_index],
                layout.birth_rates[joint_index],
                actions[joint_index],
                decision_mode,
            )
            (
                cost_total,
                cost_failure,
                cost_repair,
                _,
                leaks,
                bursts,
                lengths_by_joint[joint_index],
                depths_by_joint[joint_index],
                growth_by_joint[joint_index],
                rep1[joint_index],
                rep2[joint_index],
                pointers[joint_index],
                next_slots[joint_index],
                births[joint_index],
                active[joint_index],
                qsf_states[joint_index],
                fsf_states[joint_index],
                mv_states[joint_index],
                failed_locations[joint_index],
                expected_repairs,
                actions[joint_index],
            ) = result

            joint_cpp[joint_index, time_index] = cost_total
            joint_cff[joint_index, time_index] = cost_failure
            joint_crr[joint_index, time_index] = cost_repair
            joint_repairs[joint_index, time_index] = expected_repairs
            cumulative_leaks[joint_index, time_index] = (
                (cumulative_leaks[joint_index, time_index - 1] if time_index else 0)
                + leaks
            )
            cumulative_bursts[joint_index, time_index] = (
                (cumulative_bursts[joint_index, time_index - 1] if time_index else 0)
                + bursts
            )

            cpp[time_index] += count * cost_total
            cff[time_index] += count * cost_failure
            crr[time_index] += count * cost_repair
            num_repairs[time_index] += count * expected_repairs
            if cracks:
                cpp_cracked[time_index] += count * cost_total
            else:
                cpp_crack_free[time_index] += count * cost_total

            if capture_debug and joint_index == debug_group:
                debug_lengths[:, time_index + 1] = lengths_by_joint[joint_index][:, 0]
                debug_depths[:, time_index + 1] = depths_by_joint[joint_index][:, 0]

        cpp[time_index] += cii[time_index]

    debug = {
        "times": times,
        "crack_depths": debug_depths,
        "crack_lengths": debug_lengths,
        "active_cracks": active[debug_group][0] if capture_debug else None,
        "wall_thickness": properties[debug_group]["WT"][0] if capture_debug else None,
        "inspection_times": inspection_times.copy(),
        "sample_idx": sample_index,
        "crack_count_group": layout.num_cracks[debug_group],
        "joint_name": layout.names[debug_group],
    }
    return {
        "times": times,
        "cpp": cpp,
        "cff": cff,
        "crr": crr,
        "cii": cii,
        "cpp_cracked": cpp_cracked,
        "cpp_crack_free": cpp_crack_free,
        "repairs": num_repairs,
        "joint_cpp": joint_cpp,
        "joint_cff": joint_cff,
        "joint_crr": joint_crr,
        "joint_repairs": joint_repairs,
        "joint_leaks": cumulative_leaks,
        "joint_bursts": cumulative_bursts,
        "actions": actions,
        "debug": debug,
    }


def _system_probabilities(joint_values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return 1 - np.prod((1 - joint_values) ** counts[:, None], axis=0)


def _hazard(probabilities: np.ndarray, pipe_length_km: float) -> np.ndarray:
    result = np.zeros_like(probabilities)
    for index in range(1, len(probabilities)):
        previous = probabilities[index - 1]
        result[index] = (
            (probabilities[index] - previous) / (1 - previous)
            if previous < 1
            else 0.0
        ) / pipe_length_km
    return result


def _run_fixed_schedule(
    *,
    T,
    dt,
    nx,
    layout,
    costs,
    sim_params,
    S_err,
    q_pod,
    inspection_times,
    property_seed,
    simulation_seed,
    crack_length_std,
    crack_depth_std,
    track_debug,
    debug_sample_idx,
    debug_group,
    decision_mode,
):
    scalar_keys = (
        "cpp",
        "cff",
        "crr",
        "cii",
        "cpp_cracked",
        "cpp_crack_free",
        "repairs",
    )
    joint_keys = (
        "joint_cpp",
        "joint_cff",
        "joint_crr",
        "joint_repairs",
        "joint_leaks",
        "joint_bursts",
    )
    scalar_sums = None
    joint_sums = None
    action_counts = None
    retained = 0
    rejected = 0
    debug = None

    for sample_index in range(nx):
        capture_debug = track_debug and retained <= debug_sample_idx
        try:
            result = _simulate_one_sample(
                sample_index=sample_index,
                simulation_seed=simulation_seed,
                property_seed=property_seed,
                layout=layout,
                T=T,
                dt=dt,
                costs=costs,
                sim_params=sim_params,
                S_err=S_err,
                q_pod=q_pod,
                inspection_times=inspection_times,
                crack_length_std=crack_length_std,
                crack_depth_std=crack_depth_std,
                capture_debug=capture_debug,
                debug_group=debug_group,
                decision_mode=decision_mode,
            )
        except _InitialFailure:
            rejected += 1
            continue

        if scalar_sums is None:
            scalar_sums = {key: np.zeros_like(result[key]) for key in scalar_keys}
            joint_sums = {key: np.zeros_like(result[key]) for key in joint_keys}
            action_counts = np.zeros(
                (len(layout.names), 3, len(result["times"])), dtype=np.int64
            )
        for key in scalar_keys:
            scalar_sums[key] += result[key]
        for key in joint_keys:
            joint_sums[key] += result[key]
        for joint_index, action_history in enumerate(result["actions"]):
            for action_code in range(3):
                action_counts[joint_index, action_code] += (
                    action_history[0] == action_code
                )

        if capture_debug:
            debug = result["debug"].copy()
            debug["attempt_idx"] = sample_index
            debug["sample_idx"] = retained
        retained += 1

    if retained == 0:
        raise RuntimeError("All Monte Carlo samples failed at time zero.")

    means = {key: value / retained for key, value in scalar_sums.items()}
    joint_means = {key: value / retained for key, value in joint_sums.items()}
    action_probabilities = action_counts / retained
    joint_leak_pofs = joint_means["joint_leaks"]
    joint_burst_pofs = joint_means["joint_bursts"]
    system_leak_pofs = _system_probabilities(joint_leak_pofs, layout.counts)
    system_burst_pofs = _system_probabilities(joint_burst_pofs, layout.counts)
    system_leak_hazard = _hazard(system_leak_pofs, costs["pipe_length_km"])
    system_burst_hazard = _hazard(system_burst_pofs, costs["pipe_length_km"])
    times = np.arange(0, T + dt, dt)
    unsafe_times = [
        (time_value, system_burst_hazard[index], system_burst_pofs[index])
        for index, time_value in enumerate(times)
        if base.exceeds_safety_criteria(
            system_leak_hazard[index], system_burst_hazard[index], sim_params
        )
    ]

    if debug is None:
        debug = {
            "crack_depths": None,
            "crack_lengths": None,
            "active_cracks": None,
            "wall_thickness": None,
            "sample_idx": None,
            "crack_count_group": layout.num_cracks[debug_group],
            "joint_name": layout.names[debug_group],
        }
    debug.update(
        {
            "times": times,
            "inspection_times": inspection_times.copy(),
            "joint_names": list(layout.names),
            "joint_counts": layout.counts.copy(),
            "joint_leak_pofs": joint_leak_pofs,
            "joint_burst_pofs": joint_burst_pofs,
            "joint_cpp": joint_means["joint_cpp"],
            "joint_cff": joint_means["joint_cff"],
            "joint_crr": joint_means["joint_crr"],
            "joint_expected_repairs": joint_means["joint_repairs"],
            "joint_action_probabilities": action_probabilities,
            "action_codes": np.asarray([0, 1, 2], dtype=np.int8),
            "requested_samples": int(nx),
            "retained_samples": retained,
            "rejected_initial_failures": rejected,
            "storage_mode": "one sample at a time; joint/time aggregates retained",
        }
    )
    action_summary = [action_probabilities[index] for index in range(len(layout.names))]
    return (
        times,
        system_leak_pofs,
        system_burst_pofs,
        system_leak_hazard,
        system_burst_hazard,
        inspection_times.copy(),
        means["cpp"],
        means["cff"],
        means["crr"],
        means["cii"],
        len(unsafe_times) == 0,
        unsafe_times,
        means["cpp_cracked"],
        means["cpp_crack_free"],
        means["repairs"],
        action_summary,
        debug,
    )


def run_simulation(
    approach,
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
    """Run the model while retaining only one Monte Carlo state at a time.

    Equidistant planning requires one streaming pass.  Threshold planning uses
    repeatable streaming passes: it adds the earliest threshold-triggered
    inspection, reruns with the same sample seeds, and stops when the schedule
    no longer changes.  This avoids retaining all sample states for rollback.
    """

    if nx < 1 or int(nx) != nx:
        raise ValueError("nx must be a positive integer.")
    if dt <= 0 or T < 0:
        raise ValueError("T must be non-negative and dt must be positive.")
    if debug_sample_idx < 0:
        raise ValueError("debug_sample_idx cannot be negative.")
    if decision_mode not in DECISION_MODES:
        raise ValueError(f"decision_mode must be one of {sorted(DECISION_MODES)}.")
    if approach not in {"equidistant", "threshold"}:
        raise ValueError("approach must be 'equidistant' or 'threshold'.")
    if costs["pipe_length_km"] <= 0:
        raise ValueError("pipe_length_km must be positive.")

    simulation_seed = property_seed if simulation_seed is None else simulation_seed
    layout = _prepare_layout(
        num_cracks_list,
        joint_counts,
        pipe_props_list,
        joint_definitions,
        birth_rate,
    )
    debug_group = _debug_group_index(layout, debug_joint, debug_crack_count)
    local_sim = dict(sim_params)
    if approach == "equidistant":
        inspection_times = base._schedule_from_parameters(approach, T, dt, local_sim)
    else:
        inspection_times = np.asarray([], dtype=float)

    max_passes = len(np.arange(0, T + dt, dt)) + 1
    for pass_index in range(max_passes):
        result = _run_fixed_schedule(
            T=T,
            dt=dt,
            nx=int(nx),
            layout=layout,
            costs=costs,
            sim_params=local_sim,
            S_err=S_err,
            q_pod=q_pod,
            inspection_times=inspection_times,
            property_seed=property_seed,
            simulation_seed=simulation_seed,
            crack_length_std=crack_length_std,
            crack_depth_std=crack_depth_std,
            track_debug=track_debug,
            debug_sample_idx=debug_sample_idx,
            debug_group=debug_group,
            decision_mode=decision_mode,
        )
        result[-1]["streaming_passes"] = pass_index + 1
        if approach == "equidistant":
            return result

        combined = base.combined_hazard(result[3], result[4], costs)
        proposed_time = None
        for index, value in enumerate(combined):
            candidate = result[0][index] - dt
            if (
                value > local_sim["threshold"]
                and candidate >= 0
                and not np.any(np.isclose(inspection_times, candidate))
            ):
                proposed_time = candidate
                break
        if proposed_time is None:
            return result
        inspection_times = np.sort(np.append(inspection_times, proposed_time))

    raise RuntimeError("Threshold schedule did not converge within the available time steps.")


def run_flexible_simulation(
    approach,
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
    """Streaming interface for individually specified joints or joint groups."""

    return run_simulation(
        approach,
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


if __name__ == "__main__":
    print(
        "Import run_simulation or run_flexible_simulation from this module. "
        "The original optimized module remains unchanged."
    )
