#!/usr/bin/env python3
"""Clearer and lighter version of ``Simp_Sys_Thresh_J3.py``.

The reliability equations, repair rules, and threshold definition are retained.
The main reductions in run time and memory are structural:

* equidistant analyses do not create rollback copies that they never use;
* debug histories store one selected Monte Carlo sample, not every sample;
* representative-joint probabilities and costs are multiplied by joint count;
* joint groups with a count of zero are not simulated;
* equidistant schedules use the requested interval (or an explicit schedule).

The public ``run_simulation`` return order is kept compatible with the original
system script.
"""

from __future__ import annotations

import copy
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Sampling_2
from Models import Normal


def _load_local_corlas():
    """Load the five-output CorLAS implementation beside this script.

    The project contains several other ``CorlasP.py`` files that return only
    two values. An explicit file-based import prevents Spyder's working
    directory or module cache from selecting one of those incompatible files.
    """
    module_path = Path(__file__).resolve().with_name("CorlasP.py")
    if not module_path.is_file():  # Allows development copies outside the DBN folder.
        return importlib.import_module("CorlasP")
    spec = importlib.util.spec_from_file_location("dbn_metamodel_corlasp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the local CorLAS module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CorlasP = _load_local_corlas()


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MEAN_WT, STD_WT = 6.35, 0.6
MEAN_G, STD_G = 0.1, 0.03
MEAN_CVN = 85.0
STD_CVN = 0.0223 * MEAN_CVN**1.46
MEAN_E, STD_E = 210.0, 0.04 * 210.0
MOP = 936.0
STD_MOP = 0.02 * 1.07 * MOP
MEAN_SMYS = 1.1 * 52000
STD_SMYS = 0.035 * MEAN_SMYS
MEAN_SMTS = 1.12 * 66000
STD_SMTS = 0.035 * MEAN_SMTS
MEAN_ERR, STD_ERR = 1.27, 0.21
DI = 20 * 25.4

MU_C = 2.5e-13
STD_C = 0.54 * MU_C
MU_NEQ, STD_NEQ = 1e3, 0.3e3
_CV2_NEQ = (STD_NEQ / MU_NEQ) ** 2
LOG_STD_NEQ = np.sqrt(np.log1p(_CV2_NEQ))
LOG_MEAN_NEQ = np.log(MU_NEQ) - 0.5 * LOG_STD_NEQ**2

SF = 2
FLOW_DEFINITION = "Other"
PIPE_DIAMETER = 20
FLAW_SHAPE = "E"
FLAW_LOCATION = "E"
FD = FT = 1
MM_TO_INCH = 0.0393701
GPA_TO_PSI = 145038
SMALL_OFFSET = 1e-6

PROPERTY_NAMES = (
    "alpha", "WT", "Neq", "C", "SMYS", "SMTS", "CVN", "E", "err", "Pservice"
)
PROPERTY_TYPES = (
    "Normal", "Normal", "Normal", "Lognormal", "Normal",
    "Normal", "Lognormal", "Normal", "Normal", "Gumbel",
)
PROPERTY_MEANS = (
    MEAN_G, MEAN_WT, MU_NEQ, MU_C, MEAN_SMYS,
    MEAN_SMTS, MEAN_CVN, MEAN_E, MEAN_ERR, MOP,
)
PROPERTY_STDS = (
    STD_G, STD_WT, STD_NEQ, STD_C, STD_SMYS,
    STD_SMTS, STD_CVN, STD_E, STD_ERR, STD_MOP,
)
DEFAULT_PROPERTY_SPECS = {
    name: {"distribution": distribution, "mean": mean, "std": std}
    for name, distribution, mean, std in zip(
        PROPERTY_NAMES, PROPERTY_TYPES, PROPERTY_MEANS, PROPERTY_STDS
    )
}
BURST_VARIABLE_NAMES = ["FL", "FD", "WT", "SMYS", "SMTS", "CVN", "E"]


# ---------------------------------------------------------------------------
# Sampling and crack-birth helpers
# ---------------------------------------------------------------------------
def exp_num(t, birth_rate):
    return birth_rate * t


def generate_num_cracks(T, birth_rate):
    return np.random.poisson(exp_num(T, birth_rate))


def generate_initiation_times(n, T, birth_rate):
    """Retain the initiation-time sampling rule used by the original model."""
    if n == 0 or T <= 0 or birth_rate <= 0:
        return []
    times = []
    for _ in range(n):
        while True:
            u = np.random.uniform(0, T)
            if np.random.rand() <= exp_num(u, birth_rate) / exp_num(T, birth_rate):
                times.append(u)
                break
    return sorted(times)


def reset_births_after_failure(t_fail, T, birth_rate):
    remaining = T - t_fail
    n_new = generate_num_cracks(remaining, birth_rate)
    return [t_fail + t for t in generate_initiation_times(n_new, remaining, birth_rate)]


def _crack_means(cracks):
    """Return length/depth means from an integer or a list of crack records."""
    if isinstance(cracks, (int, np.integer)):
        if cracks < 0:
            raise ValueError("The number of cracks cannot be negative.")
        return (
            np.linspace(40, 40 + 2 * (cracks - 1), cracks),
            np.linspace(1, 1 + 0.1 * (cracks - 1), cracks),
        )

    means_l, means_d = [], []
    for index, crack in enumerate(cracks, start=1):
        if isinstance(crack, dict):
            try:
                length = crack["mean_length"]
                depth = crack["mean_depth"]
            except KeyError as exc:
                raise ValueError(
                    f"Crack {index} requires mean_length and mean_depth."
                ) from exc
        else:
            try:
                length, depth = crack
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Each crack must be a dictionary or a (mean_length, mean_depth) pair."
                ) from exc
        if length <= 0 or depth <= 0:
            raise ValueError("Initial crack means must be positive.")
        means_l.append(float(length))
        means_d.append(float(depth))
    return np.asarray(means_l), np.asarray(means_d)


def sample_initial_cracks(
    cracks, nx, seed=0, *, length_std=15.6, depth_std=0.74,
):
    """Sample initial sizes with common length/depth standard deviations."""
    means_l, means_d = _crack_means(cracks)
    num_cracks = len(means_l)
    if num_cracks == 0:
        return np.zeros((0, nx)), np.zeros((0, nx))
    if length_std < 0 or depth_std < 0:
        raise ValueError("Crack-size standard deviations cannot be negative.")
    names = [f"FL{i + 1}" for i in range(num_cracks)] + [
        f"FD{i + 1}" for i in range(num_cracks)
    ]
    samples, _ = Sampling_2.MC(
        names,
        ["Lognormal"] * (2 * num_cracks),
        np.r_[means_l, means_d].tolist(),
        [length_std] * num_cracks + [depth_std] * num_cracks,
        [], [], [], nx, Seed=seed,
    )
    return samples[:, :num_cracks].T, samples[:, num_cracks:].T


def sample_pipe_properties(num_cracks_list, nx, seed=0):
    """Sample one common pipe-property population for all joint groups.

    The original code sampled each group with the same default seed and later
    replaced every group by the first filtered population. Sampling once makes
    that existing common-property assumption explicit and avoids duplicate work.
    """
    samples, _ = Sampling_2.MC(
        list(PROPERTY_NAMES), list(PROPERTY_TYPES), list(PROPERTY_MEANS),
        list(PROPERTY_STDS), [], [], [], nx, Seed=seed,
    )
    base = {name: samples[:, i] for i, name in enumerate(PROPERTY_NAMES)}
    return [
        {**{name: values.copy() for name, values in base.items()},
         "num_initial_cracks": cracks}
        for cracks in num_cracks_list
    ]


def _resolve_property_spec(name, override):
    """Merge a concise joint override with the default property distribution."""
    spec = dict(DEFAULT_PROPERTY_SPECS[name])
    if override is None:
        return spec
    if np.isscalar(override):
        spec["mean"] = float(override)
    elif isinstance(override, (tuple, list)) and len(override) == 2:
        spec.update({"mean": float(override[0]), "std": float(override[1])})
    elif isinstance(override, dict):
        spec.update(override)
        if "type" in spec:
            spec["distribution"] = spec.pop("type")
    else:
        raise ValueError(
            f"Property {name} must be a mean, (mean, std), or a dictionary."
        )
    if spec["std"] < 0:
        raise ValueError(f"The standard deviation for {name} cannot be negative.")
    return spec


def sample_individual_joint_properties(joints, nx, seed=0):
    """Sample a separate property population for every explicitly listed joint.

    A property override can be written in any of these forms::

        "WT": 6.5
        "WT": (6.5, 0.5)
        "WT": {"mean": 6.5, "std": 0.5, "distribution": "Normal"}

    Properties omitted from a joint inherit the original model defaults.
    """
    sampled_properties = []
    for joint_index, joint in enumerate(joints):
        overrides = joint.get("properties", {})
        unknown = set(overrides) - set(PROPERTY_NAMES)
        if unknown:
            raise ValueError(f"Unknown properties: {sorted(unknown)}")
        specs = [
            _resolve_property_spec(name, overrides.get(name))
            for name in PROPERTY_NAMES
        ]
        samples, _ = Sampling_2.MC(
            list(PROPERTY_NAMES),
            [spec["distribution"] for spec in specs],
            [spec["mean"] for spec in specs],
            [spec["std"] for spec in specs],
            [], [], [], nx, Seed=seed + joint_index,
        )
        props = {name: samples[:, i] for i, name in enumerate(PROPERTY_NAMES)}
        props["num_initial_cracks"] = len(joint.get("cracks", []))
        sampled_properties.append(props)
    return sampled_properties


# ---------------------------------------------------------------------------
# Limit states and crack growth
# ---------------------------------------------------------------------------
def limit_state_burst(Name_var, var, err, pressure):
    """CorLAS burst limit state: ``g = err * failure_pressure - pressure``."""
    del Name_var  # retained in the signature for compatibility
    length, depth, wt, smys, smts, cvn, elastic_modulus = var
    length = length * MM_TO_INCH + SMALL_OFFSET
    depth = min(depth * MM_TO_INCH + SMALL_OFFSET, wt * MM_TO_INCH)
    wt *= MM_TO_INCH
    elastic_modulus *= GPA_TO_PSI
    failure_pressure, _, qf, fsf, mv = CorlasP.CorLAS(
        SF, smys, smts, elastic_modulus, cvn,
        FLOW_DEFINITION, PIPE_DIAMETER, wt, length, depth,
        FLAW_SHAPE, FLAW_LOCATION, FD, FT,
    )
    g = err * failure_pressure - pressure
    return (g, int(g < 0), failure_pressure, qf, fsf, mv)


def limit_state(depth, critical_depth):
    return critical_depth - depth, int(depth > critical_depth)


def obs_D_t(value, error, shift, n):
    return Normal.Normal_Samples(value, error, shift, n)


def POD(depth, q_pod):
    return 1 - np.exp(-q_pod * depth)


def simulate_cracks_vectorized(
    d0, l0, action, dt, gs, rep1_joint, rep2_joint, life, df,
    Neq, C, dS, m, wt, Di, Qsf, Fsf, Mv,
    resample_growth_each_step=True,
):
    """Grow every active crack in one Monte Carlo joint realization."""
    d0 = np.asarray(d0, dtype=float)

    if action == 0:
        if rep1_joint > 0:
            rep1_new, rep2_new, factor = max(rep1_joint - dt, 0), rep2_joint, df[1]
        elif rep2_joint > 0:
            rep1_new, rep2_new, factor = rep1_joint, max(rep2_joint - dt, 0), df[2]
        else:
            rep1_new, rep2_new, factor = 0, 0, df[0]
    elif action == 1:
        rep1_new, rep2_new, factor = life[1], rep2_joint, df[1]
        d0 = np.full_like(d0, np.random.uniform(1.1, 1.4))
    else:
        rep1_new, rep2_new, factor = rep1_joint, life[2], df[2]
        d0 = np.full_like(d0, np.random.uniform(1.1, 1.4))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        geometry = (
            (Di / (2 * wt))
            * ((1 - (np.pi / 4) * d0 / (wt * Mv)) /
               (1 - (np.pi / 4) * d0 / wt))
            * np.sqrt(np.pi * Qsf * Fsf * d0)
        )
        if resample_growth_each_step:
            # Backward-compatible behavior from the original script.
            annual_g = abs(np.random.normal(MEAN_G, STD_G))
            annual_neq = abs(np.random.lognormal(LOG_MEAN_NEQ, LOG_STD_NEQ))
            annual_c = abs(np.random.normal(MU_C, STD_C))
        else:
            # Use this joint's sampled properties throughout the realization.
            annual_g = abs(float(np.ravel(gs)[0]))
            annual_neq = abs(float(Neq))
            annual_c = abs(float(C))
        growth = dt * (annual_neq * annual_c * dS**m * geometry**m + annual_g)

    return d0 + factor * growth, l0, rep1_new, rep2_new


def filter_initial_failures(crack_samples, pipe_samples, num_cracks_list):
    """Remove Monte Carlo realizations that have failed at time zero."""
    nx = len(pipe_samples)
    keep = np.ones(nx, dtype=bool)
    total_cracks = sum(num_cracks_list)
    for xk in range(nx):
        props = pipe_samples[xk]
        for crack in range(total_cracks):
            length = crack_samples[xk, crack]
            depth = crack_samples[xk, total_cracks + crack]
            _, leak = limit_state(depth, props[1])
            _, burst, *_ = limit_state_burst(
                BURST_VARIABLE_NAMES,
                [length, depth, props[1], props[4], props[5], props[6], props[7]],
                props[8], props[9],
            )
            if leak or burst:
                keep[xk] = False
                break
    return crack_samples[keep], pipe_samples[keep]


def initial_system_survival_mask(crack_samples, pipe_props_list, num_cracks_list):
    """Find realizations where none of the explicitly modelled joints fails at t=0."""
    nx = crack_samples.shape[0]
    total_cracks = sum(num_cracks_list)
    keep = np.ones(nx, dtype=bool)
    crack_offset = 0
    for joint_index, num_cracks in enumerate(num_cracks_list):
        props = pipe_props_list[joint_index]
        for xk in np.flatnonzero(keep):
            for local_crack in range(num_cracks):
                crack = crack_offset + local_crack
                length = crack_samples[xk, crack]
                depth = crack_samples[xk, total_cracks + crack]
                _, leak = limit_state(depth, props["WT"][xk])
                _, burst, *_ = limit_state_burst(
                    BURST_VARIABLE_NAMES,
                    [
                        length, depth, props["WT"][xk], props["SMYS"][xk],
                        props["SMTS"][xk], props["CVN"][xk], props["E"][xk],
                    ],
                    props["err"][xk], props["Pservice"][xk],
                )
                if leak or burst:
                    keep[xk] = False
                    break
        crack_offset += num_cracks
    return keep


# Backward-compatible name used in the original script.
def fun_filter(OriginSamples_X1, OriginSamples_X2, nx, num_cracks):
    cracks, props = filter_initial_failures(
        OriginSamples_X1[:nx], OriginSamples_X2[:nx], num_cracks
    )
    return cracks, props, len(props)


# ---------------------------------------------------------------------------
# System probability and plotting helpers
# ---------------------------------------------------------------------------
def calculate_system_pofs(POFl_list, POFb_list):
    """System failure for a series system from explicitly listed joint POFs."""
    return (
        1 - np.prod(1 - np.asarray(POFl_list, dtype=float)),
        1 - np.prod(1 - np.asarray(POFb_list, dtype=float)),
    )


def calculate_grouped_system_pofs(leak_pofs, burst_pofs, counts):
    """Same series-system calculation without repeating each joint in a list."""
    counts = np.asarray(counts, dtype=int)
    return (
        1 - np.prod((1 - np.asarray(leak_pofs)) ** counts),
        1 - np.prod((1 - np.asarray(burst_pofs)) ** counts),
    )


def combined_hazard(leak_hazard, burst_hazard, costs):
    """Cost-normalized threshold measure retained from the original model.

    H_combined = H_leak * Cf_leak / Cf_burst + H_burst
    """
    if costs["Cf_burst"] <= 0:
        raise ValueError("Cf_burst must be positive for the combined-hazard threshold.")
    return leak_hazard * costs["Cf_leak"] / costs["Cf_burst"] + burst_hazard


def _output_path(output_dir, filename):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _positive_for_log(values):
    """Map exact zeros to a plotting floor without altering stored results."""
    return np.maximum(np.asarray(values, dtype=float), 1e-16)


def plot_system_pofs(times, leak_pofs, burst_pofs, insp_times, approach_name, output_dir="."):
    plt.figure(figsize=(6, 4))
    plt.semilogy(times, _positive_for_log(leak_pofs), "b-",
                 label="Leakage POF", linewidth=2)
    plt.semilogy(times, _positive_for_log(burst_pofs), "r-",
                 label="Burst POF", linewidth=2)
    plt.axhline(0.001, color="k", linestyle="--", linewidth=1.5)
    plt.axhline(0.01, color="g", linestyle=":", linewidth=1.5)
    for inspection in insp_times:
        plt.axvline(inspection, color="g", linestyle=":", alpha=0.5, linewidth=1.5)
    plt.xlabel("Time (years)", fontname="Times New Roman", fontsize=14)
    plt.ylabel("Cumulative POF", fontname="Times New Roman", fontsize=14)
    plt.title(f"System POFs over Time - {approach_name}", fontsize=14)
    plt.legend(prop={"family": "Times New Roman", "size": 14})
    plt.grid(True)
    plt.savefig(_output_path(output_dir, f"System_POFs_{approach_name}.png"),
                dpi=300, bbox_inches="tight")


def plot_system_hazard_rates(times, leak_hazard, burst_hazard, approach_name, output_dir="."):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    plt.figure(figsize=(6, 4))
    plt.axhline(0.01, color="b", linestyle=":", linewidth=1.5)
    plt.axhline(0.001, color="r", linestyle="--", linewidth=1.5)
    plt.semilogy(times, _positive_for_log(leak_hazard), "b-",
                 label="Leakage", linewidth=2)
    plt.semilogy(times, _positive_for_log(burst_hazard), "r-",
                 label="Burst", linewidth=2)
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel("Annual POF (per km-year)", fontsize=14)
    plt.legend(prop={"family": "Times New Roman", "size": 14})
    plt.grid(True)
    plt.savefig(_output_path(output_dir, f"System_Hazard_Rates_{approach_name}.png"),
                dpi=600, bbox_inches="tight")


def plot_joint_pofs(
    times, joint_names, joint_leak_pofs, joint_burst_pofs,
    insp_times, approach_name, output_dir=".",
):
    """Plot the separate POF histories of explicitly modelled joints."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for joint, name in enumerate(joint_names):
        axes[0].semilogy(times, _positive_for_log(joint_leak_pofs[joint]), label=name)
        axes[1].semilogy(times, _positive_for_log(joint_burst_pofs[joint]), label=name)
    for axis, title in zip(axes, ("Leakage POF", "Burst POF")):
        for inspection in insp_times:
            axis.axvline(inspection, color="g", linestyle=":", alpha=0.4)
        axis.set_title(title)
        axis.set_xlabel("Time (years)")
        axis.grid(True)
    axes[0].set_ylabel("Cumulative POF")
    axes[1].legend(fontsize=9)
    figure.suptitle(f"Joint-level POFs - {approach_name}")
    figure.tight_layout()
    figure.savefig(
        _output_path(output_dir, f"Joint_POFs_{approach_name}.png"),
        dpi=300, bbox_inches="tight",
    )


def plot_cost_history(times, cpp, cff, crr, cii, output_dir="."):
    plt.figure(figsize=(10, 6))
    plt.plot(times, cpp, "bo-", label="Total", linewidth=2)
    plt.plot(times, cff, "r-", label="Failure", linewidth=2)
    plt.plot(times, crr, "g-", label="Repair", linewidth=2)
    plt.plot(times, cii, "y-", label="Inspection", linewidth=2)
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel("Expected cost (million USD)", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig(_output_path(output_dir, "Cost_history.png"), dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# One planning step for one representative joint
# ---------------------------------------------------------------------------
def Joint_planning_step(
    t, k, L, D, gs, rep1, rep2, ptr, nxt, births, active,
    props, costs, sim, S_err, q_pod, Qsf_arr, Fsf_arr, M_v_arr,
    insp_times, failed_locations, birth_rate, action_arr,
):
    nx = L.shape[1]
    expected_failure = expected_repair = 0.0
    leak_count = burst_count = repair_count = 0
    wt, neq, c_paris = props["WT"], props["Neq"], props["C"]
    err, pressure = props["err"], props["Pservice"]
    rate = costs["rate"]
    is_inspection = bool(np.any(np.isclose(insp_times, t)))

    for xk in range(nx):
        while ptr[xk] < len(births[xk]) and births[xk][ptr[xk]] <= t:
            if failed_locations[xk]:
                loc = failed_locations[xk].pop()
            elif nxt[xk] < L.shape[0]:
                loc = nxt[xk]
                nxt[xk] += 1
            else:
                ptr[xk] += 1
                continue
            ptr[xk] += 1
            active[xk].append(loc)
            L[loc, xk] = np.random.uniform(40, 45)
            D[loc, xk] = np.random.uniform(0.1, 0.4)

        leak = burst = False
        failed_cracks = []
        for loc in active[xk]:
            _, burst_i, _, qsf, fsf, mv = limit_state_burst(
                BURST_VARIABLE_NAMES,
                [L[loc, xk], D[loc, xk], wt[xk], props["SMYS"][xk],
                 props["SMTS"][xk], props["CVN"][xk], props["E"][xk]],
                err[xk], pressure[xk],
            )
            Qsf_arr[loc, xk], Fsf_arr[loc, xk], M_v_arr[loc, xk] = qsf, fsf, mv
            if burst_i:
                failed_cracks = active[xk].copy()
                burst = True
                break

        if not burst:
            for loc in active[xk]:
                if limit_state(D[loc, xk], wt[xk])[1]:
                    failed_cracks = active[xk].copy()
                    leak = True
                    break

        expected_failure += (
            costs["Cf_leak"] * int(leak) + costs["Cf_burst"] * int(burst)
        ) / (1 + rate) ** t
        leak_count += int(leak)
        burst_count += int(burst)

        if leak or burst:
            for loc in failed_cracks:
                L[loc, xk] = D[loc, xk] = 0.0
            failed_locations[xk].extend(failed_cracks)
            active[xk].clear()
            births[xk] = reset_births_after_failure(t, sim["T"], birth_rate)
            ptr[xk] = 0
            action_arr[xk, k] = 0
            continue

        action = 0
        if is_inspection:
            for loc in active[xk]:
                if np.random.rand() > POD(D[loc, xk], q_pod):
                    continue
                measured_d = obs_D_t(D[loc, xk], S_err[0], 0, 1)[0]
                measured_l = obs_D_t(L[loc, xk], S_err[1], 0, 1)[0]
                _, criterion_1, *_ = limit_state_burst(
                    BURST_VARIABLE_NAMES,
                    [measured_l, measured_d, MEAN_WT, MEAN_SMYS,
                     MEAN_SMTS, MEAN_CVN, MEAN_E],
                    1, sim["FPR_rep1"] * MOP,
                )
                if measured_d < sim["D_rep1"] and not criterion_1:
                    continue
                action = 1
                for loc2 in active[xk]:
                    _, criterion_2, *_ = limit_state_burst(
                        BURST_VARIABLE_NAMES,
                        [L[loc2, xk], D[loc2, xk], MEAN_WT, MEAN_SMYS,
                         MEAN_SMTS, MEAN_CVN, MEAN_E],
                        1, sim["FPR_rep2"] * MOP,
                    )
                    if D[loc2, xk] >= sim["D_rep2"] or criterion_2:
                        action = 2
                        break
                break

            repair_count += int(action > 0)
            expected_repair += (
                costs["Cr1"] * int(action > 0) + costs["Cr2"] * int(action == 2)
            ) / (1 + rate) ** t

        action_arr[xk, k] = action
        locations = np.asarray(active[xk], dtype=int)
        if locations.size:
            D[locations, xk], L[locations, xk], rep1[xk], rep2[xk] = (
                simulate_cracks_vectorized(
                    D[locations, xk], L[locations, xk], action, sim["dt"],
                    gs[locations, xk], rep1[xk], rep2[xk],
                    sim["lifetime_arr"], sim["df_arr"], neq[xk], c_paris[xk],
                    sim["dS"], sim["m"], wt[xk], sim["Di"],
                    Qsf_arr[locations, xk], Fsf_arr[locations, xk],
                    M_v_arr[locations, xk],
                    sim.get("resample_growth_each_step", True),
                )
            )

    return (
        (expected_failure + expected_repair) / nx,
        expected_failure / nx,
        expected_repair / nx,
        0.0,
        leak_count,
        burst_count,
        L, D, gs, rep1, rep2, ptr, nxt, births, active,
        Qsf_arr, Fsf_arr, M_v_arr, failed_locations,
        repair_count / nx,
        action_arr,
    )


def _copy_array_lists(*array_lists):
    return tuple([[array.copy() for array in arrays] for arrays in array_lists])


def _copy_nested_lists(*nested_lists):
    return tuple(copy.deepcopy(item) for item in nested_lists)


def copy_array_list(*array_lists):
    copied = _copy_array_lists(*array_lists)
    return copied[0] if len(copied) == 1 else copied


def copy_nested_list(*nested_lists):
    copied = _copy_nested_lists(*nested_lists)
    return copied[0] if len(copied) == 1 else copied


def _make_checkpoint(array_groups, nested_groups):
    return _copy_array_lists(*array_groups), _copy_nested_lists(*nested_groups)


def _restore_checkpoint(checkpoint):
    arrays, nested = checkpoint
    return _copy_array_lists(*arrays), _copy_nested_lists(*nested)


def _schedule_from_parameters(approach, T, dt, sim_params):
    if approach == "equidistant":
        if "inspection_times" in sim_params:
            schedule = np.asarray(sim_params["inspection_times"], dtype=float)
        else:
            interval = sim_params["inspection_interval"]
            if interval <= 0:
                raise ValueError("inspection_interval must be positive.")
            schedule = np.arange(interval, T + dt, interval, dtype=float)
        return schedule[(schedule >= 0) & (schedule <= T)]
    if approach == "threshold":
        return np.array([], dtype=float)
    raise ValueError("approach must be 'equidistant' or 'threshold'.")


# ---------------------------------------------------------------------------
# Full system simulation
# ---------------------------------------------------------------------------
def run_simulation(
    approach, T, dt, nx, num_cracks_list, joint_counts, pipe_props_list,
    costs, sim_params, S_err, q_pod, birth_rate,
    *, track_debug=True, debug_sample_idx=10, debug_crack_count=None,
    joint_definitions=None, property_seed=0,
    crack_length_std=15.6, crack_depth_std=0.74, debug_joint=None,
):
    """Run one inspection-policy simulation.

    ``joint_counts`` maps initial crack count to the number of statistically
    represented joints, e.g. ``{0: 97, 1: 2, 2: 1}``.

    Alternatively, ``joint_definitions`` explicitly describes every joint and
    permits joint-specific properties, crack means, and birth rates. The old
    positional arguments remain unchanged for backward compatibility.
    """
    start = time.time()
    flexible_mode = joint_definitions is not None
    if flexible_mode:
        joints = [dict(joint) for joint in joint_definitions]
        if not joints:
            raise ValueError("joint_definitions must contain at least one joint.")
        joint_names = [joint.get("name", f"Joint {i + 1}") for i, joint in enumerate(joints)]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Every explicitly modelled joint must have a unique name.")
        crack_specs_list = [list(joint.get("cracks", [])) for joint in joints]
        num_cracks_list = [len(cracks) for cracks in crack_specs_list]
        raw_counts = [joint.get("count", 1) for joint in joints]
        if any(int(count) != count or count < 1 for count in raw_counts):
            raise ValueError("Each joint/group count must be a positive integer.")
        group_counts = np.asarray(raw_counts, dtype=int)
        birth_rates = np.asarray(
            [joint.get("birth_rate", birth_rate) for joint in joints], dtype=float
        )
        if np.any(birth_rates < 0):
            raise ValueError("Joint birth rates cannot be negative.")
        pipe_props_list = sample_individual_joint_properties(
            joints, nx, seed=property_seed
        )
    else:
        original_groups = list(num_cracks_list)
        props_by_group = dict(zip(original_groups, pipe_props_list))
        active_groups = [c for c in original_groups if joint_counts.get(c, 0) > 0]
        if not active_groups:
            raise ValueError("joint_counts must contain at least one positive count.")
        num_cracks_list = active_groups
        group_counts = np.asarray([joint_counts[c] for c in active_groups], dtype=int)
        pipe_props_list = [
            {key: (np.asarray(value).copy() if key != "num_initial_cracks" else value)
             for key, value in props_by_group[c].items()}
            for c in active_groups
        ]
        joint_names = [f"{cracks}-crack group" for cracks in num_cracks_list]
        crack_specs_list = [None] * len(num_cracks_list)
        birth_rates = np.full(len(num_cracks_list), birth_rate, dtype=float)

    sim_params = dict(sim_params)
    sim_params.update({"T": T, "dt": dt})
    insp_times = _schedule_from_parameters(approach, T, dt, sim_params)
    times = np.arange(0, T + dt, dt)
    nt = len(times)

    births_list = [
        [generate_initiation_times(generate_num_cracks(T, joint_birth_rate),
                                   T, joint_birth_rate)
         for _ in range(nx)]
        for joint_birth_rate in birth_rates
    ]
    total_initial_cracks = sum(num_cracks_list)
    crack_samples = np.zeros((nx, 2 * total_initial_cracks))
    for j, cracks in enumerate(num_cracks_list):
        if cracks:
            crack_input = crack_specs_list[j] if flexible_mode else cracks
            # Use a separate stream from the joint-property samples so that
            # crack sizes are not artificially correlated with pipe properties.
            crack_seed = property_seed + 1000 + j if flexible_mode else 0
            lengths, depths = sample_initial_cracks(
                crack_input, nx, seed=crack_seed,
                length_std=crack_length_std, depth_std=crack_depth_std,
            )
            start_col = sum(num_cracks_list[:j])
            crack_samples[:, start_col:start_col + cracks] = lengths.T
            crack_samples[:, total_initial_cracks + start_col:
                          total_initial_cracks + start_col + cracks] = depths.T

    survival_mask = initial_system_survival_mask(
        crack_samples, pipe_props_list, num_cracks_list
    )
    retained_indices = np.flatnonzero(survival_mask)
    crack_samples = crack_samples[survival_mask]
    nx_filtered = len(retained_indices)
    if nx_filtered == 0:
        raise RuntimeError("All Monte Carlo samples failed at time zero.")
    for props in pipe_props_list:
        for name in PROPERTY_NAMES:
            props[name] = np.asarray(props[name])[survival_mask]
    births_list = [
        [births[x] for x in retained_indices]
        for births in births_list
    ]
    print(f"Sampling/filtering time: {time.time() - start:.2f} s; "
          f"retained {nx_filtered}/{nx} samples")

    worst = int(np.ceil(exp_num(T, float(np.max(birth_rates))))) + 40
    L_states, D_states, g_states = [], [], []
    rep1, rep2, qsf_states, fsf_states, mv_states = [], [], [], [], []
    ptr, nxt, active, failed_locations, action_arr = [], [], [], [], []
    for j, cracks in enumerate(num_cracks_list):
        total_slots = cracks + worst
        lengths = np.zeros((total_slots, nx_filtered))
        depths = np.zeros_like(lengths)
        if cracks:
            start_col = sum(num_cracks_list[:j])
            lengths[:cracks] = crack_samples[:, start_col:start_col + cracks].T
            depths[:cracks] = crack_samples[
                :, total_initial_cracks + start_col:
                total_initial_cracks + start_col + cracks
            ].T
        L_states.append(lengths)
        D_states.append(depths)
        g_states.append(np.full_like(lengths, pipe_props_list[j]["alpha"]))
        rep1.append(np.zeros(nx_filtered))
        rep2.append(np.zeros(nx_filtered))
        qsf_states.append(np.zeros_like(lengths))
        fsf_states.append(np.zeros_like(lengths))
        mv_states.append(np.zeros_like(lengths))
        ptr.append([0] * nx_filtered)
        nxt.append([cracks] * nx_filtered)
        active.append([list(range(cracks)) for _ in range(nx_filtered)])
        failed_locations.append([[] for _ in range(nx_filtered)])
        action_arr.append(np.zeros((nx_filtered, nt), dtype=np.int8))

    if debug_joint is not None:
        debug_group = joint_names.index(debug_joint) if isinstance(debug_joint, str) else int(debug_joint)
        if not 0 <= debug_group < len(joint_names):
            raise ValueError("debug_joint is outside the available joint range.")
    else:
        if debug_crack_count is None:
            debug_group = next(
                (i for i, cracks in enumerate(num_cracks_list) if cracks > 0), 0
            )
            debug_crack_count = num_cracks_list[debug_group]
        else:
            debug_group = num_cracks_list.index(debug_crack_count)
    debug_sample = min(debug_sample_idx, nx_filtered - 1)
    if track_debug:
        total_slots = L_states[debug_group].shape[0]
        L_debug = np.zeros((total_slots, nt + 1))
        D_debug = np.zeros_like(L_debug)
        L_debug[:, 0] = L_states[debug_group][:, debug_sample]
        D_debug[:, 0] = D_states[debug_group][:, debug_sample]
    else:
        L_debug = D_debug = None

    system_leak_pofs = np.zeros(nt)
    system_burst_pofs = np.zeros(nt)
    system_leak_hazard = np.zeros(nt)
    system_burst_hazard = np.zeros(nt)
    cpp, cff, crr, cii = (np.zeros(nt) for _ in range(4))
    cpp_cracked, cpp_crack_free, num_repairs = (np.zeros(nt) for _ in range(3))
    cumulative_leaks = np.zeros((len(num_cracks_list), nt))
    cumulative_bursts = np.zeros_like(cumulative_leaks)
    unsafe_times = []
    previous_checkpoint = None

    arrays_for_checkpoint = lambda: (
        L_states, D_states, g_states, rep1, rep2,
        qsf_states, fsf_states, mv_states,
    )
    nested_for_checkpoint = lambda: (
        ptr, nxt, births_list, active, failed_locations,
    )

    k = 0
    while k < nt:
        current_checkpoint = None
        if approach == "threshold":
            current_checkpoint = _make_checkpoint(
                arrays_for_checkpoint(), nested_for_checkpoint()
            )
        t = times[k]
        if np.any(np.isclose(insp_times, t)):
            cii[k] = costs["total_insp_cost"] / (1 + costs["rate"]) ** t

        group_leak_pofs, group_burst_pofs = [], []
        for j, (cracks, count) in enumerate(zip(num_cracks_list, group_counts)):
            result = Joint_planning_step(
                t, k, L_states[j], D_states[j], g_states[j], rep1[j], rep2[j],
                ptr[j], nxt[j], births_list[j], active[j], pipe_props_list[j],
                costs, sim_params, S_err, q_pod, qsf_states[j], fsf_states[j],
                mv_states[j], insp_times, failed_locations[j], birth_rates[j],
                action_arr[j],
            )
            (
                joint_cpp, joint_cff, joint_crr, _, leaks, bursts,
                L_states[j], D_states[j], g_states[j], rep1[j], rep2[j],
                ptr[j], nxt[j], births_list[j], active[j], qsf_states[j],
                fsf_states[j], mv_states[j], failed_locations[j],
                expected_repairs, action_arr[j],
            ) = result
            cumulative_leaks[j, k] = (cumulative_leaks[j, k - 1] if k else 0) + leaks
            cumulative_bursts[j, k] = (cumulative_bursts[j, k - 1] if k else 0) + bursts
            group_leak_pofs.append(cumulative_leaks[j, k] / nx_filtered)
            group_burst_pofs.append(cumulative_bursts[j, k] / nx_filtered)

            cpp[k] += count * joint_cpp
            cff[k] += count * joint_cff
            crr[k] += count * joint_crr
            num_repairs[k] += count * expected_repairs
            if cracks:
                cpp_cracked[k] += count * joint_cpp
            else:
                cpp_crack_free[k] += count * joint_cpp
            if track_debug and j == debug_group:
                L_debug[:, k + 1] = L_states[j][:, debug_sample]
                D_debug[:, k + 1] = D_states[j][:, debug_sample]

        cpp[k] += cii[k]
        system_leak_pofs[k], system_burst_pofs[k] = calculate_grouped_system_pofs(
            group_leak_pofs, group_burst_pofs, group_counts
        )
        if k:
            previous_leak, previous_burst = system_leak_pofs[k - 1], system_burst_pofs[k - 1]
            system_leak_hazard[k] = (
                (system_leak_pofs[k] - previous_leak) / (1 - previous_leak)
                if previous_leak < 1 else 0
            ) / costs["pipe_length_km"]
            system_burst_hazard[k] = (
                (system_burst_pofs[k] - previous_burst) / (1 - previous_burst)
                if previous_burst < 1 else 0
            ) / costs["pipe_length_km"]

        if system_burst_hazard[k] > 1e-3 or system_leak_hazard[k] > 1e-2:
            unsafe_times.append((t, system_burst_hazard[k], system_burst_pofs[k]))

        threshold_exceeded = (
            approach == "threshold"
            and combined_hazard(system_leak_hazard[k], system_burst_hazard[k], costs)
            > sim_params["threshold"]
        )
        proposed_time = t - dt
        already_scheduled = np.any(np.isclose(insp_times, proposed_time))
        if threshold_exceeded and proposed_time >= 0 and not already_scheduled:
            inspection_k = k - 1
            if previous_checkpoint is None:
                raise RuntimeError("Missing threshold rollback checkpoint.")
            restored_arrays, restored_nested = _restore_checkpoint(previous_checkpoint)
            (
                L_states, D_states, g_states, rep1, rep2,
                qsf_states, fsf_states, mv_states,
            ) = restored_arrays
            ptr, nxt, births_list, active, failed_locations = restored_nested
            for output in (
                cpp, cff, crr, cii, cpp_cracked, cpp_crack_free, num_repairs,
                system_leak_pofs, system_burst_pofs,
                system_leak_hazard, system_burst_hazard,
            ):
                output[inspection_k] = 0
            cumulative_leaks[:, inspection_k] = 0
            cumulative_bursts[:, inspection_k] = 0
            for actions in action_arr:
                actions[:, inspection_k] = 0
            unsafe_times = [row for row in unsafe_times if row[0] < proposed_time]
            insp_times = np.sort(np.append(insp_times, proposed_time))
            print(f"Inspection scheduled at t = {proposed_time:.1f} years")
            k = inspection_k
            continue

        if approach == "threshold":
            previous_checkpoint = current_checkpoint
        k += 1

    debug_data = {
        "times": times,
        "crack_depths": D_debug,
        "crack_lengths": L_debug,
        "active_cracks": active[debug_group][debug_sample] if track_debug else None,
        "wall_thickness": pipe_props_list[debug_group]["WT"][debug_sample]
        if track_debug else None,
        "inspection_times": insp_times,
        "sample_idx": debug_sample,
        "crack_count_group": num_cracks_list[debug_group],
        "joint_name": joint_names[debug_group],
        "joint_names": joint_names,
        "joint_leak_pofs": cumulative_leaks / nx_filtered,
        "joint_burst_pofs": cumulative_bursts / nx_filtered,
    }
    is_safe = len(unsafe_times) == 0
    return (
        times, system_leak_pofs, system_burst_pofs,
        system_leak_hazard, system_burst_hazard, insp_times,
        cpp, cff, crr, cii, is_safe, unsafe_times,
        cpp_cracked, cpp_crack_free, num_repairs, action_arr, debug_data,
    )


def run_flexible_simulation(
    approach, T, dt, nx, joints, costs, sim_params, S_err, q_pod,
    *, default_birth_rate=0.0, property_seed=0,
    crack_length_std=15.6, crack_depth_std=0.74,
    track_debug=True, debug_sample_idx=10, debug_joint=None,
):
    """Convenient interface for individually specified joints or joint groups.

    Each record represents one joint by default. An optional ``count`` greater
    than one efficiently represents statistically identical joints, which is
    useful for a large group of initially crack-free joints.
    """
    return run_simulation(
        approach, T, dt, nx, [], {}, [], costs, sim_params, S_err, q_pod,
        default_birth_rate,
        track_debug=track_debug,
        debug_sample_idx=debug_sample_idx,
        joint_definitions=joints,
        property_seed=property_seed,
        crack_length_std=crack_length_std,
        crack_depth_std=crack_depth_std,
        debug_joint=debug_joint,
    )


def plot_crack_growth_debug(debug_data, output_dir="."):
    depths = debug_data["crack_depths"]
    lengths = debug_data["crack_lengths"]
    if depths is None:
        print("Debug tracking was disabled.")
        return
    times = debug_data["times"]
    inspections = debug_data["inspection_times"]

    plt.figure(figsize=(6, 4))
    for crack in range(depths.shape[0]):
        if np.any(depths[crack] > 0):
            plt.plot(times, depths[crack, :-1], label=f"Crack {crack + 1}")
    plt.axhline(debug_data["wall_thickness"], color="r", linestyle="--",
                label="Wall thickness")
    for i, inspection in enumerate(inspections):
        plt.axvline(inspection, color="g", linestyle=":", alpha=0.7,
                    label="Inspection" if i == 0 else None)
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel("Crack depth (mm)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(_output_path(output_dir, "Debug_depth.png"), dpi=600,
                bbox_inches="tight")

    plt.figure(figsize=(6, 4))
    for crack in range(lengths.shape[0]):
        if np.any(lengths[crack] > 0):
            plt.plot(times, lengths[crack, :-1], label=f"Crack {crack + 1}")
    for inspection in inspections:
        plt.axvline(inspection, color="g", linestyle=":", alpha=0.7)
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel("Crack length (mm)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(_output_path(output_dir, "Debug_length.png"), dpi=600,
                bbox_inches="tight")


def main():
    start = time.time()
    approach = "equidistant"
    T, dt, nx = 50, 1, 100
    joint_counts = {1: 1}  # Example: {0: 97, 1: 2, 2: 1}
    crack_groups = list(joint_counts)
    pipe_props = sample_pipe_properties(crack_groups, nx)
    costs = {
        "Cr1": 1.0, "Cr2": 0.0, "Cf_leak": 1.0, "Cf_burst": 1.0,
        "total_insp_cost": 0.025, "pipe_length_km": 1.0, "rate": 0.0,
    }
    sim = {
        "dS": 2, "m": 3, "Di": DI,
        "FPR_rep1": 1.25, "D_rep1": 0.5 * MEAN_WT,
        "FPR_rep2": 1.1, "D_rep2": 0.7 * MEAN_WT,
        "lifetime_arr": [0, 5, 5], "df_arr": [1, 0, 0],
        "threshold": 0.0003, "inspection_interval": 14,
    }
    result = run_simulation(
        approach, T, dt, nx, crack_groups, joint_counts, pipe_props,
        costs, sim, [0.37, 15.6 / 2], 3, 0.00008,
    )
    (
        times, leak_pof, burst_pof, leak_hazard, burst_hazard, inspections,
        cpp, cff, crr, cii, is_safe, unsafe_times,
        cpp_cracked, cpp_crack_free, num_repairs, actions, debug_data,
    ) = result

    plot_system_pofs(times, leak_pof, burst_pof, inspections, approach)
    plot_system_hazard_rates(times, leak_hazard, burst_hazard, approach)
    plot_joint_pofs(
        times, debug_data["joint_names"], debug_data["joint_leak_pofs"],
        debug_data["joint_burst_pofs"], inspections, approach,
    )
    plot_cost_history(times, cpp, cff, crr, cii)
    plot_crack_growth_debug(debug_data)
    np.savez(
        f"system_results_{approach}.npz", times=times, leak_pofs=leak_pof,
        burst_pofs=burst_pof, insp_times=inspections, approach=approach,
        cpp=cpp, is_safe=is_safe, unsafe_times=np.asarray(unsafe_times),
    )
    print(f"Approach: {approach}; inspections: {inspections}")
    print(f"Total/Failure/Repair/Inspection cost: "
          f"{cpp.sum():.4f} / {cff.sum():.4f} / {crr.sum():.4f} / {cii.sum():.4f}")
    print(f"Cracked/crack-free cost: {cpp_cracked.sum():.4f} / {cpp_crack_free.sum():.4f}")
    print(f"Expected repairs: {num_repairs.sum():.4f}; safe: {is_safe}")
    for cracks, group_actions in zip(crack_groups, actions):
        print(f"Group with {cracks} initial crack(s): "
              f"action 1={np.count_nonzero(group_actions == 1)}, "
              f"action 2={np.count_nonzero(group_actions == 2)}")
    print(f"Execution time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
