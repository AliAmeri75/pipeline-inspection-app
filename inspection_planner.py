#!/usr/bin/env python3
"""Small Streamlit interface for inspection-interval optimization."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st
from packaging.version import Version

from Simp_Sys_Thresh_J3_optimized import (
    DEFAULT_PROPERTY_SPECS,
    EXCAVATION_DECISION_MODE,
    ILI_DECISION_MODE,
    MEAN_WT,
    PROPERTY_NAMES,
    run_flexible_simulation,
)
from planning_analysis import calculate_value_of_information, select_optimal_result


APP_DIR = Path(__file__).resolve().parent
COLORS = {
    "navy": "#10243d",
    "teal": "#0f766e",
    "blue": "#2563a8",
    "amber": "#c87818",
    "coral": "#c94f3d",
    "violet": "#7556a6",
}
MODERN_CHART_WIDTH = Version(st.__version__) >= Version("1.51.0")
DEFAULT_CRACK_BIRTH_RATE = 0.00008
SAMPLE_COLUMNS = [
    "joint_id", "count", "num_cracks", "mean_lengths_mm", "mean_depths_mm",
    "WT_mean", "WT_std", "Pservice_mean", "Pservice_std", "birth_rate",
]
SAMPLE_DATA = pd.DataFrame(
    [
        {
            "joint_id": "Joint 1",
            "count": 1,
            "num_cracks": 1,
            "mean_lengths_mm": "40",
            "mean_depths_mm": "1.0",
            "WT_mean": 6.35,
            "WT_std": 0.60,
            "Pservice_mean": 936.0,
            "Pservice_std": 20.03,
            # Blank means "use the visible sidebar rate". Users can enter a
            # value here when a particular joint needs its own rate.
            "birth_rate": np.nan,
        }
    ],
    columns=SAMPLE_COLUMNS,
)


st.markdown(
    """
    <style>
    :root { --ink:#10243d; --teal:#0f766e; --paper:#f4f2ec; --line:#d9dfdf; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(244,242,236,.92); }
    [data-testid="stSidebar"] { background:#eef0ed; border-right:1px solid #cad2d2; }
    .hero { padding:1.05rem 1.3rem; margin:-.3rem 0 1rem; color:white;
            background:var(--ink); border-bottom:4px solid var(--teal); }
    .hero small { color:#8fd1c9; font-weight:800; letter-spacing:.13em;
                  text-transform:uppercase; }
    .hero h1 { margin:.18rem 0; color:white; font-family:Georgia,serif;
               font-size:2rem; font-weight:500; }
    .hero p { margin:.25rem 0 0; color:#d9e3ea; font-size:.86rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line);
                                  border-top:3px solid var(--teal); padding:.85rem; }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        background:white; border:1px solid var(--line); padding:.25rem;
    }
    .note { padding:.75rem .9rem; background:#e8efed; border-left:3px solid var(--teal);
            color:#425461; font-size:.84rem; margin:.4rem 0 .9rem; }
    .stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip()


def parse_float_list(value, field_name, joint_name):
    if is_blank(value):
        return []
    try:
        return [float(item.strip()) for item in str(value).split(";") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            f"{joint_name}: {field_name} must contain numbers separated by semicolons."
        ) from exc


def row_property(row, name):
    mean_column = f"{name}_mean"
    if mean_column not in row.index or is_blank(row[mean_column]):
        return None
    default = DEFAULT_PROPERTY_SPECS[name]
    std_value = row.get(f"{name}_std", default["std"])
    distribution = row.get(f"{name}_distribution", default["distribution"])
    return {
        "mean": float(row[mean_column]),
        "std": default["std"] if is_blank(std_value) else float(std_value),
        "distribution": default["distribution"] if is_blank(distribution) else str(distribution),
    }


def table_to_joints(table, growth, default_birth_rate, extra_crack_free):
    """Convert the editable one-row-per-joint table to model input records."""
    joints = []
    names = set()
    for row_number, (_, row) in enumerate(table.iterrows(), start=1):
        name = str(row.get("joint_id", "")).strip()
        if not name:
            raise ValueError(f"Row {row_number}: joint_id is required.")
        if name in names:
            raise ValueError(f"joint_id '{name}' appears more than once.")
        names.add(name)

        count_value = row.get("count", 1)
        count = 1 if is_blank(count_value) else int(count_value)
        cracks_value = row.get("num_cracks", 0)
        num_cracks = 0 if is_blank(cracks_value) else int(cracks_value)
        if count < 1 or num_cracks < 0:
            raise ValueError(f"{name}: count must be positive and num_cracks cannot be negative.")

        lengths = parse_float_list(row.get("mean_lengths_mm"), "mean_lengths_mm", name)
        depths = parse_float_list(row.get("mean_depths_mm"), "mean_depths_mm", name)
        if len(lengths) != num_cracks or len(depths) != num_cracks:
            raise ValueError(
                f"{name}: num_cracks={num_cracks}, but {len(lengths)} lengths and "
                f"{len(depths)} depths were provided."
            )
        if any(value <= 0 for value in lengths + depths):
            raise ValueError(f"{name}: crack length and depth means must be positive.")

        properties = copy.deepcopy(growth)
        for property_name in PROPERTY_NAMES:
            override = row_property(row, property_name)
            if override is not None:
                properties[property_name] = override
        birth_value = row.get("birth_rate", default_birth_rate)
        birth_rate = default_birth_rate if is_blank(birth_value) else float(birth_value)
        if birth_rate < 0:
            raise ValueError(f"{name}: birth_rate cannot be negative.")

        joints.append(
            {
                "name": name,
                "count": count,
                "cracks": [
                    {"mean_length": length, "mean_depth": depth}
                    for length, depth in zip(lengths, depths)
                ],
                "properties": properties,
                "birth_rate": birth_rate,
            }
        )

    if extra_crack_free:
        joints.append(
            {
                "name": "Additional crack-free joints",
                "count": int(extra_crack_free),
                "cracks": [],
                "properties": copy.deepcopy(growth),
                "birth_rate": default_birth_rate,
            }
        )
    if not joints:
        raise ValueError("Provide at least one joint or one crack-free joint group.")
    return joints


def unpack_result(raw, interval, decision_mode):
    (
        times, leak_pof, burst_pof, leak_hazard, burst_hazard, inspections,
        cpp, cff, crr, cii, is_safe, unsafe_times,
        cpp_cracked, cpp_crack_free, repairs, actions, debug,
    ) = raw
    return {
        "interval": interval,
        "decision_mode": decision_mode,
        "times": times,
        "leak_pof": leak_pof,
        "burst_pof": burst_pof,
        "leak_hazard": leak_hazard,
        "burst_hazard": burst_hazard,
        "inspections": inspections,
        "cpp": cpp,
        "total_cpp": float(np.sum(cpp)),
        "failure_cost": float(np.sum(cff)),
        "repair_cost": float(np.sum(crr)),
        "inspection_cost": float(np.sum(cii)),
        "is_safe": is_safe,
        "unsafe_times": unsafe_times,
        "expected_repairs": float(np.sum(repairs)),
        "joint_names": debug["joint_names"],
        "joint_leak_pofs": debug["joint_leak_pofs"],
        "joint_burst_pofs": debug["joint_burst_pofs"],
    }


def run_interval_study(joints, intervals, model, decision_mode):
    method_label = (
        "prior excavation"
        if decision_mode == EXCAVATION_DECISION_MODE
        else "ILI-informed inspection"
    )
    results = []
    progress = st.progress(0, text=f"Preparing {method_label} simulations…")
    log = io.StringIO()
    for index, interval in enumerate(intervals):
        sim = copy.deepcopy(model["sim"])
        sim["inspection_interval"] = interval
        np.random.seed(model["seed"])
        progress.progress(
            index / len(intervals),
            text=f"Testing {method_label}: {interval}-year interval…",
        )
        with contextlib.redirect_stdout(log):
            raw = run_flexible_simulation(
                "equidistant",
                model["horizon"],
                1,
                model["samples"],
                joints,
                model["costs"],
                sim,
                [model["depth_error"], model["length_error"]],
                model["q_pod"],
                default_birth_rate=model["birth_rate"],
                property_seed=model["seed"],
                crack_length_std=model["initial_length_std"],
                crack_depth_std=model["initial_depth_std"],
                track_debug=False,
                decision_mode=decision_mode,
            )
        results.append(unpack_result(raw, interval, decision_mode))
    progress.progress(1.0, text=f"{method_label.capitalize()} optimization complete")
    return results, log.getvalue()


def base_figure(title, x_title, y_title):
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "font": {"family": "Georgia", "size": 20}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=430,
        margin={"l": 60, "r": 25, "t": 70, "b": 55},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        font={"family": "Arial", "color": COLORS["navy"]},
        legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8ebea")
    figure.update_yaxes(showgrid=True, gridcolor="#e8ebea")
    return figure


def show_figure(figure, *, key=None, selectable=False):
    width_argument = (
        {"width": "stretch"}
        if MODERN_CHART_WIDTH
        else {"use_container_width": True}
    )
    chart_arguments = {
        "config": {"displaylogo": False},
        **width_argument,
    }
    if selectable:
        chart_arguments.update(
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    return st.plotly_chart(figure, **chart_arguments)


def _safety_label(result):
    return "Safe" if result["is_safe"] else "Unsafe"


def _cost_customdata(results):
    return np.asarray(
        [
            [
                result["total_cpp"],
                result["failure_cost"],
                result["repair_cost"],
                result["inspection_cost"],
                _safety_label(result),
            ]
            for result in results
        ],
        dtype=object,
    )


def _cost_hover_template(total_label):
    return (
        "<b>Interval: %{x} years</b><br>"
        f"{total_label}: $%{{customdata[0]:.6f}}M<br>"
        "Cf (failure): $%{customdata[1]:.6f}M<br>"
        "Cr (repair): $%{customdata[2]:.6f}M<br>"
        "Ci (inspection): $%{customdata[3]:.6f}M<br>"
        "Safety: <b>%{customdata[4]}</b><extra></extra>"
    )


def cost_figure(results, title, interval_label, total_label):
    figure = base_figure(
        title,
        interval_label,
        "Expected cost (million USD)",
    )
    styles = [
        (total_label, "total_cpp", COLORS["navy"]),
        ("Failure cost, Cf", "failure_cost", COLORS["coral"]),
        ("Repair cost, Cr", "repair_cost", COLORS["teal"]),
        ("Inspection cost, Ci", "inspection_cost", COLORS["amber"]),
    ]
    customdata = _cost_customdata(results)
    marker_symbols = ["circle" if result["is_safe"] else "x" for result in results]
    for label, key, color in styles:
        figure.add_trace(
            go.Scatter(
                x=[result["interval"] for result in results],
                y=[result[key] for result in results],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.6},
                marker={"size": 8, "symbol": marker_symbols},
                customdata=customdata,
                hovertemplate=_cost_hover_template(total_label),
            )
        )
    figure.update_layout(hovermode="closest", clickmode="event+select")
    return figure


def _selected_points(event):
    """Read selected Plotly points across Streamlit's dict and attribute APIs."""

    if event is None:
        return []
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    return list(points or [])


def _result_at_interval(results, interval):
    try:
        selected_interval = float(interval)
    except (TypeError, ValueError):
        return None
    return next(
        (
            result
            for result in results
            if np.isclose(float(result["interval"]), selected_interval)
        ),
        None,
    )


def show_selected_cost(result, total_label):
    if result is None:
        return
    st.markdown(f"**Selected interval: {result['interval']} years**")
    columns = st.columns(4)
    columns[0].metric(total_label, f"${result['total_cpp']:.6f}M")
    columns[1].metric("Cf — failure cost", f"${result['failure_cost']:.6f}M")
    columns[2].metric("Cr — repair cost", f"${result['repair_cost']:.6f}M")
    columns[3].metric("Safety", _safety_label(result))
    if result["is_safe"]:
        st.success("Safe — this interval satisfies both probability-of-failure criteria.")
    else:
        st.error("Unsafe — this interval exceeds at least one probability-of-failure criterion.")


def show_selectable_cost_figure(results, title, interval_label, total_label, key):
    st.caption("Click any point to inspect its Cf, Cr, and Safe/Unsafe result. ○ Safe   × Unsafe")
    event = show_figure(
        cost_figure(results, title, interval_label, total_label),
        key=key,
        selectable=True,
    )
    points = _selected_points(event)
    if points:
        point = points[0]
        interval = point.get("x") if isinstance(point, dict) else getattr(point, "x", None)
        show_selected_cost(_result_at_interval(results, interval), total_label)


def annual_pof_figure(result, schedule_label):
    figure = base_figure(
        f"Annual system POF at the optimal {result['interval']}-year {schedule_label.lower()} interval",
        "Year",
        "Annual probability of failure (per km-year)",
    )
    figure.add_trace(
        go.Scatter(
            x=result["times"],
            y=np.maximum(result["leak_hazard"], 1e-12),
            name="Leakage",
            mode="lines+markers",
            line={"color": COLORS["blue"], "width": 2.6},
            marker={"size": 5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result["times"],
            y=np.maximum(result["burst_hazard"], 1e-12),
            name="Burst",
            mode="lines+markers",
            line={"color": COLORS["coral"], "width": 2.6},
            marker={"size": 5},
        )
    )
    figure.add_hline(y=1e-2, line_dash="dot", line_color=COLORS["blue"],
                     annotation_text="Leak criterion")
    figure.add_hline(y=1e-3, line_dash="dash", line_color=COLORS["coral"],
                     annotation_text="Burst criterion")
    for inspection in result["inspections"]:
        figure.add_vline(x=inspection, line_dash="dot", line_color=COLORS["teal"],
                         opacity=0.45)
    figure.update_yaxes(type="log")
    return figure


def results_table(results, total_label):
    total_column = f"Total {total_label} ($M)"
    return pd.DataFrame(
        [
            {
                "Interval (years)": result["interval"],
                total_column: result["total_cpp"],
                "Cf — failure ($M)": result["failure_cost"],
                "Cr — repair ($M)": result["repair_cost"],
                "ILI inspection ($M)": result["inspection_cost"],
                "Expected repairs": result["expected_repairs"],
                "Safety": _safety_label(result),
            }
            for result in results
        ]
    )


def comparison_cost_figure(prior_results, ili_results):
    figure = base_figure(
        "Prior excavation versus ILI-informed scheduling",
        "Schedule interval (years)",
        "Expected lifecycle cost (million USD)",
    )
    figure.add_trace(
        go.Scatter(
            x=[result["interval"] for result in prior_results],
            y=[result["total_cpp"] for result in prior_results],
            name="Prior: excavation without ILI",
            mode="lines+markers",
            line={"color": COLORS["amber"], "width": 3},
            marker={
                "size": 8,
                "symbol": ["circle" if result["is_safe"] else "x" for result in prior_results],
            },
            customdata=_cost_customdata(prior_results),
            hovertemplate=_cost_hover_template("Cprior"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[result["interval"] for result in ili_results],
            y=[result["total_cpp"] for result in ili_results],
            name="Pre-posterior: ILI-informed",
            mode="lines+markers",
            line={"color": COLORS["teal"], "width": 3},
            marker={
                "size": 8,
                "symbol": ["circle" if result["is_safe"] else "x" for result in ili_results],
            },
            customdata=_cost_customdata(ili_results),
            hovertemplate=_cost_hover_template("Cpp"),
        )
    )
    figure.update_layout(hovermode="closest", clickmode="event+select")
    return figure


def show_comparison_cost_figure(prior_results, ili_results):
    st.caption("Click a Cprior or Cpp point to inspect Cf, Cr, and Safe/Unsafe. ○ Safe   × Unsafe")
    event = show_figure(
        comparison_cost_figure(prior_results, ili_results),
        key="comparison_cost_selection",
        selectable=True,
    )
    points = _selected_points(event)
    if not points:
        return
    point = points[0]
    if isinstance(point, dict):
        interval = point.get("x")
        curve_number = point.get("curve_number", 0)
    else:
        interval = getattr(point, "x", None)
        curve_number = getattr(point, "curve_number", 0)
    selected_results = prior_results if int(curve_number) == 0 else ili_results
    total_label = "Cprior" if int(curve_number) == 0 else "Cpp"
    show_selected_cost(_result_at_interval(selected_results, interval), total_label)


def comparison_table(prior_results, ili_results):
    prior_by_interval = {result["interval"]: result for result in prior_results}
    ili_by_interval = {result["interval"]: result for result in ili_results}
    return pd.DataFrame(
        [
            {
                "Interval (years)": interval,
                "Prior excavation cost ($M)": prior_by_interval[interval]["total_cpp"],
                "ILI-informed cost ($M)": ili_by_interval[interval]["total_cpp"],
                "Cost difference / VoI ($M)": (
                    prior_by_interval[interval]["total_cpp"]
                    - ili_by_interval[interval]["total_cpp"]
                ),
                "Prior safe": "Yes" if prior_by_interval[interval]["is_safe"] else "No",
                "ILI safe": "Yes" if ili_by_interval[interval]["is_safe"] else "No",
            }
            for interval in sorted(set(prior_by_interval) & set(ili_by_interval))
        ]
    )


def qr_png(url):
    image = qrcode.make(url)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.markdown(
    """
    <div class="hero">
      <small>Reliability-based planning</small>
      <h1>Pipeline Scheduling and Value of Information Lab</h1>
      <p>Compare prior excavation scheduling without ILI against pre-posterior,
         ILI-informed scheduling using the same deterioration and cost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Analysis settings")
    samples = st.number_input("Monte Carlo samples", 100, 5000, 1000, 100)
    horizon = st.number_input("Analysis horizon (years)", 5, 100, 30, 5)
    extra_crack_free = st.number_input(
        "Additional crack-free joints", 0, 10000, 0, 1,
        help="These statistically identical joints are evaluated as one efficient group.",
    )

    st.subheader("Crack initiation")
    default_birth_rate = st.number_input(
        "Crack birth rate (cracks per joint-year)",
        min_value=0.0,
        value=DEFAULT_CRACK_BIRTH_RATE,
        step=0.00001,
        format="%.6f",
        help=(
            "Expected number of newly initiated cracks in one pipe joint during one year. "
            "The simulation samples new cracks using a Poisson process. Set this to zero "
            "to model no new crack initiation."
        ),
    )
    st.caption(
        "This rate is used for every joint unless a `birth_rate` override is entered "
        "in the joint table."
    )

    st.subheader("Candidate schedule intervals")
    interval_start = st.number_input("First interval", 1, 100, 5, 1)
    interval_end = st.number_input("Last interval", 1, 100, 20, 1)
    interval_step = st.number_input("Interval step", 1, 50, 5, 1)

    with st.expander("Growth parameters"):
        mean_g = st.number_input("Mean depth growth, α (mm/year)", 0.0, 10.0, 0.10, 0.01)
        std_g = st.number_input("Std. α", 0.0, 10.0, 0.03, 0.01)
        mean_neq = st.number_input("Mean cycles/year, Neq", 0.0, 1e7, 1000.0, 100.0)
        std_neq = st.number_input("Std. Neq", 0.0, 1e7, 300.0, 50.0)
        mean_c = st.number_input("Mean Paris C", 0.0, 1e-8, 2.5e-13, 1e-14, format="%.3e")
        std_c = st.number_input("Std. Paris C", 0.0, 1e-8, 1.35e-13, 1e-14, format="%.3e")
        delta_s = st.number_input("Stress range, ΔS", 0.01, 1000.0, 2.0, 0.1)
        paris_m = st.number_input("Paris exponent, m", 0.1, 10.0, 3.0, 0.1)
    with st.expander("Costs (million USD)"):
        inspection_cost = st.number_input(
            "ILI inspection cost per event", 0.0, 1000.0, 0.015, 0.005,
            help="Applied only to the ILI-informed method.",
        )
        repair_1_cost = st.number_input(
            "Type 1 intervention cost, Cr1", 0.0, 1000.0, 0.20, 0.05,
            help="For the prior method, this is the base cost of excavating and treating a cracked joint.",
        )
        repair_2_cost = st.number_input("Additional type 2 cost, Cr2", 0.0, 1000.0, 0.0, 0.05)
        leak_cost = st.number_input("Leak failure cost", 0.0, 10000.0, 2.0, 0.5)
        burst_cost = st.number_input("Burst failure cost", 0.001, 10000.0, 20.0, 1.0)
        discount_percent = st.number_input("Discount rate (%)", 0.0, 100.0, 4.0, 0.5)

    with st.expander("Repair effectiveness"):
        coat_df = st.number_input("Recoat growth factor, df1", 0.0, 1.0, 0.0, 0.05)
        sleeve_df = st.number_input("Sleeve growth factor, df2", 0.0, 1.0, 0.0, 0.05)
        coat_period = st.number_input("Recoat effective period (years)", 0, 100, 5, 1)
        sleeve_period = st.number_input("Sleeve effective period (years)", 0, 100, 5, 1)

    with st.expander("Uncertainty and repair criteria"):
        initial_length_std = st.number_input("Initial crack-length std. (mm)", 0.0, 100.0, 15.6, 0.1)
        initial_depth_std = st.number_input("Initial crack-depth std. (mm)", 0.0, 20.0, 0.74, 0.01)
        length_error = st.number_input("Measured length error std. (mm)", 0.0, 100.0, 15.6, 0.1)
        depth_error = st.number_input("Measured depth error std. (mm)", 0.0, 20.0, 0.74, 0.01)
        q_pod = st.number_input("POD parameter, q", 0.0, 100.0, 2.0, 0.1)
        d_rep1 = st.number_input("Type 1 depth criterion (mm)", 0.0, 20.0, 0.5 * MEAN_WT, 0.1)
        d_rep2 = st.number_input("Type 2 depth criterion (mm)", 0.0, 20.0, 0.7 * MEAN_WT, 0.1)
        fpr_rep1 = st.number_input("Type 1 pressure factor", 0.0, 10.0, 1.25, 0.05)
        fpr_rep2 = st.number_input("Type 2 pressure factor", 0.0, 10.0, 1.10, 0.05)
        pipe_length = st.number_input("Pipe length (km)", 0.001, 10000.0, 1.0, 0.1)
        random_seed = st.number_input("Random seed", 0, 1000000, 0, 1)

    st.divider()
    with st.expander("Create a QR code"):
        st.caption("After deployment, paste the public app address here.")
        public_url = st.text_input("Public app URL", placeholder="https://your-app.streamlit.app")
        if public_url.startswith(("https://", "http://")):
            qr_bytes = qr_png(public_url)
            st.image(qr_bytes, caption="Scan to open the application", width=180)
            st.download_button("Download QR code", qr_bytes, "pipeline_app_qr.png", "image/png")


st.subheader("1. Define pipe joints and initial cracks")
st.markdown(
    '<div class="note">Use one row per joint. Separate multiple crack means with '
    'semicolons—for example, <b>40; 45</b>. A blank size list is required when '
    '<b>num_cracks = 0</b>. The <b>count</b> column can represent identical joints efficiently. '
    'Leave <b>birth_rate</b> blank to use the sidebar value, or enter a rate to override it '
    'for that joint or group.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="note"><b>Prior method:</b> at each candidate interval, joints '
    'known to contain initial cracks are excavated and treated with perfect '
    'observation—no ILI, POD, or measurement error. <b>Pre-posterior method:</b> '
    'ILI detection and measured crack sizes determine whether treatment occurs.</div>',
    unsafe_allow_html=True,
)

sample_csv = SAMPLE_DATA.to_csv(index=False).encode("utf-8")
left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload joint/crack CSV", type="csv")
with right:
    st.download_button("Download example CSV", sample_csv, "sample_joints.csv", "text/csv")

try:
    if uploaded is None:
        source_table = SAMPLE_DATA.copy()
        editor_key = "joint_editor_default"
    else:
        raw_bytes = uploaded.getvalue()
        source_table = pd.read_csv(io.BytesIO(raw_bytes))
        editor_key = f"joint_editor_{uploaded.name}_{len(raw_bytes)}"
    edited_table = st.data_editor(
        source_table,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=editor_key,
        column_config={
            "birth_rate": st.column_config.NumberColumn(
                "birth_rate (cracks/joint-year)",
                help=(
                    "Optional joint-specific crack birth rate. Leave blank to use the "
                    "sidebar value."
                ),
                min_value=0.0,
                format="%.6f",
            ),
        },
    )
except Exception as exc:
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

intervals = (
    list(range(int(interval_start), int(interval_end) + 1, int(interval_step)))
    if interval_start <= interval_end else []
)
growth_properties = {
    "alpha": {"mean": mean_g, "std": std_g, "distribution": "Normal"},
    "Neq": {"mean": mean_neq, "std": std_neq, "distribution": "Normal"},
    "C": {"mean": mean_c, "std": std_c, "distribution": "Lognormal"},
}

try:
    joints = table_to_joints(
        edited_table, growth_properties, default_birth_rate, extra_crack_free
    )
    total_joints = sum(joint["count"] for joint in joints)
    total_initial_cracks = sum(
        joint["count"] * len(joint["cracks"]) for joint in joints
    )
    summary = pd.DataFrame(
        [
            {
                "Joint/group": joint["name"],
                "Count": joint["count"],
                "Cracks per joint": len(joint["cracks"]),
                "Birth rate (cracks/joint-year)": joint["birth_rate"],
            }
            for joint in joints
        ]
    )
    with st.expander("Parsed model input", expanded=False):
        st.dataframe(summary, width="stretch", hide_index=True)
except ValueError as exc:
    st.error(str(exc))
    joints = []
    total_joints = total_initial_cracks = 0

expected_new_cracks = sum(
    joint["count"] * joint["birth_rate"] * horizon for joint in joints
)
metric_columns = st.columns(5)
metric_columns[0].metric("Physical joints", total_joints)
metric_columns[1].metric("Initial cracks", total_initial_cracks)
metric_columns[2].metric(
    "Expected new cracks",
    f"{expected_new_cracks:.3f}",
    help="Poisson mean over all represented joints and the full analysis horizon.",
)
metric_columns[3].metric("Candidate intervals", len(intervals))
metric_columns[4].metric("Monte Carlo samples", f"{samples:,}")

simulation_units = len(joints)
work_units = 2 * simulation_units * samples * (horizon + 1) * max(len(intervals), 1)
if work_units > 2_000_000:
    st.warning(
        "This selection is relatively heavy for an interactive demonstration. "
        "Reduce samples, intervals, horizon, or the number of individually modelled rows."
    )
allow_heavy = work_units <= 8_000_000 or st.checkbox(
    "I understand this larger run may take several minutes."
)

model = {
    "samples": int(samples),
    "horizon": int(horizon),
    "seed": int(random_seed),
    "birth_rate": float(default_birth_rate),
    "initial_length_std": float(initial_length_std),
    "initial_depth_std": float(initial_depth_std),
    "length_error": float(length_error),
    "depth_error": float(depth_error),
    "q_pod": float(q_pod),
    "costs": {
        "Cr1": float(repair_1_cost),
        "Cr2": float(repair_2_cost),
        "Cf_leak": float(leak_cost),
        "Cf_burst": float(burst_cost),
        "total_insp_cost": float(inspection_cost),
        "pipe_length_km": float(pipe_length),
        "rate": float(discount_percent) / 100,
    },
    "sim": {
        "dS": float(delta_s),
        "m": float(paris_m),
        "Di": 20 * 25.4,
        "FPR_rep1": float(fpr_rep1),
        "D_rep1": float(d_rep1),
        "FPR_rep2": float(fpr_rep2),
        "D_rep2": float(d_rep2),
        "lifetime_arr": [0, int(coat_period), int(sleeve_period)],
        "df_arr": [1, float(coat_df), float(sleeve_df)],
        "threshold": 1e-3,
        "resample_growth_each_step": False,
    },
}

run_disabled = not joints or not intervals or not allow_heavy
if st.button(
    "Run prior + pre-posterior comparison", type="primary", disabled=run_disabled
):
    try:
        with st.spinner("Running the prior excavation schedules…"):
            prior_results, prior_log = run_interval_study(
                joints, intervals, model, EXCAVATION_DECISION_MODE
            )
        with st.spinner("Running the ILI-informed schedules…"):
            ili_results, ili_log = run_interval_study(
                joints, intervals, model, ILI_DECISION_MODE
            )
        st.session_state["planning_comparison"] = {
            "prior_results": prior_results,
            "ili_results": ili_results,
            "prior_log": prior_log,
            "ili_log": ili_log,
            "joint_count": total_joints,
            "sample_count": samples,
        }
    except Exception as exc:
        st.exception(exc)


study = st.session_state.get("planning_comparison")
if study:
    prior_results = study["prior_results"]
    ili_results = study["ili_results"]
    prior_optimal, prior_has_safe = select_optimal_result(prior_results)
    ili_optimal, ili_has_safe = select_optimal_result(ili_results)
    voi = (
        calculate_value_of_information(prior_optimal, ili_optimal)
        if prior_optimal is not None and ili_optimal is not None
        else None
    )

    st.subheader("2. Minimum-cost safe schedules and VoI")
    columns = st.columns(5)
    columns[0].metric(
        "Prior safe interval",
        f"{prior_optimal['interval']} years" if prior_optimal else "Not found",
    )
    columns[1].metric(
        "Minimum safe Cprior",
        f"${prior_optimal['total_cpp']:.4f}M" if prior_optimal else "—",
    )
    columns[2].metric(
        "ILI safe interval",
        f"{ili_optimal['interval']} years" if ili_optimal else "Not found",
    )
    columns[3].metric(
        "Minimum safe Cpp",
        f"${ili_optimal['total_cpp']:.4f}M" if ili_optimal else "—",
    )
    columns[4].metric(
        "Net VoI of ILI",
        f"${voi['voi']:.4f}M" if voi else "Unavailable",
        help="Minimum safe prior cost minus minimum safe ILI-informed lifecycle cost.",
    )
    if voi is not None:
        if voi["ili_justified"]:
            st.success(voi["interpretation"])
        else:
            st.warning(voi["interpretation"])
        st.markdown(
            '<div class="note"><b>VoI = minimum safe prior excavation cost − minimum '
            'safe ILI-informed cost.</b> Each recommendation is the lowest-cost candidate '
            'that satisfies both safety criteria. The ILI-informed cost already includes '
            'discounted ILI inspection costs.</div>',
            unsafe_allow_html=True,
        )
    if not prior_has_safe or not ili_has_safe:
        missing = []
        if not prior_has_safe:
            missing.append("prior excavation")
        if not ili_has_safe:
            missing.append("ILI-informed")
        st.error(
            "No candidate interval satisfied both POF criteria for "
            + " and ".join(missing)
            + ". No unsafe interval is recommended. Test shorter intervals or revise the input assumptions."
        )

    comparison_tab, prior_tab, ili_tab = st.tabs(
        ["VoI comparison", "Prior excavation (no ILI)", "Pre-posterior (ILI)"]
    )
    with comparison_tab:
        show_comparison_cost_figure(prior_results, ili_results)
        combined = comparison_table(prior_results, ili_results)
        st.dataframe(
            combined.style.format(
                {
                    "Prior excavation cost ($M)": "{:.6f}",
                    "ILI-informed cost ($M)": "{:.6f}",
                    "Cost difference / VoI ($M)": "{:.6f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download VoI comparison",
            combined.to_csv(index=False).encode("utf-8"),
            "excavation_vs_ili_voi.csv",
            "text/csv",
        )

    with prior_tab:
        prior_columns = st.columns(4)
        prior_columns[0].metric(
            "Recommended interval",
            f"{prior_optimal['interval']} years" if prior_optimal else "Not found",
        )
        prior_columns[1].metric(
            "Minimum safe Cprior",
            f"${prior_optimal['total_cpp']:.4f}M" if prior_optimal else "—",
        )
        prior_columns[2].metric(
            "Expected interventions",
            f"{prior_optimal['expected_repairs']:.3f}" if prior_optimal else "—",
        )
        prior_columns[3].metric(
            "Safety criteria", "Satisfied" if prior_optimal else "No safe candidate"
        )
        show_selectable_cost_figure(
            prior_results,
            "Prior lifecycle cost versus excavation interval",
            "Excavation interval (years)",
            "Cprior",
            "prior_cost_selection",
        )
        if prior_optimal is not None:
            show_figure(annual_pof_figure(prior_optimal, "Excavation"))
        prior_table = results_table(prior_results, "Cprior")
        st.dataframe(
            prior_table.style.format(
                {
                    "Total Cprior ($M)": "{:.6f}",
                    "Cf — failure ($M)": "{:.6f}",
                    "Cr — repair ($M)": "{:.6f}",
                    "ILI inspection ($M)": "{:.6f}",
                    "Expected repairs": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download prior excavation results",
            prior_table.to_csv(index=False).encode("utf-8"),
            "prior_excavation_results.csv",
            "text/csv",
        )
        with st.expander("Prior simulation messages"):
            st.code(study["prior_log"] or "No simulation messages.")

    with ili_tab:
        ili_columns = st.columns(4)
        ili_columns[0].metric(
            "Recommended interval",
            f"{ili_optimal['interval']} years" if ili_optimal else "Not found",
        )
        ili_columns[1].metric(
            "Minimum safe Cpp",
            f"${ili_optimal['total_cpp']:.4f}M" if ili_optimal else "—",
        )
        ili_columns[2].metric(
            "Expected repairs",
            f"{ili_optimal['expected_repairs']:.3f}" if ili_optimal else "—",
        )
        ili_columns[3].metric(
            "Safety criteria", "Satisfied" if ili_optimal else "No safe candidate"
        )
        show_selectable_cost_figure(
            ili_results,
            "Pre-posterior lifecycle cost versus ILI interval",
            "ILI interval (years)",
            "Cpp",
            "ili_cost_selection",
        )
        if ili_optimal is not None:
            show_figure(annual_pof_figure(ili_optimal, "ILI inspection"))
        ili_table = results_table(ili_results, "Cpp")
        st.dataframe(
            ili_table.style.format(
                {
                    "Total Cpp ($M)": "{:.6f}",
                    "Cf — failure ($M)": "{:.6f}",
                    "Cr — repair ($M)": "{:.6f}",
                    "ILI inspection ($M)": "{:.6f}",
                    "Expected repairs": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download ILI-informed results",
            ili_table.to_csv(index=False).encode("utf-8"),
            "ili_informed_results.csv",
            "text/csv",
        )
        with st.expander("ILI simulation messages"):
            st.code(study["ili_log"] or "No simulation messages.")
else:
    st.info(
        "Adjust the inputs if needed, then select **Run prior + pre-posterior comparison**."
    )
