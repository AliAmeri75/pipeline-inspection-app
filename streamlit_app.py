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
    MEAN_WT,
    PROPERTY_NAMES,
    run_flexible_simulation,
)


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
            "birth_rate": 0.00008,
        }
    ],
    columns=SAMPLE_COLUMNS,
)


st.set_page_config(
    page_title="Pipeline Inspection Schedule Lab",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
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


def unpack_result(raw, interval):
    (
        times, leak_pof, burst_pof, leak_hazard, burst_hazard, inspections,
        cpp, cff, crr, cii, is_safe, unsafe_times,
        cpp_cracked, cpp_crack_free, repairs, actions, debug,
    ) = raw
    return {
        "interval": interval,
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


def run_interval_study(joints, intervals, model):
    results = []
    progress = st.progress(0, text="Preparing simulations…")
    log = io.StringIO()
    for index, interval in enumerate(intervals):
        sim = copy.deepcopy(model["sim"])
        sim["inspection_interval"] = interval
        np.random.seed(model["seed"])
        progress.progress(
            index / len(intervals), text=f"Testing a {interval}-year interval…"
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
            )
        results.append(unpack_result(raw, interval))
    progress.progress(1.0, text="Optimization complete")
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


def show_figure(figure):
    width_argument = (
        {"width": "stretch"}
        if MODERN_CHART_WIDTH
        else {"use_container_width": True}
    )
    st.plotly_chart(figure, config={"displaylogo": False}, **width_argument)


def cost_figure(results):
    figure = base_figure(
        "Expected lifecycle cost versus inspection interval",
        "Inspection interval (years)",
        "Expected cost (million USD)",
    )
    styles = [
        ("Total Cpp", "total_cpp", COLORS["navy"]),
        ("Failure", "failure_cost", COLORS["coral"]),
        ("Repair", "repair_cost", COLORS["teal"]),
        ("Inspection", "inspection_cost", COLORS["amber"]),
    ]
    for label, key, color in styles:
        figure.add_trace(
            go.Scatter(
                x=[result["interval"] for result in results],
                y=[result[key] for result in results],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.6},
                marker={"size": 7},
            )
        )
    return figure


def annual_pof_figure(result):
    figure = base_figure(
        f"Annual system POF at the optimal {result['interval']}-year interval",
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


def results_table(results):
    return pd.DataFrame(
        [
            {
                "Interval (years)": result["interval"],
                "Total Cpp ($M)": result["total_cpp"],
                "Failure ($M)": result["failure_cost"],
                "Repair ($M)": result["repair_cost"],
                "Inspection ($M)": result["inspection_cost"],
                "Expected repairs": result["expected_repairs"],
                "Safe": "Yes" if result["is_safe"] else "No",
            }
            for result in results
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
      <h1>Pipeline Inspection Schedule Lab</h1>
      <p>Compare equidistant inspection intervals using joint-specific cracks,
         Monte Carlo deterioration, repair actions, and system failure costs.</p>
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

    st.subheader("Inspection intervals")
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
        default_birth_rate = st.number_input(
            "Default crack birth rate (/year)", 0.0, 1.0, 0.00008, 0.00001,
            format="%.5f",
        )

    with st.expander("Costs (million USD)"):
        inspection_cost = st.number_input("Inspection cost per event", 0.0, 1000.0, 0.015, 0.005)
        repair_1_cost = st.number_input("Repair type 1 cost, Cr1", 0.0, 1000.0, 0.20, 0.05)
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
    '<b>num_cracks = 0</b>. The <b>count</b> column can represent identical joints efficiently.</div>',
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
                "Birth rate (/year)": joint["birth_rate"],
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

metric_columns = st.columns(4)
metric_columns[0].metric("Physical joints", total_joints)
metric_columns[1].metric("Initial cracks", total_initial_cracks)
metric_columns[2].metric("Candidate intervals", len(intervals))
metric_columns[3].metric("Monte Carlo samples", f"{samples:,}")

simulation_units = len(joints)
work_units = simulation_units * samples * (horizon + 1) * max(len(intervals), 1)
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
if st.button("Run interval optimization", type="primary", disabled=run_disabled):
    try:
        with st.spinner("Running the Monte Carlo interval comparison…"):
            results, simulation_log = run_interval_study(joints, intervals, model)
        st.session_state["interval_study"] = {
            "results": results,
            "log": simulation_log,
            "joint_count": total_joints,
            "sample_count": samples,
        }
    except Exception as exc:
        st.exception(exc)


study = st.session_state.get("interval_study")
if study:
    results = study["results"]
    optimal = min(results, key=lambda result: result["total_cpp"])
    st.subheader("2. Optimization results")
    columns = st.columns(4)
    columns[0].metric("Optimal interval", f"{optimal['interval']} years")
    columns[1].metric("Minimum Cpp", f"${optimal['total_cpp']:.4f}M")
    columns[2].metric("Expected repairs", f"{optimal['expected_repairs']:.3f}")
    columns[3].metric("Safety criteria", "Satisfied" if optimal["is_safe"] else "Exceeded")

    show_figure(cost_figure(results))
    show_figure(annual_pof_figure(optimal))

    table = results_table(results)
    st.dataframe(
        table.style.format(
            {
                "Total Cpp ($M)": "{:.6f}",
                "Failure ($M)": "{:.6f}",
                "Repair ($M)": "{:.6f}",
                "Inspection ($M)": "{:.6f}",
                "Expected repairs": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download interval results",
        table.to_csv(index=False).encode("utf-8"),
        "inspection_interval_results.csv",
        "text/csv",
    )
    with st.expander("Simulation messages"):
        st.code(study["log"] or "No simulation messages.")
else:
    st.info("Adjust the inputs if needed, then select **Run interval optimization**.")
