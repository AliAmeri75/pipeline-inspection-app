#!/usr/bin/env python3
"""Small Streamlit interface for inspection-interval optimization."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from packaging.version import Version
from streamlit_folium import st_folium
from joint_risk_map import alberta_demo, joint_reliability_table, reliability_map

from Simp_Sys_Thresh_J3_optimized import (
    DEFAULT_PROPERTY_SPECS,
    DEFAULT_MAX_ANNUAL_BURST_POF,
    DEFAULT_MAX_ANNUAL_LEAK_POF,
    EXCAVATION_DECISION_MODE,
    ILI_DECISION_MODE,
    MEAN_WT,
    POD,
    POD_DEPTH_LENGTH_MODEL,
    POD_DEPTH_ONLY_MODEL,
    PROPERTY_NAMES,
    run_flexible_simulation,
)
from planning_analysis import (
    calculate_value_of_information,
    compare_ili_intervals_to_prior_safe,
    format_optional_decimal,
    select_lowest_cost_result,
    select_optimal_result,
)
from pipeline_map import (
    haversine_distance_km,
    latest_route_from_drawings,
    map_cracks_to_joint_table,
    marker_points_from_drawings,
    point_at_chainage,
    route_length_km,
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
DEFAULT_CRACK_BIRTH_RATE = 0.00008
PIPELINE_CLASSES = ["Class 1", "Class 2", "Class 3", "Class 4", "User-defined"]
PIPELINE_CLASS_SAFETY_LIMITS = {
    "Class 1": {
        "max_annual_leak_pof": DEFAULT_MAX_ANNUAL_LEAK_POF,
        "max_annual_burst_pof": DEFAULT_MAX_ANNUAL_BURST_POF,
    },
    "Class 2": {
        "max_annual_leak_pof": 1e-3,
        "max_annual_burst_pof": 1e-4,
    },
    "Class 3": {
        "max_annual_leak_pof": 1e-4,
        "max_annual_burst_pof": 1e-5,
    },
    "Class 4": {
        "max_annual_leak_pof": 1e-5,
        "max_annual_burst_pof": 1e-6,
    },
}
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

TABLE_INPUT_MODE = "Joint table or CSV"
MAP_INPUT_MODE = "Interactive pipeline map"
MAP_SAMPLE_ROUTE = [
    [53.5350, -113.5850],
    [53.5480, -113.5350],
    [53.5570, -113.4800],
    [53.5670, -113.4300],
]
MAP_SAMPLE_CRACKS = pd.DataFrame(
    [
        {
            "crack_id": "C-001",
            "latitude": 53.5420,
            "longitude": -113.5580,
            "depth_mm": 3.5,
            "length_mm": 42.0,
        },
        {
            "crack_id": "C-002",
            "latitude": 53.5530,
            "longitude": -113.5050,
            "depth_mm": 5.1,
            "length_mm": 68.0,
        },
        {
            "crack_id": "C-003",
            "latitude": 53.5625,
            "longitude": -113.4550,
            "depth_mm": 2.7,
            "length_mm": 31.0,
        },
    ]
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
    .result-matrix { background:white; border:1px solid var(--line); border-radius:12px;
                     overflow:hidden; margin:.7rem 0 1rem; box-shadow:0 7px 22px rgba(16,36,61,.07); }
    .result-matrix table { width:100%; border-collapse:collapse; font-size:.88rem; }
    .result-matrix th { background:var(--ink); color:white; padding:.75rem .65rem;
                        text-align:center; font-weight:700; }
    .result-matrix td { padding:.78rem .65rem; text-align:center;
                        border-top:1px solid #e5e9e8; }
    .result-matrix td:first-child { text-align:left; font-weight:750; color:var(--ink); }
    .result-matrix tr:nth-child(even) td { background:#f7f9f8; }
    .badge-safe, .badge-unsafe { display:inline-block; padding:.18rem .55rem;
                                border-radius:999px; font-weight:800; }
    .badge-safe { color:#0f6a42; background:#dcf4e7; }
    .badge-unsafe { color:#9b2d22; background:#fde5e1; }
    .not-found { color:#9b2d22; font-weight:800; }
    .voi-card { border:1px solid; border-radius:12px; padding:1rem 1.15rem;
                margin:.65rem 0 1rem; box-shadow:0 7px 22px rgba(16,36,61,.07); }
    .voi-card .voi-label { font-size:.78rem; font-weight:800; letter-spacing:.06em;
                           text-transform:uppercase; }
    .voi-card .voi-value { font-size:2rem; line-height:1.15; font-weight:800;
                           margin:.18rem 0; }
    .voi-card .voi-message { font-size:.9rem; font-weight:650; }
    .voi-positive { color:#0b5d36; background:#e5f5eb; border-color:#91cba8; }
    .voi-negative { color:#8d241c; background:#fde9e6; border-color:#e7a59d; }
    .voi-neutral { color:#425461; background:#eef1f2; border-color:#cbd3d6; }
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


def render_pipeline_map_input():
    """Render the GIS input and return a simulation-ready joint table."""
    route_key = "inspection_map_route"
    cracks_key = "inspection_map_cracks"
    revision_key = "inspection_map_revision"
    if route_key not in st.session_state:
        st.session_state[route_key] = copy.deepcopy(MAP_SAMPLE_ROUTE)
    if cracks_key not in st.session_state:
        st.session_state[cracks_key] = MAP_SAMPLE_CRACKS.to_dict("records")
    if revision_key not in st.session_state:
        st.session_state[revision_key] = 0

    st.markdown(
        '<div class="note"><b>How to use the map</b><br>'
        '<b>1.</b> Draw the pipeline with the line tool: click the start and bend points, '
        'then double-click to finish.<br>'
        '<b>2.</b> Enter the depth and length of the next crack.<br>'
        '<b>3.</b> Either select the marker tool and click on or near the pipeline, or '
        'enter the crack distance from the pipeline start.<br>'
        '<b>4.</b> The app places the crack on the line, calculates its distance '
        '(chainage), and assigns it to a pipe joint automatically.</div>',
        unsafe_allow_html=True,
    )

    action_1, action_2, action_3 = st.columns([1, 1, 2])
    if action_1.button("Load Edmonton sample", width="stretch"):
        st.session_state[route_key] = copy.deepcopy(MAP_SAMPLE_ROUTE)
        st.session_state[cracks_key] = MAP_SAMPLE_CRACKS.to_dict("records")
        st.session_state[revision_key] += 1
        st.rerun()
    if action_2.button("Clear map input", width="stretch"):
        st.session_state[route_key] = []
        st.session_state[cracks_key] = []
        st.session_state[revision_key] += 1
        st.rerun()
    action_3.caption(
        "The Edmonton example is synthetic and does not represent an operating pipeline."
    )
    demo_route, demo_cracks = alberta_demo()
    if st.button("Load 100 km Alberta demo"):
        st.session_state[route_key] = demo_route
        st.session_state[cracks_key] = demo_cracks.to_dict('records')
        st.session_state[revision_key] += 1
        st.rerun()
    st.caption("Sample data: fictional 100 km route with bends, west of Edmonton; not an actual pipeline survey.")
    st.download_button("Download demo crack CSV", demo_cracks.to_csv(index=False),
                       "alberta_100km_cracks.csv", "text/csv")
    demo_geojson = json.dumps({'type': 'LineString', 'coordinates': [[lon, lat] for lat, lon in demo_route]})
    st.download_button("Download demo route GeoJSON", demo_geojson,
                       "alberta_100km_route.geojson", "application/geo+json")
    route_upload = st.file_uploader("Upload pipeline route GeoJSON (LineString)", type=['geojson', 'json'])
    if route_upload is not None:
        try:
            document = json.loads(route_upload.getvalue())
            geometry = document.get('geometry', document)
            if geometry.get('type') != 'LineString':
                raise ValueError('Use a GeoJSON LineString or a Feature containing one.')
            uploaded_route = [[float(lat), float(lon)] for lon, lat in geometry['coordinates']]
            if route_length_km(uploaded_route) <= 0:
                raise ValueError('The route must have a positive length.')
            if st.button('Apply uploaded route'):
                st.session_state[route_key] = uploaded_route
                st.session_state[revision_key] += 1
                st.rerun()
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
    crack_upload = st.file_uploader('Upload mapped crack CSV (latitude, longitude, depth_mm, length_mm)', type='csv')
    if crack_upload is not None and st.button('Apply uploaded cracks'):
        try:
            uploaded_cracks = pd.read_csv(crack_upload)
            map_cracks_to_joint_table(st.session_state[route_key], uploaded_cracks, 12.)
            st.session_state[cracks_key] = uploaded_cracks.to_dict('records')
            st.session_state[revision_key] += 1
            st.rerun()
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))

    setting_1, setting_2, setting_3, setting_4 = st.columns(4)
    with setting_1:
        joint_length_m = st.number_input(
            "Nominal pipe-joint length (m)",
            min_value=1.0,
            max_value=100.0,
            value=12.0,
            step=1.0,
            help=(
                "Used to divide the mapped route into physical joints. Replace the "
                "demonstration value with construction or asset data when available."
            ),
        )
    with setting_2:
        new_depth_mm = st.number_input(
            "New crack depth (mm)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1,
        )
    with setting_3:
        new_length_mm = st.number_input(
            "New crack length (mm)",
            min_value=0.1,
            max_value=5000.0,
            value=40.0,
            step=1.0,
        )
    with setting_4:
        snap_warning_m = st.number_input(
            "Warn when marker offset exceeds (m)",
            min_value=1.0,
            max_value=5000.0,
            value=100.0,
            step=10.0,
        )

    route = st.session_state[route_key]
    mapped_length_km = route_length_km(route) if len(route) >= 2 else 0.0

    st.markdown("#### Choose the crack location")
    location_method = st.radio(
        "How would you like to position the next crack?",
        options=["Click on the map", "Enter distance from pipeline start"],
        horizontal=True,
        help=(
            "A clicked marker is snapped to the nearest point on the pipeline. "
            "Distance from start places the crack exactly at the entered chainage."
        ),
    )
    if location_method == "Click on the map":
        st.info(
            "Below the map, select the pin-shaped marker tool and click on or near "
            "the pipeline. The orange/red crack marker will appear on the pipeline "
            "after the map refreshes."
        )
    elif mapped_length_km > 0:
        chainage_1, chainage_2 = st.columns([2, 1])
        with chainage_1:
            new_chainage_km = st.number_input(
                "Crack distance from pipeline start (km)",
                min_value=0.0,
                max_value=float(mapped_length_km),
                value=float(min(mapped_length_km / 2, mapped_length_km)),
                step=float(max(0.001, min(0.1, mapped_length_km / 20))),
                format="%.3f",
                help=(
                    f"The current pipeline runs from 0.000 to "
                    f"{mapped_length_km:.3f} km."
                ),
            )
        with chainage_2:
            st.write("")
            st.write("")
            add_at_chainage = st.button(
                "Add crack at this distance", width="stretch", type="primary"
            )
        if add_at_chainage:
            latitude, longitude = point_at_chainage(route, float(new_chainage_km))
            existing_ids = {
                str(record.get("crack_id", ""))
                for record in st.session_state[cracks_key]
            }
            crack_number = 1
            while f"C-{crack_number:03d}" in existing_ids:
                crack_number += 1
            st.session_state[cracks_key].append(
                {
                    "crack_id": f"C-{crack_number:03d}",
                    "latitude": latitude,
                    "longitude": longitude,
                    "depth_mm": float(new_depth_mm),
                    "length_mm": float(new_length_mm),
                }
            )
            st.session_state[revision_key] += 1
            st.rerun()
    else:
        st.warning(
            "Draw a pipeline centreline first. The distance-based crack option will "
            "be enabled after the route is created."
        )

    crack_columns = ["crack_id", "latitude", "longitude", "depth_mm", "length_mm"]
    crack_table = pd.DataFrame(st.session_state[cracks_key])
    for column in crack_columns:
        if column not in crack_table.columns:
            crack_table[column] = pd.Series(dtype="float64" if column != "crack_id" else "object")
    crack_table = crack_table[crack_columns]
    with st.expander("Review or manually edit crack data"):
        st.caption(
            "You normally do not need to enter coordinates. Use this table only to "
            "edit crack sizes, correct coordinates, or delete a crack row."
        )
        edited_cracks = st.data_editor(
            crack_table,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=f"pipeline_crack_editor_{st.session_state[revision_key]}",
            column_config={
                "crack_id": st.column_config.TextColumn("Crack ID"),
                "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "depth_mm": st.column_config.NumberColumn(
                    "Crack depth (mm)", min_value=0.01, format="%.2f"
                ),
                "length_mm": st.column_config.NumberColumn(
                    "Crack length (mm)", min_value=0.1, format="%.2f"
                ),
            },
        )
    edited_cracks = edited_cracks.dropna(how="all").reset_index(drop=True)
    st.session_state[cracks_key] = edited_cracks.to_dict("records")

    model_defaults = {
        "WT_mean": SAMPLE_DATA.iloc[0]["WT_mean"],
        "WT_std": SAMPLE_DATA.iloc[0]["WT_std"],
        "Pservice_mean": SAMPLE_DATA.iloc[0]["Pservice_mean"],
        "Pservice_std": SAMPLE_DATA.iloc[0]["Pservice_std"],
        "birth_rate": np.nan,
    }
    model_table = pd.DataFrame()
    located_cracks = pd.DataFrame()
    map_summary = None
    transform_error = None
    try:
        model_table, located_cracks, map_summary = map_cracks_to_joint_table(
            route,
            edited_cracks,
            joint_length_m=float(joint_length_m),
            default_row_values=model_defaults,
        )
    except ValueError as exc:
        transform_error = str(exc)

    if route:
        map_center = [
            sum(point[0] for point in route) / len(route),
            sum(point[1] for point in route) / len(route),
        ]
    else:
        map_center = [53.5461, -113.4938]
    st.markdown("#### Draw the pipeline and place crack markers")
    st.caption(
        "Line tool: draw or replace the pipeline. Marker tool: add one crack using "
        "the depth and length entered above. Green is the start (0 km), blue is the "
        "end, and orange/red circles are cracks. Click a crack circle for details."
    )
    pipeline_map = folium.Map(
        location=map_center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(
        pipeline_map
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(pipeline_map)

    if len(route) >= 2:
        folium.PolyLine(
            route,
            color="#16a6a1",
            weight=6,
            opacity=0.95,
            tooltip="Pipeline centreline",
        ).add_to(pipeline_map)
        folium.Marker(
            route[0],
            tooltip="Pipeline start — 0.000 km",
            icon=folium.Icon(color="green"),
        ).add_to(pipeline_map)
        folium.Marker(
            route[-1],
            tooltip=f"Pipeline end — {mapped_length_km:.3f} km",
            icon=folium.Icon(color="blue"),
        ).add_to(pipeline_map)
        pipeline_map.fit_bounds(route, padding=(25, 25))

    if not located_cracks.empty:
        for crack in located_cracks.itertuples(index=False):
            marker_color = "#d4493f" if crack.depth_mm >= 5 else "#d98b22"
            folium.CircleMarker(
                location=[crack.snapped_latitude, crack.snapped_longitude],
                radius=7 + min(crack.depth_mm, 8),
                color="white",
                weight=2,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.95,
                tooltip=f"{crack.crack_id} — {crack.joint_id}",
                popup=folium.Popup(
                    (
                        f"<b>{crack.crack_id}</b><br>"
                        f"{crack.joint_id}<br>"
                        f"Chainage: {crack.chainage_km:.3f} km<br>"
                        f"Depth: {crack.depth_mm:.2f} mm<br>"
                        f"Length: {crack.length_mm:.2f} mm<br>"
                        f"Snap offset: {crack.map_offset_m:.1f} m"
                    ),
                    max_width=260,
                ),
            ).add_to(pipeline_map)

    Draw(
        export=False,
        draw_options={
            "polyline": {"shapeOptions": {"color": "#16a6a1", "weight": 6}},
            "marker": True,
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(pipeline_map)
    folium.LayerControl(collapsed=False).add_to(pipeline_map)
    map_result = st_folium(
        pipeline_map,
        height=590,
        use_container_width=True,
        returned_objects=["all_drawings"],
        key=f"inspection_pipeline_map_{st.session_state[revision_key]}",
    )

    drawings = map_result.get("all_drawings") or []
    new_route = latest_route_from_drawings(drawings)
    new_points = marker_points_from_drawings(drawings)
    map_changed = False
    if new_route and new_route != route:
        st.session_state[route_key] = new_route
        map_changed = True

    existing_points = []
    for record in st.session_state[cracks_key]:
        try:
            existing_points.append(
                [float(record["latitude"]), float(record["longitude"])]
            )
        except (KeyError, TypeError, ValueError):
            continue
    for point in new_points:
        already_present = any(
            haversine_distance_km(point, existing) * 1000 < 0.5
            for existing in existing_points
        )
        if already_present:
            continue
        existing_ids = {
            str(record.get("crack_id", ""))
            for record in st.session_state[cracks_key]
        }
        crack_number = 1
        while f"C-{crack_number:03d}" in existing_ids:
            crack_number += 1
        st.session_state[cracks_key].append(
            {
                "crack_id": f"C-{crack_number:03d}",
                "latitude": point[0],
                "longitude": point[1],
                "depth_mm": float(new_depth_mm),
                "length_mm": float(new_length_mm),
            }
        )
        existing_points.append(point)
        map_changed = True

    if map_changed:
        st.session_state[revision_key] += 1
        st.rerun()

    if transform_error:
        st.error(transform_error)
        return model_table, located_cracks, map_summary

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Mapped length", f"{map_summary['pipeline_length_km']:.2f} km"
    )
    metric_columns[1].metric("Physical joints", f"{map_summary['total_joints']:,}")
    metric_columns[2].metric("Cracked joints", f"{map_summary['cracked_joints']:,}")
    metric_columns[3].metric("Initial cracks", f"{map_summary['initial_cracks']:,}")
    maximum_offset = (
        float(located_cracks["map_offset_m"].max()) if not located_cracks.empty else 0.0
    )
    metric_columns[4].metric("Maximum snap offset", f"{maximum_offset:.1f} m")
    if maximum_offset > snap_warning_m:
        st.warning(
            "At least one crack marker is farther from the pipeline than the selected "
            "offset limit. Check its position before running the analysis."
        )

    with st.expander("Crack locations on the pipeline", expanded=True):
        if located_cracks.empty:
            st.info("No cracks have been placed yet.")
        else:
            st.caption(
                "Chainage is the distance from the green pipeline-start marker. "
                "The joint number is calculated from chainage and nominal joint length. "
                f"All cracks assigned to the same {joint_length_m:g} m joint are "
                "modelled together as one pipe joint, not as separate joints."
            )
            st.dataframe(
                located_cracks[
                    [
                        "crack_id",
                        "chainage_km",
                        "joint_id",
                        "depth_mm",
                        "length_mm",
                        "map_offset_m",
                    ]
                ].rename(
                    columns={
                        "crack_id": "Crack ID",
                        "chainage_km": "Distance from start (km)",
                        "joint_id": "Assigned joint",
                        "depth_mm": "Depth (mm)",
                        "length_mm": "Length (mm)",
                        "map_offset_m": "Original click offset (m)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Map-to-joints transformation"):
        st.caption(
            "Cracks on the same physical joint are combined in one model row; its "
            "num_cracks value and semicolon-separated crack sizes retain every crack. "
            "All other physical joints are represented by one counted crack-free group."
        )
        st.dataframe(model_table, width="stretch", hide_index=True)
    return model_table, located_cracks, map_summary


def unpack_result(raw, interval, decision_mode, sim_params):
    (
        times, leak_pof, burst_pof, leak_hazard, burst_hazard, inspections,
        cpp, cff, crr, cii, is_safe, unsafe_times,
        cpp_cracked, cpp_crack_free, repairs, actions, debug,
    ) = raw
    return {
        "interval": interval,
        "decision_mode": decision_mode,
        "max_annual_leak_pof": sim_params["max_annual_leak_pof"],
        "max_annual_burst_pof": sim_params["max_annual_burst_pof"],
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
        results.append(unpack_result(raw, interval, decision_mode, sim))
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


def recommendation_matrix(prior_cost_optimal, prior_safe, ili_cost_optimal, ili_safe):
    """Build a compact result matrix for cost and safety decisions."""

    def row(method, cost_optimal, safe_result):
        safety = _safety_label(cost_optimal)
        badge_class = "badge-safe" if cost_optimal["is_safe"] else "badge-unsafe"
        safe_interval = (
            f"{safe_result['interval']} years"
            if safe_result is not None
            else '<span class="not-found">Not found</span>'
        )
        safe_cost = (
            f"${safe_result['total_cpp']:.4f}M"
            if safe_result is not None
            else "—"
        )
        return (
            "<tr>"
            f"<td>{method}</td>"
            f"<td><b>{cost_optimal['interval']} years</b></td>"
            f"<td>${cost_optimal['total_cpp']:.4f}M</td>"
            f'<td><span class="{badge_class}">{safety}</span></td>'
            f"<td>{safe_interval}</td>"
            f"<td>{safe_cost}</td>"
            "</tr>"
        )

    prior_row = row("Prior excavation (no ILI)", prior_cost_optimal, prior_safe)
    ili_row = row("Pre-posterior (ILI-informed)", ili_cost_optimal, ili_safe)
    return (
        '<div class="result-matrix"><table><thead><tr>'
        "<th>Planning method</th>"
        "<th>Cost-optimal interval</th>"
        "<th>Lowest cost</th>"
        "<th>Safety at cost optimum</th>"
        "<th>Minimum-cost safe interval</th>"
        "<th>Safe cost</th>"
        "</tr></thead><tbody>"
        f"{prior_row}{ili_row}"
        "</tbody></table></div>"
    )


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
    st.caption(
        f"Limits used: PfL ≤ {result['max_annual_leak_pof']:.1e}; "
        f"PfB ≤ {result['max_annual_burst_pof']:.1e}."
    )


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
        f"Annual system POF at the cost-optimal {result['interval']}-year "
        f"{schedule_label.lower()} interval",
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
    leak_limit = result["max_annual_leak_pof"]
    burst_limit = result["max_annual_burst_pof"]
    if leak_limit > 0:
        figure.add_hline(
            y=leak_limit,
            line_dash="dot",
            line_color=COLORS["blue"],
            annotation_text=f"PfL limit = {leak_limit:.1e}",
        )
    if burst_limit > 0:
        figure.add_hline(
            y=burst_limit,
            line_dash="dash",
            line_color=COLORS["coral"],
            annotation_text=f"PfB limit = {burst_limit:.1e}",
        )
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


def comparison_table(prior_safe_result, ili_results):
    return pd.DataFrame(
        compare_ili_intervals_to_prior_safe(prior_safe_result, ili_results)
    )


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

    joint_input_method = st.radio(
        "Pipeline and crack input",
        options=[TABLE_INPUT_MODE, MAP_INPUT_MODE],
        help=(
            "Use the existing table/CSV workflow, or draw a pipeline and place "
            "cracks on an interactive map."
        ),
    )

    with st.expander("Simulation settings", expanded=True):
        samples = st.number_input("Monte Carlo samples", 100, 10000, 1000, 100)
        horizon = st.number_input("Analysis horizon (years)", 5, 100, 30, 5)
        manual_pipe_length = st.number_input(
            "Pipe length (km)",
            0.001,
            10000.0,
            1.0,
            0.1,
            disabled=joint_input_method == MAP_INPUT_MODE,
            help=(
                "In map mode, the app calculates this value from the drawn "
                "pipeline centreline."
            ),
        )
        pipe_length = manual_pipe_length
        random_seed = st.number_input("Random seed", 0, 1000000, 0, 1)

    with st.expander("Pipeline class and safety criteria", expanded=True):
        pipeline_class = st.selectbox("Pipeline class", PIPELINE_CLASSES)
        class_limits = PIPELINE_CLASS_SAFETY_LIMITS.get(pipeline_class)
        widget_suffix = pipeline_class.lower().replace(" ", "_").replace("-", "_")
        if class_limits is not None:
            st.caption("The study's annual POF limits are selected automatically for this class.")
            max_annual_leak_pof = st.number_input(
                "Maximum annual leakage POF, PfL",
                value=class_limits["max_annual_leak_pof"],
                format="%.1e",
                disabled=True,
                key=f"pfl_{widget_suffix}",
            )
            max_annual_burst_pof = st.number_input(
                "Maximum annual burst POF, PfB",
                value=class_limits["max_annual_burst_pof"],
                format="%.1e",
                disabled=True,
                key=f"pfb_{widget_suffix}",
            )
            st.caption(
                "Classes 1–4 use an illustrative one-order-of-magnitude progression. "
                "These are study assumptions, not values prescribed directly by CSA Z662. "
                "[CSA Group research](https://www.csagroup.org/wp-content/uploads/"
                "CSA-Group-Research-Pipeline-Safety-Factors-Exec-Summary.pdf) describes "
                "the broader risk-consistent safety-class basis."
            )
        else:
            st.info("Enter the two annual POF limits for this analysis.")
            max_annual_leak_pof = st.number_input(
                "Maximum annual leakage POF, PfL",
                min_value=0.0,
                max_value=1.0,
                value=None,
                placeholder="Enter the PfL limit",
                format="%.1e",
                key=f"pfl_{widget_suffix}",
            )
            max_annual_burst_pof = st.number_input(
                "Maximum annual burst POF, PfB",
                min_value=0.0,
                max_value=1.0,
                value=None,
                placeholder="Enter the PfB limit",
                format="%.1e",
                key=f"pfb_{widget_suffix}",
            )
        safety_criteria_valid = (
            max_annual_leak_pof is not None and max_annual_burst_pof is not None
        )
        if safety_criteria_valid:
            st.caption(
                "A schedule is Safe only when "
                f"PfL ≤ {max_annual_leak_pof:.1e} and "
                f"PfB ≤ {max_annual_burst_pof:.1e}."
            )

    with st.expander("Prior excavation intervals"):
        prior_interval_start = st.number_input(
            "First excavation interval (years)", 1, 100, 5, 1
        )
        prior_interval_end = st.number_input(
            "Last excavation interval (years)", 1, 100, 20, 1
        )
        prior_interval_step = st.number_input(
            "Excavation interval step (years)", 1, 50, 5, 1
        )

    with st.expander("ILI inspection intervals"):
        ili_interval_start = st.number_input(
            "First ILI interval (years)", 1, 100, 5, 1
        )
        ili_interval_end = st.number_input(
            "Last ILI interval (years)", 1, 100, 20, 1
        )
        ili_interval_step = st.number_input(
            "ILI interval step (years)", 1, 50, 5, 1
        )

    with st.expander("Initial cracks and crack initiation"):
        extra_crack_free = st.number_input(
            "Additional crack-free joints", 0, 10000, 0, 1,
            disabled=joint_input_method == MAP_INPUT_MODE,
            help=(
                "These statistically identical joints are evaluated as one "
                "efficient group. In map mode, this count is calculated from "
                "the route length and nominal joint length."
            ),
        )
        initial_length_std = st.number_input(
            "Initial crack-length std. (mm)", 0.0, 100.0, 15.6, 0.1
        )
        initial_depth_std = st.number_input(
            "Initial crack-depth std. (mm)", 0.0, 20.0, 0.74, 0.01
        )
        default_birth_rate = st.number_input(
            "Crack birth rate (cracks per joint-year)",
            min_value=0.0,
            value=DEFAULT_CRACK_BIRTH_RATE,
            step=0.00001,
            format="%.6f",
            help=(
                "Expected number of newly initiated cracks in one pipe joint during "
                "one year. The simulation samples new cracks using a Poisson process. "
                "Set this to zero to model no new crack initiation."
            ),
        )
        st.caption(
            "The sidebar birth rate applies to every joint unless the joint table "
            "contains a specific `birth_rate` override."
        )

    with st.expander("Crack growth model"):
        mean_g = st.number_input("Mean depth growth, α (mm/year)", 0.0, 10.0, 0.10, 0.01)
        std_g = st.number_input("Std. α", 0.0, 10.0, 0.03, 0.01)
        mean_neq = st.number_input("Mean cycles/year, Neq", 0.0, 1e7, 1000.0, 100.0)
        std_neq = st.number_input("Std. Neq", 0.0, 1e7, 300.0, 50.0)
        mean_c = st.number_input("Mean Paris C", 0.0, 1e-8, 2.5e-13, 1e-14, format="%.3e")
        std_c = st.number_input("Std. Paris C", 0.0, 1e-8, 1.35e-13, 1e-14, format="%.3e")
        delta_s = st.number_input("Stress range, ΔS", 0.01, 1000.0, 2.0, 0.1)
        paris_m = st.number_input("Paris exponent, m", 0.1, 10.0, 3.0, 0.1)

    with st.expander("ILI measurement model — POD and sizing error"):
        st.caption(
            "POD controls whether ILI detects a crack; sizing errors describe the "
            "uncertainty in the reported crack dimensions."
        )
        pod_model = st.selectbox(
            "POD model",
            options=[POD_DEPTH_ONLY_MODEL, POD_DEPTH_LENGTH_MODEL],
            format_func=lambda value: {
                POD_DEPTH_ONLY_MODEL: "Depth only (current model)",
                POD_DEPTH_LENGTH_MODEL: "Depth and length",
            }[value],
            help=(
                "Choose whether detection depends only on crack depth or on both "
                "crack depth and crack length."
            ),
        )
        q_pod = st.number_input(
            "Depth coefficient, qd (1/mm)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Higher qd gives a higher POD for the same crack depth.",
        )
        if pod_model == POD_DEPTH_LENGTH_MODEL:
            q_length_pod = st.number_input(
                "Length coefficient, qL (1/mm)",
                min_value=0.0,
                max_value=10.0,
                value=0.01,
                step=0.01,
                format="%.3f",
                help="Higher qL gives a higher POD for the same crack length.",
            )
            st.latex(r"POD(d,L)=1-\exp[-(q_d d+q_L L)]")
        else:
            q_length_pod = 0.0
            st.latex(r"POD(d)=1-\exp(-q_d d)")
        example_pod = POD(
            1.0,
            q_pod,
            length=40.0,
            pod_model=pod_model,
            q_length=q_length_pod,
        )
        st.caption(
            f"Example POD for a 1 mm deep, 40 mm long crack: {example_pod:.1%}. "
            "Calibrate the coefficients using ILI vendor qualification or excavation data."
        )
        length_error = st.number_input(
            "ILI crack-length error std. (mm)", 0.0, 100.0, 15.6, 0.1
        )
        depth_error = st.number_input(
            "ILI crack-depth error std. (mm)", 0.0, 20.0, 0.74, 0.01
        )

    with st.expander("Repair criteria 1 and 2"):
        st.markdown("**Repair criterion 1**")
        d_rep1 = st.number_input(
            "Type 1 depth criterion (mm)", 0.0, 20.0, 0.5 * MEAN_WT, 0.1
        )
        fpr_rep1 = st.number_input(
            "Type 1 pressure factor", 0.0, 10.0, 1.25, 0.05
        )
        st.divider()
        st.markdown("**Repair criterion 2**")
        d_rep2 = st.number_input(
            "Type 2 depth criterion (mm)", 0.0, 20.0, 0.7 * MEAN_WT, 0.1
        )
        fpr_rep2 = st.number_input(
            "Type 2 pressure factor", 0.0, 10.0, 1.10, 0.05
        )

    with st.expander("Repair effectiveness"):
        coat_df = st.number_input("Recoat growth factor, df1", 0.0, 1.0, 0.0, 0.05)
        coat_period = st.number_input("Recoat effective period (years)", 0, 100, 5, 1)
        sleeve_df = st.number_input("Sleeve growth factor, df2", 0.0, 1.0, 0.0, 0.05)
        sleeve_period = st.number_input("Sleeve effective period (years)", 0, 100, 5, 1)

    with st.expander("Costs (million USD)"):
        inspection_cost_per_km = st.number_input(
            "ILI inspection cost per km per event ($M/km)",
            0.0,
            1000.0,
            0.015,
            0.001,
            help=(
                "Applied once across the full pipeline at every scheduled ILI. "
                "For example, 0.01 $M/km for a 10 km pipeline gives 0.10 $M "
                "per ILI event before discounting."
            ),
        )
        repair_1_cost = st.number_input(
            "Type 1 intervention cost, Cr1", 0.0, 1000.0, 0.20, 0.05,
            help="For the prior method, this is the base cost of excavating and treating a cracked joint.",
        )
        repair_2_cost = st.number_input("Additional type 2 cost, Cr2", 0.0, 1000.0, 0.0, 0.05)
        leak_cost = st.number_input("Leak failure cost", 0.0, 10000.0, 2.0, 0.5)
        burst_cost = st.number_input("Burst failure cost", 0.001, 10000.0, 20.0, 1.0)
        discount_percent = st.number_input("Discount rate (%)", 0.0, 100.0, 4.0, 0.5)

if joint_input_method == MAP_INPUT_MODE:
    st.subheader("1. Map the pipeline and its initial cracks")
    edited_table, mapped_cracks, map_summary = render_pipeline_map_input()
    model_extra_crack_free = 0
    if map_summary is not None:
        pipe_length = float(map_summary["pipeline_length_km"])
        st.caption(
            f"The scheduling model will use the mapped pipeline length: {pipe_length:.3f} km."
        )
else:
    st.subheader("1. Define pipe joints and initial cracks")
    st.markdown(
        '<div class="note">Use one row per joint. Separate multiple crack means with '
        'semicolons—for example, <b>40; 45</b>. A blank size list is required when '
        '<b>num_cracks = 0</b>. The <b>count</b> column can represent identical joints efficiently. '
        'Leave <b>birth_rate</b> blank to use the sidebar value, or enter a rate to override it '
        'for that joint or group.</div>',
        unsafe_allow_html=True,
    )

    sample_csv = SAMPLE_DATA.to_csv(index=False).encode("utf-8")
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("Upload joint/crack CSV", type="csv")
    with right:
        st.download_button(
            "Download example CSV", sample_csv, "sample_joints.csv", "text/csv"
        )

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
                        "Optional joint-specific crack birth rate. Leave blank to use "
                        "the sidebar value."
                    ),
                    min_value=0.0,
                    format="%.6f",
                ),
            },
        )
    except Exception as exc:
        st.error(f"The CSV could not be read: {exc}")
        st.stop()
    model_extra_crack_free = extra_crack_free

st.markdown(
    '<div class="note"><b>Prior method:</b> at each candidate interval, joints '
    'known to contain initial cracks are excavated and treated with perfect '
    'observation—no ILI, POD, or measurement error. <b>Pre-posterior method:</b> '
    'ILI detection and measured crack sizes determine whether treatment occurs.</div>',
    unsafe_allow_html=True,
)

prior_intervals = (
    list(
        range(
            int(prior_interval_start),
            int(prior_interval_end) + 1,
            int(prior_interval_step),
        )
    )
    if prior_interval_start <= prior_interval_end
    else []
)
ili_intervals = (
    list(
        range(
            int(ili_interval_start),
            int(ili_interval_end) + 1,
            int(ili_interval_step),
        )
    )
    if ili_interval_start <= ili_interval_end
    else []
)
growth_properties = {
    "alpha": {"mean": mean_g, "std": std_g, "distribution": "Normal"},
    "Neq": {"mean": mean_neq, "std": std_neq, "distribution": "Normal"},
    "C": {"mean": mean_c, "std": std_c, "distribution": "Lognormal"},
}

try:
    joints = table_to_joints(
        edited_table, growth_properties, default_birth_rate, model_extra_crack_free
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
metric_columns[3].metric(
    "Prior / ILI intervals", f"{len(prior_intervals)} / {len(ili_intervals)}"
)
metric_columns[4].metric("Monte Carlo samples", f"{samples:,}")

simulation_units = len(joints)
work_units = (
    simulation_units
    * samples
    * (horizon + 1)
    * max(len(prior_intervals) + len(ili_intervals), 1)
)
if work_units > 2_000_000:
    st.warning(
        "This selection is relatively heavy for an interactive demonstration. "
        "Reduce samples, either interval range, horizon, or the number of "
        "individually modelled rows."
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
        "inspection_cost_per_km": float(inspection_cost_per_km),
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
        "max_annual_leak_pof": (
            float(max_annual_leak_pof) if safety_criteria_valid else np.nan
        ),
        "max_annual_burst_pof": (
            float(max_annual_burst_pof) if safety_criteria_valid else np.nan
        ),
        "pod_model": pod_model,
        "pod_depth_coefficient": float(q_pod),
        "pod_length_coefficient": float(q_length_pod),
        "resample_growth_each_step": False,
    },
}

current_safety_signature = (
    pipeline_class,
    float(max_annual_leak_pof) if safety_criteria_valid else None,
    float(max_annual_burst_pof) if safety_criteria_valid else None,
)
current_interval_signature = (tuple(prior_intervals), tuple(ili_intervals))
run_disabled = (
    not joints
    or not prior_intervals
    or not ili_intervals
    or not allow_heavy
    or not safety_criteria_valid
)
if st.button(
    "Run prior + pre-posterior comparison", type="primary", disabled=run_disabled
):
    try:
        with st.spinner("Running the prior excavation schedules…"):
            prior_results, prior_log = run_interval_study(
                joints, prior_intervals, model, EXCAVATION_DECISION_MODE
            )
        with st.spinner("Running the ILI-informed schedules…"):
            ili_results, ili_log = run_interval_study(
                joints, ili_intervals, model, ILI_DECISION_MODE
            )
        st.session_state["planning_comparison"] = {
            "prior_results": prior_results,
            "ili_results": ili_results,
            "prior_log": prior_log,
            "ili_log": ili_log,
            "joint_count": total_joints,
            "sample_count": samples,
            "pipeline_class": pipeline_class,
            "max_annual_leak_pof": float(max_annual_leak_pof),
            "max_annual_burst_pof": float(max_annual_burst_pof),
            "safety_signature": current_safety_signature,
            "interval_signature": current_interval_signature,
            "prior_intervals": prior_intervals,
            "ili_intervals": ili_intervals,
            "map_route": copy.deepcopy(st.session_state.get('inspection_map_route')) if joint_input_method == MAP_INPUT_MODE else None,
            "mapped_cracks": mapped_cracks.copy() if joint_input_method == MAP_INPUT_MODE else None,
            "map_summary": copy.deepcopy(map_summary) if joint_input_method == MAP_INPUT_MODE else None,
        }
    except Exception as exc:
        st.exception(exc)


study = st.session_state.get("planning_comparison")
if study and study.get("safety_signature") != current_safety_signature:
    st.info(
        "The pipeline class or safety limits changed. Run the comparison again "
        "to update the recommendations."
    )
    study = None
if study and study.get("interval_signature") != current_interval_signature:
    st.info(
        "The prior or ILI interval range changed. Run the comparison again "
        "to update the recommendations."
    )
    study = None
if study:
    prior_results = study["prior_results"]
    ili_results = study["ili_results"]
    prior_cost_optimal = select_lowest_cost_result(prior_results)
    ili_cost_optimal = select_lowest_cost_result(ili_results)
    prior_safe, prior_has_safe = select_optimal_result(prior_results)
    ili_safe, ili_has_safe = select_optimal_result(ili_results)
    voi = (
        calculate_value_of_information(prior_safe, ili_safe)
        if prior_safe is not None and ili_safe is not None
        else None
    )

    st.subheader("Cost-optimal and safe scheduling results")
    if study.get('map_route') and study.get('map_summary'):
        with st.expander('Pipeline joint risk profile', expanded=True):
            method = st.radio('Map planning method', ['Prior excavation', 'ILI-informed'], horizontal=True)
            candidates = prior_results if method == 'Prior excavation' else ili_results
            preferred = prior_safe if method == 'Prior excavation' else ili_safe
            intervals = [r['interval'] for r in candidates]
            selected = st.selectbox('Map interval (years)', intervals,
                index=intervals.index(preferred['interval']) if preferred else 0)
            result = next(r for r in candidates if r['interval'] == selected)
            spatial = joint_reliability_table(result, study['map_route'], study['mapped_cracks'],
                                             study['map_summary']['joint_length_m'])
            st.caption('Red: above a selected limit. Yellow: 50–100% of a limit. Green: below 50%. '
                       'Colours use peak annual conditional failure probability per km over the analysis horizon. '
                       'These are local reliability indicators, not consequence-based risk or the whole-pipeline safety decision. '
                       'No recorded crack does not guarantee low risk: new crack initiation is included. '
                       'The 50% warning boundary is a display choice, not a regulatory criterion.')
            st_folium(reliability_map(study['map_route'], spatial), height=500,
                      width=1100, returned_objects=[], key=f'result_risk_{method}_{selected}')
            st.dataframe(spatial, hide_index=True)
            st.download_button('Download joint risk profile CSV', spatial.to_csv(index=False),
                               'joint_risk_profile.csv', 'text/csv')
    st.caption(
        f"{study['pipeline_class']} safety criteria: "
        f"PfL ≤ {study['max_annual_leak_pof']:.1e}; "
        f"PfB ≤ {study['max_annual_burst_pof']:.1e}."
    )
    st.html(
        recommendation_matrix(
            prior_cost_optimal, prior_safe, ili_cost_optimal, ili_safe
        )
    )
    if voi is not None:
        if voi["voi"] > 0:
            voi_class = "voi-positive"
        elif voi["voi"] < 0:
            voi_class = "voi-negative"
        else:
            voi_class = "voi-neutral"
        st.html(
            f'<div class="voi-card {voi_class}">'
            '<div class="voi-label">Net VoI of ILI</div>'
            f'<div class="voi-value">${voi["voi"]:.4f}M</div>'
            f'<div class="voi-message">{voi["interpretation"]}</div>'
            "</div>"
        )
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
        st.warning(
            "Minimum-cost safe interval: **Not found** for "
            + " and ".join(missing)
            + " within the tested range. The cost-optimal interval is still shown above "
            + "with its Safe/Unsafe status."
        )

    comparison_tab, prior_tab, ili_tab = st.tabs(
        ["VoI comparison", "Prior excavation (no ILI)", "Pre-posterior (ILI)"]
    )
    with comparison_tab:
        show_comparison_cost_figure(prior_results, ili_results)
        st.caption(
            "Each tested ILI interval is compared with the minimum-cost safe prior "
            "excavation solution. ILI is economically justified only when that ILI "
            "interval is safe and has a lower expected lifecycle cost."
        )
        combined = comparison_table(prior_safe, ili_results)
        st.dataframe(
            combined.style.format(
                {
                    "Prior safe cost ($M)": format_optional_decimal,
                    "ILI cost ($M)": "{:.6f}",
                    "Cost difference / VoI ($M)": format_optional_decimal,
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download VoI comparison",
            combined.to_csv(index=False).encode("utf-8"),
            "ili_intervals_vs_prior_safe.csv",
            "text/csv",
        )

    with prior_tab:
        prior_columns = st.columns(5)
        prior_columns[0].metric(
            "Cost-optimal interval",
            f"{prior_cost_optimal['interval']} years",
        )
        prior_columns[1].metric(
            "Lowest Cprior",
            f"${prior_cost_optimal['total_cpp']:.4f}M",
        )
        prior_columns[2].metric(
            "Safety at cost optimum",
            _safety_label(prior_cost_optimal),
        )
        prior_columns[3].metric(
            "Safe interval",
            f"{prior_safe['interval']} years" if prior_safe else "Not found",
        )
        prior_columns[4].metric(
            "Safe Cprior",
            f"${prior_safe['total_cpp']:.4f}M" if prior_safe else "—",
        )
        show_selectable_cost_figure(
            prior_results,
            "Prior lifecycle cost versus excavation interval",
            "Excavation interval (years)",
            "Cprior",
            "prior_cost_selection",
        )
        show_figure(annual_pof_figure(prior_cost_optimal, "Excavation"))
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
        ili_columns = st.columns(5)
        ili_columns[0].metric(
            "Cost-optimal interval",
            f"{ili_cost_optimal['interval']} years",
        )
        ili_columns[1].metric(
            "Lowest Cpp",
            f"${ili_cost_optimal['total_cpp']:.4f}M",
        )
        ili_columns[2].metric(
            "Safety at cost optimum",
            _safety_label(ili_cost_optimal),
        )
        ili_columns[3].metric(
            "Safe interval",
            f"{ili_safe['interval']} years" if ili_safe else "Not found",
        )
        ili_columns[4].metric(
            "Safe Cpp",
            f"${ili_safe['total_cpp']:.4f}M" if ili_safe else "—",
        )
        show_selectable_cost_figure(
            ili_results,
            "Pre-posterior lifecycle cost versus ILI interval",
            "ILI interval (years)",
            "Cpp",
            "ili_cost_selection",
        )
        show_figure(annual_pof_figure(ili_cost_optimal, "ILI inspection"))
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
