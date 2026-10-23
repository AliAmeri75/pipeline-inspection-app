"""Geometry helpers for converting a mapped pipeline into model joint inputs."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import pandas as pd


EARTH_RADIUS_KM = 6371.0088
REQUIRED_CRACK_COLUMNS = {
    "latitude",
    "longitude",
    "depth_mm",
    "length_mm",
}


def _point(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("Each route point must contain latitude and longitude.")
    latitude, longitude = float(point[0]), float(point[1])
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Route coordinates must be valid latitude/longitude values.")
    return latitude, longitude


def haversine_distance_km(point_1: Sequence[float], point_2: Sequence[float]) -> float:
    """Return the great-circle distance between two latitude/longitude points."""
    latitude_1, longitude_1 = _point(point_1)
    latitude_2, longitude_2 = _point(point_2)
    lat_1 = math.radians(latitude_1)
    lat_2 = math.radians(latitude_2)
    delta_lat = lat_2 - lat_1
    delta_lon = math.radians(longitude_2 - longitude_1)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(value))


def route_length_km(route: Sequence[Sequence[float]]) -> float:
    """Return the total geodesic length of a route polyline."""
    if len(route) < 2:
        return 0.0
    points = [_point(point) for point in route]
    return sum(
        haversine_distance_km(start, end)
        for start, end in zip(points[:-1], points[1:])
    )


def snap_point_to_route(
    point: Sequence[float], route: Sequence[Sequence[float]]
) -> dict[str, object]:
    """Snap a map point to the nearest route segment and calculate its chainage."""
    if len(route) < 2:
        raise ValueError("Draw a pipeline route with at least two points first.")

    target_latitude, target_longitude = _point(point)
    route_points = [_point(route_point) for route_point in route]
    cumulative_km = 0.0
    best: dict[str, object] | None = None

    for segment_index, (start, end) in enumerate(
        zip(route_points[:-1], route_points[1:])
    ):
        segment_km = haversine_distance_km(start, end)
        if segment_km == 0:
            continue

        reference_latitude = math.radians(
            (target_latitude + start[0] + end[0]) / 3
        )

        def local_xy(latitude: float, longitude: float) -> tuple[float, float]:
            x = EARTH_RADIUS_KM * math.radians(longitude) * math.cos(reference_latitude)
            y = EARTH_RADIUS_KM * math.radians(latitude)
            return x, y

        target_x, target_y = local_xy(target_latitude, target_longitude)
        start_x, start_y = local_xy(*start)
        end_x, end_y = local_xy(*end)
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        denominator = segment_x**2 + segment_y**2
        fraction = (
            ((target_x - start_x) * segment_x + (target_y - start_y) * segment_y)
            / denominator
        )
        fraction = max(0.0, min(1.0, fraction))
        snapped = (
            start[0] + fraction * (end[0] - start[0]),
            start[1] + fraction * (end[1] - start[1]),
        )
        offset_m = 1000 * haversine_distance_km(
            (target_latitude, target_longitude), snapped
        )
        candidate = {
            "chainage_km": cumulative_km + fraction * segment_km,
            "snapped_latitude": snapped[0],
            "snapped_longitude": snapped[1],
            "offset_m": offset_m,
            "segment_index": segment_index,
        }
        if best is None or offset_m < float(best["offset_m"]):
            best = candidate
        cumulative_km += segment_km

    if best is None:
        raise ValueError("The pipeline route cannot contain only repeated points.")
    return best


def chainage_to_joint_number(
    chainage_km: float, joint_length_m: float, total_joints: int
) -> int:
    """Map distance along the pipeline to a one-based physical joint number."""
    if joint_length_m <= 0:
        raise ValueError("Pipe-joint length must be positive.")
    if total_joints < 1:
        raise ValueError("The pipeline must contain at least one joint.")
    joint_number = math.floor(max(0.0, chainage_km) * 1000 / joint_length_m) + 1
    return min(joint_number, total_joints)


def map_cracks_to_joint_table(
    route: Sequence[Sequence[float]],
    cracks: pd.DataFrame,
    joint_length_m: float,
    default_row_values: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Transform mapped cracks into the table consumed by the simulation.

    Cracks that fall on the same physical joint are grouped into one row. All
    joints without an initial crack are represented by one counted group so the
    simulation does not need hundreds of duplicate rows.
    """
    total_length_km = route_length_km(route)
    if total_length_km <= 0:
        raise ValueError("Draw a pipeline route with a positive length first.")
    if joint_length_m <= 0:
        raise ValueError("Pipe-joint length must be positive.")

    total_joints = max(1, math.ceil(total_length_km * 1000 / joint_length_m))
    missing = REQUIRED_CRACK_COLUMNS.difference(cracks.columns)
    if missing:
        raise ValueError(
            "Crack data are missing required columns: " + ", ".join(sorted(missing))
        )

    located_records: list[dict[str, object]] = []
    for row_number, (_, crack) in enumerate(cracks.iterrows(), start=1):
        try:
            latitude = float(crack["latitude"])
            longitude = float(crack["longitude"])
            depth_mm = float(crack["depth_mm"])
            length_mm = float(crack["length_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Crack row {row_number} contains a non-numeric value.") from exc
        if not all(math.isfinite(value) for value in (latitude, longitude, depth_mm, length_mm)):
            raise ValueError(f"Crack row {row_number} contains a missing or infinite value.")
        if depth_mm <= 0 or length_mm <= 0:
            raise ValueError(f"Crack row {row_number}: depth and length must be positive.")

        snapped = snap_point_to_route((latitude, longitude), route)
        joint_number = chainage_to_joint_number(
            float(snapped["chainage_km"]), joint_length_m, total_joints
        )
        crack_id_value = crack.get("crack_id", f"C-{row_number:03d}")
        crack_id = (
            f"C-{row_number:03d}"
            if pd.isna(crack_id_value) or not str(crack_id_value).strip()
            else str(crack_id_value).strip()
        )
        located_records.append(
            {
                "crack_id": crack_id,
                "latitude": latitude,
                "longitude": longitude,
                "depth_mm": depth_mm,
                "length_mm": length_mm,
                "chainage_km": float(snapped["chainage_km"]),
                "joint_number": joint_number,
                "joint_id": f"Joint {joint_number:05d}",
                "snapped_latitude": float(snapped["snapped_latitude"]),
                "snapped_longitude": float(snapped["snapped_longitude"]),
                "map_offset_m": float(snapped["offset_m"]),
            }
        )

    located = pd.DataFrame(located_records)
    defaults = dict(default_row_values or {})
    model_rows: list[dict[str, object]] = []
    if not located.empty:
        for joint_number, group in located.groupby("joint_number", sort=True):
            row = {
                "joint_id": f"Joint {int(joint_number):05d}",
                "count": 1,
                "num_cracks": len(group),
                "mean_lengths_mm": "; ".join(
                    f"{value:g}" for value in group["length_mm"]
                ),
                "mean_depths_mm": "; ".join(
                    f"{value:g}" for value in group["depth_mm"]
                ),
                "crack_ids": "; ".join(group["crack_id"].astype(str)),
                "chainages_km": "; ".join(
                    f"{value:.4f}" for value in group["chainage_km"]
                ),
            }
            row.update(defaults)
            model_rows.append(row)

    remaining_crack_free = total_joints - len(model_rows)
    if remaining_crack_free > 0:
        row = {
            "joint_id": "Remaining crack-free joints",
            "count": remaining_crack_free,
            "num_cracks": 0,
            "mean_lengths_mm": "",
            "mean_depths_mm": "",
            "crack_ids": "",
            "chainages_km": "",
        }
        row.update(defaults)
        model_rows.append(row)

    summary: dict[str, float | int] = {
        "pipeline_length_km": total_length_km,
        "joint_length_m": joint_length_m,
        "total_joints": total_joints,
        "cracked_joints": len(model_rows) - (1 if remaining_crack_free > 0 else 0),
        "crack_free_joints": remaining_crack_free,
        "initial_cracks": len(located),
    }
    return pd.DataFrame(model_rows), located, summary


def latest_route_from_drawings(
    drawings: Iterable[Mapping[str, object]] | None,
) -> list[list[float]] | None:
    """Return the most recently drawn GeoJSON LineString as [lat, lon] points."""
    for drawing in reversed(list(drawings or [])):
        geometry = drawing.get("geometry", {})
        if not isinstance(geometry, Mapping) or geometry.get("type") != "LineString":
            continue
        coordinates = geometry.get("coordinates", [])
        return [[float(latitude), float(longitude)] for longitude, latitude in coordinates]
    return None


def marker_points_from_drawings(
    drawings: Iterable[Mapping[str, object]] | None,
) -> list[list[float]]:
    """Return drawn GeoJSON point markers as [latitude, longitude] points."""
    points: list[list[float]] = []
    for drawing in drawings or []:
        geometry = drawing.get("geometry", {})
        if not isinstance(geometry, Mapping) or geometry.get("type") != "Point":
            continue
        longitude, latitude = geometry.get("coordinates", [None, None])
        if latitude is not None and longitude is not None:
            points.append([float(latitude), float(longitude)])
    return points
