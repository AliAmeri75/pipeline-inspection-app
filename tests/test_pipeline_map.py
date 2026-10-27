import pandas as pd
import pytest

from pipeline_map import (
    chainage_to_joint_number,
    map_cracks_to_joint_table,
    point_at_chainage,
    route_length_km,
    snap_point_to_route,
)


def test_route_length_and_snap_chainage():
    route = [[53.5, -113.5], [53.5, -113.35]]
    length = route_length_km(route)
    snapped = snap_point_to_route([53.51, -113.425], route)

    assert length == pytest.approx(9.94, rel=0.03)
    assert snapped["chainage_km"] == pytest.approx(length / 2, rel=0.03)
    assert snapped["offset_m"] > 1000


def test_endpoint_is_assigned_to_last_joint():
    assert chainage_to_joint_number(0.120, joint_length_m=12, total_joints=10) == 10


def test_point_at_chainage_places_crack_on_route():
    route = [[53.5, -113.5], [53.5, -113.4], [53.5, -113.3]]
    total_length = route_length_km(route)

    midpoint = point_at_chainage(route, total_length / 2)

    assert midpoint[0] == pytest.approx(53.5)
    assert midpoint[1] == pytest.approx(-113.4, abs=1e-5)
    assert point_at_chainage(route, 0) == pytest.approx(route[0])
    assert point_at_chainage(route, total_length) == pytest.approx(route[-1])


def test_point_at_chainage_rejects_distance_beyond_route():
    route = [[53.5, -113.5], [53.5, -113.4]]
    with pytest.raises(ValueError, match="between 0"):
        point_at_chainage(route, route_length_km(route) + 0.1)


def test_map_transform_groups_cracks_and_crack_free_joints():
    route = [[53.5, -113.5], [53.5, -113.35]]
    cracks = pd.DataFrame(
        [
            {
                "crack_id": "C-1",
                "latitude": 53.5,
                "longitude": -113.49,
                "depth_mm": 2.0,
                "length_mm": 30.0,
            },
            {
                "crack_id": "C-2",
                "latitude": 53.5,
                "longitude": -113.48995,
                "depth_mm": 3.0,
                "length_mm": 40.0,
            },
            {
                "crack_id": "C-3",
                "latitude": 53.5,
                "longitude": -113.42,
                "depth_mm": 4.0,
                "length_mm": 50.0,
            },
        ]
    )

    table, located, summary = map_cracks_to_joint_table(
        route,
        cracks,
        joint_length_m=100,
        default_row_values={"WT_mean": 6.35, "birth_rate": float("nan")},
    )

    assert summary["initial_cracks"] == 3
    assert summary["cracked_joints"] == 2
    assert summary["crack_free_joints"] == summary["total_joints"] - 2
    cracked_rows = table[table["num_cracks"] > 0]
    assert sorted(cracked_rows["num_cracks"].tolist()) == [1, 2]
    assert table["count"].sum() == summary["total_joints"]
    assert located["joint_id"].nunique() == 2


def test_transform_rejects_non_positive_crack_size():
    cracks = pd.DataFrame(
        [{"latitude": 53.5, "longitude": -113.4, "depth_mm": 0, "length_mm": 20}]
    )
    with pytest.raises(ValueError, match="depth and length must be positive"):
        map_cracks_to_joint_table(
            [[53.5, -113.5], [53.5, -113.3]], cracks, joint_length_m=12
        )
