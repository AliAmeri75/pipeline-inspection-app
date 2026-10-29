import numpy as np
import pandas as pd
import pytest
from joint_risk_map import alberta_demo, joint_reliability_table, reliability_map
from pipeline_map import map_cracks_to_joint_table, route_length_km
from joint_risk_map import sliding_window_reliability


def test_sliding_combines_same_year_and_catches_cluster():
    route = [[53.5, -113.5], [53.5, -113.46]]
    count = int(np.ceil(route_length_km(route) / .012))
    leak = np.zeros((count, 3))
    # Two nearby joints, each below the limit, jointly exceed it in year one.
    leak[80:82] = [0, .006, .006]
    result = dict(times=[0., 1., 2.],
        joint_names=[f'Joint {i+1:05d}' for i in range(count)],
        joint_leak_pofs=leak, joint_burst_pofs=np.zeros_like(leak),
        max_annual_leak_pof=.01, max_annual_burst_pof=.001)
    windows, spatial = sliding_window_reliability(result, route, pd.DataFrame(), 12)
    assert windows.peak_annual_leak_pof.max() == pytest.approx(1 - .994**2)
    assert 'High' in set(windows.category)
    assert spatial.iloc[80].category == 'High'
    assert np.allclose(windows.section_length_km, 1.)
    assert windows.end_km.iloc[-1] == pytest.approx(route_length_km(route))
    # Different-year events must not be added as simultaneous annual peaks.
    leak[81] = [0, 0, .006]
    windows, _ = sliding_window_reliability(result, route, pd.DataFrame(), 12)
    assert windows.peak_annual_leak_pof.max() == pytest.approx(.006)


def test_short_route_and_saturated_probability():
    route = [[53.5, -113.5], [53.5, -113.4999]]
    result = dict(times=[0., 1., 2.], joint_names=['Remaining crack-free joints'],
                  joint_leak_pofs=[[0, 1, 1]], joint_burst_pofs=[[0, 0, 0]],
                  max_annual_leak_pof=.01, max_annual_burst_pof=.001)
    windows, spatial = sliding_window_reliability(result, route, pd.DataFrame({'joint_number': [1]}), 12)
    assert len(windows) == 1
    assert windows.peak_annual_leak_pof.iloc[0] == 1
    assert windows.section_length_km.iloc[0] < 1
    assert spatial.category.iloc[0] == 'High'
    assert 'Worst enclosing window' in reliability_map(route, spatial).get_root().render()


def test_demo_length_grouping_and_bends():
    route, cracks = alberta_demo()
    assert route_length_km(route) == pytest.approx(100, abs=1e-8)
    assert len(route) > 2
    _, located, summary = map_cracks_to_joint_table(route, cracks, 12.)
    assert len(located) == 12
    assert summary['cracked_joints'] < 12
    assert located.iloc[0].joint_number == located.iloc[1].joint_number


def test_local_probability_group_mapping_and_categories():
    route = [[53.5, -113.5], [53.5, -113.4995]]
    length = route_length_km(route)
    joint_m = length * 1000 / 2
    result = dict(times=np.array([0., 1., 2.]),
                  joint_names=['Joint 00001', 'Remaining crack-free joints'],
                  joint_leak_pofs=np.array([[0, .01, .02], [0, 0, 0]]),
                  joint_burst_pofs=np.zeros((2, 3)),
                  max_annual_leak_pof=.01, max_annual_burst_pof=.001)
    table = joint_reliability_table(result, route, pd.DataFrame({'joint_number': [1, 1]}), joint_m)
    assert table.iloc[0].category == 'High'
    assert table.iloc[0].initial_cracks == 2
    assert table.iloc[1].category == 'Low'
    assert table.iloc[0].peak_annual_leak_per_km == pytest.approx(.01/.99/(joint_m/1000))
    assert 'OpenStreetMap' in reliability_map(route, table).get_root().render()


def test_crack_free_group_can_be_high():
    route = [[53.5, -113.5], [53.5, -113.4999]]
    result = dict(times=[0., 1.], joint_names=['Remaining crack-free joints'],
                  joint_leak_pofs=[[0, .1]], joint_burst_pofs=[[0, 0]],
                  max_annual_leak_pof=.01, max_annual_burst_pof=.001)
    table = joint_reliability_table(result, route, pd.DataFrame(), 12.)
    assert (table.category == 'High').all()
