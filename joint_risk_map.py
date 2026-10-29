"""Spatial reliability output and a reproducible, fictional Alberta example."""
import math
import numpy as np
import pandas as pd
import folium
from pipeline_map import route_length_km, point_at_chainage


def alberta_demo():
    """Return a bent 100 km fictional route west of Edmonton and sample cracks."""
    origin = (53.6, -114.5)
    offsets = [(0, 0), (.05, .2), (-.03, .4), (.08, .65), (.02, .9),
               (.15, 1.1), (.12, 1.35)]
    low, high = 0., 3.
    for _ in range(60):
        scale = (low + high) / 2
        route = [[origin[0] + a * scale, origin[1] + b * scale] for a, b in offsets]
        if route_length_km(route) < 100:
            low = scale
        else:
            high = scale
    records = []
    for i, distance in enumerate([2.001, 2.002, 8.10, 15.30, 24.0, 38.1, 51.2,
                                   62.401, 62.403, 76.2, 88.8, 96.1]):
        lat, lon = point_at_chainage(route, distance)
        records.append(dict(crack_id=f"DEMO-{i+1:03d}", latitude=lat, longitude=lon,
                            depth_mm=[.6, 1., 2., 3.5, 4.5, 1.5][i % 6],
                            length_mm=[20., 40., 80., 140., 200., 60.][i % 6]))
    return route, pd.DataFrame(records)


def joint_reliability_table(result, route, cracks, joint_length_m):
    """Peak annual conditional failure probability per km, for each physical joint.

    Values use the same conditional-increment convention as the system model.
    These are local indicators, not a substitute for the system safety decision.
    The aggregated crack-free group describes each member joint, not its sum.
    """
    length = route_length_km(route)
    count = math.ceil(length * 1000 / joint_length_m)
    times = np.asarray(result['times'], dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError('Result times must increase.')
    probabilities = {}
    for index, name in enumerate(result['joint_names']):
        peaks = []
        for key in ('joint_leak_pofs', 'joint_burst_pofs'):
            values = np.asarray(result[key][index], dtype=float)
            annual = np.divide(np.diff(values), 1 - values[:-1],
                               out=np.zeros_like(dt), where=values[:-1] < 1) / dt
            peaks.append(float(np.maximum(annual, 0).max(initial=0)))
        probabilities[name] = peaks
    crack_counts = cracks.groupby('joint_number').size().to_dict() if not cracks.empty else {}
    rows = []
    for number in range(1, count + 1):
        name = f'Joint {number:05d}'
        start = (number - 1) * joint_length_m / 1000
        end = min(number * joint_length_m / 1000, length)
        local = probabilities.get(name, probabilities.get('Remaining crack-free joints'))
        if local is None:
            raise ValueError(f'Missing simulation results for {name}')
        leak, burst = np.asarray(local) / (end - start)
        ratio = max(leak / result['max_annual_leak_pof'], burst / result['max_annual_burst_pof'])
        category = 'High' if ratio > 1 else 'Medium' if ratio >= .5 else 'Low'
        rows.append(dict(joint_id=name, start_km=start, end_km=end,
                         initial_cracks=crack_counts.get(number, 0),
                         peak_annual_leak_per_km=leak, peak_annual_burst_per_km=burst,
                         limit_ratio=ratio, category=category))
    return pd.DataFrame(rows)


def reliability_map(route, table):
    m = folium.Map(location=route[0], tiles='OpenStreetMap')
    colors = {'High': '#dc2626', 'Medium': '#eab308', 'Low': '#16a34a'}
    # Merge adjacent equal categories: thousands of healthy joints stay fast.
    groups = (table.category != table.category.shift()).cumsum()
    vertex_distances = [route_length_km(route[:i+1]) for i in range(1, len(route))]
    for _, group in table.groupby(groups):
        start, end = float(group.start_km.iloc[0]), float(group.end_km.iloc[-1])
        distances = [start] + [d for d in vertex_distances if start < d < end] + [end]
        category = group.category.iloc[0]
        folium.PolyLine([point_at_chainage(route, d) for d in distances],
                        color=colors[category], weight=7,
                        tooltip=f'{category}: {start:.3f}–{end:.3f} km').add_to(m)
    # Mark cracked joints so 12 m hotspots remain visible at a 100 km zoom.
    for row in table[table.initial_cracks > 0].itertuples():
        folium.CircleMarker(point_at_chainage(route, (row.start_km + row.end_km)/2),
            radius=6, color=colors[row.category], fill=True, fill_opacity=.9,
            tooltip=f'{row.joint_id}: {row.category}',
            popup=f'{row.joint_id}<br>{row.initial_cracks} initial cracks<br>'
                  f'Peak annual leakage/km: {row.peak_annual_leak_per_km:.3g}<br>'
                  f'Peak annual burst/km: {row.peak_annual_burst_per_km:.3g}').add_to(m)
    m.fit_bounds(route)
    return m
