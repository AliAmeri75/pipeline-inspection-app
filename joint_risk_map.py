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
                  + (f'Worst enclosing window: {row.critical_window_start_km:.3f}–{row.critical_window_end_km:.3f} km<br>'
                     f'Peak annual window leakage Pf: {row.peak_annual_window_leak_pof:.3g}<br>'
                     f'Peak annual window burst Pf: {row.peak_annual_window_burst_pof:.3g}'
                     if 'critical_window_start_km' in table.columns else
                     f'Peak annual leakage/km: {row.peak_annual_leak_per_km:.3g}<br>'
                     f'Peak annual burst/km: {row.peak_annual_burst_per_km:.3g}')).add_to(m)
    m.fit_bounds(route)
    return m


def sliding_window_reliability(result, route, cracks, joint_length_m):
    """Evaluate overlapping 1 km windows, advancing one physical joint.

    Combine cumulative joint Pf as 1-product(1-Pf), matching the simulator's
    independent-joint assumption, BEFORE calculating annual increments. Never
    combine individual joints' maxima from different years. Include a complete
    boundary joint conservatively because the model has no within-joint failure
    location distribution. Lines shorter than 1 km use their actual length and
    return a short-section indicator without extrapolation to 1 km.
    """
    if not math.isfinite(joint_length_m) or joint_length_m <= 0:
        raise ValueError('Joint length must be positive and finite.')
    length = route_length_km(route)
    if length <= 0:
        raise ValueError('Route must have positive length.')
    times = np.asarray(result['times'], dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('At least two increasing result times are required.')
    count = math.ceil(length * 1000 / joint_length_m)
    edges = np.minimum(np.arange(count + 1) * joint_length_m / 1000, length)
    terminal = max(0., length - 1.)
    starts = np.unique(np.r_[edges[:-1][edges[:-1] < terminal], terminal])
    ends = np.minimum(starts + 1., length)
    left = np.searchsorted(edges, starts + 1e-10, side='right') - 1
    right = np.searchsorted(edges, ends - 1e-10, side='left')
    names = {name: i for i, name in enumerate(result['joint_names'])}
    indices = []
    for number in range(1, count + 1):
        index = names.get(f'Joint {number:05d}', names.get('Remaining crack-free joints'))
        if index is None:
            raise ValueError(f'Missing simulation results for joint {number}')
        indices.append(index)
    peaks, years = [], []
    for key in ('joint_leak_pofs', 'joint_burst_pofs'):
        cumulative = np.asarray(result[key], dtype=float)[indices]
        if cumulative.shape != (count, len(times)) or not np.isfinite(cumulative).all():
            raise ValueError('Joint probabilities must be finite and match result times.')
        if (cumulative < 0).any() or (cumulative > 1).any() or (np.diff(cumulative, axis=1) < -1e-12).any():
            raise ValueError('Cumulative joint probabilities must increase within [0, 1].')
        # Prefix sums compute all window products efficiently, including Pf=1.
        logs = np.log1p(-np.minimum(cumulative, 1 - np.finfo(float).eps))
        prefix = np.vstack([np.zeros(len(times)), np.cumsum(logs, axis=0)])
        saturated = np.vstack([np.zeros(len(times)), np.cumsum(cumulative == 1, axis=0)])
        log_survival = prefix[right] - prefix[left]
        log_survival[(saturated[right] - saturated[left]) > 0] = -np.inf
        previous = log_survival[:, :-1]
        annual = np.zeros_like(previous)
        valid = np.isfinite(previous)
        difference = np.zeros_like(previous)
        np.subtract(log_survival[:, 1:], previous, out=difference, where=valid)
        annual[valid] = -np.expm1(np.minimum(difference[valid], 0))
        annual /= np.diff(times)
        peak_index = np.argmax(annual, axis=1)
        peaks.append(annual[np.arange(len(starts)), peak_index])
        years.append(times[1:][peak_index])
    ratio = np.maximum(peaks[0] / result['max_annual_leak_pof'],
                       peaks[1] / result['max_annual_burst_pof'])
    categories = np.where(ratio > 1, 'High', np.where(ratio >= .5, 'Medium', 'Low'))
    windows = pd.DataFrame(dict(start_km=starts, end_km=ends,
        section_length_km=ends-starts, peak_annual_leak_pof=peaks[0],
        peak_annual_burst_pof=peaks[1], leak_peak_year=years[0], burst_peak_year=years[1],
        limit_ratio=ratio, category=categories))
    crack_counts = cracks.groupby('joint_number').size().to_dict() if not cracks.empty else {}
    rows = []
    for i in range(count):
        # Colour each joint by the worst sliding window that contains it.
        a = np.searchsorted(ends, edges[i] + 1e-10, side='right')
        b = np.searchsorted(starts, edges[i+1] - 1e-10, side='left')
        selected = a + int(np.argmax(ratio[a:b]))
        rows.append(dict(joint_id=f'Joint {i+1:05d}', start_km=edges[i], end_km=edges[i+1],
            initial_cracks=crack_counts.get(i+1, 0), category=categories[selected],
            limit_ratio=ratio[selected], critical_window_start_km=starts[selected],
            critical_window_end_km=ends[selected],
            peak_annual_window_leak_pof=peaks[0][selected],
            peak_annual_window_burst_pof=peaks[1][selected]))
    return windows, pd.DataFrame(rows)
