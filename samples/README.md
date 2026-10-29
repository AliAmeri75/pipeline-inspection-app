# Alberta 100 km teaching example

These are synthetic sample data for a fictional route west of Edmonton, Alberta.
They are not an operating pipeline route, surveyed asset locations, or inspection records.

In the map input mode, click **Load 100 km Alberta demo** to load everything.
Alternatively upload `alberta_100km_route.geojson`, apply it, then upload and apply
`alberta_100km_cracks.csv`. The joint CSV is for the existing joint-table input mode;
it does not contain route geometry.

The route is 100 km with six directional segments, 8,334 nominal 12 m joints
(the final joint is shorter), and 12 cracks in 10 joints. Two joints each contain
two cracks. These sizes are examples, not measurements or calibrated uncertainty
distributions. Review model assumptions before running the study.

After a mapped analysis, expand **Pipeline joint risk profile** and select the
planning method and interval. Red indicates a local indicator above a selected
limit, yellow indicates 50–100%, and green indicates below 50%. The 50% boundary
is a visualization convention. Local indicators are peak annual conditional
failure probabilities normalized by each physical joint's length in km, matching
the system model's conditional-increment convention. The full-pipeline safety
decision remains in the scheduling results. These colours do not include spatial
consequences, and therefore do not constitute quantitative location-specific risk.

Joints without initial cracks share the simulated crack-free group distribution;
they are not forced green, because crack birth is included. A zero Monte Carlo
failure estimate is not proof of zero failure probability. The displayed map is
the saved route from the completed run. Run again after changing inputs.

Adjacent equal-colour segments are combined for map performance. Coloured markers
make cracked joints visible when the whole 100 km route is shown. Download the
joint profile CSV for individual joint results and distances from the start.
