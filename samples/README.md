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
is a visualization convention. Overlapping 1 km windows advance by one nominal
joint, with a final window ending at the route endpoint. Window cumulative Pf is
`1 - product(1 - joint cumulative Pf)` for each year, using the simulation's
independent-joint assumption. Annual conditional increments are then calculated
from each window's curve; joint maxima from different years are never combined.
The largest annual value is compared with the selected project limits.

An intersected boundary joint is included in full, conservatively, because
within-joint failure locations are unknown. Thus the joint coverage may slightly
exceed 1 km. Routes shorter than 1 km use the actual short section without
extrapolation. Each mapped joint inherits the worst category of all windows
overlapping it. The critical-window table gives the precise evaluated start/end
positions, leakage/burst probabilities and their peak years.

This is a selected assessment method, not a verified CSA compliance implementation.
The full-pipeline scheduling safety calculation and optimum selection still use
the existing whole-route calculation. The map may therefore flag a critical
local section even when the whole-route average passes. These colours do not
include spatial consequences and are not quantitative consequence-based risk.

Joints without initial cracks share the simulated crack-free group distribution;
they are not forced green, because crack birth is included. A zero Monte Carlo
failure estimate is not proof of zero failure probability. The displayed map is
the saved route from the completed run. Run again after changing inputs.

Adjacent equal-colour segments are combined for map performance. Coloured markers
make cracked joints visible when the whole 100 km route is shown. Download the
joint profile CSV for individual joint results and distances from the start.
