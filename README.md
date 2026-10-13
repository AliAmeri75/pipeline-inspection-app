# Pipeline Scheduling and Value of Information Lab

This folder is a self-contained Streamlit demonstration comparing the optimized
J3 ILI schedule with a prior excavation schedule adapted from
`System_regular_repair.py` and `optimize_regular_repair.py`.

The comparison uses common pipe, crack, deterioration, cost, candidate-interval,
and Monte Carlo inputs. For each method, the recommended interval is the
lowest-cost candidate that satisfies both safety criteria. An unsafe candidate
is never presented as the optimum. It reports net VoI as:

`minimum safe prior excavation cost - minimum safe ILI-informed cost`

Click any cost-curve point to see the interval's total cost, failure cost
(`Cf`), repair cost (`Cr`), and its **Safe** or **Unsafe** status.

## Start locally

On macOS, double-click `run_app.command`. On Windows, double-click
`run_app.bat`. The first start creates a private Python environment and installs
the packages in `requirements.txt`.

The application opens on a separate **Introduction** page. Use the navigation
menu to open **Scheduling and VoI**.

## Edit the introduction page

- Edit `content/ABSTRACT.md` to change the short abstract.
- Edit `content/DOCUMENTATION.md` to change the explanation and documentation.
- Replace the files in `assets/` to update the two portraits or university logo.

The scheduling interface remains in `inspection_planner.py`; the shared
simulation logic is in `Simp_Sys_Thresh_J3_optimized.py`, and the pure
optimization/VoI helpers are in `planning_analysis.py`. The small
`streamlit_app.py` file only configures navigation, so the Streamlit Community
Cloud entry point remains unchanged.

See [`docs/INTRODUCTION_PAGE_SETUP.md`](docs/INTRODUCTION_PAGE_SETUP.md) for the
step-by-step implementation and editing workflow.

## Joint/crack CSV

Use one row for each unique joint or statistically identical joint group:

- `joint_id`: unique display name.
- `count`: number of statistically identical joints represented by the row.
- `num_cracks`: number of initial cracks in each represented joint.
- `mean_lengths_mm`: semicolon-separated mean initial crack lengths.
- `mean_depths_mm`: semicolon-separated mean initial crack depths.
- `WT_mean`, `WT_std`: wall-thickness distribution.
- `Pservice_mean`, `Pservice_std`: service-pressure distribution.
- `birth_rate`: crack-initiation rate per year.

For zero initial cracks, set `num_cracks` to zero and leave both size fields
blank. Additional property columns can be supplied using the pattern
`PROPERTY_mean`, `PROPERTY_std`, and `PROPERTY_distribution`, where PROPERTY is
one of `alpha`, `WT`, `Neq`, `C`, `SMYS`, `SMTS`, `CVN`, `E`, `err`, or
`Pservice`.

## Public deployment and QR code

1. Put this folder in a GitHub repository.
2. Create an app at https://share.streamlit.io.
3. Select the repository and `streamlit_app.py` as the entrypoint.
4. Make the app public and copy its stable `streamlit.app` URL.
5. Paste that URL into **Create a QR code** in the app sidebar, then download
   the PNG for a poster or presentation.

The default one-joint, 1,000-sample example is intended for an interactive
demonstration. Large numbers of individually different joints require
proportionally more computation; use the `count` column for identical groups.
