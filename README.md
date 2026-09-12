# Pipeline Inspection Schedule Lab

This folder is a self-contained Streamlit demonstration of the optimized J3
inspection-schedule model. The original DBN scripts are not modified.

## Start locally

On macOS, double-click `run_app.command`. On Windows, double-click
`run_app.bat`. The first start creates a private Python environment and installs
the packages in `requirements.txt`.

The application opens on a separate **Introduction** page. Use the navigation
menu to open **Inspection scheduling**.

## Edit the introduction page

- Edit `content/ABSTRACT.md` to change the short abstract.
- Edit `content/DOCUMENTATION.md` to change the explanation and documentation.
- Replace the files in `assets/` to update the two portraits or university logo.

The scheduling calculations remain in `inspection_planner.py`. The small
`streamlit_app.py` file only configures the two-page navigation, so the
Streamlit Community Cloud entry point remains unchanged.

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
