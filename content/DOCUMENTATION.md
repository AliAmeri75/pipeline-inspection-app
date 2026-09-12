## What the application does

The scheduling page evaluates a user-defined set of **equidistant inspection
intervals** over a selected analysis horizon. Each candidate interval is run
with the same random seed so the economic and reliability results can be
compared consistently.

The model can represent:

- multiple individual pipe joints or groups of statistically identical joints;
- one or more initial cracks per joint, as well as initially crack-free joints;
- uncertain crack growth and future crack initiation;
- inspection probability of detection and crack-sizing error;
- two repair criteria and configurable repair effectiveness; and
- inspection, repair, leak, and burst costs with economic discounting.

## Typical workflow

1. Open **Inspection scheduling** from the navigation menu.
2. Enter the Monte Carlo sample size, analysis horizon, and candidate interval range.
3. Upload a joint/crack CSV or edit the example table directly.
4. Review growth, cost, repair, uncertainty, and safety assumptions in the sidebar.
5. Run the optimization and compare the candidate schedules.
6. Download the interval-results table for reporting or sensitivity analysis.

## Main outputs

The application reports the preferred candidate interval, expected present
life-cycle cost, inspection/repair/failure cost components, expected number of
repairs, and whether the leak and burst safety criteria are satisfied. It also
plots the cost comparison and annual leak/burst probabilities for the selected
schedule.

## Scope and limitations

The current optimization compares only the candidate fixed intervals supplied
by the user. Results depend on the selected physical, probabilistic, inspection,
repair, consequence, and economic assumptions. A production integrity decision
should include data-quality review, model validation, sensitivity analysis, and
assessment by qualified pipeline-integrity professionals.
