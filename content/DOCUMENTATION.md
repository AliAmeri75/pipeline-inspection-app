## What the application does

The scheduling page evaluates a user-defined set of **equidistant intervention
intervals** over a selected analysis horizon. Each candidate is run with the
same random seed for two methods:

- **Prior excavation without ILI:** joints known to contain initial cracks are
  excavated at the scheduled times. Observation is perfect, no POD or sizing
  error is used, and at least the type-1 intervention is applied. A type-2
  intervention is added when its depth or pressure criterion is reached.
- **Pre-posterior ILI scheduling:** the current POD and sizing-error models
  determine whether detected cracks trigger an intervention. Discounted ILI
  costs are included.

The model can represent:

- multiple individual pipe joints or groups of statistically identical joints;
- one or more initial cracks per joint, as well as initially crack-free joints;
- uncertain crack growth and future crack initiation;
- ILI probability of detection and crack-sizing error;
- two repair criteria and configurable repair effectiveness; and
- inspection, repair, leak, and burst costs with economic discounting.

## Typical workflow

1. Open **Inspection scheduling** from the navigation menu.
2. Enter the Monte Carlo sample size, analysis horizon, and candidate interval range.
3. Upload a joint/crack CSV or edit the example table directly.
4. Review growth, cost, repair, uncertainty, and safety assumptions in the sidebar.
5. Run the prior and pre-posterior optimizations together.
6. Review the optimized schedules, cost curves, POF plots, and net VoI.
7. Download the comparison or method-specific tables.

## Main outputs

The application reports the preferred safe candidate for each method, expected
present life-cycle cost, ILI/repair/failure cost components, expected number of
interventions, and whether the leak and burst criteria are satisfied. It also
plots the cost comparison and annual leak/burst probabilities.

The reported net value of information is:

**VoI = minimum prior excavation cost − minimum ILI-informed cost**

The ILI-informed cost already includes discounted ILI costs. Positive VoI means
the ILI-informed strategy has the lower expected lifecycle cost under the stated
assumptions.

## Scope and limitations

The prior method follows the supplied regular-repair model and targets only
joints represented with initial cracks; initially crack-free joints continue
without scheduled excavation. The optimization compares only the candidate
fixed intervals supplied by the user. Results depend on the selected physical,
probabilistic, inspection, repair, consequence, and economic assumptions. A
production integrity decision should include data-quality review, model
validation, sensitivity analysis, and assessment by qualified
pipeline-integrity professionals.
