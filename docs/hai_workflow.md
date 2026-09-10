# PI-STGAD HAI method workflow

This document connects the public HAI materials to the principal stages of the
PI-STGAD case-study analysis. Each section identifies the stage’s purpose,
inputs, outputs, and corresponding repository files.

## 1. HAI process data

**Purpose.** Provide the normal-operation records used to establish data
preparation parameters and the test record containing the selected diagnostic
case.

**Input.** Official HAI 21.03 `train1.csv`–`train3.csv` and `test5.csv` files.

**Output.** Public P1 boiler variables sampled at 1 Hz.

**Files.** `data/README.md` and `configs/hai_data_schema.json`.

## 2. Data preparation

**Purpose.** Convert the official P1 variables into consistently scaled model
inputs while retaining the temporal structure of the process.

The eight physical sensor channels are filtered with an 11-point, third-order
Savitzky–Golay filter. Per-variable min–max parameters are fitted on the three
normal-operation training files and then applied to the selected test record.
The preparation step does not load per-sample attack-label columns for
model-input construction.

**Input.** P1 sensor, actuator, and operational-context columns.

**Output.** Normalized sensor arrays, normalized operational-context arrays,
normalization parameters, and process variables used by the case figure.

**Files.** `scripts/prepare_hai_inputs.py` and
`configs/hai_data_schema.json`.

## 3. Graph-window construction

**Purpose.** Organize each 120-s sensor window as a fixed 17-node process graph
with a separate nine-channel operational-context sequence.

The graph contains eight sensor nodes and nine actuator nodes. Its edges are
fixed from documented P1 boiler connectivity projected onto the observed
variables. Direct process links have weight 1.0, while retained two-hop links
have weight 0.5. The test windows advance by 5 s. The prediction target is the
normalized 17-node process state immediately following the input window.

**Input.** Normalized P1 arrays and fixed process topology.

**Output.** `X_sensor` with node-by-time structure, `X_op` with
operational-variable-by-time structure, and the one-step target `y_target`.

**Files.** `configs/hai_graph.json`, `configs/hai_data_schema.json`, and
`scripts/prepare_hai_inputs.py`.

## 4. Temporal representation

**Purpose.** Summarize the recent dynamics of each graph node before spatial
aggregation.

**Input.** A 120-s sequence for each sensor or actuator node.

**Output.** Per-node temporal representations for graph processing.

**Files.** The representation settings used for the HAI experiment are
recorded in `configs/hai_test5_case.json`; this stage is documented here to
locate the released HAI materials within the PI-STGAD method flow.

## 5. Topology-aware graph aggregation

**Purpose.** Combine each node’s temporal representation with information from
its process-connected neighborhood.

**Input.** Per-node temporal representations, process-connectivity edges, and
fixed edge weights.

**Output.** Topology-aware node representations.

**Files.** `configs/hai_graph.json` records the inspectable graph used by the
HAI analysis.

## 6. Operational-context-conditioned one-step prediction

**Purpose.** Predict the next normalized process state while accounting for
the operating conditions represented by the nine context variables.

**Input.** Topology-aware node representations and the operational-context
window.

**Output.** One-step node predictions aligned with `y_target`.

**Files.** The inputs and case settings are defined in
`configs/hai_data_schema.json` and `configs/hai_test5_case.json`.

## 7. Prediction-error and physics-residual evidence

**Purpose.** Describe departures from learned next-step behavior together with
departures from the reduced-order process relations used in the HAI analysis.

**Input.** Case-aligned predictions, process states, and residual channels.

**Output.** Prediction-error, pressure-residual, and heat-exchange-residual
evidence for the selected diagnostic interval.

**Files.** `artifacts/hai_case_artifact_seed42.npz` contains the case-specific
model-derived values used by the figure.

## 8. OAAD score enhancement

**Purpose.** Compare the case score before and after continuous local-context
enhancement.

**Input.** Case-aligned score responses.

**Output.** Score traces before and after OAAD enhancement.

**Files.** `artifacts/hai_case_artifact_seed42.npz` and
`configs/hai_test5_case.json`.

## 9. Representative HAI case-study analysis

**Purpose.** Present process context, anomaly evidence, and the OAAD response
in a single diagnostic figure.

**Input.** Prepared process variables, the case artifact, and the fixed case
configuration.

**Output.** `outputs/hai_case_study.pdf`,
`outputs/hai_case_study.svg`, and `outputs/hai_case_study.png`.

**Files.** `scripts/reproduce_hai_case_figure.py` and
`assets/hai_case_reference.png`.

The three panels show target-loop variables and context, prediction-error and
physics-residual evidence, and score responses before and after OAAD
enhancement. Together, they provide an inspectable diagnostic example of the
HAI analysis path.
