"""Validate the public PI-STGAD HAI case-study release."""
from __future__ import annotations

import json
import importlib.util
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"

REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "environment.yml",
    ROOT / ".gitignore",
    ROOT / ".gitattributes",
    CONFIG_DIR / "hai_test5_case.json",
    CONFIG_DIR / "hai_data_schema.json",
    CONFIG_DIR / "hai_graph.json",
    ROOT / "docs" / "hai_workflow.md",
    ROOT / "scripts" / "prepare_hai_inputs.py",
    ROOT / "scripts" / "reproduce_hai_case_figure.py",
    ROOT / "scripts" / "validate_release.py",
    ROOT / "artifacts" / "README.md",
    ROOT / "artifacts" / "hai_case_artifact_seed42.npz",
    ROOT / "assets" / "hai_case_reference.png",
    ROOT / "data" / "README.md",
]

PREPARED_INPUTS = OUTPUT_DIR / "hai_test5_case_inputs.npz"
EXPECTED_FIGURES = [
    OUTPUT_DIR / "hai_case_study.pdf",
    OUTPUT_DIR / "hai_case_study.svg",
    OUTPUT_DIR / "hai_case_study.png",
]
FORBIDDEN_ARRAY_TOKENS = ("attack", "anomaly", "label")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse {path.relative_to(ROOT)}: {exc}") from exc


def validate_layout() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    require(not missing, f"Missing repository files: {missing}")
    require(not (ROOT / "LICENSE").exists(), "Unexpected LICENSE file in release candidate")


def validate_configuration() -> tuple[dict, dict, dict]:
    case = load_json(CONFIG_DIR / "hai_test5_case.json")
    schema = load_json(CONFIG_DIR / "hai_data_schema.json")
    graph = load_json(CONFIG_DIR / "hai_graph.json")

    graph_input = schema["graph_input"]
    operational = schema["operational_context"]
    windowing = schema["windowing"]
    nodes = graph_input["node_order"]
    node_names = [entry["name"] for entry in nodes]
    node_types = [entry["type"] for entry in nodes]

    require(graph_input["node_count"] == 17, "Expected 17 graph nodes")
    require(len(nodes) == 17 and len(set(node_names)) == 17, "Invalid node ordering")
    require(node_types.count("sensor") == 8, "Expected 8 sensor nodes")
    require(node_types.count("actuator") == 9, "Expected 9 actuator nodes")
    require(operational["variable_count"] == 9, "Expected 9 operational variables")
    require(len(operational["variables"]) == 9, "Operational-variable list mismatch")
    require(windowing["input_window_length_seconds"] == 120, "Expected 120-s window")
    require(windowing["training_stride_seconds"] == 60, "Expected 60-s training stride")
    require(windowing["testing_stride_seconds"] == 5, "Expected 5-s test stride")
    require(case["window_size_seconds"] == 120, "Case window mismatch")
    require(case["test_stride_seconds"] == 5, "Case test-stride mismatch")
    require(case["output_stem"] == "hai_case_study", "Unexpected output stem")
    canonical_case_fields = {
        "attack_start",
        "attack_end",
        "plot_padding_before_seconds",
        "plot_padding_after_seconds",
    }
    require(
        "case" in case and isinstance(case["case"], dict),
        "Missing canonical case configuration",
    )
    require(
        canonical_case_fields.issubset(case["case"]),
        "Canonical case configuration is incomplete",
    )
    require(
        canonical_case_fields.isdisjoint(case),
        "Duplicate case fields found at the configuration top level",
    )
    experiment = case["experiment_parameters"]
    expected_experiment = {
        "epochs": 180,
        "batch_size": 128,
        "learning_rate": 0.001,
        "tcn_hidden_dimension": 32,
        "gat_attention_heads": 4,
        "gat_per_head_dimension": 32,
        "operational_context_hidden_dimension": 32,
        "physics_loss_weight": 0.05,
        "dropout": 0.3,
        "early_stopping_patience": 25,
    }
    for key, expected in expected_experiment.items():
        require(experiment[key] == expected, f"Experiment-configuration mismatch: {key}")

    graph_names = [entry["name"] for entry in graph["nodes"]]
    require(graph["node_count"] == 17 and graph_names == node_names, "Graph node mismatch")
    require(graph["edge_count"] == 31, "Expected 31 undirected graph edges")
    require(len(graph["edges"]) == 31, "Graph edge-list length mismatch")
    seen_edges: set[tuple[str, str]] = set()
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        require(source in node_names and target in node_names, "Graph edge has unknown node")
        require(source != target, "Graph contains a self-loop")
        key = tuple(sorted((source, target)))
        require(key not in seen_edges, f"Duplicate graph edge: {key}")
        seen_edges.add(key)
        expected_weight = 1.0 / int(edge["process_hops"])
        require(np.isclose(edge["weight"], expected_weight), f"Edge-weight mismatch: {key}")

    preparation_path = ROOT / "scripts" / "prepare_hai_inputs.py"
    module_spec = importlib.util.spec_from_file_location("hai_public_preparation", preparation_path)
    require(module_spec is not None and module_spec.loader is not None, "Cannot load preparation script")
    preparation = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(preparation)
    schema_columns = [entry["csv_column"] for entry in nodes]
    operational_columns = [entry["csv_column"] for entry in operational["variables"]]
    sensor_filter = schema["preprocessing"]["sensor_filter"]
    require(preparation.SENSOR_COLUMNS == schema_columns, "Preparation/schema node-column mismatch")
    require(preparation.OPERATIONAL_COLUMNS == operational_columns, "Preparation/schema context-column mismatch")
    require(preparation.N_PHYSICAL_SENSORS == 8, "Preparation sensor count mismatch")
    require(preparation.FILTER_WINDOW == sensor_filter["window_length"], "Filter-window mismatch")
    require(preparation.FILTER_POLYORDER == sensor_filter["polynomial_order"], "Filter-order mismatch")
    require(preparation.WINDOW_SIZE == windowing["input_window_length_seconds"], "Preparation window mismatch")
    require(preparation.TEST_STRIDE == windowing["testing_stride_seconds"], "Preparation stride mismatch")

    return case, schema, graph


def validate_arrays(case: dict, schema: dict) -> dict[str, int]:
    require(PREPARED_INPUTS.is_file(), "Prepared inputs are missing; run data preparation first")
    artifact_path = ROOT / case["case_output_artifact"]
    require(artifact_path.is_file(), "Case artifact is missing")

    prepared = np.load(PREPARED_INPUTS, allow_pickle=False)
    artifact = np.load(artifact_path, allow_pickle=False)
    forbidden_prepared = [
        key for key in prepared.files if any(token in key.lower() for token in FORBIDDEN_ARRAY_TOKENS)
    ]
    forbidden_artifact = [
        key for key in artifact.files if any(token in key.lower() for token in FORBIDDEN_ARRAY_TOKENS)
    ]
    require(not forbidden_prepared, f"Forbidden prepared-input fields: {forbidden_prepared}")
    require(not forbidden_artifact, f"Forbidden artifact fields: {forbidden_artifact}")

    prepared_required = {
        "X_sensor",
        "X_op",
        "y_target",
        "timestamps",
        "window_size",
        "process_rel_min",
    }
    artifact_required = {
        "rel_min",
        "prediction_error",
        "pressure_residual",
        "heat_exchange_residual",
        "score_before_oaad",
        "score_after_oaad",
    }
    require(not prepared_required.difference(prepared.files), "Prepared-input fields are missing")
    require(not artifact_required.difference(artifact.files), "Case-artifact fields are missing")

    x_sensor = prepared["X_sensor"]
    x_op = prepared["X_op"]
    y_target = prepared["y_target"]
    timestamps = prepared["timestamps"]
    case_points = artifact["rel_min"]
    window_size = schema["windowing"]["input_window_length_seconds"]
    test_stride = schema["windowing"]["testing_stride_seconds"]

    require(x_sensor.ndim == 3 and x_sensor.shape[1:] == (17, window_size), "X_sensor shape mismatch")
    require(x_op.ndim == 3 and x_op.shape[1:] == (9, window_size), "X_op shape mismatch")
    require(y_target.shape == (x_sensor.shape[0], 17), "y_target shape mismatch")
    require(x_op.shape[0] == x_sensor.shape[0] == timestamps.shape[0], "Prepared-array length mismatch")
    require(case_points.shape == (x_sensor.shape[0],), "Case/artifact window-count mismatch")
    for key in artifact_required:
        require(artifact[key].shape == case_points.shape, f"Artifact alignment mismatch: {key}")
        require(np.isfinite(artifact[key]).all(), f"Non-finite artifact values: {key}")
    require(np.all(np.diff(timestamps) == test_stride), "Prepared timestamps do not follow the test stride")
    require(
        np.allclose(np.diff(case_points) * 60.0, test_stride, atol=1e-8),
        "Case relative times do not follow the test stride",
    )

    return {
        "case_windows": int(x_sensor.shape[0]),
        "graph_nodes": int(x_sensor.shape[1]),
        "operational_variables": int(x_op.shape[1]),
        "window_seconds": int(x_sensor.shape[2]),
    }


def validate_figures() -> dict[str, float | list[int]]:
    missing = [str(path.relative_to(ROOT)) for path in EXPECTED_FIGURES if not path.is_file()]
    require(not missing, f"Missing generated figures: {missing}")
    for path in EXPECTED_FIGURES:
        require(path.stat().st_size > 0, f"Empty generated figure: {path.name}")

    generated = mpimg.imread(OUTPUT_DIR / "hai_case_study.png").astype(np.float64)
    reference = mpimg.imread(ROOT / "assets" / "hai_case_reference.png").astype(np.float64)
    require(generated.shape == reference.shape, "Generated/reference image-shape mismatch")
    mean_absolute_error = float(np.mean(np.abs(generated - reference)))
    require(mean_absolute_error <= 0.01, "Generated image differs from the reference")
    return {
        "png_shape": list(generated.shape),
        "reference_mean_absolute_error": mean_absolute_error,
    }


def validate_git_exclusions() -> None:
    if not (ROOT / ".git").exists():
        return
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tracked = {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}
    forbidden = [
        path
        for path in tracked
        if path.startswith("outputs/")
        or path.startswith("data/hai-21.03/")
        or "__pycache__" in path
        or path == "LICENSE"
    ]
    require(not forbidden, f"Generated, third-party, or excluded files are tracked: {forbidden}")


def main() -> int:
    validate_layout()
    case, schema, _ = validate_configuration()
    array_summary = validate_arrays(case, schema)
    figure_summary = validate_figures()
    validate_git_exclusions()
    print("Release validation passed.")
    print(json.dumps({**array_summary, **figure_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
