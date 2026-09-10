"""Prepare PI-STGAD inputs for the plotted HAI 21.03 test-5 case.

This is the data-input portion of the archived experiment pipeline. It fits
normalization only on the official normal training files, applies the same
sensor filter, and creates the 120-s test windows that overlap the case figure.
Per-sample attack-label columns are not loaded or written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hai_test5_case.json"
TRAIN_FILES = ("train1.csv", "train2.csv", "train3.csv")

SENSOR_COLUMNS = [
    "P1_TIT01",
    "P1_TIT02",
    "P1_PIT01",
    "P1_PIT02",
    "P1_FT01",
    "P1_FT02",
    "P1_FT03",
    "P1_LIT01",
    "P1_PCV01Z",
    "P1_PCV02Z",
    "P1_LCV01Z",
    "P1_FCV01Z",
    "P1_FCV02Z",
    "P1_FCV03Z",
    "P1_PP01AR",
    "P1_PP01BR",
    "P1_PP02R",
]

OPERATIONAL_COLUMNS = [
    "P1_B2004",
    "P1_B2016",
    "P1_B3004",
    "P1_B3005",
    "P1_B4002",
    "P1_B4022",
    "P1_B4005",
    "P1_B400B",
    "P1_STSP",
]

N_PHYSICAL_SENSORS = 8
FILTER_WINDOW = 11
FILTER_POLYORDER = 3
WINDOW_SIZE = 120
TEST_STRIDE = 5

FIGURE_COLUMNS = [
    "P1_B3004",
    "P1_LIT01",
    "P1_LCV01D",
    "P1_FCV03D",
    "P1_FT03",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hai-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "hai_test5_case_inputs.npz",
    )
    return parser.parse_args()


def load_values(path: Path) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    columns = ["time", *SENSOR_COLUMNS, *OPERATIONAL_COLUMNS]
    frame = pd.read_csv(path, usecols=columns, parse_dates=["time"])
    sensor = frame[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    operational = frame[OPERATIONAL_COLUMNS].to_numpy(dtype=np.float32)
    return sensor, operational, frame["time"]


def filter_physical_sensors(values: np.ndarray) -> np.ndarray:
    filtered = values.copy()
    for index in range(N_PHYSICAL_SENSORS):
        filtered[:, index] = savgol_filter(
            values[:, index],
            window_length=FILTER_WINDOW,
            polyorder=FILTER_POLYORDER,
        )
    return filtered


def fit_minmax(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate(arrays, axis=0)
    minimum = joined.min(axis=0, keepdims=True)
    maximum = joined.max(axis=0, keepdims=True)
    value_range = np.where(maximum - minimum < 1e-8, 1.0, maximum - minimum)
    return minimum, value_range


def normalize(values: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, value_range = scaler
    return ((values - minimum) / value_range).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hai_root = args.hai_root.resolve()

    train_sensors: list[np.ndarray] = []
    train_operational: list[np.ndarray] = []
    for filename in TRAIN_FILES:
        sensor, operational, _ = load_values(hai_root / filename)
        train_sensors.append(filter_physical_sensors(sensor))
        train_operational.append(operational)

    sensor_scaler = fit_minmax(train_sensors)
    operational_scaler = fit_minmax(train_operational)

    test_path = hai_root / config["test_file"]
    test_columns = ["time", *SENSOR_COLUMNS, *OPERATIONAL_COLUMNS, *FIGURE_COLUMNS]
    test_columns = list(dict.fromkeys(test_columns))
    test_frame = pd.read_csv(test_path, usecols=test_columns, parse_dates=["time"])
    test_sensor = test_frame[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    test_operational = test_frame[OPERATIONAL_COLUMNS].to_numpy(dtype=np.float32)
    test_time = test_frame["time"]
    test_sensor = filter_physical_sensors(test_sensor)
    sensor_normalized = normalize(test_sensor, sensor_scaler)
    operational_normalized = normalize(test_operational, operational_scaler)

    case_config = config["case"]
    attack_start = pd.Timestamp(case_config["attack_start"])
    attack_end = pd.Timestamp(case_config["attack_end"])
    plot_start = attack_start - pd.Timedelta(
        seconds=case_config["plot_padding_before_seconds"]
    )
    plot_end = attack_end + pd.Timedelta(
        seconds=case_config["plot_padding_after_seconds"]
    )
    base_time = test_time.iloc[0]
    first_start = int((plot_start - base_time).total_seconds())
    final_start = int((plot_end - base_time).total_seconds())

    starts = np.arange(0, len(test_sensor) - WINDOW_SIZE, TEST_STRIDE, dtype=np.int64)
    starts = starts[(starts >= first_start) & (starts <= final_start)]
    windows_sensor = np.stack(
        [sensor_normalized[start : start + WINDOW_SIZE] for start in starts], axis=0
    ).transpose(0, 2, 1)
    windows_operational = np.stack(
        [operational_normalized[start : start + WINDOW_SIZE] for start in starts], axis=0
    ).transpose(0, 2, 1)
    targets = sensor_normalized[starts + WINDOW_SIZE]

    process_mask = (test_time >= plot_start) & (test_time <= plot_end)
    process_frame = test_frame.loc[process_mask, ["time", *FIGURE_COLUMNS]].copy()
    process_rel_min = (
        (process_frame["time"] - plot_start).dt.total_seconds().to_numpy(dtype=float)
        / 60.0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_sensor=windows_sensor,
        X_op=windows_operational,
        y_target=targets,
        timestamps=starts,
        sensor_min=sensor_scaler[0],
        sensor_range=sensor_scaler[1],
        operational_min=operational_scaler[0],
        operational_range=operational_scaler[1],
        window_size=np.array([WINDOW_SIZE], dtype=np.int64),
        process_rel_min=process_rel_min,
        **{
            f"process_{column}": process_frame[column].to_numpy(dtype=np.float64)
            for column in FIGURE_COLUMNS
        },
    )
    print(
        f"Saved {args.output} with {len(starts)} case windows; "
        f"X_sensor={windows_sensor.shape}, X_op={windows_operational.shape}."
    )


if __name__ == "__main__":
    main()
