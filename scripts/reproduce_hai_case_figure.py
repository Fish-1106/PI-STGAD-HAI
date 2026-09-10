"""Generate the representative HAI case-study figure from prepared case data.

The script reads process variables prepared from the official HAI 21.03
``test5.csv`` and the PI-STGAD case-output artifact shipped beside it. It
performs no training, score-channel selection, threshold search, or metric
calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hai_test5_case.json"

PROCESS_COLUMNS = [
    "P1_B3004",
    "P1_LIT01",
    "P1_LCV01D",
    "P1_FCV03D",
    "P1_FT03",
]

COLORS = {
    "attack": "#E76F51",
    "level_sp": "#4D4D4D",
    "level": "#2A9D8F",
    "level_valve": "#E69F00",
    "flow_valve": "#CC6677",
    "flow": "#009E73",
    "prediction": "#0072B2",
    "pressure": "#7E57C2",
    "heat_exchange": "#E69F00",
    "after_oaad": "#0072B2",
    "threshold": "#A33A3A",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.72,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-inputs",
        type=Path,
        default=ROOT / "outputs" / "hai_test5_case_inputs.npz",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def minmax_scale(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    base = array if reference is None else np.asarray(reference, dtype=float)
    low = float(np.nanmin(base))
    high = float(np.nanmax(base))
    if not np.isfinite(high - low) or abs(high - low) < 1e-12:
        return np.full_like(array, 0.5, dtype=float)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def relative_log_residual(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=float), 0.0)
    base = np.maximum(np.asarray(reference, dtype=float), 0.0)
    baseline = np.nanmedian(base) + 1e-9
    return np.log10(1.0 + array / baseline)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_process_case(prepared_path: Path) -> pd.DataFrame:
    if not prepared_path.exists():
        raise SystemExit(
            f"Missing {prepared_path}. Run scripts/prepare_hai_inputs.py first."
        )
    data = np.load(prepared_path)
    required = {"process_rel_min", *(f"process_{name}" for name in PROCESS_COLUMNS)}
    missing = required.difference(data.files)
    if missing:
        raise SystemExit(f"Missing prepared process arrays: {sorted(missing)}")
    segment = pd.DataFrame({"rel_min": data["process_rel_min"].astype(float)})
    for name in PROCESS_COLUMNS:
        segment[name] = data[f"process_{name}"].astype(float)
    return segment


def load_case_outputs(config: dict) -> dict[str, np.ndarray]:
    artifact = ROOT / config["case_output_artifact"]
    data = np.load(artifact)
    required = {
        "rel_min",
        "prediction_error",
        "pressure_residual",
        "heat_exchange_residual",
        "score_before_oaad",
        "score_after_oaad",
    }
    missing = required.difference(data.files)
    if missing:
        raise SystemExit(f"Missing case-output arrays: {sorted(missing)}")
    return {name: data[name].astype(float) for name in required}


def add_attack_span(ax: plt.Axes, config: dict) -> None:
    case_config = config["case"]
    start = case_config["plot_padding_before_seconds"] / 60.0
    duration = (
        pd.Timestamp(case_config["attack_end"])
        - pd.Timestamp(case_config["attack_start"])
    ).total_seconds() / 60.0
    ax.axvspan(start, start + duration, color=COLORS["attack"], alpha=0.14, lw=0, zorder=0)


def add_bottom_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.5, -0.16, f"({label})", transform=ax.transAxes, ha="center", va="top", fontsize=10.0)


def plot_process_panel(ax: plt.Axes, segment: pd.DataFrame, config: dict) -> None:
    add_attack_span(ax, config)
    curves = [
        ("P1_B3004", "B3004 level SP", COLORS["level_sp"]),
        ("P1_LIT01", "LIT01 level", COLORS["level"]),
        ("P1_LCV01D", "LCV01 valve", COLORS["level_valve"]),
        ("P1_FCV03D", "FCV03 outflow valve", COLORS["flow_valve"]),
        ("P1_FT03", "FT03 flow", COLORS["flow"]),
    ]
    for column, label, color in curves:
        y = minmax_scale(segment[column].to_numpy())
        ax.plot(segment["rel_min"], smooth(y, 5), color=color, lw=1.18, label=label)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Target-loop\nvariables")
    ax.grid(axis="y", color="#e7eaee", lw=0.45)


def plot_residual_panel(ax: plt.Axes, case: dict[str, np.ndarray], config: dict) -> None:
    add_attack_span(ax, config)
    curves = [
        ("prediction_error", "Prediction error", COLORS["prediction"], 1.33),
        ("pressure_residual", "Pressure residual", COLORS["pressure"], 1.18),
        ("heat_exchange_residual", "Heat-exchange residual", COLORS["heat_exchange"], 1.18),
    ]
    for key, label, color, width in curves:
        y = relative_log_residual(case[key], case[key])
        y = 0.86 * minmax_scale(y) + 0.04
        ax.plot(case["rel_min"], smooth(y, 3), color=color, lw=width, label=label)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Normalized\nresidual")
    ax.grid(axis="y", color="#e7eaee", lw=0.45)


def plot_score_panel(ax: plt.Axes, case: dict[str, np.ndarray], config: dict) -> None:
    add_attack_span(ax, config)
    threshold = float(config["figure_operating_point"])
    denominator = threshold + 1e-9
    pre = smooth((case["score_before_oaad"] - threshold) / denominator, 3)
    post = smooth((case["score_after_oaad"] - threshold) / denominator, 3)
    reference = np.r_[pre, post, 0.0]
    pre_scaled = minmax_scale(pre, reference)
    post_scaled = minmax_scale(post, reference)
    threshold_scaled = float(minmax_scale(np.array([0.0]), reference)[0])
    x = case["rel_min"]

    ax.fill_between(
        x,
        pre_scaled,
        post_scaled,
        where=post_scaled >= pre_scaled,
        color=COLORS["after_oaad"],
        alpha=0.22,
        linewidth=0,
        label="OAAD lift",
    )
    ax.plot(x, pre_scaled, color="#333333", lw=1.75, ls=(0, (4, 2)), label="Before OAAD", zorder=4)
    ax.plot(x, post_scaled, color=COLORS["after_oaad"], lw=1.85, label="After OAAD", zorder=5)
    ax.axhline(
        threshold_scaled,
        color=COLORS["threshold"],
        lw=0.9,
        ls=(0, (3, 2)),
        label="Detection threshold",
    )
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Normalized\nscore")
    ax.grid(False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    segment = load_process_case(args.prepared_inputs.resolve())
    case = load_case_outputs(config)

    fig = plt.figure(figsize=(5.35, 4.85))
    grid = GridSpec(
        3,
        1,
        height_ratios=[1.28, 1.05, 1.12],
        hspace=0.55,
        left=0.135,
        right=0.985,
        top=0.88,
        bottom=0.15,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]

    plot_process_panel(axes[0], segment, config)
    plot_residual_panel(axes[1], case, config)
    plot_score_panel(axes[2], case, config)

    x_max = float(segment["rel_min"].max())
    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.tick_params(axis="x", pad=2)
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel("")

    for ax, label in zip(axes, "abc"):
        add_bottom_panel_label(ax, label)

    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 1.18),
        ncol=5,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.65,
    )
    attack_patch = Patch(
        facecolor=COLORS["attack"],
        alpha=0.16,
        edgecolor="none",
        label="Ground-truth attack",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        [attack_patch] + handles,
        ["Ground-truth attack"] + labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 1.15),
        ncol=4,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.75,
    )
    axes[2].legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 1.15),
        ncol=4,
        frameon=False,
        handlelength=1.55,
        columnspacing=0.8,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / config["output_stem"]
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
