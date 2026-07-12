#!/usr/bin/env python3
"""Plot current-drive contours from an existing JSONL file.  No NC reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from plot_alpha_beta_rho_ratio_surface import (
    PARULA,
    SCRIPT_DIR,
    _bold_tick_labels,
    configure_plot_style,
    mode_tag,
    normalize_suffix,
)


SAVE_DPI = 300


def load_jsonl(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record:
                continue
            for key in ("frqncy", "ioxm", "alfast", "betast", "driven_current_ka"):
                if key not in record:
                    raise KeyError(f"{jsonl_path}:{line_number} missing field {key}")
            # Retain only the five scalars used by this plot.  In particular,
            # do not keep rho_bin_center/powden_e profile arrays for all cases.
            rows.append({
                "frqncy": float(record["frqncy"]),
                "ioxm": int(record["ioxm"]),
                "alfast": float(record["alfast"]),
                "betast": float(record["betast"]),
                "driven_current_ka": float(record["driven_current_ka"]),
            })
    if not rows:
        raise ValueError(f"No usable current-drive records in {jsonl_path}")
    return rows


def _qk(value: float) -> float:
    return round(float(value), 6)


def _build_grid(rows: list[dict], frequency: float, ioxm: int, value_key: str):
    subset = [
        row for row in rows
        if _qk(row["frqncy"]) == _qk(frequency) and int(row["ioxm"]) == int(ioxm)
    ]
    if not subset:
        return None
    alphas = sorted({_qk(row["alfast"]) for row in subset})
    betas = sorted({_qk(row["betast"]) for row in subset})
    if len(alphas) < 2 or len(betas) < 2:
        return None
    alpha_index = {value: index for index, value in enumerate(alphas)}
    beta_index = {value: index for index, value in enumerate(betas)}
    values = np.full((len(betas), len(alphas)), np.nan, dtype=float)
    for row in subset:
        values[beta_index[_qk(row["betast"])], alpha_index[_qk(row["alfast"])]] = float(row[value_key])
    x_grid, y_grid = np.meshgrid(np.asarray(alphas), np.asarray(betas))
    return x_grid, y_grid, values


def _levels(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, count)
    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    if minimum == maximum:
        delta = max(abs(minimum) * 0.05, 1.0e-12)
        minimum -= delta
        maximum += delta
    return np.linspace(minimum, maximum, count)


def plot_current_case(
    rows: list[dict],
    frequency: float,
    ioxm: int,
    out_path: Path,
    value_key: str,
    level_count: int,
    label_every: int,
) -> bool:
    grid = _build_grid(rows, frequency, ioxm, value_key)
    if grid is None:
        print(f"Skip {frequency:g} GHz {mode_tag(ioxm)}: need at least a 2 x 2 alpha-beta grid")
        return False

    x_grid, y_grid, values = grid
    levels = _levels(values, level_count)
    masked_values = np.ma.masked_invalid(values)
    fig, axis = plt.subplots(figsize=(12.0, 8.0))
    filled = axis.contourf(x_grid, y_grid, masked_values, levels=levels, cmap=PARULA, extend="both")
    contours = axis.contour(x_grid, y_grid, masked_values, levels=levels, colors="0.12", linewidths=0.95)
    label_levels = contours.levels[::max(2, int(label_every))]
    if len(label_levels):
        axis.clabel(contours, levels=label_levels, inline=True, fontsize=13, fmt="%.3g")
    axis.set_xlabel(r"$\alpha$ (deg)")
    axis.set_ylabel(r"$\beta$ (deg)")
    axis.set_title(f"{frequency:g} GHz, {mode_tag(ioxm)} | current-drive contour", pad=12)
    # One-degree reference grid for reading exact alpha/beta coordinates.
    # Keep it behind the contours and use low opacity to avoid visual clutter.
    axis.xaxis.set_minor_locator(MultipleLocator(1.0))
    axis.yaxis.set_minor_locator(MultipleLocator(1.0))
    # ``'line'`` places grid lines above contourf patches but below contour
    # strokes, so the 1° mesh remains visible without obscuring the data.
    axis.set_axisbelow("line")
    axis.grid(which="minor", color="0.25", linewidth=0.45, alpha=0.22)
    _bold_tick_labels(axis)
    colorbar = fig.colorbar(filled, ax=axis, shrink=0.82, pad=0.025)
    colorbar.set_label(r"$I_{\mathrm{CD}}$ (kA)", fontweight="bold")
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def plot_current_maps(
    rows: list[dict],
    out_dir: Path,
    suffix: str = "",
    value_key: str = "driven_current_ka",
    level_count: int = 15,
    label_every: int = 2,
) -> int:
    normalized_suffix = normalize_suffix(suffix)
    cases = sorted({(_qk(row["frqncy"]), int(row["ioxm"])) for row in rows})
    written = 0
    for frequency, ioxm in cases:
        filename = f"alpha_beta_current_drive_{frequency:g}GHz_{mode_tag(ioxm)}{normalized_suffix}.png"
        written += int(plot_current_case(
            rows,
            frequency,
            ioxm,
            out_dir / filename,
            value_key,
            max(3, level_count),
            label_every,
        ))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot current-drive contours from JSONL (no NC reading).")
    parser.add_argument("jsonl", type=Path, help="JSONL produced by build_genray_jsonl_from_nc.py")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--value-key", default="driven_current_ka")
    parser.add_argument("--levels", type=int, default=15)
    parser.add_argument("--label-every", type=int, default=2)
    args = parser.parse_args()

    configure_plot_style()
    rows = load_jsonl(args.jsonl)
    written = plot_current_maps(rows, args.out_dir, args.suffix, args.value_key, args.levels, args.label_every)
    print(f"Finished: {written} figure(s)")


if __name__ == "__main__":
    main()
