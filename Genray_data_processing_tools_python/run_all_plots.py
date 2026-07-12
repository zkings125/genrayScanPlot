#!/usr/bin/env python3
"""Run the four standard GENRAY plotting workflows from one JSONL file."""
from __future__ import annotations

import argparse
from pathlib import Path

import plot_absorption_peak_ranges as absorption
import plot_alpha_beta_freq_2x2 as freq_2x2
import plot_alpha_beta_rho_ratio_surface as surface
import plot_current_contours_from_jsonl as current


def run_all_plots(jsonl: str | Path, out_dir: str | Path | None = None,
                  suffix: str = "", current_var: str = "driven_current_ka",
                  levels: int = 15, label_every: int = 2) -> dict[str, object]:
    """Generate all four plot families from one JSONL without reading NC files."""
    jsonl_path = Path(jsonl).expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    output = Path(out_dir).expanduser().resolve() if out_dir else jsonl_path.parent / "figs"
    output.mkdir(parents=True, exist_ok=True)
    sfx = surface.normalize_suffix(suffix)
    surface.configure_plot_style()

    rows = surface.load_rows(jsonl_path)
    cases = sorted({(row["frqncy"], int(row["ioxm"])) for row in rows})
    surface_files = []
    for frequency, ioxm in cases:
        path = output / f"alpha_beta_rhoPeak_powRatio_{frequency:g}GHz_{surface.mode_tag(ioxm)}{sfx}.png"
        surface.plot_one_case(rows, frequency, ioxm, path)
        surface_files.append(path)

    freq_files = []
    for frequency in sorted({row["frqncy"] for row in rows}):
        path = output / f"alpha_beta_freq_{frequency:g}GHz_OX_2x2{sfx}.png"
        freq_2x2.plot_one_frequency(rows, frequency, path)
        freq_files.append(path)

    peak_rows = absorption.load_and_analyze(jsonl_path)
    absorption.write_table_csv(peak_rows, output / f"{absorption.TABLE_CSV_STEM}{sfx}.csv")
    peak_summary = absorption.summary_by_freq_mode(peak_rows)
    absorption.write_summary_csv(peak_summary, output / f"{absorption.SUMMARY_CSV_STEM}{sfx}.csv")
    absorption.plot_distribution_panels(peak_rows, output / f"absorption_peak_rho_distribution{sfx}.png")
    absorption.plot_fwhm_width_heatmaps(peak_rows, output / f"absorption_peak_fwhm_width_alpha_beta{sfx}.png")
    absorption.plot_interval_strips(peak_rows, output / f"absorption_peak_interval_strip{sfx}.png")
    absorption.plot_summary_boxes(peak_summary, output / f"absorption_peak_summary_median_bands{sfx}.png")

    current_rows = current.load_jsonl(jsonl_path)
    current.configure_plot_style()
    current_count = current.plot_current_maps(
        current_rows, output, sfx, current_var, max(3, levels), max(1, label_every)
    )
    result = {"jsonl": jsonl_path, "out_dir": output, "records": len(rows),
              "surface_files": surface_files, "frequency_2x2_files": freq_files,
              "current_files": current_count}
    print(f"Completed all four plot families: {len(rows)} records -> {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all four standard GENRAY plotting functions.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--current-var", default="driven_current_ka")
    parser.add_argument("--levels", type=int, default=15)
    parser.add_argument("--label-every", type=int, default=2)
    args = parser.parse_args()
    run_all_plots(args.jsonl, args.out_dir, args.suffix, args.current_var, args.levels, args.label_every)


if __name__ == "__main__":
    main()
