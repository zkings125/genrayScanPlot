#!/usr/bin/env python3
"""Compatibility wrapper for the former combined NC-to-JSONL-and-plot command.

New workflows should call build_genray_jsonl_from_nc.py and
plot_current_contours_from_jsonl.py separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_genray_jsonl_from_nc import DEFAULT_OUT_JSONL, load_nc_folder, write_jsonl
from plot_alpha_beta_rho_ratio_surface import SCRIPT_DIR, configure_plot_style
from plot_current_contours_from_jsonl import plot_current_maps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper; prefer the separate JSONL builder and JSONL plotter."
    )
    parser.add_argument("nc_dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--coord-method", choices=("first", "mean"), default="first")
    parser.add_argument("--alpha-mode", choices=("auto", "raw", "plus180", "from180"), default="auto")
    parser.add_argument("--alpha-shift", type=float, default=0.0)
    parser.add_argument("--current-var", default="parallel_cur_total")
    parser.add_argument("--levels", type=int, default=15)
    parser.add_argument("--label-every", type=int, default=2)
    parser.add_argument("--no-current-plot", action="store_true")
    args = parser.parse_args()

    rows = load_nc_folder(
        args.nc_dir,
        args.recursive,
        args.coord_method,
        args.alpha_mode,
        args.alpha_shift,
        args.current_var,
    )
    write_jsonl(rows, args.jsonl)
    print(f"Wrote {args.jsonl} ({len(rows)} records)")
    if not args.no_current_plot:
        configure_plot_style()
        plot_current_maps(rows, args.out_dir, args.suffix, "driven_current_ka", args.levels, args.label_every)


if __name__ == "__main__":
    main()
