#!/usr/bin/env python3
"""Extract GENRAY NetCDF results into JSONL.  This module does no plotting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_JSONL = SCRIPT_DIR / "genray_scan_from_nc.jsonl"


def _as_array(ds: nc.Dataset, name: str) -> np.ndarray:
    return np.asarray(ds.variables[name][:], dtype=float)


def _first_finite(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    return float(flat[0]) if flat.size else float("nan")


def _mean_finite(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    return float(np.mean(flat)) if flat.size else float("nan")


def _scalar(ds: nc.Dataset, names: tuple[str, ...], method: str = "first") -> float:
    take = _mean_finite if method == "mean" else _first_finite
    for name in names:
        if name in ds.variables:
            return take(_as_array(ds, name))
    return float("nan")


def _freq_ghz(ds: nc.Dataset) -> float:
    value = _scalar(ds, ("frqncy", "freqcy", "frequency"))
    if not np.isfinite(value):
        return value
    return value / 1.0e9 if abs(value) > 1.0e6 else value


def _jsonable_array(values: np.ndarray) -> list | float:
    array = np.asarray(values, dtype=float)
    return float(array) if array.ndim == 0 else array.tolist()


def _filename_number(path: Path, pattern: str, fallback: float) -> tuple[float, bool]:
    match = re.search(pattern, path.stem)
    if not match:
        return fallback, False
    try:
        return float(match.group(1)), True
    except ValueError:
        return fallback, False


def extract_record(nc_path: Path, coord_method: str, current_var: str) -> dict:
    """Extract one plotting-independent record from a GENRAY NC file."""
    with nc.Dataset(str(nc_path), "r") as ds:
        for name in ("rho_bin_center", "powden_e", "power_total", "power_inj_total"):
            if name not in ds.variables:
                raise KeyError(f"{nc_path} missing NetCDF variable {name}")

        rho = _as_array(ds, "rho_bin_center")
        power_density = _as_array(ds, "powden_e")
        rho_1d = rho.ravel()
        power_density_1d = power_density.ravel()
        if rho_1d.size != power_density_1d.size or rho_1d.size == 0:
            raise ValueError(f"{nc_path} has incompatible rho_bin_center/powden_e shapes")

        peak_index = int(np.nanargmax(power_density_1d))
        power_total = float(_as_array(ds, "power_total"))
        power_injected = float(_as_array(ds, "power_inj_total"))
        alpha_raw = _scalar(ds, ("alfast", "alphast"), method=coord_method)
        beta_raw = _scalar(ds, ("betast",), method=coord_method)
        frequency = _freq_ghz(ds)
        ioxm = int(round(_scalar(ds, ("ioxm",))))

        if current_var not in ds.variables:
            raise KeyError(f"{nc_path} missing NetCDF variable {current_var}")
        current_a = float(_as_array(ds, current_var))
        current_density = (
            _as_array(ds, "s_cur_den_parallel")
            if "s_cur_den_parallel" in ds.variables
            else np.asarray([], dtype=float)
        )

    alpha, alpha_from_filename = _filename_number(nc_path, r"_a([-+0-9.]+)(?:_|$)", alpha_raw)
    beta, _ = _filename_number(nc_path, r"_b([-+0-9.]+)_a", beta_raw)
    rho_cd_peak = float("nan")
    current_density_peak = float("nan")
    if current_density.size:
        count = min(rho_1d.size, current_density.size)
        rho_cd = rho_1d[:count]
        current_cd = np.abs(current_density.ravel()[:count])
        valid = np.isfinite(rho_cd) & np.isfinite(current_cd)
        if np.any(valid):
            rho_cd = rho_cd[valid]
            current_cd = current_cd[valid]
            current_peak_index = int(np.argmax(current_cd))
            rho_cd_peak = float(rho_cd[current_peak_index])
            current_density_peak = float(current_cd[current_peak_index])
    return {
        "source_nc": str(nc_path),
        "frqncy": frequency,
        "ioxm": ioxm,
        "alfast": alpha,
        "alfast_nc": alpha_raw,
        "betast": beta,
        "betast_nc": beta_raw,
        "alpha_from_filename": alpha_from_filename,
        "rho_bin_center": _jsonable_array(rho),
        "powden_e": _jsonable_array(power_density),
        "s_cur_den_parallel": _jsonable_array(current_density),
        "rho_peak": float(rho_1d[peak_index]),
        "powden_e_peak": float(power_density_1d[peak_index]),
        "power_total_1e10": power_total / 1.0e10,
        "power_inj_total_1e10": power_injected / 1.0e10,
        "driven_current_var": current_var,
        "driven_current_a": current_a,
        "driven_current_ka": current_a / 1.0e3,
        "rho_cd_peak": rho_cd_peak,
        "parallel_current_density_peak_a_cm2": current_density_peak,
        "_shapes": {
            "rho_bin_center": list(np.asarray(rho).shape),
            "powden_e": list(np.asarray(power_density).shape),
            "s_cur_den_parallel": list(np.asarray(current_density).shape),
        },
    }


def normalize_alpha(rows: list[dict], alpha_mode: str, alpha_shift: float) -> None:
    raw = np.asarray([row["alfast_nc"] for row in rows], dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        offset = 0.0
    elif alpha_mode == "auto":
        offset = 180.0 if float(np.nanmax(finite)) <= 0.0 else 0.0
    elif alpha_mode == "plus180":
        offset = 180.0
    elif alpha_mode in ("from180", "raw"):
        offset = 0.0
    else:
        raise ValueError(f"Unknown alpha mode: {alpha_mode}")

    for row in rows:
        if row.get("alpha_from_filename", False):
            row["alfast"] = float(row["alfast"]) + alpha_shift
            row["alpha_mode"] = "filename"
            row["alpha_offset_applied"] = alpha_shift
            continue
        raw_alpha = float(row["alfast_nc"])
        row["alfast"] = (
            180.0 - raw_alpha + alpha_shift
            if alpha_mode == "from180"
            else raw_alpha + offset + alpha_shift
        )
        row["alpha_mode"] = alpha_mode
        row["alpha_offset_applied"] = offset + alpha_shift


def load_nc_folder(
    nc_dir: Path,
    recursive: bool = False,
    coord_method: str = "first",
    alpha_mode: str = "auto",
    alpha_shift: float = 0.0,
    current_var: str = "parallel_cur_total",
) -> list[dict]:
    pattern = "**/*.nc" if recursive else "*.nc"
    paths = sorted(path for path in nc_dir.glob(pattern) if path.is_file())
    rows: list[dict] = []
    failures: list[str] = []
    for path in paths:
        try:
            rows.append(extract_record(path, coord_method, current_var))
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print("Skipped NetCDF files:")
        for item in failures:
            print(f"  {item}")
    if not rows:
        raise ValueError(f"No usable .nc files found under {nc_dir}")
    normalize_alpha(rows, alpha_mode, alpha_shift)
    return sorted(rows, key=lambda row: (row["frqncy"], row["ioxm"], row["alfast"], row["betast"]))


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_jsonl_streaming(
    nc_dir: Path,
    out_path: Path,
    recursive: bool = False,
    coord_method: str = "first",
    alpha_mode: str = "auto",
    alpha_shift: float = 0.0,
    current_var: str = "parallel_cur_total",
) -> tuple[int, int]:
    """Extract and write one NC at a time without retaining the scan in RAM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = nc_dir.rglob("*.nc") if recursive else nc_dir.glob("*.nc")
    written = 0
    failed = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            if not path.is_file():
                continue
            try:
                row = extract_record(path, coord_method, current_var)
                normalize_alpha([row], alpha_mode, alpha_shift)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                failed += 1
                print(f"Skipped {path}: {exc}")
    if written == 0:
        raise ValueError(f"No usable .nc files found under {nc_dir}")
    return written, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GENRAY NetCDF files into JSONL (no plotting).")
    parser.add_argument("nc_dir", type=Path, help="Folder containing GENRAY .nc files")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--coord-method", choices=("first", "mean"), default="first")
    parser.add_argument("--alpha-mode", choices=("auto", "raw", "plus180", "from180"), default="auto")
    parser.add_argument("--alpha-shift", type=float, default=0.0)
    parser.add_argument("--current-var", default="parallel_cur_total")
    args = parser.parse_args()

    written, failed = build_jsonl_streaming(
        args.nc_dir,
        args.jsonl,
        args.recursive,
        args.coord_method,
        args.alpha_mode,
        args.alpha_shift,
        args.current_var,
    )
    print(f"Wrote {args.jsonl} ({written} records, {failed} skipped)")


if __name__ == "__main__":
    main()
