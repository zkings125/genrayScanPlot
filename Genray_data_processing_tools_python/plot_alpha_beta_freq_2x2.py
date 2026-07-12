#!/usr/bin/env python3
"""
对每个频率单独出一张 2×2 图（对应 MATLAB 子图顺序 221–224）：

  ┌─────────────────┬─────────────────┐
  │ 221 O模          │ 222 O模          │
  │ α–β 吸收比例    │ α–β ρ_peak      │
  ├─────────────────┼─────────────────┤
  │ 223 X模          │ 224 X模          │
  │ α–β 吸收比例    │ α–β ρ_peak      │
  └─────────────────┴─────────────────┘

吸收比例：power_total_1e10 / power_inj_total_1e10（即 P_tot/P_inj）。
ρ_peak：powden_e 全局最大处的 rho_bin_center。

数据来自 genray_scan_from_nc.jsonl。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_alpha_beta_rho_ratio_surface import (
    PARULA,
    SCRIPT_DIR,
    _bold_tick_labels,
    build_grids,
    configure_plot_style,
    draw_labeled_contours,
    load_rows,
    normalize_suffix,
)

DEFAULT_JSONL = SCRIPT_DIR / "genray_scan_from_nc.jsonl"
SAVE_DPI = 300


def _norm_from_arrays(arrays: list[np.ndarray]) -> mpl.colors.Normalize:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a.size])
    if vals.size == 0:
        return mpl.colors.Normalize(0.0, 1.0)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1e-15
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def _contour_levels(norm: mpl.colors.Normalize, n: int = 14) -> np.ndarray:
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    if vmin == vmax:
        vmax = vmin + 1e-15
    return np.linspace(vmin, vmax, n)


def _plot_contourf(
    ax: mpl.axes.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    data: np.ndarray,
    norm: mpl.colors.Normalize,
    title: str,
) -> mpl.contour.QuadContourSet:
    dm = np.ma.masked_invalid(data)
    levels = _contour_levels(norm)
    cf = ax.contourf(
        X,
        Y,
        dm,
        levels=levels,
        cmap=PARULA,
        norm=norm,
        extend="both",
    )
    draw_labeled_contours(
        ax,
        X,
        Y,
        data,
        levels=levels,
        label_every=2,
        fontsize=11,
    )
    ax.set_title(title, fontweight="bold", fontsize=14, pad=8)
    ax.set_xlabel(r"$\alpha$ (deg)", fontweight="bold")
    ax.set_ylabel(r"$\beta$ (deg)", fontweight="bold")
    ax.set_aspect("auto")
    ax.tick_params(axis="both", which="major", pad=4)
    _bold_tick_labels(ax)
    return cf


def plot_one_frequency(rows: list[dict], frqncy: float, out_path: Path) -> None:
    g_o = build_grids(rows, frqncy, 1)
    g_x = build_grids(rows, frqncy, -1)

    ratio_arrays: list[np.ndarray] = []
    rho_arrays: list[np.ndarray] = []
    for g in (g_o, g_x):
        if g is None:
            continue
        _x, _y, z_rho, c_ratio = g
        ratio_arrays.append(c_ratio)
        rho_arrays.append(z_rho)

    norm_ratio = _norm_from_arrays(ratio_arrays)
    norm_rho = _norm_from_arrays(rho_arrays)

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 18))
    fig.subplots_adjust(
        left=0.07,
        right=0.82,
        bottom=0.06,
        top=0.93,
        hspace=0.38,
        wspace=0.35,
    )

    # 221 O模 — 吸收比例；222 O模 — rho_peak
    # 223 X模 — 吸收比例；224 X模 — rho_peak
    if g_o is not None:
        Xo, Yo, Zo_rho, Co_ratio = g_o
        _plot_contourf(
            axes[0, 0],
            Xo,
            Yo,
            Co_ratio,
            norm_ratio,
            f"{frqncy:g} GHz · O-mode · $P_{{\\mathrm{{tot}}}}/P_{{\\mathrm{{inj}}}}$",
        )
        _plot_contourf(
            axes[0, 1],
            Xo,
            Yo,
            Zo_rho,
            norm_rho,
            f"{frqncy:g} GHz · O-mode · $\\rho_{{\\mathrm{{peak}}}}$",
        )
    else:
        axes[0, 0].set_visible(False)
        axes[0, 1].set_visible(False)

    if g_x is not None:
        Xx, Yx, Zx_rho, Cx_ratio = g_x
        _plot_contourf(
            axes[1, 0],
            Xx,
            Yx,
            Cx_ratio,
            norm_ratio,
            f"{frqncy:g} GHz · X-mode · $P_{{\\mathrm{{tot}}}}/P_{{\\mathrm{{inj}}}}$",
        )
        _plot_contourf(
            axes[1, 1],
            Xx,
            Yx,
            Zx_rho,
            norm_rho,
            f"{frqncy:g} GHz · X-mode · $\\rho_{{\\mathrm{{peak}}}}$",
        )
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    fig.suptitle(
        f"{frqncy:g} GHz — α–β maps (221–222: O-mode, 223–224: X-mode)",
        fontweight="bold",
        fontsize=16,
        y=0.98,
    )

    cax_r = fig.add_axes([0.865, 0.52, 0.022, 0.38])
    cax_z = fig.add_axes([0.865, 0.10, 0.022, 0.38])
    sm_r = mpl.cm.ScalarMappable(norm=norm_ratio, cmap=PARULA)
    sm_z = mpl.cm.ScalarMappable(norm=norm_rho, cmap=PARULA)
    cb_r = fig.colorbar(sm_r, cax=cax_r)
    cb_z = fig.colorbar(sm_z, cax=cax_z)
    cb_r.set_label(r"$P_{\mathrm{tot}}/P_{\mathrm{inj}}$", fontweight="bold", labelpad=10)
    cb_z.set_label(r"$\rho_{\mathrm{peak}}$", fontweight="bold", labelpad=10)
    for cb in (cb_r, cb_z):
        cb.ax.tick_params(labelsize=13)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight("bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.28,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="每个频率一张 2×2：O/X 模的吸收比与 ρ_peak 等高线填色图",
    )
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="文件名后缀，例如 _2T → ..._OX_2x2_2T.png",
    )
    args = ap.parse_args()

    configure_plot_style()
    rows = load_rows(args.jsonl)
    freqs = sorted({r["frqncy"] for r in rows})
    sfx = normalize_suffix(args.suffix)

    for fq in freqs:
        out = args.out_dir / f"alpha_beta_freq_{fq:g}GHz_OX_2x2{sfx}.png"
        plot_one_frequency(rows, fq, out)
        print(f"写入 {out}")


if __name__ == "__main__":
    main()
