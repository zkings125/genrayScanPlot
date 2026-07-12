#!/usr/bin/env python3
"""
从 genray_scan_from_nc.jsonl 读取扫描结果，对每个 (α, β) 取 powden_e 峰值对应的 ρ，
绘制 α–β 平面二维热力图：颜色为 P_total / P_inj；
叠加黑色等高线表示 ρ（powden_e 最大处的 rho_bin_center）。

每个 (frqncy, ioxm) 组合输出一张 PNG（文件名 alpha_beta_rhoPeak_powRatio_<GHz>_<Xm|Om>.png）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = SCRIPT_DIR / "genray_scan_from_nc.jsonl"

# Parula（与 MATLAB 常用版本一致，来源：BIDS/colormap；MathWorks 保留权利）
_PARULA_RGB = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538],
]


PARULA = LinearSegmentedColormap.from_list("parula", _PARULA_RGB, N=256)


def configure_plot_style() -> None:
    """Times New Roman，16pt，加粗；适合论文出图。"""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.fontset": "stix",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _bold_tick_labels(ax: mpl.axes.Axes) -> None:
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_fontweight("bold")


def contour_levels_from_data(data: np.ndarray, count: int = 15) -> np.ndarray:
    vals = np.asarray(data, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, count)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        delta = max(abs(vmin) * 0.05, 1.0e-12)
        vmin -= delta
        vmax += delta
    return np.linspace(vmin, vmax, count)


def draw_labeled_contours(
    ax: mpl.axes.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    data: np.ndarray,
    *,
    levels=15,
    label_every: int = 2,
    fmt: str = "%.3g",
    fontsize: int = 13,
    colors: str = "0.12",
    linewidths: float = 0.95,
    alpha: float = 1.0,
) -> mpl.contour.QuadContourSet | None:
    data_arr = np.asarray(data, dtype=float)
    zmask = np.ma.masked_invalid(data_arr)
    if not np.any(np.isfinite(data_arr)):
        return None
    if isinstance(levels, int):
        levels = contour_levels_from_data(data_arr, levels)
    cs = ax.contour(
        X,
        Y,
        zmask,
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        alpha=alpha,
    )
    step = max(2, int(label_every))
    label_levels = cs.levels[::step]
    if len(label_levels):
        ax.clabel(cs, levels=label_levels, inline=True, fontsize=fontsize, fmt=fmt)
    return cs


FIG_WIDTH = 12.0
FIG_HEIGHT = 8.0
SAVE_DPI = 300


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            rho = np.asarray(rec["rho_bin_center"], dtype=float)
            pde = np.asarray(rec["powden_e"], dtype=float)
            if rho.size != pde.size or rho.size == 0:
                continue
            imax = int(np.argmax(pde))
            rho_peak = float(rho[imax])
            pt = float(rec["power_total_1e10"])
            pj = float(rec["power_inj_total_1e10"])
            ratio = pt / pj if pj != 0 else np.nan
            rows.append(
                {
                    "frqncy": float(rec["frqncy"]),
                    "ioxm": int(rec["ioxm"]),
                    "alfast": float(rec["alfast"]),
                    "betast": float(rec["betast"]),
                    "rho_peak": rho_peak,
                    "pow_ratio": ratio,
                }
            )
    if not rows:
        raise ValueError(f"未读到有效记录: {path}")
    return rows


def mode_tag(ioxm: int) -> str:
    return "Xm" if ioxm == -1 else "Om" if ioxm == 1 else f"ioxm{ioxm}"


def _qk(x: float) -> float:
    return round(float(x), 6)


def build_grids(rows: list[dict], frqncy: float, ioxm: int):
    """返回 X,Y,Z,C 及 alphas, betas。"""
    sub = [r for r in rows if r["frqncy"] == frqncy and r["ioxm"] == ioxm]
    if not sub:
        return None

    alphas = sorted({_qk(r["alfast"]) for r in sub})
    betas = sorted({_qk(r["betast"]) for r in sub})
    ia = {a: i for i, a in enumerate(alphas)}
    ib = {b: i for i, b in enumerate(betas)}

    nz = len(betas)
    na = len(alphas)
    Z = np.full((nz, na), np.nan, dtype=float)
    C = np.full((nz, na), np.nan, dtype=float)

    for r in sub:
        ai = ia[_qk(r["alfast"])]
        bi = ib[_qk(r["betast"])]
        Z[bi, ai] = r["rho_peak"]
        C[bi, ai] = r["pow_ratio"]

    X, Y = np.meshgrid(np.asarray(alphas), np.asarray(betas))
    return X, Y, Z, C


def plot_one_case(
    rows: list[dict],
    frqncy: float,
    ioxm: int,
    out_path: Path,
) -> None:
    grids = build_grids(rows, frqncy, ioxm)
    if grids is None:
        return
    X, Y, Z, C = grids

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    valid_c = np.isfinite(C)
    if valid_c.any():
        vmin = float(np.nanmin(C[valid_c]))
        vmax = float(np.nanmax(C[valid_c]))
        if vmin == vmax:
            vmax = vmin + 1e-15
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    Cm = np.ma.masked_invalid(C)
    pcm = ax.pcolormesh(
        X,
        Y,
        Cm,
        shading="nearest",
        cmap=PARULA,
        norm=norm,
        edgecolors="0.4",
        linewidth=0.35,
    )

    if np.any(np.isfinite(Z)):
        try:
            draw_labeled_contours(
                ax,
                X,
                Y,
                Z,
                levels=10,
                label_every=2,
                fmt="%.2f",
                fontsize=13,
            )
        except ValueError:
            pass

    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    title = (
        f"{frqncy:g} GHz, {mode_tag(ioxm)} | "
        r"heatmap: $P_{\mathrm{tot}}/P_{\mathrm{inj}}$ · "
        r"contours: $\rho$ @ max($P_{\mathrm{den},e}$)"
    )
    ax.set_title(title, pad=12)
    ax.set_aspect("auto")
    _bold_tick_labels(ax)

    cb = fig.colorbar(pcm, ax=ax, shrink=0.82, pad=0.025)
    cb.set_label(r"$P_{\mathrm{tot}}/P_{\mathrm{inj}}$", fontweight="bold")
    cb.ax.tick_params(labelsize=16)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize_suffix(s: str) -> str:
    """非空后缀统一为以 '_' 开头，便于拼到文件名中。"""
    s = (s or "").strip()
    if not s:
        return ""
    return s if s.startswith("_") else f"_{s}"


def main() -> None:
    ap = argparse.ArgumentParser(description="α–β 二维热力图（吸收功率比），ρ 等高线")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="扫描结果 jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="PNG 输出目录",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="插入在 .png 之前，例如 _2T → ..._105GHz_Xm_2T.png（不写则与旧文件名一致）",
    )
    args = ap.parse_args()

    configure_plot_style()
    rows = load_rows(args.jsonl)
    cases = sorted({(r["frqncy"], r["ioxm"]) for r in rows})
    sfx = normalize_suffix(args.suffix)

    for fq, ix in cases:
        name = f"alpha_beta_rhoPeak_powRatio_{fq:g}GHz_{mode_tag(ix)}{sfx}.png"
        out = args.out_dir / name
        plot_one_case(rows, fq, ix, out)
        print(f"写入 {out}")


if __name__ == "__main__":
    main()
