#!/usr/bin/env python3
"""
读取扫描 jsonl，绘制 3×2 子图（行：105 / 140 / 170 GHz；列：O 模 / X 模）：

  图 1：α–β 平面上 P_tot / P_inj（Parula，六子图共用一个 colorbar）。
  图 2：α–β 平面上 ρ_peak（powden_e 最大处的 rho_bin_center），同上。

数据默认：genray_scan_alpha_beta_adaptive.jsonl（与 genray_alpha_beta_adaptive_scan.py 一致）。

按扫描阶段过滤（字段 ``scan_phase``）：

  - ``coarse``：粗扫（步长 10°）；无该字段的旧记录视为粗扫。
  - ``fine``：精扫（``fine_d5`` / ``fine_d3``）。
  - ``all``：全部（同一 (α,β) 若有多条，后读入的覆盖先读入的）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_alpha_beta_rho_ratio_surface import (
    PARULA,
    SCRIPT_DIR,
    _bold_tick_labels,
    configure_plot_style,
    mode_tag,
)

DEFAULT_JSONL = SCRIPT_DIR / "genray_scan_alpha_beta_adaptive.jsonl"
FREQ_ORDER = [105.0, 140.0, 170.0]
IOXM_COLS = [1, -1]


def qk(x: float) -> float:
    return round(float(x), 6)


def _record_scan_phase(rec: dict) -> str:
    """jsonl 中 scan_phase；缺省视为粗扫（与早期记录兼容）。"""
    ph = rec.get("scan_phase")
    if ph is None or ph == "":
        return "coarse"
    return str(ph)


def _keep_phase(scan_phase: str, phase_filter: str) -> bool:
    if phase_filter == "all":
        return True
    if phase_filter == "coarse":
        return scan_phase == "coarse"
    if phase_filter == "fine":
        return scan_phase in ("fine_d5", "fine_d3")
    raise ValueError(f"未知 phase_filter: {phase_filter}")


def load_heatmap_rows(path: Path, phase_filter: str = "all") -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            sp = _record_scan_phase(rec)
            if not _keep_phase(sp, phase_filter):
                continue
            fq = float(rec["frqncy"])
            mx = int(rec["ioxm"])
            a = float(rec["alfast"])
            b = float(rec["betast"])
            if "pow_ratio" in rec and rec["pow_ratio"] is not None and "rho_peak" in rec:
                pr = float(rec["pow_ratio"])
                rp = float(rec["rho_peak"])
            else:
                rho = np.asarray(rec["rho_bin_center"], dtype=float)
                pde = np.asarray(rec["powden_e"], dtype=float)
                if rho.size != pde.size or rho.size == 0:
                    continue
                imax = int(np.argmax(pde))
                rp = float(rho[imax])
                pt = float(rec["power_total_1e10"])
                pj = float(rec["power_inj_total_1e10"])
                pr = pt / pj if pj != 0 else float("nan")
            rows.append(
                {
                    "frqncy": fq,
                    "ioxm": mx,
                    "alfast": a,
                    "betast": b,
                    "rho_peak": rp,
                    "pow_ratio": pr,
                    "scan_phase": sp,
                }
            )
    if not rows:
        raise ValueError(f"未读到有效记录: {path}")
    return rows


def build_grid(rows: list[dict], frqncy: float, ioxm: int):
    sub = [
        r
        for r in rows
        if abs(float(r["frqncy"]) - frqncy) < 0.01 and int(r["ioxm"]) == ioxm
    ]
    if not sub:
        return None
    alphas = sorted({qk(r["alfast"]) for r in sub})
    betas = sorted({qk(r["betast"]) for r in sub})
    ia = {al: i for i, al in enumerate(alphas)}
    ib = {be: i for i, be in enumerate(betas)}
    nz, na = len(betas), len(alphas)
    C = np.full((nz, na), np.nan, dtype=float)
    Z = np.full((nz, na), np.nan, dtype=float)
    for r in sub:
        C[ib[qk(r["betast"])], ia[qk(r["alfast"])]] = r["pow_ratio"]
        Z[ib[qk(r["betast"])], ia[qk(r["alfast"])]] = r["rho_peak"]
    X, Y = np.meshgrid(np.asarray(alphas), np.asarray(betas))
    return X, Y, C, Z


def _shared_norm(arrays: list[np.ndarray]) -> mpl.colors.Normalize:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a.size])
    if vals.size == 0:
        return mpl.colors.Normalize(0.0, 1.0)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1e-15
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def plot_panel(
    ax: mpl.axes.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    data: np.ndarray,
    norm: mpl.colors.Normalize,
    title: str,
) -> mpl.collections.QuadMesh:
    pcm = ax.pcolormesh(
        X,
        Y,
        np.ma.masked_invalid(data),
        shading="nearest",
        cmap=PARULA,
        norm=norm,
        edgecolors="0.35",
        linewidth=0.18,
    )
    ax.set_title(title, fontweight="bold", fontsize=13, pad=6)
    ax.set_xlabel(r"$\alpha$ (deg)", fontweight="bold")
    ax.set_ylabel(r"$\beta$ (deg)", fontweight="bold")
    ax.set_aspect("auto")
    ax.tick_params(axis="both", which="major", pad=3)
    _bold_tick_labels(ax)
    return pcm


def plot_figure(
    rows: list[dict],
    quantity: str,
    out_path: Path,
    *,
    suptitle_suffix: str = "",
) -> None:
    """quantity: 'ratio' | 'rho'"""
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 18.5))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.06, top=0.92, hspace=0.36, wspace=0.28)

    gather: list[np.ndarray] = []
    grid_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, fq in enumerate(FREQ_ORDER):
        for j, mx in enumerate(IOXM_COLS):
            g = build_grid(rows, fq, mx)
            if g is None:
                continue
            X, Y, C, Z = g
            data = C if quantity == "ratio" else Z
            gather.append(np.asarray(data, dtype=float))
            grid_data[(i, j)] = (X, Y, np.asarray(data, dtype=float))

    norm = _shared_norm(gather)
    if not gather:
        raise ValueError("没有可用的 (频率, 模式) 网格数据，无法绘图。")
    last_pcm: mpl.collections.QuadMesh | None = None
    for i in range(3):
        for j in range(2):
            if (i, j) not in grid_data:
                axes[i, j].set_visible(False)
                continue
            X, Y, data = grid_data[(i, j)]
            fq = FREQ_ORDER[i]
            mx = IOXM_COLS[j]
            ttl = f"{fq:g} GHz · {mode_tag(mx)}"
            last_pcm = plot_panel(axes[i, j], X, Y, data, norm, ttl)

    if last_pcm is not None:
        cax = fig.add_axes([0.905, 0.15, 0.022, 0.68])
        cb = fig.colorbar(last_pcm, cax=cax)
        if quantity == "ratio":
            cb.set_label(r"$P_{\mathrm{tot}}/P_{\mathrm{inj}}$", fontweight="bold", labelpad=10)
            fig.suptitle(
                r"$\alpha$–$\beta$ absorption efficiency ($P_{\mathrm{tot}}/P_{\mathrm{inj}}$)"
                + suptitle_suffix,
                fontweight="bold",
                fontsize=15,
                y=0.98,
            )
        else:
            cb.set_label(r"$\rho_{\mathrm{peak}}$", fontweight="bold", labelpad=10)
            fig.suptitle(
                r"$\alpha$–$\beta$ · $\rho$ at max($P_{\mathrm{den},e}$)" + suptitle_suffix,
                fontweight="bold",
                fontsize=15,
                y=0.98,
            )
        cb.ax.tick_params(labelsize=12)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight("bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="3×2 Parula 热力图（三频 × O/X）")
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    ap.add_argument(
        "--phase",
        choices=("all", "coarse", "fine"),
        default="all",
        help="使用 jsonl 中的 scan_phase 过滤：coarse=粗扫，fine=精扫(d5+d3)，all=全部",
    )
    ap.add_argument(
        "--coarse-only",
        action="store_true",
        help="等价于 --phase coarse，且输出文件名带 _coarse",
    )
    args = ap.parse_args()

    phase = "coarse" if args.coarse_only else args.phase
    tag = "_coarse" if phase == "coarse" else ("_fine" if phase == "fine" else "")
    suffix = ""
    if phase == "coarse":
        suffix = "\n(coarse scan: Δα = Δβ = 10°)"
    elif phase == "fine":
        suffix = "\n(fine scan: step 5° / 3°)"

    configure_plot_style()
    rows = load_heatmap_rows(args.jsonl, phase_filter=phase)

    out1 = args.out_dir / f"alpha_beta_3x2_pow_ratio_parula{tag}.png"
    out2 = args.out_dir / f"alpha_beta_3x2_rho_peak_parula{tag}.png"
    plot_figure(rows, "ratio", out1, suptitle_suffix=suffix)
    print(f"写入 {out1}")
    plot_figure(rows, "rho", out2, suptitle_suffix=suffix)
    print(f"写入 {out2}")


if __name__ == "__main__":
    main()
