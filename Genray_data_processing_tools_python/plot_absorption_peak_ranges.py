#!/usr/bin/env python3
"""
根据 genray_scan_from_nc.jsonl 中每条记录的 powden_e 剖面，计算最强吸收峰对应的：
  - rho_peak：峰值所在的 bin 中心
  - rho_fwhm_lo / rho_fwhm_hi：半高全宽（FWHM）对应的 ρ 区间端点（峰值一半高度处与剖面相交）

输出：
  1) absorption_peak_table.csv — 逐条扫描工况的完整表（可用 --suffix 避免覆盖旧文件）
  2) absorption_peak_summary_by_freq_mode.csv — 按频率×模式汇总统计
  3) 若干 PNG：沉积 ρ 分布、FWHM 宽度在 α–β 上的热力图、ρ 区间条带图

用法：在 88890 目录下 python3 plot_absorption_peak_ranges.py [--suffix _2T]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_alpha_beta_rho_ratio_surface import (
    PARULA,
    configure_plot_style,
    draw_labeled_contours,
    normalize_suffix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = SCRIPT_DIR / "genray_scan_from_nc.jsonl"
TABLE_CSV_STEM = "absorption_peak_table"
SUMMARY_CSV_STEM = "absorption_peak_summary_by_freq_mode"


def rho_fwhm_and_peak(
    rho: np.ndarray, pde: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    返回 (rho_fwhm_lo, rho_fwhm_hi, rho_peak, powden_e_peak, rho_fwhm_width)。
    峰值取全局最大；FWHM 为峰值一半高度处与剖面相交的 ρ 区间（离散 bin 间线性插值）。
    """
    rho = np.asarray(rho, dtype=float).ravel()
    pde = np.asarray(pde, dtype=float).ravel()
    if rho.shape != pde.shape or rho.size < 2:
        return (np.nan,) * 5

    imax = int(np.argmax(pde))
    peak_val = float(pde[imax])
    rho_peak = float(rho[imax])

    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.nan, np.nan, rho_peak, peak_val, np.nan

    half = 0.5 * peak_val
    n = len(rho)

    def interp(i: int, j: int) -> float:
        ri, rj = rho[i], rho[j]
        yi, yj = pde[i], pde[j]
        if not np.isfinite(yi) or not np.isfinite(yj):
            return float((ri + rj) / 2.0)
        if abs(yj - yi) < 1e-300:
            return float((ri + rj) / 2.0)
        t = (half - yi) / (yj - yi)
        t = float(np.clip(t, 0.0, 1.0))
        return ri + t * (rj - ri)

    j = imax
    while j > 0 and pde[j - 1] >= half:
        j -= 1
    if j == 0 and pde[0] >= half:
        rho_lo = float(rho[0])
    else:
        rho_lo = interp(j - 1, j)

    k = imax
    while k < n - 1 and pde[k + 1] >= half:
        k += 1
    if k == n - 1 and pde[-1] >= half:
        rho_hi = float(rho[-1])
    else:
        rho_hi = interp(k, k + 1)

    width = float(rho_hi - rho_lo) if np.isfinite(rho_lo) and np.isfinite(rho_hi) else np.nan
    return rho_lo, rho_hi, rho_peak, peak_val, width


def mode_tag(ioxm: int) -> str:
    return "Xm" if ioxm == -1 else "Om" if ioxm == 1 else f"ioxm{ioxm}"


def load_and_analyze(path: Path) -> list[dict]:
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
            rlo, rhi, rpk, ppeak, w = rho_fwhm_and_peak(rho, pde)
            pt = float(rec.get("power_total_1e10", float("nan")))
            pj = float(rec.get("power_inj_total_1e10", float("nan")))
            ratio = pt / pj if pj and np.isfinite(pj) and pj != 0 else float("nan")

            rows.append(
                {
                    "frqncy_GHz": float(rec["frqncy"]),
                    "ioxm": int(rec["ioxm"]),
                    "mode": mode_tag(int(rec["ioxm"])),
                    "alfast_deg": float(rec["alfast"]),
                    "betast_deg": float(rec["betast"]),
                    "rho_peak": rpk,
                    "powden_e_peak": ppeak,
                    "rho_fwhm_lo": rlo,
                    "rho_fwhm_hi": rhi,
                    "rho_fwhm_width": w,
                    "power_total_1e10": pt,
                    "power_inj_total_1e10": pj,
                    "P_tot_over_P_inj": ratio,
                }
            )
    if not rows:
        raise ValueError(f"无有效记录: {path}")
    return rows


def write_table_csv(rows: list[dict], out: Path) -> None:
    fields = [
        "frqncy_GHz",
        "ioxm",
        "mode",
        "alfast_deg",
        "betast_deg",
        "rho_peak",
        "powden_e_peak",
        "rho_fwhm_lo",
        "rho_fwhm_hi",
        "rho_fwhm_width",
        "power_total_1e10",
        "power_inj_total_1e10",
        "P_tot_over_P_inj",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summary_by_freq_mode(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["frqncy_GHz"], r["ioxm"])].append(r)

    out: list[dict] = []

    def stats(a: np.ndarray) -> tuple[float, float, float]:
        m = np.isfinite(a)
        if not m.any():
            return float("nan"), float("nan"), float("nan")
        v = a[m]
        return (
            float(np.median(v)),
            float(np.percentile(v, 25)),
            float(np.percentile(v, 75)),
        )

    for (fq, ix) in sorted(groups.keys(), key=lambda t: (t[0], t[1])):
        g = groups[(fq, ix)]
        rpk = np.asarray([x["rho_peak"] for x in g], dtype=float)
        rlo = np.asarray([x["rho_fwhm_lo"] for x in g], dtype=float)
        rhi = np.asarray([x["rho_fwhm_hi"] for x in g], dtype=float)
        rw = np.asarray([x["rho_fwhm_width"] for x in g], dtype=float)

        med_pk, p25_pk, p75_pk = stats(rpk)
        med_lo, p25_lo, p75_lo = stats(rlo)
        med_hi, p25_hi, p75_hi = stats(rhi)
        med_w, p25_w, p75_w = stats(rw)

        out.append(
            {
                "frqncy_GHz": fq,
                "ioxm": ix,
                "mode": mode_tag(ix),
                "n_cases": len(g),
                "rho_peak_median": med_pk,
                "rho_peak_p25": p25_pk,
                "rho_peak_p75": p75_pk,
                "rho_fwhm_lo_median": med_lo,
                "rho_fwhm_lo_p25": p25_lo,
                "rho_fwhm_lo_p75": p75_lo,
                "rho_fwhm_hi_median": med_hi,
                "rho_fwhm_hi_p25": p25_hi,
                "rho_fwhm_hi_p75": p75_hi,
                "rho_fwhm_width_median": med_w,
                "rho_fwhm_width_p25": p25_w,
                "rho_fwhm_width_p75": p75_w,
            }
        )
    return out


def write_summary_csv(summary: list[dict], out: Path) -> None:
    if not summary:
        return
    fields = list(summary[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)


def _qk(x: float) -> float:
    return round(float(x), 6)


def build_grid(
    rows: list[dict], frqncy: float, ioxm: int, key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sub = [r for r in rows if r["frqncy_GHz"] == frqncy and r["ioxm"] == ioxm]
    if not sub:
        return None
    alphas = sorted({_qk(r["alfast_deg"]) for r in sub})
    betas = sorted({_qk(r["betast_deg"]) for r in sub})
    ia = {a: i for i, a in enumerate(alphas)}
    ib = {b: i for i, b in enumerate(betas)}
    nz, na = len(betas), len(alphas)
    Z = np.full((nz, na), np.nan, dtype=float)
    for r in sub:
        Z[ib[_qk(r["betast_deg"])], ia[_qk(r["alfast_deg"])]] = float(r[key])
    X, Y = np.meshgrid(np.asarray(alphas), np.asarray(betas))
    return X, Y, Z


def plot_distribution_panels(rows: list[dict], out: Path) -> None:
    cases = sorted({(r["frqncy_GHz"], r["ioxm"]) for r in rows})
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True)
    freq_order = sorted({r["frqncy_GHz"] for r in rows})
    mode_order = [-1, 1]
    for ax in axes.ravel():
        ax.set_visible(False)
    pos = {(fq, m): (freq_order.index(fq), 1 - mode_order.index(m)) for fq in freq_order for m in mode_order}
    for fq, ix in cases:
        g = [r for r in rows if r["frqncy_GHz"] == fq and r["ioxm"] == ix]
        rpk = np.asarray([x["rho_peak"] for x in g], dtype=float)
        rpk = rpk[np.isfinite(rpk)]
        ri, cj = pos[(fq, ix)]
        ax = axes[ri, cj]
        ax.set_visible(True)
        ax.hist(rpk, bins=32, range=(0.0, 1.0), color="0.45", edgecolor="0.15", alpha=0.88)
        med = float(np.median(rpk)) if rpk.size else float("nan")
        ax.axvline(med, color="crimson", lw=2.2, label=f"median ρ={med:.3f}")
        ax.set_title(f"{fq:g} GHz · {mode_tag(ix)}", fontweight="bold")
        ax.set_ylabel("count", fontweight="bold")
        ax.legend(loc="upper right", fontsize=12)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\rho$ at max($P_{\mathrm{den},e}$)", fontweight="bold")
    fig.suptitle(
        "Distribution of peak absorption ρ (all α–β combinations)",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contour_levels(norm: mpl.colors.Normalize, n: int = 14) -> np.ndarray:
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    if vmin == vmax:
        vmax = vmin + 1e-15
    return np.linspace(vmin, vmax, n)


def plot_fwhm_width_heatmaps(rows: list[dict], out: Path) -> None:
    cases = sorted({(r["frqncy_GHz"], r["ioxm"]) for r in rows})
    # 加大画布与子图间距，避免 3×2 等高线图与 colorbar 挤在一起
    fig, axes = plt.subplots(3, 2, figsize=(18.5, 20))
    freq_order = sorted({r["frqncy_GHz"] for r in rows})
    mode_order = [-1, 1]
    for ax in axes.ravel():
        ax.set_visible(False)
    pos = {(fq, m): (freq_order.index(fq), 1 - mode_order.index(m)) for fq in freq_order for m in mode_order}

    all_w = np.asarray([r["rho_fwhm_width"] for r in rows], dtype=float)
    all_w = all_w[np.isfinite(all_w)]
    vmin, vmax = float(np.min(all_w)), float(np.max(all_w))
    if vmin == vmax:
        vmax = vmin + 1e-12
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    levels = _contour_levels(norm)

    for fq, ix in cases:
        g = build_grid(rows, fq, ix, "rho_fwhm_width")
        if g is None:
            continue
        X, Y, Z = g
        ri, cj = pos[(fq, ix)]
        ax = axes[ri, cj]
        ax.set_visible(True)
        Zm = np.ma.masked_invalid(Z)
        ax.contourf(
            X,
            Y,
            Zm,
            levels=levels,
            cmap=PARULA,
            norm=norm,
            extend="both",
        )
        draw_labeled_contours(
            ax,
            X,
            Y,
            Z,
            levels=levels,
            label_every=2,
            fontsize=11,
        )
        ax.set_title(
            f"{fq:g} GHz · {mode_tag(ix)}\nFWHM Δρ",
            fontweight="bold",
            fontsize=15,
            pad=10,
        )
        ax.set_xlabel(r"$\alpha$ (deg)", fontweight="bold", labelpad=8)
        ax.set_ylabel(r"$\beta$ (deg)", fontweight="bold", labelpad=8)
        ax.set_aspect("auto")
        ax.tick_params(axis="both", which="major", pad=6)
        for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lb.set_fontweight("bold")

    fig.subplots_adjust(
        left=0.065,
        right=0.835,
        bottom=0.045,
        top=0.915,
        hspace=0.52,
        wspace=0.42,
    )

    cax = fig.add_axes([0.865, 0.14, 0.018, 0.62])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=PARULA)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(
        r"FWHM $\Delta\rho = \rho_{\mathrm{hi}}-\rho_{\mathrm{lo}}$",
        fontweight="bold",
        labelpad=12,
    )
    cb.ax.tick_params(labelsize=14)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    fig.suptitle(
        r"Absorption peak width (half-maximum $\rho$ interval)",
        fontweight="bold",
        fontsize=17,
        y=0.98,
    )
    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.35,
        facecolor="white",
    )
    plt.close(fig)


def plot_interval_strips(rows: list[dict], out: Path) -> None:
    """每个频率×模式一条子图：横轴 ρ，纵轴为按 ρ_peak 排序的工况；线段为 [ρ_FWHM_lo, ρ_FWHM_hi]，点标 ρ_peak。"""
    cases = sorted({(r["frqncy_GHz"], r["ioxm"]) for r in rows})
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), sharex=True)
    freq_order = sorted({r["frqncy_GHz"] for r in rows})
    mode_order = [-1, 1]
    for ax in axes.ravel():
        ax.set_visible(False)
    pos = {(fq, m): (freq_order.index(fq), 1 - mode_order.index(m)) for fq in freq_order for m in mode_order}

    for fq, ix in cases:
        g = sorted(
            [r for r in rows if r["frqncy_GHz"] == fq and r["ioxm"] == ix],
            key=lambda r: r["rho_peak"],
        )
        ri, cj = pos[(fq, ix)]
        ax = axes[ri, cj]
        ax.set_visible(True)
        for i, r in enumerate(g):
            rlo, rhi = r["rho_fwhm_lo"], r["rho_fwhm_hi"]
            rpk = r["rho_peak"]
            if np.isfinite(rlo) and np.isfinite(rhi):
                ax.plot([rlo, rhi], [i, i], color="steelblue", lw=1.8, solid_capstyle="round")
            if np.isfinite(rpk):
                ax.scatter([rpk], [i], color="crimson", s=22, zorder=5, linewidths=0)
        ax.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{fq:g} GHz · {mode_tag(ix)}", fontweight="bold")
        ax.set_xlabel(r"$\rho$", fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.45)

    fig.suptitle(
        r"FWHM $\rho$ interval (blue) and $\rho_{\mathrm{peak}}$ (red dot) per α–β case "
        r"(sorted by $\rho_{\mathrm{peak}}$)",
        fontweight="bold",
        y=1.005,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary_boxes(summary: list[dict], out: Path) -> None:
    """汇总：每个频率×模式的 ρ_FWHM 区间中位数与 IQR（横向条形示意）。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f'{s["frqncy_GHz"]:g} GHz\n{mode_tag(int(s["ioxm"]))}' for s in summary]
    y = np.arange(len(summary))
    lo = np.asarray([s["rho_fwhm_lo_median"] for s in summary], dtype=float)
    hi = np.asarray([s["rho_fwhm_hi_median"] for s in summary], dtype=float)
    pk = np.asarray([s["rho_peak_median"] for s in summary], dtype=float)

    for i in range(len(summary)):
        ax.plot([lo[i], hi[i]], [i, i], color="steelblue", lw=10, solid_capstyle="round", alpha=0.65)
        ax.scatter(pk[i], i, color="crimson", s=120, zorder=6, edgecolors="0.2", linewidths=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel(r"normalized $\rho$", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        r"Median FWHM deposition band $[\rho_{\mathrm{lo}},\rho_{\mathrm{hi}}]$ (blue bar) "
        r"and median $\rho_{\mathrm{peak}}$ (red)",
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="插入在 .csv/.png 扩展名之前，例如 _2T → absorption_peak_table_2T.csv",
    )
    args = ap.parse_args()

    configure_plot_style()
    sfx = normalize_suffix(args.suffix)

    rows = load_and_analyze(args.jsonl)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / f"{TABLE_CSV_STEM}{sfx}.csv"
    write_table_csv(rows, table_path)
    print(f"写入 {table_path}")

    summary = summary_by_freq_mode(rows)
    summary_path = out_dir / f"{SUMMARY_CSV_STEM}{sfx}.csv"
    write_summary_csv(summary, summary_path)
    print(f"写入 {summary_path}")

    plot_distribution_panels(rows, out_dir / f"absorption_peak_rho_distribution{sfx}.png")
    print(f"写入 {out_dir / f'absorption_peak_rho_distribution{sfx}.png'}")
    plot_fwhm_width_heatmaps(rows, out_dir / f"absorption_peak_fwhm_width_alpha_beta{sfx}.png")
    print(f"写入 {out_dir / f'absorption_peak_fwhm_width_alpha_beta{sfx}.png'}")
    plot_interval_strips(rows, out_dir / f"absorption_peak_interval_strip{sfx}.png")
    print(f"写入 {out_dir / f'absorption_peak_interval_strip{sfx}.png'}")
    plot_summary_boxes(summary, out_dir / f"absorption_peak_summary_median_bands{sfx}.png")
    print(f"写入 {out_dir / f'absorption_peak_summary_median_bands{sfx}.png'}")


if __name__ == "__main__":
    main()
