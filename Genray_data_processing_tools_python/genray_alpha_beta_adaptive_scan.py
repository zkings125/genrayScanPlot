#!/usr/bin/env python3
"""
GENRAY α–β 自适应扫描（与任务说明一致）：

  - frqncy ∈ {105, 140, 170} GHz；ioxm ∈ {-1 (X), 1 (O)}
  - betast ∈ [-60, 0]，alfast ∈ [140, 220]（直接写入 genray.dat 的 &eccone，无 +180 偏移）
  - 粗扫：Δα=Δβ=10（β=-60,-50,…,0；α=140,150,…,220）
  - 精扫：在每个 (frqncy,ioxm) 粗扫完成后，取吸收效率较高的一批格点作种子，在 ±10° 内以步长 5 加密；
    再对全局最优格点作步长 3 的 ±6° 局部加密。所有点去重后依次运行。
  - 每轮：写 genray.dat → 运行 ../../xgenray（即仓库外同级的 xgenray 可执行文件）→
    读取 mnemonic.nc，将标量与一维剖面写入 jsonl → 删除 .nc

断点续跑：jsonl 中已存在的 (frqncy, ioxm, alfast, betast) 会跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "data_processing_tools" else SCRIPT_DIR
XGENRAY = WORK_DIR.parent.parent / "xgenray"
DEFAULT_JSONL = WORK_DIR / "genray_scan_alpha_beta_adaptive.jsonl"

FREQS_GHZ = [105.0, 140.0, 170.0]
IOXMS = [-1, 1]  # X, O

COARSE_BETAS = list(range(-60, 1, 10))
COARSE_ALPHAS = list(range(140, 221, 10))


def qk(x: float) -> float:
    return round(float(x), 6)


def load_jsonl_keys(path: Path) -> set[tuple[float, int, float, float]]:
    done: set[tuple[float, int, float, float]] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            done.add(
                (float(rec["frqncy"]), int(rec["ioxm"]), qk(rec["alfast"]), qk(rec["betast"]))
            )
    return done


def read_genray_template(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"mnemonic\s*=\s*'([^']*)'", text)
    if not m:
        raise ValueError("无法在 genray.dat 中解析 mnemonic='...'")
    return text, m.group(1)


def patch_genray(text: str, frqncy: float, ioxm: int, alfast: float, betast: float) -> str:
    s = text
    s = re.sub(r"(?m)^(\s*frqncy\s*=\s*)[^\n]+", rf"\g<1>{frqncy}", s, count=1)
    s = re.sub(r"(?m)^(\s*ioxm\s*=\s*)[^\n]+", rf"\g<1>{ioxm}", s, count=1)
    s = re.sub(r"(?m)^(\s*alfast\s*=\s*)[^\n]+", rf"\g<1>{alfast}", s, count=1)
    s = re.sub(r"(?m)^(\s*betast\s*=\s*)[^\n]+", rf"\g<1>{betast}", s, count=1)
    return s


def nc_array_to_json(var: np.ndarray) -> list | float:
    if var.ndim == 0:
        return float(var)
    return var.tolist()


def extract_run(nc_path: Path) -> dict:
    with nc.Dataset(str(nc_path), "r") as ds:
        need = ("rho_bin_center", "powden_e", "power_total", "power_inj_total")
        for name in need:
            if name not in ds.variables:
                raise KeyError(f"{nc_path} 缺少变量 {name}")
        rho = np.asarray(ds.variables["rho_bin_center"][:], dtype=float)
        pde = np.asarray(ds.variables["powden_e"][:], dtype=float)
        ptot = float(np.asarray(ds.variables["power_total"][:]))
        pinj = float(np.asarray(ds.variables["power_inj_total"][:]))
    imax = int(np.argmax(pde))
    rho_peak = float(rho[imax])
    pde_peak = float(pde[imax])
    ratio = ptot / pinj if pinj != 0.0 else float("nan")
    return {
        "rho_bin_center": nc_array_to_json(rho),
        "powden_e": nc_array_to_json(pde),
        "power_total_1e10": ptot / 1.0e10,
        "power_inj_total_1e10": pinj / 1.0e10,
        "rho_peak": rho_peak,
        "powden_e_peak": pde_peak,
        "pow_ratio": ratio,
        "_shapes": {
            "rho_bin_center": list(np.asarray(rho).shape),
            "powden_e": list(np.asarray(pde).shape),
        },
    }


def run_one(
    cwd: Path,
    genray_text: str,
    mnemonic: str,
    frqncy: float,
    ioxm: int,
    alfast: float,
    betast: float,
    scan_phase: str,
    out_jsonl: Path,
) -> None:
    patched = patch_genray(genray_text, frqncy, ioxm, alfast, betast)
    (cwd / "genray.dat").write_text(patched, encoding="utf-8")

    if not XGENRAY.is_file():
        raise FileNotFoundError(f"未找到可执行文件: {XGENRAY}")

    r = subprocess.run(
        [str(XGENRAY)],
        cwd=str(cwd),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    nc_path = cwd / f"{mnemonic}.nc"
    if r.returncode != 0:
        err = {
            "error": "xgenray_failed",
            "returncode": r.returncode,
            "frqncy": frqncy,
            "ioxm": ioxm,
            "alfast": alfast,
            "betast": betast,
            "scan_phase": scan_phase,
            "stderr_tail": (r.stderr or "")[-4000:],
            "stdout_tail": (r.stdout or "")[-4000:],
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
        raise RuntimeError(f"xgenray 退出码 {r.returncode}，已记录到 {out_jsonl}")

    if not nc_path.is_file():
        raise FileNotFoundError(f"运行结束但未找到输出文件: {nc_path}")

    payload = {
        "frqncy": frqncy,
        "ioxm": ioxm,
        "alfast": alfast,
        "betast": betast,
        "scan_phase": scan_phase,
    }
    payload.update(extract_run(nc_path))

    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    nc_path.unlink(missing_ok=True)


def coarse_points() -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a in COARSE_ALPHAS for b in COARSE_BETAS]


def fine_points_step5(
    seeds: list[tuple[float, float]],
    existing: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set(existing)
    for a0, b0 in seeds:
        for da in range(-10, 11, 5):
            for db in range(-10, 11, 5):
                a = float(np.clip(a0 + da, 140.0, 220.0))
                b = float(np.clip(b0 + db, -60.0, 0.0))
                key = (qk(a), qk(b))
                if key in seen:
                    continue
                seen.add(key)
                out.append((key[0], key[1]))
    return out


def fine_points_step3_around(
    center: tuple[float, float],
    existing: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    a0, b0 = center
    out: list[tuple[float, float]] = []
    seen = set(existing)
    for da in (-6, -3, 0, 3, 6):
        for db in (-6, -3, 0, 3, 6):
            a = float(np.clip(a0 + da, 140.0, 220.0))
            b = float(np.clip(b0 + db, -60.0, 0.0))
            key = (qk(a), qk(b))
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1]))
    return out


def load_records_for_pair(
    path: Path, frqncy: float, ioxm: int
) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            if float(rec["frqncy"]) != frqncy or int(rec["ioxm"]) != ioxm:
                continue
            if rec.get("scan_phase") != "coarse":
                continue
            rows.append(rec)
    return rows


def pick_seeds_from_coarse(coarse_recs: list[dict], max_seeds: int = 6) -> list[tuple[float, float]]:
    if not coarse_recs:
        return []
    scored: list[tuple[float, float, float]] = []
    for rec in coarse_recs:
        r = rec.get("pow_ratio")
        if r is None or not np.isfinite(float(r)):
            continue
        scored.append((float(r), float(rec["alfast"]), float(rec["betast"])))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0], reverse=True)
    max_r = scored[0][0]
    thr = max(0.65 * max_r, float(np.percentile([s[0] for s in scored], 70.0)))
    seeds: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for rv, a, b in scored:
        if rv < thr:
            break
        key = (qk(a), qk(b))
        if key in seen:
            continue
        seen.add(key)
        seeds.append((key[0], key[1]))
        if len(seeds) >= max_seeds:
            break
    if not seeds:
        seeds = [(qk(s[1]), qk(s[2])) for s in scored[: min(3, len(scored))]]
    return seeds


def best_center_from_all_recs(recs: list[dict]) -> tuple[float, float] | None:
    best_r = -1.0
    best_ab: tuple[float, float] | None = None
    for rec in recs:
        r = rec.get("pow_ratio")
        if r is None or not np.isfinite(float(r)):
            continue
        rf = float(r)
        if rf > best_r:
            best_r = rf
            best_ab = (float(rec["alfast"]), float(rec["betast"]))
    return best_ab


def _load_all_ok(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="GENRAY α–β 粗扫+精扫，汇总 jsonl 并删 nc")
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_JSONL,
        help="输出 jsonl（默认 genray_scan_alpha_beta_full.jsonl）",
    )
    ap.add_argument(
        "--coarse-only",
        action="store_true",
        help="只做粗扫（步长 10），不生成精扫任务",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印任务数与若干示例",
    )
    args = ap.parse_args()

    genray_path = WORK_DIR / "genray.dat"
    genray_text, mnemonic = read_genray_template(genray_path)
    done = load_jsonl_keys(args.out)

    coarse_keys = {(qk(a), qk(b)) for a, b in coarse_points()}
    n_coarse_total = len(FREQS_GHZ) * len(IOXMS) * len(coarse_points())
    n_coarse_left = sum(
        1
        for fq in FREQS_GHZ
        for mx in IOXMS
        for a, b in coarse_points()
        if (qk(fq), mx, qk(a), qk(b)) not in done
    )

    print(f"xgenray: {XGENRAY}")
    print(f"mnemonic → {mnemonic}.nc")
    print(f"汇总: {args.out.resolve()}")
    print(
        f"粗网格: |α|={len(COARSE_ALPHAS)}, |β|={len(COARSE_BETAS)} → "
        f"每 (f,m) {len(coarse_points())} 点，粗扫总槽位 {n_coarse_total}，待跑粗扫 {n_coarse_left}"
    )

    if args.dry_run:
        print("(dry-run) 示例粗扫键:", list(sorted(done))[:3])
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def run_if_new(fq: float, mx: int, a: float, b: float, phase: str, counter: list[int]) -> None:
        key = (qk(fq), mx, qk(a), qk(b))
        if key in done:
            return
        counter[0] += 1
        print(f"[{counter[0]}] {phase} f={fq} m={mx} α={a} β={b}")
        run_one(WORK_DIR, genray_text, mnemonic, fq, mx, a, b, phase, args.out)
        done.add(key)

    ctr = [0]
    for fq in FREQS_GHZ:
        for mx in IOXMS:
            for a, b in coarse_points():
                run_if_new(fq, mx, a, b, "coarse", ctr)

    if args.coarse_only:
        print("粗扫阶段结束（--coarse-only）。")
        return

    for fq in FREQS_GHZ:
        for mx in IOXMS:
            coarse_recs = load_records_for_pair(args.out, fq, mx)
            seeds = pick_seeds_from_coarse(coarse_recs, max_seeds=6)
            existing_rm = coarse_keys | {
                (qk(float(r["alfast"])), qk(float(r["betast"])))
                for r in _load_all_ok(args.out)
                if float(r["frqncy"]) == fq and int(r["ioxm"]) == mx
            }
            fine5 = fine_points_step5(seeds, existing_rm)
            for a, b in fine5:
                run_if_new(fq, mx, a, b, "fine_d5", ctr)

    for fq in FREQS_GHZ:
        for mx in IOXMS:
            all_recs = [
                r
                for r in _load_all_ok(args.out)
                if float(r["frqncy"]) == fq and int(r["ioxm"]) == mx
            ]
            ctr_ab = best_center_from_all_recs(all_recs)
            if ctr_ab is None:
                continue
            done_local = {(qk(float(r["alfast"])), qk(float(r["betast"]))) for r in all_recs}
            fine3 = fine_points_step3_around(ctr_ab, done_local)
            for a, b in fine3:
                run_if_new(fq, mx, a, b, "fine_d3", ctr)

    print("全部阶段完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        sys.exit(130)
