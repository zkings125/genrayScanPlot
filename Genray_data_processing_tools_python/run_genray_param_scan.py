#!/usr/bin/env python3
"""
GENRAY 参数扫描：仅修改 genray.dat 中的 frqncy、ioxm、alfast、betast。
环向角扫描网格为 0:5:35（8 点），写入 genray 时 alfast = 扫描值 + 180（例如扫描 5° → alfast=185）。
每轮运行 ../../xgenray，读取 mnemonic.nc，将结果追加到 genray_scan_results.jsonl，
然后删除该 nc 以节省空间。

变量说明见同目录 genrayOutVar.json。输出 nc 中 rho_bin_center、powden_e 一般为
与 NR-1 一致长度的一维数组；若为多维则整体序列化为嵌套列表。
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
DEFAULT_OUT = WORK_DIR / "genray_scan_results.jsonl"
VARDEF_PATH = WORK_DIR / "genrayOutVar.json"

# 扫描网格
FREQS_GHZ = [105, 140, 170]
IOXMS = [-1, 1]
# 环向角扫描值 0:5:35 → 8 点；写入 genray.dat 时为该值 + ALFAST_GENRAY_OFFSET
ALFAST_GENRAY_OFFSET = 180
ALFASTS = list(range(0, 40, 5))
# 极向角：-60:5:0（步长 5，含 0）共 13 点 → 3*2*8*13 = 624 组（全网格）
BETASTS = list(range(-60, 5, 5))
# 相对旧网格 -40:5:0 新增的 4 个极向角（不含 -40；仅补算时用 --only-extra-betast）
BETASTS_EXTRA = [-60, -55, -50, -45]


def load_json_variable_names() -> set[str]:
    if not VARDEF_PATH.is_file():
        return set()
    with open(VARDEF_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["name"] for entry in data.get("variables", [])}


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
    """将 NetCDF 变量读入 numpy 后转为 JSON 可序列化形式。"""
    if var.ndim == 0:
        return float(var)
    return var.tolist()


def extract_run(nc_path: Path) -> dict:
    with nc.Dataset(str(nc_path), "r") as ds:
        need = ("rho_bin_center", "powden_e", "power_total", "power_inj_total")
        for name in need:
            if name not in ds.variables:
                raise KeyError(f"{nc_path} 缺少变量 {name}")
        rho = np.asarray(ds.variables["rho_bin_center"][:])
        pde = np.asarray(ds.variables["powden_e"][:])
        ptot = float(np.asarray(ds.variables["power_total"][:]))
        pinj = float(np.asarray(ds.variables["power_inj_total"][:]))
    return {
        "rho_bin_center": nc_array_to_json(rho),
        "powden_e": nc_array_to_json(pde),
        "power_total_1e10": ptot / 1.0e10,
        "power_inj_total_1e10": pinj / 1.0e10,
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
    alfast_scan: float,
    betast: float,
    out_jsonl: Path,
    dry_run: bool,
) -> None:
    alfast_genray = alfast_scan + ALFAST_GENRAY_OFFSET
    patched = patch_genray(genray_text, frqncy, ioxm, alfast_genray, betast)
    genray_path = cwd / "genray.dat"
    if not dry_run:
        genray_path.write_text(patched, encoding="utf-8")

    nc_path = cwd / f"{mnemonic}.nc"
    if dry_run:
        print(f"[dry-run] would write genray.dat and run {XGENRAY}, then read {nc_path}")
        return

    if not XGENRAY.is_file():
        raise FileNotFoundError(f"未找到可执行文件: {XGENRAY}")

    env = os.environ.copy()
    # 在运行目录执行，保证相对路径 ../../xgenray 与 genray.dat、平衡一致
    r = subprocess.run(
        [str(XGENRAY)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = {
            "error": "xgenray_failed",
            "returncode": r.returncode,
            "frqncy": frqncy,
            "ioxm": ioxm,
            "alfast": alfast_scan,
            "alfast_genray": alfast_genray,
            "betast": betast,
            "stderr_tail": (r.stderr or "")[-4000:],
            "stdout_tail": (r.stdout or "")[-4000:],
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"xgenray 退出码 {r.returncode}；详情已写入 {out_jsonl}"
        )

    if not nc_path.is_file():
        raise FileNotFoundError(f"运行结束但未找到输出文件: {nc_path}")

    payload = {
        "frqncy": frqncy,
        "ioxm": ioxm,
        "alfast": alfast_scan,
        "alfast_genray": alfast_genray,
        "betast": betast,
    }
    payload.update(extract_run(nc_path))

    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    nc_path.unlink(missing_ok=True)


def load_existing_keys(jsonl_path: Path) -> set[tuple[float, int, float, float]]:
    """已写入 jsonl 的成功工况键 (frqncy, ioxm, alfast, betast)。"""
    keys: set[tuple[float, int, float, float]] = set()
    if not jsonl_path.is_file():
        return keys
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            keys.add(
                (
                    float(rec["frqncy"]),
                    int(rec["ioxm"]),
                    float(rec["alfast"]),
                    float(rec["betast"]),
                )
            )
    return keys


def _qk_combo(frqncy: float, ioxm: int, alfast: float, betast: float) -> tuple[float, int, float, float]:
    return (round(frqncy, 6), ioxm, round(alfast, 6), round(betast, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description="GENRAY 四参数扫描并汇总到 jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"汇总输出路径（默认 {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印组合数量与示例，不写 genray、不运行 xgenray",
    )
    parser.add_argument(
        "--only-extra-betast",
        action="store_true",
        help=f"仅扫描新增极向角 {BETASTS_EXTRA}（相对旧 -40:5:0 网格补 4 点）",
    )
    parser.add_argument(
        "--betast",
        type=float,
        nargs="+",
        metavar="DEG",
        help="仅扫描所列极向角（度），覆盖默认/only-extra-betast",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若 --out 指向的 jsonl 中已有同 (频率,模式,α,β) 成功记录则跳过",
    )
    args = parser.parse_args()

    names = load_json_variable_names()
    if names:
        for v in ("rho_bin_center", "powden_e", "power_total", "power_inj_total"):
            if v not in names:
                print(f"警告: genrayOutVar.json 中未列出 {v}", file=sys.stderr)

    genray_path = WORK_DIR / "genray.dat"
    genray_text, mnemonic = read_genray_template(genray_path)

    if args.betast is not None:
        betasts = sorted(set(args.betast))
    elif args.only_extra_betast:
        betasts = BETASTS_EXTRA
    else:
        betasts = BETASTS

    combos = [
        (f, m, a, b)
        for f in FREQS_GHZ
        for m in IOXMS
        for a in ALFASTS
        for b in betasts
    ]

    existing: set[tuple[float, int, float, float]] = set()
    if args.skip_existing:
        existing = {_qk_combo(*k) for k in load_existing_keys(args.out)}
        before = len(combos)
        combos = [c for c in combos if _qk_combo(*c) not in existing]
        print(f"--skip-existing: 跳过 {before - len(combos)} 组，待算 {len(combos)} 组")

    print(
        f"极向角列表: {betasts}"
    )
    print(
        f"扫描规模: {len(combos)} 组 "
        f"(frqncy×ioxm×alfast×betast = {len(FREQS_GHZ)}×{len(IOXMS)}×{len(ALFASTS)}×{len(betasts)})"
    )
    print(f"mnemonic → {mnemonic}.nc")
    print(f"汇总文件: {args.out.resolve()}")

    if args.dry_run:
        for i, c in enumerate(combos[:3]):
            ag = c[2] + ALFAST_GENRAY_OFFSET
            print(
                f"  示例 {i+1}: frqncy={c[0]}, ioxm={c[1]}, "
                f"alfast_scan={c[2]}→genray={ag}, betast={c[3]}"
            )
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for idx, (f, m, a, b) in enumerate(combos, start=1):
        print(
            f"[{idx}/{len(combos)}] frqncy={f} ioxm={m} "
            f"alfast_scan={a}→{a + ALFAST_GENRAY_OFFSET} betast={b}"
        )
        run_one(
            WORK_DIR,
            genray_text,
            mnemonic,
            float(f),
            int(m),
            float(a),
            float(b),
            args.out,
            dry_run=False,
        )

    print("全部完成。")


if __name__ == "__main__":
    main()
