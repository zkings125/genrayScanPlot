# GENRAY 88890 数据处理工具

本目录用于从 GENRAY/CQL3D 的 NetCDF 结果提取 JSONL，并基于 JSONL 绘图。数据提取与绘图相互独立：生成图像时不再重复读取 NC 文件，适合大规模扫描和断点续算后的结果汇总。

## Python 快速使用

```bash
cd /home/jqlin/my/experiment/88890/data_processing_tools
python3 -m pip install --user numpy matplotlib netCDF4 scipy
```

从 `gres` 目录流式生成 JSONL：

```bash
python3 build_genray_jsonl_from_nc.py \
  /mnt/f/genrayCal/88890/gres \
  --jsonl ./genray_scan_from_nc.jsonl
```

一次运行全部四类绘图：

```bash
python3 run_all_plots.py genray_scan_from_nc.jsonl \
  --out-dir ./figs --suffix _Z2.5
```

四类输出包括：

1. 吸收功率比和 `rho_peak` 的 alpha-beta 图；
2. 每个频率的 O/X 2×2 对比图；
3. 吸收峰/FWHM 统计图和 CSV；
4. 电流驱动等高线图。电流图包含 alpha/beta 每 1° 的低透明度网格。

也可以单独运行：

```bash
python3 plot_alpha_beta_rho_ratio_surface.py --jsonl genray_scan_from_nc.jsonl --out-dir figs
python3 plot_alpha_beta_freq_2x2.py --jsonl genray_scan_from_nc.jsonl --out-dir figs
python3 plot_absorption_peak_ranges.py --jsonl genray_scan_from_nc.jsonl --out-dir figs
python3 plot_current_contours_from_jsonl.py genray_scan_from_nc.jsonl --out-dir figs
```

上述绘图脚本默认使用 `genray_scan_from_nc.jsonl`，也可用 `--jsonl` 指定其他文件。`--suffix` 会追加到输出文件名，便于保留不同扫描结果。

## JSONL 内容

每行对应一个算例，通常包含频率、`ioxm`、`alpha/beta`、吸收功率、`rho_peak`、`powden_e` 剖面以及电流驱动总量/剖面等字段。`build_genray_jsonl_from_nc.py` 采用逐文件处理，不把全部 NC 数据一次性载入内存。

## MATLAB 版本

Windows 版位于：`F:\genrayCal\data_processing_tools`。在 MATLAB 中加入该目录后运行：

```matlab
run_all_plots('F:\genrayCal\88890\gres\genray_scan_from_nc.jsonl', ...
              'F:\genrayCal\88890\gres\figs_matlab', '_Z2.5');
```

MATLAB 版本读取同一 JSONL，生成吸收功率比、O/X 2×2、电流等高线和峰值统计图；每张图都会同时保存 `.png` 和可编辑的 `.fig`。不需要 MATLAB 时，推荐使用 Python 版本。

## 直接扫描（可选）

`run_genray_param_scan.py` 和 `genray_alpha_beta_adaptive_scan.py` 用于修改 `genray.dat` 并调用 `xgenray`，正式运行前请先执行 `--dry-run` 并确认输出路径。扫描完成后，推荐使用 `build_genray_jsonl_from_nc.py` 从最终 `gres` 目录重建统一 JSONL。

## 内存与断点续算建议

- 计算过程中保留 NC；结束或中止后再用流式脚本重建 JSONL。
- 绘图只读取 JSONL，不扫描 NC，也不会把 NetCDF 原始变量长期缓存。
- 断点续算时使用已有 NC 文件名匹配，跳过已完成算例；新增结果可再次重建 JSONL。
