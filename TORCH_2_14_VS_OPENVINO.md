# U-Net CPU 性能对比：PyTorch 2.14.1 与 OpenVINO 2026.3.0

对比版本：**PyTorch 2.14.1+cpu**（eager、Inductor compile）与 **OpenVINO 2026.3.0**。

测试口径：Intel Xeon 6952P，同一 `ov2630` 环境，CPU 2–9、8 线程，batch size 为 1；同一组 10 张图片，完整图片集预热 3 轮、计时 10 轮，关闭 oneDNN verbose。仅测 `core` 推理，不包含预处理、模型加载和首次编译耗时。

## 平均每张耗时

单位：**ms/张，越低越好**。

| Scale | 精度 | Torch 2.14 eager | Torch 2.14 compile | OpenVINO | Compile 相对 eager 加速 | OV 相对 eager 加速 | OV 相对 compile 加速 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | FP32 | 208.79 | 159.44 | 153.31 | 1.310x | 1.362x | 1.040x |
| 0.5 | BF16 | 103.51 | 31.42 | 25.76 | 3.294x | 4.019x | 1.220x |
| 0.5 | FP16 | 105.06 | 32.02 | 26.80 | 3.281x | 3.920x | 1.195x |
| 1.0 | FP32 | 863.60 | 635.93 | 614.96 | 1.358x | 1.404x | 1.034x |
| 1.0 | BF16 | 449.00 | 111.81 | 97.94 | 4.016x | 4.585x | 1.142x |
| 1.0 | FP16 | 439.62 | 117.20 | 105.71 | 3.751x | 4.159x | 1.109x |

平均每张耗时 = 原始 `mean_ms / 10`，是输入集平均值。Compile 相对 eager 加速比 = eager 耗时 / compile 耗时；OV 相对 Torch 加速比 = 对应 Torch 耗时 / OpenVINO 耗时。加速比均使用未四舍五入的耗时计算，大于 1 表示被比较的优化模式更快。

## 串行处理速率

单位：**FPS，越高越好**。

| Scale | 精度 | Torch 2.14 eager | Torch 2.14 compile | OpenVINO |
| --- | --- | ---: | ---: | ---: |
| 0.5 | FP32 | 4.7894 | 6.2718 | 6.5227 |
| 0.5 | BF16 | 9.6605 | 31.8229 | 38.8252 |
| 0.5 | FP16 | 9.5179 | 31.2309 | 37.3078 |
| 1.0 | FP32 | 1.1579 | 1.5725 | 1.6261 |
| 1.0 | BF16 | 2.2271 | 8.9440 | 10.2106 |
| 1.0 | FP16 | 2.2747 | 8.5327 | 9.4601 |

FPS 为串行处理该输入集的速率，不是并发吞吐。以上为单轮实验数据，未进行多次独立实验的显著性检验。

## 原始数据

- [本轮性能汇总 CSV](results/torch2141_20261021_142106/performance_summary.csv)
- [Torch eager 原始 JSON](results/torch2141_20261021_142106/performance/eager/benchmark_20261021_142656.json)
- [Torch compile 原始 JSON](results/torch2141_20261021_142106/performance/compile/benchmark_20261021_143033.json)
- [OpenVINO 原始 JSON](results/torch2141_20261021_142106/performance/openvino/benchmark_20261021_143252.json)
