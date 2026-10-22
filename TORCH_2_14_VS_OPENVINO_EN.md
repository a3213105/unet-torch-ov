# U-Net CPU Performance: PyTorch 2.14.1 vs. OpenVINO 2026.3.0

Versions compared: **PyTorch 2.14.1+cpu** (eager and Inductor compile) and **OpenVINO 2026.3.0**.

Test configuration: Intel Xeon 6952P, the same `ov2630` environment, CPU affinity 2-9, 8 threads, and batch size 1. All modes use the same 10 images, with 3 full-dataset warmup passes and 10 full-dataset timed passes. oneDNN verbose is disabled. Only `core` inference is measured; preprocessing, model loading, and initial compilation are excluded.

## Average Time per Image

Unit: **ms/image; lower is better**.

| Scale | Precision | Torch 2.14 eager | Torch 2.14 compile | OpenVINO | Compile speedup over eager | OV speedup over eager | OV speedup over compile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | FP32 | 208.79 | 159.44 | 153.31 | 1.310x | 1.362x | 1.040x |
| 0.5 | BF16 | 103.51 | 31.42 | 25.76 | 3.294x | 4.019x | 1.220x |
| 0.5 | FP16 | 105.06 | 32.02 | 26.80 | 3.281x | 3.920x | 1.195x |
| 1.0 | FP32 | 863.60 | 635.93 | 614.96 | 1.358x | 1.404x | 1.034x |
| 1.0 | BF16 | 449.00 | 111.81 | 97.94 | 4.016x | 4.585x | 1.142x |
| 1.0 | FP16 | 439.62 | 117.20 | 105.71 | 3.751x | 4.159x | 1.109x |

Average time per image is the original `mean_ms / 10`, averaged across the input dataset. Compile speedup over eager is eager time divided by compile time. OV speedup over Torch is the corresponding Torch time divided by OpenVINO time. All speedups are calculated using unrounded times; values greater than 1 indicate that compile or OpenVINO, respectively, is faster than its reference.

## Serial Processing Rate

Unit: **FPS; higher is better**.

| Scale | Precision | Torch 2.14 eager | Torch 2.14 compile | OpenVINO |
| --- | --- | ---: | ---: | ---: |
| 0.5 | FP32 | 4.7894 | 6.2718 | 6.5227 |
| 0.5 | BF16 | 9.6605 | 31.8229 | 38.8252 |
| 0.5 | FP16 | 9.5179 | 31.2309 | 37.3078 |
| 1.0 | FP32 | 1.1579 | 1.5725 | 1.6261 |
| 1.0 | BF16 | 2.2271 | 8.9440 | 10.2106 |
| 1.0 | FP16 | 2.2747 | 8.5327 | 9.4601 |

FPS represents the serial processing rate for this input dataset, not concurrent throughput. These results come from a single experiment; no statistical significance testing across multiple independent experiments was performed.

## Source Data

- [Performance summary CSV](results/torch2141_20261021_142106/performance_summary.csv)
- [Torch eager raw JSON](results/torch2141_20261021_142106/performance/eager/benchmark_20261021_142656.json)
- [Torch compile raw JSON](results/torch2141_20261021_142106/performance/compile/benchmark_20261021_143033.json)
- [OpenVINO raw JSON](results/torch2141_20261021_142106/performance/openvino/benchmark_20261021_143252.json)
