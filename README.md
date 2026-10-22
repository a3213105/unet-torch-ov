# Carvana U-Net CPU / OpenVINO Benchmark

This repository benchmarks two U-Net checkpoints:

Run all commands below from the repository root.

| Variant | PyTorch | OpenVINO |
| --- | --- | --- |
| scale 0.5 | `UNet-models/unet_carvana_scale0.5_epoch2.pth` | `models_ov/unet_carvana_scale0_5_epoch2/` |
| scale 1.0 | `UNet-models/unet_carvana_scale1.0_epoch2.pth` | `models_ov/unet_carvana_scale1_0_epoch2/` |

Each OpenVINO directory contains two uncompressed FP32 IR variants:

- Standard IR: outputs `[B,2,H,W]` logits for numerical comparison with PyTorch.
- `_all` IR: performs resize and argmax inside the graph, and outputs an `int64` mask at original image size for end-to-end latency and mask consistency checks.

The benchmark runs FP32, BF16, and FP16. OpenVINO is configured with `PERFORMANCE_HINT=LATENCY` and `NUM_STREAMS=1`, and reports effective precision, AMX node usage, NRMSE, cosine similarity, pixel agreement, and foreground IoU.

## Installation

Use the minimal in-repo setup script:

```bash
bash setup_benchmark_env.sh
```

Note: the repository's `setup_env.sh` calls `../setup_benchmark_env.sh` and requires that script in the parent directory. Use the in-repo setup command above for a standalone checkout.

## Running the Benchmark

The benchmark retains three execution modes: PyTorch eager, PyTorch `torch.compile(model, backend="inductor")`, and OpenVINO. Use `--mode eager`, `--mode compile`, or `--mode openvino` to run one mode independently. Without an explicit mode, it runs compile and OpenVINO.

PyTorch inference runs under `torch.inference_mode()`, with gradients disabled. BF16/FP16 use the corresponding CPU autocast dtype; FP32 does not use autocast. Compile mode calls `torch.compile(model.eval(), backend="inductor")` with `torch._inductor.config.freezing=True` and `torch._inductor.config.cpp.weight_prepack=True`. A scoped configuration patch stays active throughout model tracing, initial execution, warmup, timing, and output checks, so both the Dynamo frontend and Inductor backend see these settings for every lazy compilation. Original configuration values are restored after each precision run or before an eager fallback. The complete input dataset is executed before formal timing. Compilation time for that initial pass is recorded separately and excluded from measured latency; subsequent lazy compilation, if any, is not separately deducted. If compilation setup or the first execution fails, the benchmark emits `[WARN]` and falls back to eager at the same precision. Failures in eager execution are still recorded.

The legacy `--torch-compile 0` option still selects eager, and `--torch-compile 1` selects compile. The JSON fields `torch_compile_backend`, `torch_inductor_freezing`, `torch_inductor_weight_prepack`, `torch_compile_error`, and `torch_compile_seconds` record the actual execution path and compilation status. The freezing/prepacking fields indicate enabled options, not a guarantee that every kernel avoids runtime reorders; verify the steady-state trace for that behavior. Existing core/all switches and timing boundaries are unchanged; the current CLI entry point forces core-only testing.

By default, the benchmark tests all 10 ADAS images under `test_images/`, both scales, and all three precisions:

```bash
conda run -n torch_ov python unet_benchmark.py \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images
```

Print configuration and file-existence information without running performance loops:

```bash
conda run -n torch_ov python unet_benchmark.py --check-only \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images
```

Example arguments for images, variants, and precisions:

```bash
conda run -n torch_ov python unet_benchmark.py \
  --variants scale0.5 scale1.0 \
  --images ./test_images/adas_coco_bike.jpg \
  --precisions FP32 BF16 FP16 \
  --warmup 3 --iterations 10 \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images
```

## Configurable Paths

```bash
python unet_benchmark.py \
  --model-dir ./data/UNet-models \
  --openvino-dir ./data/models_ov \
  --image-dir ./data/test_images
```

The corresponding environment variables are `UNET_MODEL_DIR`, `UNET_OPENVINO_DIR`, `UNET_IMAGE_DIR`, and `UNET_RESULT_DIR`. Override individual model files with the following repeatable arguments:

```bash
--checkpoint scale0.5=./data/model.pth
--openvino-xml scale0.5=./data/model.xml
--openvino-all-xml scale0.5=./data/model_all.xml
```

Results are saved as `results/benchmark_*.json` and CSV files.

## Capturing oneDNN Verbose for All Three Modes

Run the following after installing the required dependencies:

```bash
CONDA_ENV=didi_t213 CPU_LIST=2-9 bash run_onednn_verbose.sh
```

The script runs eager, compile, and OpenVINO sequentially. By default, it uses both scales, all 10 images, all three precisions, and the benchmark's default warmup/iteration settings. Each mode starts a separate process with `ONEDNN_VERBOSE=2` and the compatibility variable `DNNL_VERBOSE=2`. Standard output and standard error are captured together.

Logs and JSON/CSV reports are stored under `results/onednn_<timestamp>/{eager,compile,openvino}/`. The top-level `run.txt` records commands and source checksums; `status.tsv` records exit codes. If a mode fails, the script continues testing the remaining modes and ultimately returns a nonzero exit code. A compile-to-eager fallback is explicitly recorded in the log and JSON; fallback results must not be treated as compiled results.

Set `UNET_VERBOSE_DIR` to choose the output directory. Use `UNET_MODEL_DIR`, `UNET_OPENVINO_DIR`, and `UNET_IMAGE_DIR` to override resource paths. Additional benchmark arguments are forwarded to all three modes, for example:

```bash
CONDA_ENV=didi_t213 CPU_LIST=2-9 bash run_onednn_verbose.sh \
  --precisions BF16 --warmup 1 --iterations 1
```

`OPENVINO_CONDA_ENV` selects a separate OpenVINO environment and defaults to `CONDA_ENV`. On didi, OpenVINO 2026.4.1 in `didi_t213` failed with `could not execute a primitive` when verbose was enabled, while a probe using the same input succeeded with verbose disabled. The existing OpenVINO 2026.3.0 environment supports verbose capture. To use separate environments on that machine:

```bash
CONDA_ENV=didi_t213 OPENVINO_CONDA_ENV=ov2630 CPU_LIST=2-9 \
  bash run_onednn_verbose.sh
```

Ensure that NumPy and Pillow are also installed in the OpenVINO environment. Actual package versions are recorded in each mode's JSON report. Retain the original failure logs for diagnosis.

Verbose logging affects timing. Use verbose runs for execution-path analysis, and run performance comparisons separately with verbose disabled.

### Measuring Performance Before Capturing Verbose in the Same ov2630 Environment

The `ov2630` environment must contain CPU-only PyTorch, OpenVINO, NumPy, and Pillow. On didi, it currently uses Torch 2.14.1+cpu and OpenVINO 2026.3.0. Both passes below use the same environment, CPU affinity 2-9, both scales, all 10 images, and all three precisions. All performance tests finish before verbose capture begins:

```bash
RUN_DIR="./results/ov2630_$(date +%Y%m%d_%H%M%S)"
CONDA_ENV=ov2630 OPENVINO_CONDA_ENV=ov2630 CPU_LIST=2-9 \
  UNET_ONEDNN_VERBOSE=0 UNET_VERBOSE_DIR="$RUN_DIR/performance" \
  bash run_onednn_verbose.sh --warmup 30 --iterations 100 &&
CONDA_ENV=ov2630 OPENVINO_CONDA_ENV=ov2630 CPU_LIST=2-9 \
  UNET_ONEDNN_VERBOSE=2 UNET_VERBOSE_DIR="$RUN_DIR/verbose" \
  bash run_onednn_verbose.sh --warmup 30 --iterations 100
```

`UNET_ONEDNN_VERBOSE` defaults to `2`. Setting it to `0` disables both `ONEDNN_VERBOSE` and `DNNL_VERBOSE` and changes the log filename to `benchmark.log`. JSON/CSV reports, logs, and exit statuses are saved separately for the performance and verbose passes.

The current entry point divides warmup/iteration counts by the number of images and rounds up. With 10 images, `--warmup 30 --iterations 100` produces 3 full-dataset warmup passes and 10 full-dataset timed passes. Each result's `mean_ms` is the average time for one pass over all 10 images, not single-image latency. Divide it by `samples_per_iteration` to obtain the average time per image. Initial compilation execution time is recorded separately and excluded from formal latency measurements.
