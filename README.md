# Carvana U-Net CPU / OpenVINO Benchmark

This repository benchmarks two U-Net checkpoints:

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

The default conda environment name is `torch_ov` (override with `BENCHMARK_CONDA_ENV`).

## Run

Default run over all 10 ADAS images in `test_images/`, both scales, and all three precisions:

```bash
conda run -n torch_ov python unet_benchmark.py \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images
```

Run configuration and file checks only (no performance loop):

```bash
conda run -n torch_ov python unet_benchmark.py --check-only \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images
```

Run with explicit image, variants, and precisions:

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

All commands above are expected to be run from the repository root.

## Backend-Specific Runs and oneDNN Logs

The commands below run PyTorch and OpenVINO separately using the same model, same input, same thread count, and same CPU affinity, while exporting oneDNN verbose logs.

```bash
mkdir -p logs

THREADS=8
CPU_LIST=0-7
COMMON_ARGS="\
  --variants scale0.5 scale1.0 \
  --images ./test_images/adas_coco_bike.jpg \
  --precisions BF16 \
  --warmup 3 --iterations 10 \
  --threads ${THREADS} \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images"
```

PyTorch-only run (skip OpenVINO):

```bash
ONEDNN_VERBOSE=2 DNNL_VERBOSE=2 \
taskset -c "${CPU_LIST}" \
conda run -n torch_ov python unet_benchmark.py \
  ${COMMON_ARGS} \
  --skip-ov \
  2>&1 | tee ./logs/torch_onednn.log
```

OpenVINO-only run (skip PyTorch):

```bash
ONEDNN_VERBOSE=2 DNNL_VERBOSE=2 \
taskset -c "${CPU_LIST}" \
conda run -n torch_ov python unet_benchmark.py \
  ${COMMON_ARGS} \
  --skip-pytorch \
  2>&1 | tee ./logs/openvino_onednn.log
```

Notes:

- Set both `ONEDNN_VERBOSE=2` and `DNNL_VERBOSE=2` for compatibility across different oneDNN versions.
- If your machine has a different CPU layout, adjust `THREADS` and `CPU_LIST`, but keep them identical across both runs.

## Inspect Logs and Compare Performance

Inspect key oneDNN verbose output (first 40 lines):

```bash
grep -E "onednn_verbose|dnnl_verbose" ./logs/torch_onednn.log | head -n 40
grep -E "onednn_verbose|dnnl_verbose" ./logs/openvino_onednn.log | head -n 40
```

Inspect summary performance lines (`ALL_IMAGES`) from each run:

```bash
grep "ALL_IMAGES" ./logs/torch_onednn.log
grep "ALL_IMAGES" ./logs/openvino_onednn.log
```

Compare both framework summaries side by side:

```bash
grep "ALL_IMAGES" ./logs/torch_onednn.log > ./logs/torch_summary.txt
grep "ALL_IMAGES" ./logs/openvino_onednn.log > ./logs/openvino_summary.txt
paste ./logs/torch_summary.txt ./logs/openvino_summary.txt
```

To get direct speedup output, run a combined benchmark once (the script prints a `Same-model PyTorch vs OpenVINO` table):

```bash
conda run -n torch_ov python unet_benchmark.py \
  --variants scale0.5 scale1.0 \
  --images ./test_images/adas_coco_bike.jpg \
  --precisions FP32 BF16 FP16 \
  --warmup 3 --iterations 10 \
  --threads 8 \
  --model-dir ./UNet-models \
  --openvino-dir ./models_ov \
  --image-dir ./test_images \
  2>&1 | tee ./logs/torch_vs_openvino.log
```

## Configurable Paths

```bash
python unet_benchmark.py \
  --model-dir /data/UNet-models \
  --openvino-dir /data/models_ov \
  --image-dir /data/test_images
```

Related environment variables are `UNET_MODEL_DIR`, `UNET_OPENVINO_DIR`, `UNET_IMAGE_DIR`, and `UNET_RESULT_DIR`. You can also override per-file paths with the following repeatable arguments:

```bash
--checkpoint scale0.5=/data/model.pth
--openvino-xml scale0.5=/data/model.xml
--openvino-all-xml scale0.5=/data/model_all.xml
```

Results are written to `results/benchmark_*.json` and CSV files.