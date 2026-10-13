# Carvana U-Net CPU / OpenVINO Benchmark

本目录测试两个 U-Net checkpoint：

| 变体 | PyTorch | OpenVINO |
| --- | --- | --- |
| scale 0.5 | `../models/UNet-models/unet_carvana_scale0.5_epoch2.pth` | `../models_ov/unet_carvana_scale0_5_epoch2/` |
| scale 1.0 | `../models/UNet-models/unet_carvana_scale1.0_epoch2.pth` | `../models_ov/unet_carvana_scale1_0_epoch2/` |

每个 OpenVINO 目录包含两套未压缩 FP32 IR：

- 普通 IR：输出 `[B,2,H,W]` logits，用于与 PyTorch 同精度比较。
- `_all` IR：在图内完成 resize 和 argmax，输出原图尺寸的 `int64` mask，用于端到端延迟与 mask 一致率测试。

测试 FP32、BF16 和 FP16，OpenVINO 固定 `PERFORMANCE_HINT=LATENCY`、`NUM_STREAMS=1`，并记录实际 precision、AMX 节点、NRMSE、cosine、pixel agreement 和 foreground IoU。

## 安装

与 YOLO-World、DINOv3 共用 `didi` 环境：

```bash
cd /home/sgui/Didi/2026/unet
./setup_env.sh
```

`setup_env.sh` 会调用上级目录的 `../setup_benchmark_env.sh`，统一安装当前已验证的最新兼容组合：PyTorch 2.13 CPU、TorchVision 0.28 CPU、MMCV 2.1 CPU ops、MMDetection/MMYOLO、Transformers 4.56.2 与 OpenVINO 2026.4。

## 运行

默认测试 `../test_images/` 下全部 10 张 ADAS 图片、两个 scale 和三种精度：

```bash
./run_benchmark.sh
```

指定绑核、图片、变体和精度：

```bash
./run_benchmark.sh --start-cpu 64 --core-count 6 \
  --variants scale0.5 scale1.0 \
  --images ../test_images/adas_coco_bike.jpg \
  --precisions FP32 BF16 FP16 \
  --warmup 3 --iterations 10
```

`--core-count` 是物理核数，启动器自动加入每个物理核的 HT sibling。只测网络 core 时添加 `--skip-all`。

## 可配置路径

```bash
./run_benchmark.sh \
  --model-dir /data/UNet-models \
  --openvino-dir /data/models_ov \
  --test-images-dir /data/test_images
```

对应环境变量为 `UNET_MODEL_DIR`、`UNET_OPENVINO_DIR`、`UNET_TEST_IMAGE_DIR` 和 `UNET_RESULT_DIR`。单文件可使用以下可重复参数覆盖：

```bash
--checkpoint scale0.5=/data/model.pth
--openvino-xml scale0.5=/data/model.xml
--openvino-all-xml scale0.5=/data/model_all.xml
```

结果写入 `results/benchmark_*.json` 和 CSV。