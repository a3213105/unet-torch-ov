# Carvana U-Net CPU / OpenVINO Benchmark

本目录测试两个 U-Net checkpoint：

| 变体 | PyTorch | OpenVINO |
| --- | --- | --- |
| scale 0.5 | `UNet-models/unet_carvana_scale0.5_epoch2.pth` | `models_ov/unet_carvana_scale0_5_epoch2/` |
| scale 1.0 | `UNet-models/unet_carvana_scale1.0_epoch2.pth` | `models_ov/unet_carvana_scale1_0_epoch2/` |

每个 OpenVINO 目录包含两套未压缩 FP32 IR：

- 普通 IR：输出 `[B,2,H,W]` logits，用于与 PyTorch 同精度比较。
- `_all` IR：在图内完成 resize 和 argmax，输出原图尺寸的 `int64` mask，用于端到端延迟与 mask 一致率测试。

测试 FP32、BF16 和 FP16，OpenVINO 固定 `PERFORMANCE_HINT=LATENCY`、`NUM_STREAMS=1`，并记录实际 precision、AMX 节点、NRMSE、cosine、pixel agreement 和 foreground IoU。

## 安装

建议复用上级目录 `/home/sgui/2026` 的共享环境脚本：

```bash
cd /home/sgui/unet
bash /home/sgui/2026/setup_benchmark_env.sh
```

说明：仓库内的 `setup_env.sh` 会调用 `../setup_benchmark_env.sh`。仅当当前目录结构本身就是 `/home/sgui/2026/unet` 时，直接执行 `./setup_env.sh` 才会成功。

## 运行

默认测试 `test_images/` 下全部 10 张 ADAS 图片、两个 scale 和三种精度：

```bash
conda run -n didi_t213 python unet_benchmark.py \
  --model-dir /home/sgui/unet/UNet-models \
  --openvino-dir /home/sgui/unet/models_ov \
  --image-dir /home/sgui/unet/test_images
```

只做配置和文件完整性检查（不跑性能循环）：

```bash
conda run -n didi_t213 python unet_benchmark.py --check-only \
  --model-dir /home/sgui/unet/UNet-models \
  --openvino-dir /home/sgui/unet/models_ov \
  --image-dir /home/sgui/unet/test_images
```

指定图片、变体和精度：

```bash
conda run -n didi_t213 python unet_benchmark.py \
  --variants scale0.5 scale1.0 \
  --images /home/sgui/unet/test_images/adas_coco_bike.jpg \
  --precisions FP32 BF16 FP16 \
  --warmup 3 --iterations 10 \
  --model-dir /home/sgui/unet/UNet-models \
  --openvino-dir /home/sgui/unet/models_ov \
  --image-dir /home/sgui/unet/test_images
```

## 可配置路径

```bash
python unet_benchmark.py \
  --model-dir /data/UNet-models \
  --openvino-dir /data/models_ov \
  --image-dir /data/test_images
```

对应环境变量为 `UNET_MODEL_DIR`、`UNET_OPENVINO_DIR`、`UNET_IMAGE_DIR` 和 `UNET_RESULT_DIR`。单文件可使用以下可重复参数覆盖：

```bash
--checkpoint scale0.5=/data/model.pth
--openvino-xml scale0.5=/data/model.xml
--openvino-all-xml scale0.5=/data/model_all.xml
```

结果写入 `results/benchmark_*.json` 和 CSV。