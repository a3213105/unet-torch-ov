ONEDNN_VERBOSE=2 taskset -c 2-9 python unet_benchmark.py --model-dir ./UNet-models --image-dir ./test_images --skip-ov --skip-all --warmup 1 --precisions BF16 --iterations 1 > unet_torch214_bf16.log

ONEDNN_VERBOSE=2 taskset -c 2-9 python unet_benchmark.py --openvino-dir ./models_ov --image-dir ./test_images --skip-pytorch --skip-all --warmup 1 --precisions BF16 --iterations 1 > unet_ov2603_bf16.log

taskset -c 2-9 python unet_benchmark.py --model-dir ./UNet-models --openvino-dir ./models_ov --image-dir ./test_images --skip-all --warmup 1 --precisions {FP32,BF16} --iterations 1 > unet_all_bf16.log
