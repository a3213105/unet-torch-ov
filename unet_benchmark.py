#!/usr/bin/env python3
"""Benchmark Carvana U-Net models with PyTorch and OpenVINO CPU backends."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT.parent / "models" / "UNet-models"
DEFAULT_OPENVINO_DIR = ROOT.parent / "models_ov"
DEFAULT_IMAGE_DIR = ROOT.parent / "test_images"
DEFAULT_RESULT_DIR = ROOT / "results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

VARIANTS = {
    "scale0.5": {
        "scale": 0.5,
        "stem": "unet_carvana_scale0.5_epoch2",
        "ov_dir": "unet_carvana_scale0_5_epoch2",
    },
    "scale1.0": {
        "scale": 1.0,
        "stem": "unet_carvana_scale1.0_epoch2",
        "ov_dir": "unet_carvana_scale1_0_epoch2",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    scale: float
    checkpoint: Path
    xml_path: Path
    all_xml_path: Path

    @property
    def name(self) -> str:
        return self.xml_path.stem


def build_unet() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.double_conv(inputs)

    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

        def forward(self, inputs: Any) -> Any:
            return self.maxpool_conv(inputs)

    class Up(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, inputs: Any, skip: Any) -> Any:
            inputs = self.up(inputs)
            diff_y = skip.size()[2] - inputs.size()[2]
            diff_x = skip.size()[3] - inputs.size()[3]
            inputs = functional.pad(
                inputs,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
            return self.conv(torch.cat([skip, inputs], dim=1))

    class OutConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, inputs: Any) -> Any:
            return self.conv(inputs)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inc = DoubleConv(3, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            self.up1 = Up(1024, 512)
            self.up2 = Up(512, 256)
            self.up3 = Up(256, 128)
            self.up4 = Up(128, 64)
            self.outc = OutConv(64, 2)

        def forward(self, inputs: Any) -> Any:
            level1 = self.inc(inputs)
            level2 = self.down1(level1)
            level3 = self.down2(level2)
            level4 = self.down3(level3)
            inputs = self.down4(level4)
            inputs = self.up1(inputs, level4)
            inputs = self.up2(inputs, level3)
            inputs = self.up3(inputs, level2)
            inputs = self.up4(inputs, level1)
            return self.outc(inputs)

    return UNet()


def parse_overrides(values: Sequence[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects VARIANT=PATH, got: {value}")
        variant, path = value.split("=", 1)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant in {option}: {variant}")
        result[variant] = Path(path).expanduser().resolve()
    return result


def collect_images_from_dir(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Carvana U-Net PyTorch and OpenVINO FP32/BF16/FP16 performance."
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("UNET_MODEL_DIR", DEFAULT_MODEL_DIR)),
    )
    parser.add_argument(
        "--openvino-dir",
        type=Path,
        default=Path(os.environ.get("UNET_OPENVINO_DIR", DEFAULT_OPENVINO_DIR)),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(os.environ.get("UNET_IMAGE_DIR", DEFAULT_IMAGE_DIR)),
        help="Directory containing images to benchmark.",
    )
    parser.add_argument("--images", type=Path, nargs="+", help="Images to test; defaults to every image in --image-dir.")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="VARIANT=PATH")
    parser.add_argument("--openvino-xml", action="append", default=[], metavar="VARIANT=PATH")
    parser.add_argument("--openvino-all-xml", action="append", default=[], metavar="VARIANT=PATH")
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("FP32", "BF16", "FP16"),
        default=["FP32", "BF16", "FP16"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timed loops. Each loop runs once over the full input image set.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--skip-all", default=True, action="store_true", help="Skip end-to-end resize and argmax paths.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmark paths.")
    parser.add_argument(
        "--mode",
        choices=("eager", "compile", "openvino"),
        help="Run only the selected backend. Defaults to compile plus OpenVINO.",
    )
    parser.add_argument(
        "--torch-compile",
        type=int,
        choices=(0, 1),
        default=None,
        help="Legacy PyTorch selector: Inductor (1) or eager (0).",
    )
    parser.add_argument("--skip-ov", action="store_true", help="Skip OpenVINO benchmark paths.")

    args = parser.parse_args()

    args.skip_all = True
    if args.mode is not None:
        if args.torch_compile is not None:
            parser.error("--mode cannot be combined with --torch-compile")
        if args.skip_pytorch or args.skip_ov:
            parser.error("--mode cannot be combined with --skip-pytorch or --skip-ov")
        args.skip_pytorch = args.mode == "openvino"
        args.skip_ov = args.mode != "openvino"
        args.torch_compile = int(args.mode == "compile")
    elif args.torch_compile is None:
        args.torch_compile = 1
    if args.skip_pytorch and args.skip_ov:
        parser.error("--skip-pytorch and --skip-ov cannot both be set")
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1 or args.threads < 0:
        parser.error("invalid batch-size, warmup, iterations, or threads")
    try:
        args.checkpoint = parse_overrides(args.checkpoint, "--checkpoint")
        args.openvino_xml = parse_overrides(args.openvino_xml, "--openvino-xml")
        args.openvino_all_xml = parse_overrides(args.openvino_all_xml, "--openvino-all-xml")
    except ValueError as error:
        parser.error(str(error))
    args.model_dir = args.model_dir.expanduser().resolve()
    args.openvino_dir = args.openvino_dir.expanduser().resolve()

    image_dir = args.image_dir.expanduser().resolve() if args.image_dir else None
    if image_dir is not None:
        args.images = collect_images_from_dir(image_dir)
        args.image_dir = image_dir
    elif args.images:
        resolved_images = [path.expanduser().resolve() for path in args.images]
        if len(resolved_images) == 1 and resolved_images[0].is_dir():
            args.image_dir = resolved_images[0]
            args.images = collect_images_from_dir(args.image_dir)
        else:
            args.image_dir = None
            args.images = resolved_images
    else:
        args.images = collect_images_from_dir(args.image_dir)

    if not args.images:
        parser.error(f"No supported images found in directory: {args.image_dir}")
    return args


def build_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = []
    for variant in args.variants:
        data = VARIANTS[variant]
        ov_dir = args.openvino_dir / data["ov_dir"]
        stem = data["stem"]
        specs.append(ModelSpec(
            variant=variant,
            scale=data["scale"],
            checkpoint=args.checkpoint.get(variant, args.model_dir / f"{stem}.pth"),
            xml_path=args.openvino_xml.get(variant, ov_dir / f"{stem}.xml"),
            all_xml_path=args.openvino_all_xml.get(variant, ov_dir / f"{stem}_all.xml"),
        ))
    return specs


def cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def default_thread_count() -> int:
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    physical_cores = set()
    for cpu in affinity:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            physical_cores.add(((topology / "physical_package_id").read_text().strip(), (topology / "core_id").read_text().strip()))
        except OSError:
            return len(affinity)
    return len(physical_cores) or len(affinity)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "numpy", "pillow", "openvino"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


def environment_report(specs: Sequence[ModelSpec], args: argparse.Namespace) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    report = {
        "python": platform.python_version(),
        "cpu": cpu_model(),
        "cpu_affinity": affinity,
        "affinity_logical_cpus": len(affinity),
        "taskset_binding": {
            "enabled": os.environ.get("UNET_CPU_BOUND") == "1",
            "start_cpu": os.environ.get("UNET_START_CPU"),
            "physical_core_count": os.environ.get("UNET_CORE_COUNT"),
            "logical_cpu_list": os.environ.get("UNET_CPU_LIST"),
        },
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "recommended_threads": default_thread_count(),
        "amx_flags": sorted(cpu_flags() & {"amx_bf16", "amx_fp16", "amx_int8", "amx_tile"}),
        "packages": package_versions(),
        "image_dir": str(args.image_dir) if args.image_dir else None,
        "image_count": len(args.images),
        "images": [str(path) for path in args.images],
        "models": [
            {
                "variant": spec.variant,
                "scale": spec.scale,
                "checkpoint": str(spec.checkpoint),
                "checkpoint_exists": spec.checkpoint.is_file(),
                "openvino_xml": str(spec.xml_path),
                "openvino_exists": spec.xml_path.is_file(),
                "openvino_all_xml": str(spec.all_xml_path),
                "openvino_all_exists": spec.all_xml_path.is_file(),
            }
            for spec in specs
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def require_environment(specs: Sequence[ModelSpec], args: argparse.Namespace) -> None:
    versions = package_versions()
    required_packages = ["numpy", "pillow"]
    if not args.skip_pytorch:
        required_packages.append("torch")
    if not args.skip_ov:
        required_packages.append("openvino")
    missing_packages = [name for name in required_packages if versions.get(name) == "NOT INSTALLED"]
    for spec in specs:
        if not spec.all_xml_path.is_file():
            args.skip_all = True
    missing_files = []
    for spec in specs:
        if not args.skip_pytorch and not spec.checkpoint.is_file():
            missing_files.append(str(spec.checkpoint))
        if not args.skip_ov:
            if not spec.xml_path.is_file():
                missing_files.append(str(spec.xml_path))
            if not args.skip_all and not spec.all_xml_path.is_file():
                missing_files.append(str(spec.all_xml_path))
    missing_images = [str(path) for path in args.images if not path.is_file()]
    if missing_packages:
        raise SystemExit("Missing packages: " + ", ".join(missing_packages))
    if missing_files:
        raise SystemExit("Missing model files:\n  " + "\n  ".join(missing_files))
    if missing_images or not args.images:
        raise SystemExit("Missing test images:\n  " + "\n  ".join(missing_images))


def load_model(spec: ModelSpec) -> tuple[Any, list[Any], float]:
    import torch

    start = time.perf_counter()
    state = torch.load(spec.checkpoint, map_location="cpu")
    mask_values = state.pop("mask_values", [0, 1])
    model = build_unet().eval()
    model.load_state_dict(state)
    return model, mask_values, time.perf_counter() - start


def preprocess_image(path: Path, scale: float, batch_size: int) -> tuple[Any, tuple[int, int]]:
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    resized_width = max(1, int(scale * width))
    resized_height = max(1, int(scale * height))
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    batched = np.repeat(array[None, ...], batch_size, axis=0).astype(np.float32, copy=False)
    return batched, (height, width)


def percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timed_iterations(call: Callable[[], Any], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        call()
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        call()
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return latencies


def summarize(spec: ModelSpec, image: str, backend: str, path: str, precision: str, args: argparse.Namespace, threads: int, latencies: Sequence[float], samples_per_iteration: int, **extra: Any) -> dict[str, Any]:
    mean_ms = statistics.mean(latencies)
    result = {
        "model": spec.name,
        "variant": spec.variant,
        "scale": spec.scale,
        "image": image,
        "backend": backend,
        "path": path,
        "precision": precision,
        "batch_size": args.batch_size,
        "threads": threads,
        "iterations": len(latencies),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(percentile(latencies, 50), 4),
        "p90_ms": round(percentile(latencies, 90), 4),
        "p99_ms": round(percentile(latencies, 99), 4),
        "samples_per_iteration": samples_per_iteration,
        "throughput_fps": round(samples_per_iteration * 1000.0 / mean_ms, 4),
    }
    result.update(extra)
    return result


def autocast_dtype(precision: str) -> Any | None:
    import torch
    return {"BF16": torch.bfloat16, "FP16": torch.float16}.get(precision)


def compare_arrays(reference: Any, actual: Any) -> dict[str, Any]:
    import numpy as np

    expected = reference.astype(np.float32, copy=False)
    observed = actual.astype(np.float32, copy=False)
    finite = np.isfinite(expected).all() and np.isfinite(observed).all()
    result = {"valid": bool(finite), "reference_shape": list(expected.shape), "actual_shape": list(observed.shape)}
    if not finite or expected.shape != observed.shape:
        return result
    difference = expected - observed
    expected_squared = float(np.sum(expected * expected, dtype=np.float64))
    observed_squared = float(np.sum(observed * observed, dtype=np.float64))
    dot = float(np.sum(expected * observed, dtype=np.float64))
    squared_error = float(np.sum(difference * difference, dtype=np.float64))
    result.update({
        "mean_abs": round(float(np.abs(difference).mean()), 8),
        "max_abs": round(float(np.abs(difference).max()), 8),
        "normalized_rmse": round((squared_error / max(expected_squared, 1e-12)) ** 0.5, 8),
        "cosine_similarity": round(dot / max((expected_squared * observed_squared) ** 0.5, 1e-12), 8),
    })
    return result


def compare_masks(reference: Any, actual: Any) -> dict[str, Any]:
    import numpy as np

    expected = reference.astype(np.int64, copy=False)
    observed = actual.astype(np.int64, copy=False)
    if expected.shape != observed.shape:
        return {"valid": False, "reference_shape": list(expected.shape), "actual_shape": list(observed.shape)}
    intersection = np.logical_and(expected == 1, observed == 1).sum()
    union = np.logical_or(expected == 1, observed == 1).sum()
    return {
        "valid": True,
        "pixel_agreement": round(float((expected == observed).mean()), 8),
        "foreground_iou": round(float(intersection / union), 8) if union else 1.0,
        "different_pixels": int((expected != observed).sum()),
    }


def ov_property(compiled: Any, name: str) -> str:
    try:
        return str(compiled.get_property(name))
    except Exception:
        return "unavailable"


def compile_openvino(path: Path, precision: str, threads: int) -> Any:
    import openvino as ov

    config = {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": {"FP32": "f32", "BF16": "bf16", "FP16": "f16"}[precision],
        "INFERENCE_NUM_THREADS": threads,
        "NUM_STREAMS": 1,
        "PERF_COUNT": "YES",
    }
    return ov.Core().compile_model(path, "CPU", config)


def benchmark_dataset(
    spec: ModelSpec,
    model_torch: Any,
    args: argparse.Namespace,
    threads: int,
    compiled_core: dict[str, Any],
    compiled_all: dict[str, Any],
    iterations,
    warmup,
    failures: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    import numpy as np

    OutputStore = dict[tuple[str, str, str], dict[str, Any]]

    torch = None
    functional = None
    if not args.skip_pytorch:
        import torch
        import torch.nn.functional as functional

    prepared_inputs = []
    for image in args.images:
        input_data, original_shape = preprocess_image(image, spec.scale, args.batch_size)
        item = {
            "input_data": input_data,
            "original_shape": original_shape,
            "full_shape": np.asarray(original_shape, dtype=np.int32),
        }
        if torch is not None:
            item["input_tensor"] = torch.from_numpy(input_data.copy())
        prepared_inputs.append(item)
    first = prepared_inputs[0]
    image_label = f"ALL_IMAGES({len(args.images)})"
    samples_per_iteration = len(prepared_inputs) * args.batch_size
    results = []
    outputs: OutputStore = {}

    def store_output(backend: str, path: str, precision: str, index: int, logits: Any | None = None, mask: Any | None = None) -> None:
        outputs[(backend, path, precision)] = {
            "index": index,
            "backend": backend,
            "path": path,
            "precision": precision,
            "logits": logits,
            "mask": mask,
        }

    def apply_consistency_metrics() -> None:
        fp32_reference: dict[str, tuple[str, str, str]] = {}
        for path in ("core", "all"):
            torch_fp32 = ("pytorch", path, "FP32")
            ov_fp32 = ("openvino", path, "FP32")
            if torch_fp32 in outputs:
                fp32_reference[path] = torch_fp32
            elif ov_fp32 in outputs:
                fp32_reference[path] = ov_fp32

        for key, item in outputs.items():
            backend, path, precision = key
            reference_key = None

            # If FP32 exists for this path, every other run compares to that FP32 baseline.
            if path in fp32_reference:
                if key != fp32_reference[path]:
                    reference_key = fp32_reference[path]
            # Fallback: without FP32, OpenVINO compares to same-precision PyTorch when available.
            elif backend == "openvino":
                torch_peer = ("pytorch", path, precision)
                if torch_peer in outputs:
                    reference_key = torch_peer

            if reference_key is None:
                continue

            reference = outputs[reference_key]
            current_result = results[item["index"]]

            if item.get("logits") is not None and reference.get("logits") is not None:
                reference_logits = reference["logits"]
                current_logits = item["logits"]
                current_result["output_difference"] = compare_arrays(reference_logits, current_logits)
                current_result["mask_difference"] = compare_masks(
                    reference_logits.argmax(axis=1),
                    current_logits.argmax(axis=1),
                )
            elif item.get("mask") is not None and reference.get("mask") is not None:
                current_result["mask_difference"] = compare_masks(reference["mask"], item["mask"])

    for precision in args.precisions:
        try:
            if not args.skip_pytorch:
                cast_dtype = autocast_dtype(precision)

                def torch_core_single(single_input: Any) -> Any:
                    if cast_dtype is None:
                        return opt_model(single_input)
                    with torch.autocast(device_type="cpu", dtype=cast_dtype):
                        return opt_model(single_input)

                def torch_core() -> Any:
                    last_logits = None
                    for item in prepared_inputs:
                        last_logits = torch_core_single(item["input_tensor"])
                    return last_logits

                def torch_all() -> Any:
                    last_mask = None
                    for item in prepared_inputs:
                        logits = torch_core_single(item["input_tensor"])
                        last_mask = functional.interpolate(logits, item["original_shape"], mode="bilinear").argmax(dim=1)
                    return last_mask

                with torch.inference_mode(), ExitStack() as compile_context:
                    compile_error = ""
                    opt_model = model_torch
                    compile_start = time.perf_counter()
                    if args.torch_compile == 1:
                        try:
                            import torch._inductor.config as inductor_cfg

                            # Dynamo must also see freezing during every trace.
                            compile_context.enter_context(inductor_cfg.patch({
                                "freezing": True,
                                "cpp.weight_prepack": True,
                            }))
                            opt_model = torch.compile(
                                model_torch.eval(),
                                backend="inductor",
                            )
                            torch_core()
                        except Exception as error:
                            compile_context.close()
                            compile_error = f"{type(error).__name__}: {error}"
                            print(f"[WARN] {spec.variant} {precision} compile failed; falling back to eager: {compile_error}", flush=True)
                            opt_model = model_torch
                    compile_seconds = time.perf_counter() - compile_start if args.torch_compile else 0.0
                    if compile_error:
                        torch_core()
                    compiled = args.torch_compile == 1 and not compile_error
                    compile_metadata = {
                        "torch_compile_backend": "inductor" if compiled else "eager",
                        "torch_inductor_freezing": compiled,
                        "torch_inductor_weight_prepack": compiled,
                        "torch_compile_error": compile_error,
                        "torch_compile_seconds": round(compile_seconds, 4),
                    }
                    core_latencies = timed_iterations(torch_core, warmup, iterations)
                    torch_logits = torch_core_single(first["input_tensor"]).detach().float().numpy().copy()
                    results.append(summarize(spec, image_label, "pytorch", "core", precision, args, threads, core_latencies, samples_per_iteration, output_shape=list(torch_logits.shape), **compile_metadata))
                    store_output("pytorch", "core", precision, len(results) - 1, logits=torch_logits)
                    torch_mask = functional.interpolate(
                        torch_core_single(first["input_tensor"]),
                        first["original_shape"],
                        mode="bilinear",
                    ).argmax(dim=1).detach().cpu().numpy().copy()
                    if not args.skip_all:
                        all_latencies = timed_iterations(torch_all, warmup, iterations)
                        results.append(summarize(spec, image_label, "pytorch", "all", precision, args, threads, all_latencies, samples_per_iteration, output_shape=list(torch_mask.shape), **compile_metadata))
                        store_output("pytorch", "all", precision, len(results) - 1, mask=torch_mask)

            if args.skip_ov:
                continue
            compiled = compiled_core[precision]
            request = compiled.create_infer_request()

            def ov_core_single(single_input_data: Any) -> Any:
                return request.infer({compiled.input(0): single_input_data}, share_inputs=True, share_outputs=True)[compiled.output(0)]

            def ov_core() -> Any:
                last_logits = None
                for item in prepared_inputs:
                    last_logits = ov_core_single(item["input_data"])
                return last_logits

            core_latencies = timed_iterations(ov_core, warmup, iterations)
            ov_logits = ov_core_single(first["input_data"]).copy()
            profiling = request.profiling_info
            amx_nodes = {item.node_name for item in profiling if "amx" in str(item.exec_type).lower()}
            results.append(summarize(
                spec, image_label, "openvino", "core", precision, args, threads, core_latencies, samples_per_iteration,
                effective_performance_hint=ov_property(compiled, "PERFORMANCE_HINT"),
                effective_precision=ov_property(compiled, "INFERENCE_PRECISION_HINT"),
                amx_profiled=bool(amx_nodes), amx_node_count=len(amx_nodes),
                output_shape=list(ov_logits.shape),
            ))
            store_output("openvino", "core", precision, len(results) - 1, logits=ov_logits)

            if not args.skip_all:
                compiled_all_model = compiled_all[precision]
                request_all = compiled_all_model.create_infer_request()

                def ov_all_single(single_input_data: Any, single_full_shape: Any) -> Any:
                    return request_all.infer(
                        {compiled_all_model.input(0): single_input_data, compiled_all_model.input(1): single_full_shape},
                        share_inputs=True, share_outputs=True,
                    )[compiled_all_model.output(0)]

                def ov_all() -> Any:
                    last_mask = None
                    for item in prepared_inputs:
                        last_mask = ov_all_single(item["input_data"], item["full_shape"])
                    return last_mask

                all_latencies = timed_iterations(ov_all, warmup, iterations)
                ov_mask = ov_all_single(first["input_data"], first["full_shape"]).copy()
                profiling_all = request_all.profiling_info
                amx_all = {item.node_name for item in profiling_all if "amx" in str(item.exec_type).lower()}
                results.append(summarize(
                    spec, image_label, "openvino", "all", precision, args, threads, all_latencies, samples_per_iteration,
                    effective_performance_hint=ov_property(compiled_all_model, "PERFORMANCE_HINT"),
                    effective_precision=ov_property(compiled_all_model, "INFERENCE_PRECISION_HINT"),
                    amx_profiled=bool(amx_all), amx_node_count=len(amx_all),
                    output_shape=list(ov_mask.shape),
                ))
                store_output("openvino", "all", precision, len(results) - 1, mask=ov_mask)
        except Exception as error:
            if failures is not None:
                record_failure(failures, f"{spec.variant}/{spec.name}", "pipeline", precision, "benchmark", error)
            continue

    apply_consistency_metrics()
    return results


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    table = [tuple(headers), *[tuple(row) for row in rows]]
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    for index, row in enumerate(table):
        print("  ".join(value.ljust(widths[column]) for column, value in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def print_results(results: Sequence[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        mask = result.get("mask_difference") or {}
        difference = result.get("output_difference") or {}
        rows.append((
            result["variant"], Path(result["image"]).name, result["backend"], result["path"], result["precision"],
            f'{result["mean_ms"]:.3f}', f'{result["throughput_fps"]:.2f}',
            f'{difference.get("normalized_rmse"):.6g}' if difference.get("normalized_rmse") is not None else "-",
            f'{mask.get("pixel_agreement"):.6f}' if mask.get("pixel_agreement") is not None else "-",
        ))
    print_table(("variant", "image", "backend", "path", "compute", "mean ms", "FPS", "NRMSE", "mask agree"), rows)


def print_comparisons(results: Sequence[dict[str, Any]]) -> None:
    indexed = {
        (result["variant"], result["image"], result["precision"], result["path"], result["backend"]): result
        for result in results
    }
    rows = []
    for variant in VARIANTS:
        images = sorted({result["image"] for result in results if result["variant"] == variant})
        for image in images:
            for precision in ("FP32", "BF16", "FP16"):
                for path in ("core", "all"):
                    torch_result = indexed.get((variant, image, precision, path, "pytorch"))
                    ov_result = indexed.get((variant, image, precision, path, "openvino"))
                    if not torch_result or not ov_result:
                        continue
                    output = ov_result.get("output_difference") or {}
                    mask = ov_result.get("mask_difference") or {}
                    rows.append((
                        variant,
                        Path(image).name,
                        precision,
                        path,
                        f'{torch_result["mean_ms"]:.3f}',
                        f'{ov_result["mean_ms"]:.3f}',
                        f'{torch_result["mean_ms"] / ov_result["mean_ms"]:.2f}x',
                        f'{output["normalized_rmse"]:.6g}' if output.get("normalized_rmse") is not None else "-",
                        f'{mask["pixel_agreement"]:.6f}' if mask.get("pixel_agreement") is not None else "-",
                        f'{mask["foreground_iou"]:.6f}' if mask.get("foreground_iou") is not None else "-",
                    ))
    if rows:
        print("\nSame-model PyTorch vs OpenVINO")
        print_table(("variant", "image", "compute", "path", "Torch ms", "OV ms", "speedup", "NRMSE", "mask agree", "fg IoU"), rows)


def save_results(
    results: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    environment: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    result_dir = Path(os.environ.get("UNET_RESULT_DIR", DEFAULT_RESULT_DIR)).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = result_dir / f"benchmark_{timestamp}.json"
    csv_path = result_dir / f"benchmark_{timestamp}.csv"
    failure_csv_path = result_dir / f"benchmark_failures_{timestamp}.csv"
    payload = {
        "environment": environment,
        "arguments": vars(args),
        "results": list(results),
        "failures": list(failures),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    fields = ("model", "variant", "scale", "image", "backend", "path", "precision", "batch_size", "threads", "iterations", "samples_per_iteration", "mean_ms", "p50_ms", "p90_ms", "p99_ms", "throughput_fps", "effective_performance_hint", "effective_precision", "amx_profiled", "amx_node_count")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    failure_fields = ("target", "backend", "precision", "stage", "error_type", "error_message", "traceback")
    with failure_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=failure_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print(f"Failure CSV: {failure_csv_path}")


def record_failure(
    failures: list[dict[str, Any]],
    target: str,
    backend: str,
    precision: str,
    stage: str,
    error: Exception,
) -> None:
    item = {
        "target": target,
        "backend": backend,
        "precision": precision,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    failures.append(item)
    print(
        f"[WARN] {target} {backend} {precision} {stage} failed: "
        f"{item['error_type']}: {item['error_message']}"
    )


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    environment = environment_report(specs, args)
    if args.check_only:
        return
    require_environment(specs, args)

    threads = args.threads or environment["recommended_threads"]
    if not args.skip_pytorch:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    print(f"Using {threads} inference threads")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    iterations = max(1, math.ceil(args.iterations / len(args.images)))
    warmup = max(1, math.ceil(args.warmup / len(args.images)))

    for spec in specs:
        print(f"\n[{spec.variant} / {spec.name}]")
        target = f"{spec.variant}/{spec.name}"
        if not args.skip_pytorch:
            try:
                model, mask_values, setup_seconds = load_model(spec)
                print(f"checkpoint setup={setup_seconds:.3f}s, mask_values={mask_values}")
            except Exception as error:
                record_failure(failures, target, "setup", "N/A", "load_model", error)
                gc.collect()
        else:
            model = None

        compiled_core: dict[str, Any] = {}
        compiled_all: dict[str, Any] = {}
        if not args.skip_ov:
            for precision in args.precisions:
                try:
                    compiled_core[precision] = compile_openvino(spec.xml_path, precision, threads)
                except Exception as error:
                    record_failure(failures, target, "openvino", precision, "compile_core", error)
            if not args.skip_all:
                for precision in args.precisions:
                    try:
                        compiled_all[precision] = compile_openvino(spec.all_xml_path, precision, threads)
                    except Exception as error:
                        record_failure(failures, target, "openvino", precision, "compile_all", error)
        print(f"  [images] count={len(args.images)}")
        results.extend(benchmark_dataset(
            spec, model, args, threads, compiled_core, compiled_all, 
            iterations, warmup, failures,
        ))
        gc.collect()

    if results:
        print_results(results)
        print_comparisons(results)
    if failures:
        print(f"\nFailure summary: {len(failures)} failed task(s)")
    if results or failures:
        save_results(results, failures, environment, args)
    if failures and os.environ.get("UNET_FAIL_ON_ERROR", "1") != "0":
        raise SystemExit(1)


if __name__ == "__main__":
    main()