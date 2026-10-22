# U-Net BF16: PyTorch 2.14.1 vs. OpenVINO oneDNN Analysis

## Summary

The main gap between Torch eager and OpenVINO is **not missing AMX support**. All three modes use AMX BF16 convolution kernels. Eager spends substantial time materializing contiguous tensors and moving data between layouts, and executes batch normalization, ReLU, and max pooling separately.

Torch compile removes much of this overhead and fuses ReLU into convolutions. OpenVINO retains a smaller advantage over compile through fewer steady-state reorders, lower convolution/deconvolution times in this capture, and less work outside the logged oneDNN primitives.

The layout explanation is supported by a controlled experiment: changing only eager's model/input memory format to channels-last reduces average BF16 time from **103.515 to 40.413 ms/image** at scale 0.5 and from **449.005 to 148.135 ms/image** at scale 1.0. This experiment does not modify the production benchmark.

## Configuration and Measurement Boundaries

- Host: gnr-ap02, Intel Xeon 6952P.
- Shared environment: `ov2630`, Torch **2.14.1+cpu**, OpenVINO **2026.3.0**.
- CPU affinity: 2-9; 8 inference threads; batch size 1.
- Input: the same 10 ADAS images, both scales, BF16 only.
- Capture: `ONEDNN_VERBOSE=2`, `DNNL_VERBOSE=2`.
- For each mode/scale: 3 full-dataset warmup passes, then 10 full-dataset timed passes, totaling 100 measured images.
- The capture wrapper adds `TRACE_TIMED_BEGIN` / `TRACE_TIMED_END` markers around formal timing. Primitive statistics include only `primitive,exec` records between these markers. Compilation, primitive creation, warmup, and output checks are excluded.
- Clean performance numbers below come from the existing matching Torch 2.14.1/OpenVINO run with verbose disabled. Verbose primitive timings and separate CPU-profiler measurements are diagnostic evidence, not replacements for clean latency.
- All six BF16 capture results succeeded. Both compile results report `inductor`, with no eager fallback.

## Clean BF16 Performance

Average time per image, in milliseconds:

| Scale | Torch eager | Torch compile | OpenVINO | Compile / eager speedup | OV / eager speedup | OV / compile speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 103.515 | 31.424 | 25.756 | 3.294x | 4.019x | 1.220x |
| 1.0 | 449.005 | 111.807 | 97.938 | 4.016x | 4.585x | 1.142x |

Each speedup is reference latency divided by optimized-mode latency. Times exclude preprocessing, loading, and initial compilation.

## What the oneDNN Logs Show

### 1. All modes already use AMX

Each mode executes **19 convolutions and 4 transposed convolutions per image**. In each scale's measured region, all **2,300 convolution/deconvolution executions** use AMX implementations:

```text
brg_conv_fwd:avx10_1_512_amx
brgconv_1x1:avx10_1_512_amx
brg_deconv:avx10_1_512_amx+brgconv_strided:avx10_1_512_amx
```

Consequently, the performance difference cannot be explained as "Torch does not use AMX" or "Torch runs these convolutions in FP32."

The libraries are not identical despite sharing one Conda environment:

| Backend | Embedded oneDNN | CPU threading runtime |
| --- | --- | --- |
| Torch eager / compile | 3.12.0 | OpenMP, 8 threads |
| OpenVINO | 3.10.2 | Threadpool, 8 threads |

These differences may affect scheduling and kernel timing, but this experiment does not isolate their causal contribution.

### 2. Eager has many more reorders; compile still repacks weights

Steady-state calls per image:

| Primitive | Eager | Compile | OpenVINO |
| --- | ---: | ---: | ---: |
| Convolution | 19 | 19 | 19 |
| Deconvolution | 4 | 4 | 4 |
| Reorder | 69 | 23 | 10 |
| Pooling visible in oneDNN | 0 | 0 | 4 |

The absence of pooling in Torch's oneDNN log does **not** mean pooling is absent from the model. Eager uses native ATen pooling; compile handles it outside these logged oneDNN calls.

At scale 1.0:

- Eager has 23 BF16 `abcd -> acdb` activation reorders per image, totaling **24.924 ms/image**. The tags represent NCHW and NHWC layouts. A further 23 `acdb -> acdb` reorders total **15.653 ms/image**; these are copies, not changes of the displayed layout tag. The remaining 23 reorders pack convolution weights.
- Compile's 23 reorders pack weights into blocked formats such as `AcdB16b32a2b`, totaling **2.474 ms/image**. Thus, weight-packing reorders still occur during formal execution in this specific compiled path, even after warmup.
- OpenVINO has one FP32 NCHW-to-BF16 NHWC input reorder, one BF16 NHWC-to-FP32 NCHW output reorder, and eight small `s32` reorders per image. Together they total **0.241 ms/image**. No steady-state weight-packing reorder appears in the captured region.

Reorder counts include copies, layout changes, dtype conversion, and weight packing; they must not all be described as BF16 casts.

### 3. Compile and OpenVINO fuse convolution post-operations

For each image:

- Eager has no convolution `eltwise_relu` post-operation in these logs.
- Compile and OpenVINO each have **18 convolutions with `attr-post-ops:eltwise_relu`**.
- Their convolution descriptors include bias, whereas eager's batch-normalized convolutions have no bias.
- The CPU profiler sees 18 separate `aten::native_batch_norm` and 18 `aten::clamp_min_` calls per image in eager, but neither appears as a separate ATen operation in compile.

This supports a transformed/fused inference path, consistent with batch-normalization folding and convolution-plus-ReLU fusion. It does not establish an exact standalone timing for every compiler-generated fused operation.

### 4. Primitive execution-time breakdown

Sum of measured oneDNN execution times divided by 100 measured images, in **ms/image**:

| Scale | Mode | Convolution | Deconvolution | Reorder | Pooling | oneDNN sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | Eager | 20.551 | 2.864 | 11.028 | Not logged | 34.443 |
| 0.5 | Compile | 21.362 | 2.903 | 2.346 | Not logged | 26.612 |
| 0.5 | OpenVINO | 20.504 | 2.465 | 0.083 | 0.306 | 23.357 |
| 1.0 | Eager | 82.801 | 10.658 | 42.765 | Not logged | 136.224 |
| 1.0 | Compile | 84.449 | 10.787 | 2.474 | Not logged | 97.709 |
| 1.0 | OpenVINO | 79.720 | 8.297 | 0.241 | 1.258 | 89.516 |

Eager's convolution kernel time is not dramatically worse than compile's; most of eager's wall-clock gap is elsewhere. At scale 1.0, compile versus OpenVINO differs by approximately **4.729 ms/image in convolution**, **2.490 ms/image in deconvolution**, and **2.233 ms/image in reorders**. OpenVINO additionally logs 1.258 ms/image of pooling, while Torch's pooling is outside these oneDNN totals.

These are diagnostic primitive-time sums, not a complete additive partition of clean wall time. Framework operations, generated kernels, scheduling, allocation, and logging overhead are not fully represented; OpenVINO graph execution may also overlap work.

## CPU Profiler: Why Eager Is Much Slower

A separate run disables verbose, warms up the models, and profiles one complete 10-image pass. The following values are averaged per image:

| Eager operation | Scale 0.5 (ms) | Scale 1.0 (ms) | Timing type |
| --- | ---: | ---: | --- |
| `aten::copy_` | 50.872 | 227.148 | Self CPU |
| `aten::contiguous` | 48.226 | 222.105 | Inclusive CPU |
| `aten::_to_copy` | 2.480 | 2.768 | Inclusive CPU |
| `aten::max_pool2d_with_indices` | 8.367 | 32.039 | Self CPU |
| `aten::native_batch_norm` | 3.254 | 17.610 | Self CPU |
| `aten::clamp_min_` | 2.644 | 15.061 | Self CPU |
| `aten::cat` | 1.299 | 6.562 | Self CPU |

**Do not add these rows together:** inclusive `contiguous` and `_to_copy` timings contain nested copy operations already represented by `copy_`.

The profiler reports **23 `contiguous` calls per image**. Their inclusive cost dominates dtype-conversion cost: at scale 1.0, about **222.105 ms/image** versus **2.768 ms/image** for `_to_copy`. This is evidence for expensive contiguous/layout materialization, rather than BF16 casting alone, as the largest eager bottleneck.

Compile uses `mkldnn::_convolution_pointwise` 19 times per image. Separate native BN, ReLU, and eager copy operations disappear from the profiled operator list. Compiler-generated work remains inside compiled-graph regions and cannot be decomposed accurately into pooling, concatenation, padding, and dispatch costs from this operator summary alone.

## Controlled Check: Channels-Last Eager

An isolated wrapper converts model weights to channels-last and converts each input to channels-last inside `forward`. Input-conversion cost remains included in measured inference. The model, weights, BF16 autocast, images, CPU binding, and warmup/timing counts are otherwise unchanged.

| Scale | Original eager (ms/image) | Channels-last eager (ms/image) | Speedup |
| --- | ---: | ---: | ---: |
| 0.5 | 103.515 | 40.413 | 2.56x |
| 1.0 | 449.005 | 148.135 | 3.03x |

The first image at each scale produced identical BF16 logits before and after the layout change (`max_abs=0`, `NRMSE=0`). This is a sampled correctness check, not a complete accuracy evaluation across the dataset.

The ablation strongly supports the layout/materialization explanation. Remaining eager overhead is consistent with unfused operations, native pooling, packing, and framework execution. This is an experimental result, not a change to the benchmark's default eager mode.

## Why OpenVINO Still Leads Compile

The clean gap is **5.667 ms/image** at scale 0.5 and **13.870 ms/image** at scale 1.0.

Evidence supports three contributors:

1. **Less steady-state packing/copy work.** Compile still executes 23 weight reorders per image; OpenVINO's measured region contains only boundary and small integer reorders.
2. **Lower observed convolution/deconvolution cost.** Both use AMX, but their library versions, threading runtimes, descriptors, and execution contexts differ. These logs establish the timing difference, not which individual implementation choice causes it.
3. **Graph-level work outside oneDNN.** At scale 1.0, verbose wall time is 113.910 ms/image for compile and 98.550 ms/image for OpenVINO; their oneDNN sums are 97.709 and 89.516 ms/image. The remaining differences are consistent with generated/native operations and runtime overhead, but are not isolated kernel-level measurements.

It would be unsupported to attribute the whole remaining gap to AMX, BF16 support, Python dispatch, or the oneDNN version alone.

## Artifacts

- Raw oneDNN logs: [Torch eager](results/bf16_analysis_20261021_145529/eager/onednn_verbose.log), [Torch compile](results/bf16_analysis_20261021_145529/compile/onednn_verbose.log), [OpenVINO](results/bf16_analysis_20261021_145529/openvino/onednn_verbose.log).
- [Steady-state primitive summary CSV](results/bf16_analysis_20261021_145529/primitive_summary.csv).
- [Trace summary JSON, including implementations, descriptors, and source line numbers](results/bf16_analysis_20261021_145529/trace_summary.json).
- [CPU profiler and channels-last attribution summary](results/bf16_analysis_20261021_145529/cpu_attribution_summary.json).
- CPU-profiler samples: [eager scale 0.5](results/bf16_analysis_20261021_145529/profiler/eager/profile_scale0.5.json), [eager scale 1.0](results/bf16_analysis_20261021_145529/profiler/eager/profile_scale1.0.json), [compile scale 0.5](results/bf16_analysis_20261021_145529/profiler/compile/profile_scale0.5.json), [compile scale 1.0](results/bf16_analysis_20261021_145529/profiler/compile/profile_scale1.0.json).
- [Channels-last ablation log and sampled output comparison](results/bf16_analysis_20261021_145529/channels_last/benchmark.log).
- [Clean performance CSV used as the latency baseline](results/torch2141_20261021_142106/performance_summary.csv).
- [Capture commands and source checksums](results/bf16_analysis_20261021_145529/run.txt).
- [Trace capture wrapper](results/bf16_analysis_20261021_145529/capture_unet_trace.py) and [analysis script](results/bf16_analysis_20261021_145529/analyze_traces.py).
