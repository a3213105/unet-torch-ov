#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-didi_t213}"
OPENVINO_CONDA_ENV="${OPENVINO_CONDA_ENV:-$CONDA_ENV}"
CPU_LIST="${CPU_LIST:-2-9}"
VERBOSE_LEVEL="${UNET_ONEDNN_VERBOSE:-2}"
case "$VERBOSE_LEVEL" in
    0|1|2) ;;
    *)
        printf 'UNET_ONEDNN_VERBOSE must be 0, 1 or 2.\n' >&2
        exit 2
        ;;
esac
OUTPUT_PREFIX="onednn"
LOG_NAME="onednn_verbose.log"
if [[ "$VERBOSE_LEVEL" == "0" ]]; then
    OUTPUT_PREFIX="performance"
    LOG_NAME="benchmark.log"
fi
OUTPUT_DIR="${UNET_VERBOSE_DIR:-$SCRIPT_DIR/results/${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"

printf 'mode\texit_code\tlog\n' > "$OUTPUT_DIR/status.tsv"
{
    date --iso-8601=seconds
    hostname
    printf 'conda_env=%s\nopenvino_conda_env=%s\ncpu_list=%s\n' "$CONDA_ENV" "$OPENVINO_CONDA_ENV" "$CPU_LIST"
    printf 'onednn_verbose=%s\n' "$VERBOSE_LEVEL"
    sha256sum "$SCRIPT_DIR/unet_benchmark.py" "$SCRIPT_DIR/run_onednn_verbose.sh"
} > "$OUTPUT_DIR/run.txt"

failed=0
for mode in eager compile openvino; do
    mode_dir="$OUTPUT_DIR/$mode"
    mkdir -p "$mode_dir"
    mode_env="$CONDA_ENV"
    if [[ "$mode" == "openvino" ]]; then
        mode_env="$OPENVINO_CONDA_ENV"
    fi
    command=(
        conda run --no-capture-output -n "$mode_env"
        taskset -c "$CPU_LIST"
        env ONEDNN_VERBOSE="$VERBOSE_LEVEL" DNNL_VERBOSE="$VERBOSE_LEVEL" PYTHONUNBUFFERED=1
        UNET_RESULT_DIR="$mode_dir"
        python "$SCRIPT_DIR/unet_benchmark.py"
        --mode "$mode"
        --model-dir "${UNET_MODEL_DIR:-$SCRIPT_DIR/UNet-models}"
        --openvino-dir "${UNET_OPENVINO_DIR:-$SCRIPT_DIR/models_ov}"
        --image-dir "${UNET_IMAGE_DIR:-$SCRIPT_DIR/test_images}"
        "$@"
    )
    printf '[%s] Starting %s; log: %s\n' "$(date --iso-8601=seconds)" "$mode" "$mode_dir/$LOG_NAME"
    printf '%q ' "${command[@]}" >> "$OUTPUT_DIR/run.txt"
    printf '\n' >> "$OUTPUT_DIR/run.txt"
    if "${command[@]}" > "$mode_dir/$LOG_NAME" 2>&1; then
        exit_code=0
    else
        exit_code=$?
        failed=1
    fi
    printf '%s\t%s\t%s\n' "$mode" "$exit_code" "$mode_dir/$LOG_NAME" >> "$OUTPUT_DIR/status.tsv"
    printf '[%s] Finished %s; exit code: %s\n' "$(date --iso-8601=seconds)" "$mode" "$exit_code"
done

printf 'Logs and reports: %s\n' "$OUTPUT_DIR"
exit "$failed"
