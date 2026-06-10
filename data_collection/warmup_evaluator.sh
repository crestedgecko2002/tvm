#!/bin/bash

# =========================
# Path configuration
# =========================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# First argument: tuning log directory
WORK_DIR=$1

# Second argument: output CSV file
OUT_FILE=$2

# Defaults
if [ -z "$WORK_DIR" ]; then
  WORK_DIR="${SCRIPT_DIR}/tuning_logs_quen"
fi

if [ -z "$OUT_FILE" ]; then
  OUT_FILE="${SCRIPT_DIR}/warmup_power_debug.csv"
fi

# TVM build path
TVM_BUILD="${SCRIPT_DIR}/../build"

# =========================
# Warmup debug configuration
# =========================
# New WarmupEvaluator meaning:
#
# total observed kernel calls = NUMBER * REPEAT
#
# Example:
# NUMBER=1, REPEAT=20
# -> observe first 20 kernel calls one by one
#
# NUMBER=5, REPEAT=5
# -> observe 25 kernel calls, grouped as 5 repeats x 5 calls
#
# MIN_REPEAT_MS is kept only because the Python/C++ signature still has it.
# The new per-kernel-call debug WarmupEvaluator does not really use it.

TARGET="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16"

NUMBER=1
REPEAT=20
MIN_REPEAT_MS=0
START_IDX=0
TIMEOUT_SEC=160

CPU_FREQ_KHZ=3100000

# =========================
# CPU frequency setting
# =========================

echo "[INFO] Fixing CPU frequency..."
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u ${CPU_FREQ_KHZ}
sudo cpupower frequency-set -d ${CPU_FREQ_KHZ}

echo "[INFO] Disabling Turbo Boost..."
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

echo "[INFO] Current CPU policy:"
cpupower frequency-info | grep "frequency should be" || true

echo "[INFO] Runtime CPU frequency sample:"
cat /proc/cpuinfo | grep "MHz" | head -n 5 || true

echo "[INFO] Current governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true

# =========================
# Print configuration
# =========================

echo "================================="
echo "Warmup Power Debug Evaluation"
echo "================================="
echo "Script dir    : ${SCRIPT_DIR}"
echo "Work dir      : ${WORK_DIR}"
echo "Output file   : ${OUT_FILE}"
echo "TVM build     : ${TVM_BUILD}"
echo "Target        : ${TARGET}"
echo "Number        : ${NUMBER}"
echo "Repeat        : ${REPEAT}"
echo "Total calls   : $((NUMBER * REPEAT))"
echo "Min repeat ms : ${MIN_REPEAT_MS}"
echo "Start idx     : ${START_IDX}"
echo "Timeout sec   : ${TIMEOUT_SEC}"
echo "CPU freq kHz  : ${CPU_FREQ_KHZ}"
echo "================================="

# =========================
# Check TVM build
# =========================

echo "[INFO] Checking TVM build..."

if [ ! -f "${TVM_BUILD}/libtvm.so" ]; then
    echo "[INFO] libtvm.so not found. Building TVM..."
    mkdir -p "${TVM_BUILD}"
    cd "${TVM_BUILD}" || exit 1
    cmake ..
    make -j"$(nproc)"
    cd - >/dev/null || exit 1
else
    echo "[INFO] Rebuilding TVM because profiling.cc may have changed..."
    cd "${TVM_BUILD}" || exit 1
    make -j"$(nproc)"
    cd - >/dev/null || exit 1
fi

# =========================
# Run warmup evaluator
# =========================

echo "================================="
echo "Running warmup evaluator"
echo "================================="

python "${SCRIPT_DIR}/warmup_evaluator.py" \
  --work-dir "$WORK_DIR" \
  --out "$OUT_FILE" \
  --target "$TARGET" \
  --number "$NUMBER" \
  --repeat "$REPEAT" \
  --min-repeat-ms "$MIN_REPEAT_MS" \
  --start-idx "$START_IDX" \
  --timeout-sec "$TIMEOUT_SEC"

echo ""
echo "================================="
echo "All done."
echo "Output saved to:"
echo "$OUT_FILE"
echo "================================="