#!/bin/bash

# =========================
# User configuration
# =========================

WORK_DIR="tuning_roberta_base"                         # Directory where MetaSchedule logs will be stored
OUT_PATH="./roberta_base_number5_repeat5_freq2600.csv"     # Output dataset file
TARGET="llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16"
NUMBER=1 #this is just the initial number not the final one
REPEAT=5
MIN_REPEAT_MS=400
TVM_BUILD="../build" 

# Dataset extraction options (dataset.py의 argument)
START_IDX=0

# Number of tuning trials (tuning.py의 argument)
TOTAL_TRIALS=5361     

#each repeat maximum time
TIMEOUT_SEC=15

# =========================
# Reset tuning logs
# =========================


#####frequency만 바꿀시 여기 comment out ###########

# echo "[INFO] Resetting tuning log directory..."

# if [ -d "$WORK_DIR" ]; then
#     echo "[INFO] Removing existing tuning logs..."
#     rm -rf "$WORK_DIR" 
# fi

# mkdir -p "$WORK_DIR"

# echo "[INFO] Fresh tuning log directory created."

# =========================
# Build TVM if needed
# =========================

#profiling.cc 가 제대로 빌드 되었는지 확인 안되어있으면 다시 빌드

echo "[INFO] Checking TVM build..."

if [ ! -f "$TVM_BUILD/libtvm.so" ]; then #libtvm 없으면 처음부터 빌드하고
    echo "[INFO] libtvm.so not found. Building TVM..."
    mkdir -p $TVM_BUILD
    cd $TVM_BUILD
    cmake ..
    make -j$(nproc)
    cd -
else
    echo "[INFO] Rebuilding TVM (profiling.cc updates)..." #libtvm있으면 최근 업데이트된거만 빌드
    cd $TVM_BUILD
    make -j$(nproc)
    cd -
fi


# =========================
# Logging configuration
# =========================

echo "===================================="
echo "TVM Tuning + Dataset Generation"
echo "===================================="
echo "Work Directory : $WORK_DIR"
echo "Output Dataset : $OUT_PATH"
echo "Total Trials   : $TOTAL_TRIALS"
echo "Target         : $TARGET"
echo "===================================="

# =========================
# Run MetaSchedule tuning 
# =========================

####frequency만 바꿀시 여기 comment out ###########

# echo "[INFO] Running TVM MetaSchedule tuning..."

# python tuning_roberta_base.py \
#     --work-dir "$WORK_DIR" \
#     --target "$TARGET" \
#     --total_trials "$TOTAL_TRIALS"

# =========================
# fix frequency
# =========================

echo "[INFO] Fixing CPU frequency..."
sudo cpupower frequency-set -g performance #governer 설정 PU frequency scaling 정책 설정
sudo cpupower frequency-set -u 2200000
sudo cpupower frequency-set -d 2200000

# disable turbo boost (optional but recommended)
echo "[INFO] Disabling Turbo Boost..."
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

echo "[INFO] Current CPU policy:"
cpupower frequency-info | grep "frequency should be"

echo "[INFO] Runtime CPU frequency (sample):"
cat /proc/cpuinfo | grep "MHz" | head -n 5

echo "[INFO] Current governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# =========================
# Extract dataset
# =========================

echo "[INFO] Extracting dataset from tuning logs..."

######print arguments to make sure everything is working
echo "Number        : $NUMBER"
echo "Repeat        : $REPEAT"
echo "Min Repeat MS : $MIN_REPEAT_MS"
echo "Timeout Sec   : $TIMEOUT_SEC"

python dataset.py \
    --work-dir "$WORK_DIR" \
    --out "$OUT_PATH" \
    --target "$TARGET" \
    --number "$NUMBER" \
    --repeat "$REPEAT" \
    --start-idx "$START_IDX" \
    --min-repeat-ms "$MIN_REPEAT_MS" \
    --timeout-sec "$TIMEOUT_SEC"
  
  

echo "===================================="
echo "[DONE] Dataset saved to:"
echo "$OUT_PATH"
echo "===================================="