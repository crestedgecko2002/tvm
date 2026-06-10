# TVM-Based CPU Power Measurement and Prediction Dataset Collection

This repository is a customized TVM environment for collecting kernel-level CPU power, frequency, and latency data from multiple deep-learning workloads.

The project extends TVM runtime profiling so that model tuning results can be evaluated under controlled CPU-frequency conditions. The collected measurements are stored as CSV files and can later be used for power analysis or machine-learning-based power prediction.

---

## 1. Project Overview

The overall workflow is:

```text
Model-specific TVM tuning
        ↓
Generated tuning logs
        ↓
Warmup stability evaluation
        ↓
Kernel-level power, frequency, and latency measurement
        ↓
CSV dataset generation
        ↓
Power analysis or prediction model training
```

The repository includes:

* model-specific TVM tuning scripts
* a dataset-collection script
* a shell script for running the full experiment workflow
* warmup-evaluation scripts
* custom runtime-level profiling functions added to TVM
* generated CSV measurement results

---

## 2. Repository Structure

The main project files are located in:

```text
data_collection/
```

Current structure:

```text
data_collection/
├── dataset.py
├── run_experiment.sh
├── tune_quen2.5_3b.py
├── tuning_bert.py
├── tuning_densenet.py
├── tuning_distilbert.py
├── tuning_gpt2.py
├── tuning_llama.py
├── tuning_mobilenet.py
├── tuning_resnet.py
├── tuning_roberta_base.py
├── warmup_evaluator.py
└── warmup_evaluator.sh
```

### File Roles

| File                     | Description                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset.py`             | Measures kernel-level power, CPU frequency, and latency using previously generated TVM tuning logs. Results are written to CSV files. |
| `run_experiment.sh`      | Runs the complete measurement workflow across configured frequencies and workloads.                                                   |
| `tune_quen2.5_3b.py`     | TVM tuning script for Qwen2.5 3B workloads.                                                                                           |
| `tuning_bert.py`         | TVM tuning script for BERT workloads.                                                                                                 |
| `tuning_densenet.py`     | TVM tuning script for DenseNet workloads.                                                                                             |
| `tuning_distilbert.py`   | TVM tuning script for DistilBERT workloads.                                                                                           |
| `tuning_gpt2.py`         | TVM tuning script for GPT-2 workloads.                                                                                                |
| `tuning_llama.py`        | TVM tuning script for LLaMA workloads.                                                                                                |
| `tuning_mobilenet.py`    | TVM tuning script for MobileNet workloads.                                                                                            |
| `tuning_resnet.py`       | TVM tuning script for ResNet workloads.                                                                                               |
| `tuning_roberta_base.py` | TVM tuning script for RoBERTa-base workloads.                                                                                         |
| `warmup_evaluator.py`    | Checks CPU warmup behavior and initial measurement stability before the main experiment.                                              |
| `warmup_evaluator.sh`    | Convenience wrapper for running `warmup_evaluator.py` with predefined arguments.                                                      |

Generated tuning logs and CSV measurement files may also appear inside `data_collection/` or related tuning-log directories.

---

## 3. Custom TVM Runtime Modifications

This repository is not an unmodified TVM source tree.

The runtime profiling implementation has been extended in:

```text
src/runtime/profiling.cc
```

The custom implementation adds runtime-level evaluators used by the measurement scripts.

### Added Evaluators

| Evaluator                                  | Purpose                                                                                                                        |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Warmup evaluator                           | Runs workloads repeatedly before the main experiment to verify that the CPU reaches a sufficiently stable operating condition. |
| Power / energy-latency-frequency evaluator | Measures workload execution while collecting latency, CPU frequency, and power-related data.                                   |

The relevant runtime registry entries are intended to expose functions such as:

```text
runtime.profiling.warmup_evaluator
runtime.profiling.energy_latency_freq_evaluator
```

These functions can be called from Python through TVM's runtime registry.

### Why Runtime-Level Modifications Are Needed

Standard timing evaluation measures execution latency, but this project also needs power-related measurements under controlled CPU conditions.

The custom runtime flow is conceptually:

```text
Python experiment script
        ↓
TVM runtime registry call
        ↓
Custom evaluator in profiling.cc
        ↓
Repeated kernel execution
        ↓
Latency, CPU-frequency, and power-related measurement
        ↓
Return measurement values to Python
        ↓
Write results to CSV
```

Because `profiling.cc` is compiled into TVM, TVM must be rebuilt whenever this file is modified.

---

## 4. Supported Workload Categories

The current scripts cover multiple model families.

| Category                         | Models                         |
| -------------------------------- | ------------------------------ |
| Large language models            | Qwen2.5 3B, LLaMA, GPT-2       |
| Encoder-based transformer models | BERT, DistilBERT, RoBERTa-base |
| Computer-vision CNN models       | ResNet, DenseNet, MobileNet    |

This makes it possible to compare power behavior across workloads with different operator distributions, tensor shapes, and computational characteristics.

---

## 5. Environment Setup

### Clone the Repository

Because TVM uses third-party submodules, clone the repository recursively:

```bash
git clone --recursive https://github.com/crestedgecko2002/tvm.git
cd tvm
```

For an existing clone, initialize or update the submodules with:

```bash
git submodule update --init --recursive
```

### Conda Environment

The development environment used for this project is:

```text
tvm19
```

Activate it with:

```bash
conda activate tvm19
```

The exact package installation steps may differ depending on the machine configuration.

---

## 6. Build TVM

Create and enter the build directory:

```bash
mkdir -p build
cd build
```

Prepare the TVM CMake configuration:

```bash
cp ../cmake/config.cmake .
```

Configure and build:

```bash
cmake ..
make -j$(nproc)
```

Return to the repository root:

```bash
cd ..
```

If `src/runtime/profiling.cc` is modified, rebuild TVM before running the Python measurement scripts:

```bash
cd build
make -j$(nproc)
cd ..
```

Depending on the local Python setup, the TVM Python path may also need to be exported:

```bash
export PYTHONPATH="$PWD/python:${PYTHONPATH}"
```

---

## 7. CPU Frequency Control

For reproducible power measurements, the CPU frequency should be controlled before each experiment.

A typical setup is:

```bash
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 3100000
sudo cpupower frequency-set -d 3100000
```

This example fixes the CPU frequency near:

```text
3.1 GHz
```

The frequency value should be adjusted for each experiment.

Examples:

```text
2200000 → approximately 2.2 GHz
2600000 → approximately 2.6 GHz
2900000 → approximately 2.9 GHz
3000000 → approximately 3.0 GHz
3100000 → approximately 3.1 GHz
```

Check the current CPU-frequency configuration with:

```bash
cpupower frequency-info
```

---

## 8. Run Model Tuning

Each model has its own TVM tuning script.

Example:

```bash
cd data_collection
python tuning_resnet.py
```

Other examples:

```bash
python tuning_bert.py
python tuning_densenet.py
python tuning_distilbert.py
python tuning_gpt2.py
python tuning_llama.py
python tuning_mobilenet.py
python tuning_roberta_base.py
python tune_quen2.5_3b.py
```

Each tuning script generates tuning logs for its corresponding workload.

The generated log directory should be checked before running the dataset-collection stage.

---

## 9. Run Warmup Evaluation

Before the main dataset collection, run the warmup evaluator:

```bash
cd data_collection
bash warmup_evaluator.sh
```

The shell script acts as a wrapper around:

```text
warmup_evaluator.py
```

and provides the necessary arguments in a convenient form.

The warmup stage is used to inspect whether the CPU operating condition has stabilized sufficiently before power measurements begin.

---

## 10. Collect Power Measurement Data

The main dataset-generation script is:

```text
dataset.py
```

A representative execution format is:

```bash
python dataset.py \
  --work-dir tuning_logs_resnet \
  --out ./resnet_number5_repeat5_freq3100.csv \
  --target "llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512 -num-cores=16" \
  --number 1 \
  --repeat 5 \
  --start-idx 0 \
  --min-repeat-ms 400 \
  --timeout-sec 15
```

### Common Arguments

| Argument          | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `--work-dir`      | Directory containing TVM tuning logs.                   |
| `--out`           | Output CSV file path.                                   |
| `--target`        | TVM LLVM target configuration.                          |
| `--number`        | Number of function calls grouped into one measurement.  |
| `--repeat`        | Number of repeated measurements.                        |
| `--start-idx`     | Starting index in the tuning-log workload list.         |
| `--min-repeat-ms` | Minimum execution duration used for stable measurement. |
| `--timeout-sec`   | Timeout threshold for abnormally long workloads.        |

The exact arguments may be adjusted depending on the model, workload size, measurement duration, and CPU configuration.

---

## 11. Run the Full Experiment Workflow

To run the configured end-to-end experiment workflow:

```bash
cd data_collection
bash run_experiment.sh
```

This script is intended to automate repeated experiments across the selected:

* models
* CPU frequencies
* tuning-log directories
* measurement settings
* output CSV paths

Before running it, inspect and update the script variables to match the local environment.

---

## 12. Output CSV Files

Generated CSV files follow a naming pattern similar to:

```text
bert_number5_repeat5_freq2200.csv
bert_number5_repeat5_freq2600.csv
bert_number5_repeat5_freq2900.csv
bert_number5_repeat5_freq3000.csv
bert_number5_repeat5_freq3100.csv
```

The name encodes:

```text
model name
measurement number
repeat count
CPU frequency
```

For example:

```text
bert_number5_repeat5_freq3100.csv
```

indicates a BERT measurement dataset collected with:

```text
number = 5
repeat = 5
frequency ≈ 3.1 GHz
```

The CSV contents are intended for downstream analysis and power-prediction experiments.

---

## 13. Recommended Experimental Procedure

For consistent measurements, use the following order:

```text
1. Activate the tvm19 environment
2. Confirm that TVM has been rebuilt after runtime modifications
3. Fix the CPU governor to performance mode
4. Set the desired CPU frequency
5. Verify the current CPU-frequency state
6. Run the warmup evaluator
7. Run model-specific tuning if tuning logs do not already exist
8. Run dataset.py or run_experiment.sh
9. Check the generated CSV file
10. Repeat for each target frequency and model
```

---

## 14. Notes on Reproducibility

Power measurements can vary due to:

* CPU temperature
* background processes
* frequency-governor behavior
* system load
* workload duration
* cache state
* measurement interval
* timeout settings
* model-specific operator distributions

For more stable results:

```text
- use a fixed CPU governor
- set upper and lower frequency bounds to the same value
- run a warmup phase before data collection
- keep the machine workload as consistent as possible
- use the same target configuration across experiments
- inspect timeout cases separately
```

---

## 15. Git Notes

This repository includes TVM third-party submodules under:

```text
3rdparty/
```

If submodule directories appear modified after cloning or building, inspect them separately:

```bash
git -C 3rdparty/cutlass status --short
git -C 3rdparty/cutlass_fpA_intB_gemm status --short
git -C 3rdparty/libflash_attn status --short
```

Generated CSV files can become large. If they do not need to be committed, add an ignore rule:

```gitignore
data_collection/*.csv
```

---

## 16. Summary

This repository extends TVM runtime profiling for CPU power-measurement experiments.

The main components are:

```text
src/runtime/profiling.cc
        → custom runtime-level measurement support

data_collection/tuning_*.py
        → model-specific TVM tuning

data_collection/warmup_evaluator.py
        → warmup and stability evaluation

data_collection/dataset.py
        → kernel-level power, frequency, and latency collection

data_collection/run_experiment.sh
        → automated experiment execution

CSV outputs
        → datasets for power analysis and prediction
```

The project is designed to collect reproducible CPU power datasets across multiple deep-learning workload categories and CPU-frequency conditions.
