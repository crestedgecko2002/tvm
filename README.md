# TVM-Based CPU Power Measurement Dataset Collection

This repository is a customized TVM environment for collecting kernel-level CPU power, frequency, and latency measurements from multiple deep-learning workloads.

The project extends the TVM runtime profiling implementation and provides scripts for:

* model-specific TVM tuning
* CPU warmup evaluation
* automated power-data collection
* repeated experiments across multiple CPU-frequency settings
* CSV dataset generation for downstream analysis and power prediction

---

## 1. Overview

The overall workflow is:

```text
Model-specific TVM tuning
        ↓
Generated tuning logs
        ↓
CPU warmup evaluation
        ↓
Kernel-level power, frequency, and latency measurement
        ↓
CSV dataset generation
        ↓
Power analysis or prediction-model training
```

For convenience, the complete experiment workflow can be executed through a shell script.

---

## 2. Run the Complete Experiment Workflow

The simplest way to execute the configured experiment is:

```bash
cd data_collection
bash run_experiment.sh
```

`run_experiment.sh` is the main entry point for the experiment. It is intended to automate repeated dataset collection across the configured:

* model workloads
* tuning-log directories
* CPU-frequency settings
* measurement parameters
* output CSV paths

Conceptually, the script performs the following sequence:

```text
Set CPU-frequency configuration
        ↓
Select the tuning-log directory
        ↓
Run dataset.py with the configured arguments
        ↓
Save the measured results as a CSV file
        ↓
Repeat for the next frequency or workload
```

Before running the script, inspect its configuration values and update them for the target machine if necessary.

```bash
nano data_collection/run_experiment.sh
```

Typical values that may need adjustment include:

```text
CPU frequencies
model names
tuning-log directories
output file names
TVM target configuration
minimum measurement duration
timeout threshold
```

---

## 3. Repository Structure

The main experiment scripts are located in:

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

| File                     | Description                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `run_experiment.sh`      | Runs the configured power-measurement workflow across multiple workloads and CPU-frequency settings.        |
| `dataset.py`             | Collects kernel-level latency, CPU-frequency, and power-related measurements from existing TVM tuning logs. |
| `tune_quen2.5_3b.py`     | TVM tuning script for Qwen2.5 3B workloads.                                                                 |
| `tuning_bert.py`         | TVM tuning script for BERT workloads.                                                                       |
| `tuning_densenet.py`     | TVM tuning script for DenseNet workloads.                                                                   |
| `tuning_distilbert.py`   | TVM tuning script for DistilBERT workloads.                                                                 |
| `tuning_gpt2.py`         | TVM tuning script for GPT-2 workloads.                                                                      |
| `tuning_llama.py`        | TVM tuning script for LLaMA workloads.                                                                      |
| `tuning_mobilenet.py`    | TVM tuning script for MobileNet workloads.                                                                  |
| `tuning_resnet.py`       | TVM tuning script for ResNet workloads.                                                                     |
| `tuning_roberta_base.py` | TVM tuning script for RoBERTa-base workloads.                                                               |
| `warmup_evaluator.py`    | Evaluates CPU warmup behavior and initial measurement stability.                                            |
| `warmup_evaluator.sh`    | Runs `warmup_evaluator.py` with predefined arguments.                                                       |

Generated tuning logs and CSV datasets may also appear inside `data_collection/` or the corresponding tuning-log directories.

---

## 4. Supported Workloads

The current scripts cover several deep-learning workload categories.

| Category                         | Models                         |
| -------------------------------- | ------------------------------ |
| Large language models            | Qwen2.5 3B, LLaMA, GPT-2       |
| Encoder-based transformer models | BERT, DistilBERT, RoBERTa-base |
| Computer-vision CNN models       | ResNet, DenseNet, MobileNet    |

Using multiple workload categories makes it possible to compare CPU power behavior across different operator distributions, tensor shapes, and computational characteristics.

---

## 5. Detailed Usage

The automated shell script is recommended for repeated experiments. The individual Python scripts can also be run separately when tuning a new model, testing a specific configuration, or resuming an interrupted measurement.

### 5.1 Run Model-Specific TVM Tuning

Each model has a dedicated TVM tuning script.

For example:

```bash
cd data_collection
python tuning_resnet.py
```

Available tuning scripts include:

```bash
python tune_quen2.5_3b.py
python tuning_bert.py
python tuning_densenet.py
python tuning_distilbert.py
python tuning_gpt2.py
python tuning_llama.py
python tuning_mobilenet.py
python tuning_resnet.py
python tuning_roberta_base.py
```

Each script generates tuning logs for its corresponding workload. These logs are later used by `dataset.py` during power-data collection.

The exact internal configuration may differ by model because the workload shapes, model architecture, and tuning directories are model-specific.

---

### 5.2 Run CPU Warmup Evaluation

Before collecting the main dataset, run the warmup evaluator:

```bash
cd data_collection
bash warmup_evaluator.sh
```

The shell script is a convenience wrapper for:

```text
warmup_evaluator.py
```

Its purpose is to simplify execution by passing the required arguments automatically.

The warmup stage helps verify that the CPU operating condition is sufficiently stable before the main power-measurement workflow begins.

---

### 5.3 Run Dataset Collection Manually

The main power-data collection script is:

```text
dataset.py
```

A representative command is:

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

### `dataset.py` Arguments

| Argument          | Description                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--work-dir`      | Directory containing the TVM tuning logs for the target model.                                                            |
| `--out`           | Output path for the generated CSV dataset.                                                                                |
| `--target`        | TVM LLVM target configuration used for compilation and execution.                                                         |
| `--number`        | Initial number of kernel executions grouped into one timing measurement.                                                  |
| `--repeat`        | Number of repeated measurements collected for each kernel configuration.                                                  |
| `--start-idx`     | Index from which dataset collection begins. Useful when resuming an interrupted experiment.                               |
| `--min-repeat-ms` | Minimum measurement duration in milliseconds. Kernel execution may be repeated internally until this duration is reached. |
| `--timeout-sec`   | Timeout threshold for abnormally long workloads.                                                                          |

The arguments should be selected according to the CPU platform, model workload, desired measurement stability, and acceptable experiment duration.

---

## 6. Custom TVM Runtime Modifications

This repository is not an unmodified TVM source tree.

The TVM runtime profiling implementation has been extended in:

```text
src/runtime/profiling.cc
```

The modified runtime includes custom evaluator functions for the power-measurement workflow.

### Added Runtime Evaluators

| Evaluator                                  | Purpose                                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Warmup evaluator                           | Repeatedly executes workloads to evaluate CPU warmup behavior and measurement stability.   |
| Power / energy-latency-frequency evaluator | Measures kernel execution while collecting latency, CPU-frequency, and power-related data. |

The corresponding TVM runtime registry functions are intended to be callable from Python scripts through entries such as:

```text
runtime.profiling.warmup_evaluator
runtime.profiling.energy_latency_freq_evaluator
```

The measurement flow is:

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
Return results to Python
        ↓
Write the dataset to a CSV file
```

Because `profiling.cc` is compiled into TVM, TVM must be rebuilt whenever this file is modified.

---

## 7. Environment Setup

### 7.1 Clone the Repository

Because TVM uses third-party submodules, clone the repository recursively:

```bash
git clone --recursive https://github.com/crestedgecko2002/tvm.git
cd tvm
```

For an existing clone, initialize or update the submodules with:

```bash
git submodule update --init --recursive
```

### 7.2 Activate the Conda Environment

The development environment used for this project is:

```text
tvm19
```

Activate it with:

```bash
conda activate tvm19
```

The required package installation steps may differ depending on the machine configuration.

---

## 8. Build TVM

Create and enter the build directory:

```bash
mkdir -p build
cd build
```

Prepare the TVM CMake configuration:

```bash
cp ../cmake/config.cmake .
```

Configure and build TVM:

```bash
cmake ..
make -j$(nproc)
```

Return to the repository root:

```bash
cd ..
```

Depending on the local Python setup, the TVM Python path may also need to be exported:

```bash
export PYTHONPATH="$PWD/python:${PYTHONPATH}"
```

After modifying `src/runtime/profiling.cc`, rebuild TVM:

```bash
cd build
make -j$(nproc)
cd ..
```

---

## 9. CPU-Frequency Control

For reproducible measurements, the CPU-frequency configuration should be controlled before running the experiment.

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

Example frequency values:

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

## 10. Output CSV Files

Generated CSV files follow a naming pattern similar to:

```text
bert_number5_repeat5_freq2200.csv
bert_number5_repeat5_freq2600.csv
bert_number5_repeat5_freq2900.csv
bert_number5_repeat5_freq3000.csv
bert_number5_repeat5_freq3100.csv
```

The file name records the experiment configuration:

```text
model name
number parameter
repeat parameter
CPU frequency
```

For example:

```text
bert_number5_repeat5_freq3100.csv
```

represents a BERT dataset collected using:

```text
min_number = 5
min_repeat = 5
CPU frequency ≈ 3.1 GHz
```

---

## 11. Recommended Experimental Procedure

For consistent measurements, use the following order:

```text
1. Activate the tvm19 environment
2. Confirm that TVM has been rebuilt after runtime modifications
3. Fix the CPU governor to performance mode
4. Set the desired CPU frequency
5. Verify the CPU-frequency state
6. Run the warmup evaluator and check stable runtime for the CPU
7. Run run_experiment.sh for automated collection
8. Check the generated CSV files
9. Repeat or resume individual cases with dataset.py if necessary
```

---

## 12. Notes on Reproducibility

Power measurements can vary due to:

* CPU temperature
* background processes
* CPU-frequency governor behavior
* system load
* workload duration
* cache state
* measurement interval
* timeout settings
* model-specific operator distributions

For more stable results:

```text
- use a fixed CPU governor
- set the upper and lower CPU-frequency bounds to the same value
- run a warmup phase before dataset collection
- keep the machine workload as consistent as possible
- use the same TVM target configuration across experiments
- inspect timeout cases separately
```

---

## 13. Git Notes

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

Generated CSV files can become large. If they do not need to be committed, add the following rule to `.gitignore`:

```gitignore
data_collection/*.csv
```
