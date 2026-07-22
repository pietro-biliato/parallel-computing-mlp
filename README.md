# Neural Network Evaluation on GPU

![Course](https://img.shields.io/badge/Course-Modern%20Computing%20for%20Physics-blue)
![Degree](https://img.shields.io/badge/Master-Physics%20of%20Data-orange)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![OpenMP](https://img.shields.io/badge/OpenMP-Supported-blue)

## Project Objective
The goal of this project is to parallelize the forward pass of a simple neural network for a supervised classification problem, using a GPU. Neural network operations (primarily matrix multiplications and element-wise functions) map naturally to parallel architectures, making GPUs critical for achieving high-performance execution.

The chosen architecture is a **3-layer Multi-Layer Perceptron (MLP)** (2 hidden layers + output layer) trained on the **MNIST** handwritten digits dataset.

## Repository Structure

The project follows a progressive optimization approach, starting from a high-level framework down to fine-grained CUDA optimizations.

### 1. Model Preparation
* `build_NN.py`: Python script utilizing **PyTorch** to build, train, and extract the weights for the 3-layer MLP model on the MNIST dataset.

### 2. `serial/` - CPU Baseline
Contains the serial CPU implementations of the forward pass with incremental optimizations.
* `serial_v0`: Baseline, simplest version.
* `serial_v1`: Accumulates partial results using a local `sum` variable mapped to a CPU register.
* `serial_v2`: Initializes `sum` to `b[j]`, saving one addition per neuron per image.
* `serial_v3`: Avoids recomputing constant index expressions to save ALU cycles.
* `serial_v4`: Applies ReLU directly inside the matrix-vector product function.
* **Utils:** 
  * `run_serial.sh`: Automates execution across different batch sizes with `-O0` and `-O3` compiler flags.
  * `MKL.py`: Standard script to estimate the CPU’s 'Peak Sustained Compute' (GFLOPS).
  * `STREAM.c`: Standard benchmark to measure the CPU’s effective memory bandwidth.

### 3. `OpenMP/` - CPU Parallelization
Explores multi-threading on the CPU using OpenMP directives.
* `omp_v2`: Based on `serial_v2`, parallelized with `#pragma omp parallel for collapse(2)`.
* `omp_v2_relu`: Adds parallelization to the ReLU function.
* `omp_v3`: Based on `serial_v3`; reduces parallelism to `collapse(1)` to use pointer arithmetic.
* `omp_v3_relu`: Adds ReLU parallelization to v3.
* `omp_v4`: Applies the `omp_v2` strategy to `serial_v4`.
* `omp_v4_singlefunct`: Implements the forward pass of all 3 layers inside a single function.
* `omp_v5`: Like `omp_v4_singlefunct`, but uses pointer arithmetic (`collapse(1)`).
* **Utils:** `run_omp.sh` for automated testing over various batch sizes and thread counts.

### 4. `OpenACC/` - Directive-based GPU Offloading
Transitioning to the GPU using OpenACC directives.
* `oacc_v1`: Simplest working attempt at GPU parallelization.
* `oacc_v2`: The simplest truly parallel version.
* `oacc_v3`: Extensive use of the `kernels` directive.
* `oacc_v4`: Replaces `kernels` with `parallel loop` for better mapping to this specific context.
* **Profiling:** `report_oacc_vx` files contain NSight Compute reports for performance assessment.

### 5. `CUDA/` - Fine-grained GPU Optimization
Custom kernels exploiting the CUDA programming model.
* `cuda_naive_v1`: The simplest fully functional CUDA version.
* `cuda_naive_v2`: Improves v1 by increasing warp utilization.
* `cuda_opt_v1`: Implements **2D Tiling** and Shared Memory to optimize memory bandwidth.
* `cuda_opt_v2`: Resolves inefficiencies in `opt_v1` by implementing bank-conflict prevention.
* **Profiling:** `report_cuda_vx` files contain detailed NSight Compute reports for kernel analysis.

## How to Run

1. **Train the Model:** Run the Python script to generate weights and biases.
   ```bash
   python build_NN.py
   ```
2. **Run CPU Benchmarks (Serial & OpenMP):**
   ```bash
   cd serial && ./run_serial.sh
   cd ../OpenMP && ./run_omp.sh
   ```
3. **Run GPU Implementations:** Compile using `nvcc` (for CUDA) or an OpenACC-enabled compiler (e.g., NVIDIA HPC SDK) and profile with `ncu` (NSight Compute) as required.
