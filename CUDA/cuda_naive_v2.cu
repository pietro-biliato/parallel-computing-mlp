#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <cuda_runtime.h>

// ==========================================
// MACROS AND HELPER FUNCTIONS
// ==========================================

// Macro for CUDA error checking
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Read binary files
template <typename T>
std::vector<T> read_binary_file(const std::string& filename, size_t expected_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);
    std::vector<T> data(expected_elements);
    file.read(reinterpret_cast<char*>(data.data()), expected_elements * sizeof(T));
    if (!file) throw std::runtime_error("Error reading file: " + filename);
    return data;
}

// Argmax for predictions
std::vector<int> argmax(const std::vector<float>& logits, int B, int O) {
    std::vector<int> predictions(B, 0);
    for (int i = 0; i < B; ++i) {
        float max_val = logits[i * O];
        int best_class = 0;
        for (int j = 1; j < O; ++j) {
            if (logits[i * O + j] > max_val) {
                max_val = logits[i * O + j];
                best_class = j;
            }
        }
        predictions[i] = best_class;
    }
    return predictions;
}

// ==========================================
// CUDA KERNEL
// ==========================================

 //Kernel naive
__global__ void linear_forward_naive(const float* X, const float* W, const float* b, float* Y, 
                                     int B, int I, int O, bool apply_relu) {
    
    // Calculation of global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output neuron index

    // Prevent out-of-bounds threads from reading/writing unallocated memory
    if (row < B && col < O) {
        
        // Initialize the accumulator directly with the bias.
        // This saves a subsequent addition and a separate memory access
        float sum = b[col];

        // Dot product
        for (int k = 0; k < I; ++k) {
            sum += X[row * I + k] * W[col * I + k];
        }

        // Apply ReLU directly in the thread registers
        // Avoid writing to global memory and then re-reading in a separate kernel
        if (apply_relu && sum < 0.0f) {
            sum = 0.0f;
        }

        // Single write to global memory
        Y[row * O + col] = sum;
    }
}

// Function to dynamically compute the launch configuration
void compute_launch_config(int B, int O, dim3& blockSize, dim3& gridSize) {
    const int WARP_SIZE = 32;
    const int NUM_SM = 16; // Number of SMs on my GTX 1650 Max‑Q

    // 1. X‑AXIS OPTIMIZATION (warp alignment): block_x forced to be a multiple of 32 
    // --> each warp always works on a single row of the batch (Y)
    int block_x = WARP_SIZE; 
    
    // 2. Y‑AXIS OPTIMIZATION (occcupancy): maximizing threads per block 
    // (32×32 = 1024 threads, the hardware limit)
    int block_y = 32; 

    // Preliminary grid calculation
    int grid_x = (O + block_x - 1) / block_x;
    int grid_y = (B + block_y - 1) / block_y;
    int total_blocks = grid_x * grid_y;

    // 3. EDGE CASE HANDLING (SM underutilization)
    // If the total number of blocks is smaller than the number of SMs (e.g., Layer 3),
    // reduce block_y, to “spread” the batch rows across more blocks, forcing all SMs to be active
    if (total_blocks < NUM_SM) {
        block_y = B / NUM_SM; 
        if (block_y == 0) block_y = 1; // Safety check for very small batches
    }

    blockSize = dim3(block_x, block_y);
    gridSize = dim3((O + blockSize.x - 1) / blockSize.x,
                    (B + blockSize.y - 1) / blockSize.y);
}


int main() {

    // ==========================================
    // INITIALIZATION
    // ==========================================
    const std::string dir = "./export/";

    // Architecture Dimensions
    const int B = 1000;
    const int I1 = 784, O1 = 256;
    const int I2 = 256, O2 = 128;
    const int I3 = 128, O3 = 10;

    std::cout << "-> Loading weights and inputs into Host memory (CPU)..." << std::endl;

    auto h_W1 = read_binary_file<float>(dir + "fc1_weight.bin", O1 * I1);
    auto h_b1 = read_binary_file<float>(dir + "fc1_bias.bin", O1);

    auto h_W2 = read_binary_file<float>(dir + "fc2_weight.bin", O2 * I2);
    auto h_b2 = read_binary_file<float>(dir + "fc2_bias.bin", O2);

    auto h_W3 = read_binary_file<float>(dir + "fc3_weight.bin", O3 * I3);
    auto h_b3 = read_binary_file<float>(dir + "fc3_bias.bin", O3);

    auto h_X  = read_binary_file<float>(dir + "test_inputs.bin", B * I1);
    
    auto expected_logits = read_binary_file<float>(dir + "test_logits.bin", B * O3);
    auto expected_preds  = read_binary_file<int>(dir + "test_predictions.bin", B);

    std::vector<float> h_Y(B * O3, 0.0f); // Vector to download the results

    // ==========================================
    // DEVICE MEMORY ALLOCATION (GPU)
    // ==========================================
    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    float *d_H1, *d_H2, *d_Y;

    checkCuda(cudaMalloc((void**)&d_X,  B * I1 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_W1, O1 * I1 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b1, O1 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_H1, B * O1 * sizeof(float))); // output layer 1

    checkCuda(cudaMalloc((void**)&d_W2, O2 * I2 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b2, O2 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_H2, B * O2 * sizeof(float))); // output layer 2

    checkCuda(cudaMalloc((void**)&d_W3, O3 * I3 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b3, O3 * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_Y,  B * O3 * sizeof(float))); // logits

    // ==========================================
    // TIMERS SETUP
    // ==========================================
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);

    // ==========================================
    // HOST-TO-DEVICE TRANSFER
    // ==========================================
    std::cout << "-> Transferring data to GPU (H2D)..." << std::endl;
    cudaEventRecord(start_h2d);

    checkCuda(cudaMemcpy(d_X,  h_X.data(),  B * I1 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_W1, h_W1.data(), O1 * I1 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b1, h_b1.data(), O1 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_W2, h_W2.data(), O2 * I2 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b2, h_b2.data(), O2 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_W3, h_W3.data(), O3 * I3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b3, h_b3.data(), O3 * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    // ==========================================
    // GRID AND BLOCK CONFIGURATION
    // ==========================================
    dim3 blockSize1, gridSize1;
    compute_launch_config(B, O1, blockSize1, gridSize1);

    dim3 blockSize2, gridSize2;
    compute_launch_config(B, O2, blockSize2, gridSize2);

    dim3 blockSize3, gridSize3;
    compute_launch_config(B, O3, blockSize3, gridSize3);

    // ==========================================
    // KERNEL EXECUTION (FORWARD PASS)
    // ==========================================
    std::cout << "-> Executing Forward Pass on GPU..." << std::endl;
    cudaEventRecord(start_kernel);

    linear_forward_naive<<<gridSize1, blockSize1>>>(d_X, d_W1, d_b1, d_H1, B, I1, O1, true);
    checkCuda(cudaGetLastError());

    linear_forward_naive<<<gridSize2, blockSize2>>>(d_H1, d_W2, d_b2, d_H2, B, I2, O2, true);
    checkCuda(cudaGetLastError());

    linear_forward_naive<<<gridSize3, blockSize3>>>(d_H2, d_W3, d_b3, d_Y, B, I3, O3, false);
    checkCuda(cudaGetLastError());

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // ==========================================
    // DEVICE-TO-HOST TRANSFER
    // ==========================================
    std::cout << "-> Downloading results (D2H)..." << std::endl;
    cudaEventRecord(start_d2h);

    checkCuda(cudaMemcpy(h_Y.data(), d_Y, B * O3 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    // ==========================================
    // CALCULATION OF TIMES AND METRICS
    // ==========================================
    float time_h2d = 0, time_kernel = 0, time_d2h = 0;
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // Validation
    auto predictions = argmax(h_Y, B, O3);
    int correct_preds = 0;
    float max_logit_diff = 0.0f;

    for (int i = 0; i < B; ++i) {
        if (predictions[i] == expected_preds[i]) correct_preds++;
        for (int j = 0; j < O3; ++j) {
            float diff = std::abs(h_Y[i * O3 + j] - expected_logits[i * O3 + j]);
            if (diff > max_logit_diff) max_logit_diff = diff;
        }
    }

    const double PEAK_GFLOPS = 2550; // Peak FP32 per GTX 1650 Max-Q
    const double PEAK_BANDWIDTH = 112.1; // Peak bandwidth (GB/s)  (https://www.techpowerup.com/gpu-specs/geforce-gtx-1650-max-q.c3383)

    // FLOPs and bytes (same formulas as CPU)
    double total_flops = (B * O1 * 2.0 * I1 + B * O1) + 
                         (B * O2 * 2.0 * I2 + B * O2) + 
                         (B * O3 * 2.0 * I3);
    
    // Calculate bytes read/written to global memory during kernels
    double total_bytes = ((B*I1 + O1*I1 + O1 + B*O1) + 
                          (B*I2 + O2*I2 + O2 + B*O2) + 
                          (B*I3 + O3*I3 + O3 + B*O3)) * 4.0;

    double time_kernel_sec = time_kernel / 1000.0;
    double gflops = (total_flops / time_kernel_sec) / 1e9;
    double bandwidth_gbps = (total_bytes / time_kernel_sec) / 1e9;

    double gflops_utilization = (gflops / PEAK_GFLOPS) * 100.0;
    double bandwidth_utilization = (bandwidth_gbps / PEAK_BANDWIDTH) * 100.0;    

    std::cout << "\n==========================================" << std::endl;
    std::cout << "   RESULTS AND BENCHMARK (CUDA NAIVE)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Accuracy vs Python    : " << correct_preds << "/" << B << " (" << (correct_preds*100.0/B) << "%)" << std::endl;
    std::cout << "Max Logit Error       : " << std::scientific << max_logit_diff << std::fixed << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "H2D Time (Weights+In) : " << std::setprecision(3) << time_h2d << " ms" << std::endl;
    std::cout << "Kernel Time (Compute) : " << std::setprecision(3) << time_kernel << " ms" << std::endl;
    std::cout << "D2H Time (Logits)     : " << std::setprecision(3) << time_d2h << " ms" << std::endl;
    std::cout << "Total Time (End2End)  : " << std::setprecision(3) << (time_h2d + time_kernel + time_d2h) << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Performance (GFLOPS)  : " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << "-> GPU Compute Usage  : " << std::setprecision(4) << gflops_utilization << " %" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Effective Bandwidth   : " << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "-> GPU Memory Usage   : " << std::setprecision(4) << bandwidth_utilization << " %" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    // Cleanup
    cudaFree(d_X); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_H1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_H2);
    cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_Y);

    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);

    return 0;
}