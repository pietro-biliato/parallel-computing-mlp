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

#define TILE_WIDTH 16

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

// Optimized kernel
// V2: Tiling 2D + shared memory + preventing bank conlicts
__global__ void linear_forward_tiled(const float* X, const float* W, const float* b, float* Y, 
                                     int B, int I, int O, bool apply_relu) {
    
    __shared__ float Xs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ws[TILE_WIDTH][TILE_WIDTH + 1]; // shared memory padding to avoid bank conflicts.
    // (hen the threads of a warp read Ws[tx][k] (same column, different rows), they will now access 
    // different physical memory banks, lowering latency)

    // Thread indices inside the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices of the output matrix Y
    int row = blockIdx.y * TILE_WIDTH + ty; // B
    int col = blockIdx.x * TILE_WIDTH + tx; // O

    float sum = 0.0f; //stored in registers

    // Number of tiles needed to cover the inner dimension (I)
    int num_tiles = (I + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < num_tiles; ++m) {  // Loop over tiles
        
        // LOADING INTO SHARED MEMORY:

        // Load tile of X: thread (ty, tx) loads element X[row, m*TILE + tx]
        if (row < B && (m * TILE_WIDTH + tx) < I) {
            Xs[ty][tx] = X[row * I + (m * TILE_WIDTH + tx)];
        } else {
            Xs[ty][tx] = 0.0f; // zero padding for out‑of‑bounds threads
        }

        // Load tile of W:
        // W has shape [O, I] --> for coalesced reads from Global Memory,
        // tx is mapped to the I‑axis (contiguous in memory) and ty to the O‑axis
        int w_row = blockIdx.x * TILE_WIDTH + ty;
        int w_col = m * TILE_WIDTH + tx;
        
        if (w_row < O && w_col < I) {
            Ws[ty][tx] = W[w_row * I + w_col];
        } else {
            Ws[ty][tx] = 0.0f; // padding for Layer 3 (because O = 10)
        }

        __syncthreads(); // wait for all threads to finish loading the tile

        // PARTIAL DOT PRODUCT:

        for (int k = 0; k < TILE_WIDTH; ++k) {
            // Xs[ty][k] -> element from row 'ty' of X
            // Ws[tx][k] -> element from row 'tx' of W (<->column 'tx' of W^T)
            sum += Xs[ty][k] * Ws[tx][k];
        }

        __syncthreads(); //ensure all threads finish computing before 
        // overwriting shared memory with the next tile
    }

    // WRITING TO GLOBAL MEMORY:

    if (row < B && col < O) { // a valid element of Y is needed
        sum += b[col]; // add bias

        if (apply_relu && sum < 0.0f) {
            sum = 0.0f; // Kernel fusion: ReLU applied on‑the‑fly
        }

        Y[row * O + col] = sum;
    }
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
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    
    dim3 gridSize1((O1 + TILE_WIDTH - 1) / TILE_WIDTH, (B + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 gridSize2((O2 + TILE_WIDTH - 1) / TILE_WIDTH, (B + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 gridSize3((O3 + TILE_WIDTH - 1) / TILE_WIDTH, (B + TILE_WIDTH - 1) / TILE_WIDTH);

    // ==========================================
    // KERNEL EXECUTION (FORWARD PASS)
    // ==========================================
    std::cout << "-> Esecuzione Forward Pass su GPU (Tiling 2D)..." << std::endl;
    cudaEventRecord(start_kernel);

    linear_forward_tiled<<<gridSize1, blockSize>>>(d_X, d_W1, d_b1, d_H1, B, I1, O1, true);
    checkCuda(cudaGetLastError());

    linear_forward_tiled<<<gridSize2, blockSize>>>(d_H1, d_W2, d_b2, d_H2, B, I2, O2, true);
    checkCuda(cudaGetLastError());

    linear_forward_tiled<<<gridSize3, blockSize>>>(d_H2, d_W3, d_b3, d_Y, B, I3, O3, false);
    checkCuda(cudaGetLastError());

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // ==========================================
    // DEVICE-TO-HOST TRANSFER
    // ==========================================
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

    std::cout << "\n==========================================" << std::endl;
    std::cout << "   RESULTS AND BENCHMARK (CUDA TILED 2)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Accuracy vs Python    : " << correct_preds << "/" << B << " (" << (correct_preds*100.0/B) << "%)" << std::endl;
    std::cout << "Max Logit Error       : " << std::scientific << max_logit_diff << std::fixed << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "H2D Time (Weights+In) : " << std::setprecision(3) << time_h2d << " ms" << std::endl;
    std::cout << "Kernel Time (Compute) : " << std::setprecision(3) << time_kernel << " ms" << std::endl;
    std::cout << "D2H Time (Logits)     : " << std::setprecision(3) << time_d2h << " ms" << std::endl;
    std::cout << "Total Time (End2End)  : " << std::setprecision(3) << (time_h2d + time_kernel + time_d2h) << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    cudaFree(d_X); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_H1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_H2);
    cudaFree(d_W3); cudaFree(d_b3); cudaFree(d_Y);
    
    return 0;
}