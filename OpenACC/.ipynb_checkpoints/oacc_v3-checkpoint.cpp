#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <string>


// ==========================================
// 1. SUPPORT FUNCTIONS (I/O and Validation)
// ==========================================

// Template function to read raw binary files into std::vector
template <typename T>
std::vector<T> read_binary_file(const std::string& filename, size_t expected_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    std::vector<T> data(expected_elements);
    file.read(reinterpret_cast<char*>(data.data()), expected_elements * sizeof(T));
    if (!file) {
        throw std::runtime_error("Error during reading or incorrect file size: " + filename);
    }
    return data;
}

// ==========================================
// 2. MATHEMATICAL CORE
// ==========================================

// Forward pass of a linear layer: Y = X * W^T + b
// V3: optimal 'kernel' based version
void linear_layer_v4_step3(const float* restrict X, 
                              const float* restrict W, 
                              const float* restrict b, 
                              float* restrict Y, 
                              int B, int I, int O, bool apply_relu) {
    #pragma acc kernels present (X[0:B*I], W[0:O*I], b[0:O], Y[0:B*O])
    { // Notice the 'present' keyword: this function will be called inside a structured data region
        
        #pragma acc loop independent
        for (int i = 0; i < B; ++i) {
            
            #pragma acc loop independent
            for (int j = 0; j < O; ++j) {
                float sum = b[j];
                
                //#pragma acc loop reduction(+:sum)
                for (int k = 0; k < I; ++k) {
                    sum += X[i * I + k] * W[j * I + k];
                }
                
                if (apply_relu) {
                    Y[i * O + j] = sum > 0.0f ? sum : 0.0f;
                } else {
                    Y[i * O + j] = sum;
                }
            }
        }
    }
}

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

int main() {

    // ==========================================
    // INITIALIZATION
    // ==========================================
    const std::string dir = "./export/";
    
    const int B = 100;
    
    // Architecture Dimensions
    const int I1 = 784, O1 = 256;
    const int I2 = 256, O2 = 128;
    const int I3 = 128, O3 = 10;

    std::cout << "-> Loading weights and inputs into memory..." << std::endl;
    
    auto W1 = read_binary_file<float>(dir + "fc1_weight.bin", O1 * I1);
    auto b1 = read_binary_file<float>(dir + "fc1_bias.bin", O1);
    auto W2 = read_binary_file<float>(dir + "fc2_weight.bin", O2 * I2);
    auto b2 = read_binary_file<float>(dir + "fc2_bias.bin", O2);
    auto W3 = read_binary_file<float>(dir + "fc3_weight.bin", O3 * I3);
    auto b3 = read_binary_file<float>(dir + "fc3_bias.bin", O3);
    
    auto X = read_binary_file<float>(dir + "test_inputs.bin", B * I1);
    auto expected_logits = read_binary_file<float>(dir + "test_logits.bin", B * O3);
    auto expected_preds  = read_binary_file<int>(dir + "test_predictions.bin", B);

    // Pre‑allocating CPU memory for the intermediate and final outputs
    std::vector<float> H1(B * O1);
    std::vector<float> H2(B * O2);
    std::vector<float> Logits(B * O3);
    
    // Extracting the raw pointers for safety when using OpenACC
    float* ptr_X  = X.data();
    float* ptr_W1 = W1.data(); float* ptr_b1 = b1.data(); float* ptr_H1 = H1.data();
    float* ptr_W2 = W2.data(); float* ptr_b2 = b2.data(); float* ptr_H2 = H2.data();
    float* ptr_W3 = W3.data(); float* ptr_b3 = b3.data(); float* ptr_Logits = Logits.data();

    // ==========================================
    // FORWARD PASS
    // ==========================================
    std::cout << "-> Starting Forward Pass (Inference)..." << std::endl;

    std::chrono::duration<double> elapsed_l1;
    std::chrono::duration<double> elapsed_l2;
    std::chrono::duration<double> elapsed_l3;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // Structured data region
    #pragma acc data copyin(ptr_X[0:B*I1], \
                            ptr_W1[0:O1*I1], ptr_b1[0:O1], \
                            ptr_W2[0:O2*I2], ptr_b2[0:O2], \
                            ptr_W3[0:O3*I3], ptr_b3[0:O3]) \
                     create(ptr_H1[0:B*O1], ptr_H2[0:B*O2]) \
                     copyout(ptr_Logits[0:B*O3])
    {
        // layer 1
        auto start_l1 = std::chrono::high_resolution_clock::now();
        linear_layer_v4_step3(ptr_X, ptr_W1, ptr_b1, ptr_H1, B, I1, O1, true);
        auto end_l1 = std::chrono::high_resolution_clock::now();
        elapsed_l1 = end_l1 - start_l1;

        // layer 2
        auto start_l2 = std::chrono::high_resolution_clock::now();
        linear_layer_v4_step3(ptr_H1, ptr_W2, ptr_b2, ptr_H2, B, I2, O2, true);
        auto end_l2 = std::chrono::high_resolution_clock::now();
        elapsed_l2 = end_l2 - start_l2;
        
        // layer 3
        auto start_l3 = std::chrono::high_resolution_clock::now();
        linear_layer_v4_step3(ptr_H2, ptr_W3, ptr_b3, ptr_Logits, B, I3, O3, false);
        auto end_l3 = std::chrono::high_resolution_clock::now();
        elapsed_l3 = end_l3 - start_l3;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Durations
    std::chrono::duration<double> elapsed_total = end_time - start_time;
    
    double time_l1 = elapsed_l1.count();
    double time_l2 = elapsed_l2.count();
    double time_l3 = elapsed_l3.count();
    double time_sec = elapsed_total.count();

    // Predictions
    auto predictions = argmax(Logits, B, O3);

    // ==========================================
    // GROUND TRUTH VALIDATION
    // ==========================================
    std::cout << "\n-> Results Validation:" << std::endl;
    int correct_preds = 0;
    float max_logit_diff = 0.0f;

    for (int i = 0; i < B; ++i) {
        if (predictions[i] == expected_preds[i]) correct_preds++;
        for (int j = 0; j < O3; ++j) {
            float diff = std::abs(Logits[i * O3 + j] - expected_logits[i * O3 + j]);
            if (diff > max_logit_diff) max_logit_diff = diff;
        }
    }
    
    std::cout << "   Accuracy vs Python: " << correct_preds << "/" << B << " (" << (correct_preds*100.0/B) << "%)" << std::endl;
    std::cout << "   Maximum Logits Error: " << std::scientific << max_logit_diff << std::fixed << " (Should be < 1e-5)" << std::endl;

    std::cout << "\n-> Time Breakdown per Layer:" << std::endl;
    std::cout << "   Layer 1 (FC + ReLU) : " << std::setprecision(4) << time_l1 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 2 (FC + ReLU) : " << std::setprecision(4) << time_l2 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 3 (FC Only)   : " << std::setprecision(4) << time_l3 * 1000.0 << " ms" << std::endl;
    std::cout << "   Data transfering    : " << std::setprecision(4) << (time_sec - (time_l1+time_l2+time_l3)) * 1000.0 << " ms" << std::endl;

    return 0;
}