#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
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
// X: [B, I], W: [O, I], b: [O] -> Y: [B, O]
// V2: Bias Trick (Arithmetic Instruction Optimization)
std::vector<float> linear_layer_v2(const std::vector<float>& X, const std::vector<float>& W, const std::vector<float>& b, int B, int I, int O) {
    std::vector<float> Y(B * O, 0.0f);
    
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < O; ++j) {
            
            float sum = b[j]; // initialize the register directly with the bias
            
            for (int k = 0; k < I; ++k) {
                sum += X[i * I + k] * W[j * I + k];
            }
            
            Y[i * O + j] = sum; // removed the final + b[j]
        }
    }
    return Y;
}


// In-place ReLU activation function: f(x) = max(0, x)
void relu_inplace(std::vector<float>& X) {
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i] < 0.0f) X[i] = 0.0f;
    }
}

// Calculates the predicted class (Argmax) for each image in the batch
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

// Function to save data to a CSV file
void log_to_csv(const std::string& filename, const std::string& file_version, int B, const std::string& opt, int run,
                double t_l1, double t_l2, double t_l3, double t_tot, 
                double gflops, double bandwidth, double cgma) {
    std::ifstream check_file(filename);
    bool file_exists = check_file.good();
    check_file.close();

    std::ofstream csv(filename, std::ios::app);
    if (!file_exists) {
        csv << "Version,B,Optimization,Run,Time_L1_ms,Time_L2_ms,Time_L3_ms,Total_Time_ms,GFLOPS,Bandwidth_GBps,CGMA\n"; //header
    }
    csv << file_version << "," << B << "," << opt << "," << run << ","
        << t_l1 << "," << t_l2 << "," << t_l3 << "," << t_tot << "," 
        << gflops << "," << bandwidth << "," << cgma << "\n";
}

int main(int argc, char* argv[]) {

    // ==========================================
    // INITIALIZATION
    // ==========================================
    // Same 4 input arguments:
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <B> <file_version> <opt/no_opt> <run_id>\n";
        return 1;
    }
    const int B = std::stoi(argv[1]);
    const std::string file_version = argv[2];
    const std::string opt_status = argv[3];
    const int run_id = std::stoi(argv[4]);
    
    const std::string dir = "./export/";
    
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

    // ==========================================
    // FORWARD PASS
    // ==========================================
    std::cout << "-> Starting Forward Pass (Inference)..." << std::endl;

    // --- START TIMING ---
    // Layer 1
    auto start_l1 = std::chrono::high_resolution_clock::now();
    auto H1 = linear_layer_v2(X, W1, b1, B, I1, O1);
    relu_inplace(H1);
    auto end_l1 = std::chrono::high_resolution_clock::now();

    // Layer 2
    auto start_l2 = std::chrono::high_resolution_clock::now();
    auto H2 = linear_layer_v2(H1, W2, b2, B, I2, O2);
    relu_inplace(H2);
    auto end_l2 = std::chrono::high_resolution_clock::now();

    // Layer 3 (Logits)
    auto start_l3 = std::chrono::high_resolution_clock::now();
    auto Logits = linear_layer_v2(H2, W3, b3, B, I3, O3);
    auto end_l3 = std::chrono::high_resolution_clock::now();
    // --- END TIMING ---

    // Durations
    std::chrono::duration<double> elapsed_l1 = end_l1 - start_l1;
    std::chrono::duration<double> elapsed_l2 = end_l2 - start_l2;
    std::chrono::duration<double> elapsed_l3 = end_l3 - start_l3;
    
    double time_l1 = elapsed_l1.count();
    double time_l2 = elapsed_l2.count();
    double time_l3 = elapsed_l3.count();
    double time_sec = time_l1 + time_l2 + time_l3;

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
    std::cout << "   Maximum Logits Error: " << std::scientific << max_logit_diff << std::fixed << " (should be < 1e-5)" << std::endl;

    // ==========================================
    // FIGURES OF MERIT CALCULATION
    // ==========================================
    double flops_l1 = B * O1 * 2.0 * I1 + (B * O1);
    double flops_l2 = B * O2 * 2.0 * I2 + (B * O2);
    double flops_l3 = B * O3 * 2.0 * I3;
    double total_flops = flops_l1 + flops_l2 + flops_l3;

    double bytes_l1 = (B*I1 + O1*I1 + O1 + B*O1) * 4.0;
    double bytes_l2 = (B*I2 + O2*I2 + O2 + B*O2) * 4.0;
    double bytes_l3 = (B*I3 + O3*I3 + O3 + B*O3) * 4.0;
    double total_bytes = bytes_l1 + bytes_l2 + bytes_l3;

    double gflops = (total_flops / time_sec) / 1e9;
    double bandwidth_gbps = (total_bytes / time_sec) / 1e9;
    double cgma = total_flops / total_bytes; // Compute to Global Memory Access

    std::cout << "\n-> Time Breakdown per Layer:" << std::endl;
    std::cout << "   Layer 1 (FC + ReLU) : " << std::setprecision(4) << time_l1 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 2 (FC + ReLU) : " << std::setprecision(4) << time_l2 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 3 (Only FC)   : " << std::setprecision(4) << time_l3 * 1000.0 << " ms" << std::endl;

    std::cout << "\n-> HPC Figures of Merit (Batch Size = " << B << "):" << std::endl;
    std::cout << "   Inference Time     : " << std::setprecision(6) << time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "   Total FLOPs        : " << total_flops / 1e6 << " MFLOPs" << std::endl;
    std::cout << "   Total Memory I/O   : " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "   --------------------------------------" << std::endl;
    std::cout << "   Performance (GFLOPS): " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << "   Bandwidth (GB/s)    : " << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "   CGMA (FLOPs/Byte)   : " << cgma << " (Arithmetic Intensity)" << std::endl;
    std::cout << "   --------------------------------------" << std::endl;

    log_to_csv("benchmark_results.csv", file_version, B, opt_status, run_id, 
                   time_l1*1000.0, time_l2*1000.0, time_l3*1000.0, time_sec*1000.0, 
                   gflops, bandwidth_gbps, cgma);

    return 0;
}