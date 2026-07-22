#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <omp.h>

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

// Full forward pass of a 3‑layer network inside a SINGLE parallel region
// <-> attempt to avoid multiple fork‑join overheads by reusing the same thread pool
std::vector<float> mlp_forward_unified(
    const std::vector<float>& X,
    const std::vector<float>& W1, const std::vector<float>& b1,
    const std::vector<float>& W2, const std::vector<float>& b2,
    const std::vector<float>& W3, const std::vector<float>& b3,
    int B,
    double& time_l1, double& time_l2, double& time_l3, double& time_sec
) {
    const int I1 = 784, O1 = 256;
    const int I2 = 256, O2 = 128;
    const int I3 = 128, O3 = 10;

    std::vector<float> H1(B * O1, 0.0f);
    std::vector<float> H2(B * O2, 0.0f);
    std::vector<float> Logits(B * O3, 0.0f);

    double t_l1_end = 0.0, t_l2_end = 0.0; // timestamp variables

    double t_start = omp_get_wtime(); // START TIMING --> Includes the upcoming Fork overhead

    #pragma omp parallel default(none) \
                         shared(X, W1, b1, H1, W2, b2, H2, W3, b3, Logits, B, \
                                t_l1_end, t_l2_end)
    {
        int tid = omp_get_thread_num();

        // ==========================================
        // LAYER 1
        // ==========================================
        #pragma omp for collapse(2) nowait 
        // 'nowait' removes the implicit barrier at the end of the loop
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < O1; ++j) {
                const float *x_row = &X[i * I1];
                const float *w_row = &W1[j * I1];
                float sum = b1[j];
                for (int k = 0; k < I1; ++k) sum += x_row[k] * w_row[k];
                H1[i * O1 + j] = std::max(0.0f, sum);
            }
        }
        // Explicit barrier
        #pragma omp barrier 
        
        if (tid == 0) t_l1_end = omp_get_wtime(); // layer 1 time

        // ==========================================
        // LAYER 2
        // ==========================================
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < O2; ++j) {
                const float *h1_row = &H1[i * I2]; 
                const float *w_row = &W2[j * I2];
                float sum = b2[j];
                for (int k = 0; k < I2; ++k) sum += h1_row[k] * w_row[k];
                H2[i * O2 + j] = std::max(0.0f, sum);
            }
        }
        #pragma omp barrier 
        
        if (tid == 0) t_l2_end = omp_get_wtime(); // layer 2 time

        // ==========================================
        // LAYER 3
        // ==========================================
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < O3; ++j) {
                const float *h2_row = &H2[i * I3]; 
                const float *w_row = &W3[j * I3];
                float sum = b3[j];
                for (int k = 0; k < I3; ++k) sum += h2_row[k] * w_row[k];
                Logits[i * O3 + j] = sum; 
            }
        }
    }  // The parallel region ends here --> implicit join of all threads
    double t_end = omp_get_wtime(); // END TIMING

    // Durations
    time_l1 = t_l1_end - t_start; // Arithmetic time for L1 + fork overhead
    time_l2 = t_l2_end - t_l1_end; // Arithmetic time for L2 + barrier overhead
    time_l3 = t_end - t_l2_end;    // Arithmetic time for L3 + barrier + join overhead
    time_sec = t_end - t_start;

    return Logits;
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

// Modified function to create separate files for each version
void log_to_csv(const std::string& version, int B, int num_threads, const std::string& opt, int run,
                double t_l1, double t_l2, double t_l3, double t_tot, 
                double gflops, double bandwidth, double cgma) {
    
    std::string filename = "results_" + version + ".csv";
    std::ifstream check_file(filename);
    bool file_exists = check_file.good();
    check_file.close();

    std::ofstream csv(filename, std::ios::app);
    if (!file_exists) {
        csv << "Version,B,Threads,Optimization,Run,Time_L1_ms,Time_L2_ms,Time_L3_ms,Total_Time_ms,GFLOPS,Bandwidth_GBps,CGMA\n";
    }
    csv << version << "," << B << "," << num_threads << "," << opt << "," << run << ","
        << t_l1 << "," << t_l2 << "," << t_l3 << "," << t_tot << "," 
        << gflops << "," << bandwidth << "," << cgma << "\n";
}


int main(int argc, char* argv[]) {
    
    // ==========================================
    // INITIALIZATION
    // ==========================================
    
   // Same 5 arguments
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <B> <num_threads> <file_version> <opt/no_opt> <run_id>\n";
        return 1;
    }

    const int B = std::stoi(argv[1]);
    const int num_threads = std::stoi(argv[2]); // n_threads
    const std::string file_version = argv[3];
    const std::string opt_status = argv[4];
    const int run_id = std::stoi(argv[5]);

    omp_set_num_threads(num_threads);
    
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

    double time_l1 = 0.0, time_l2 = 0.0, time_l3 = 0.0, time_sec=0.0;
    // The timing of each layer is perfromed within the function
    auto Logits = mlp_forward_unified(X, W1, b1, W2, b2, W3, b3, B, time_l1, time_l2, time_l3, time_sec);

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

    // ==========================================
    // FIGURES OF MERIT
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
    double cgma = total_flops / total_bytes;

    std::cout << "\n-> Time Breakdown per Layer:" << std::endl;
    std::cout << "   Layer 1 (FC + ReLU) : " << std::setprecision(4) << time_l1 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 2 (FC + ReLU) : " << std::setprecision(4) << time_l2 * 1000.0 << " ms" << std::endl;
    std::cout << "   Layer 3 (FC Only)   : " << std::setprecision(4) << time_l3 * 1000.0 << " ms" << std::endl;

    std::cout << "\n-> HPC Figures of Merit (Batch Size = " << B << "):" << std::endl;
    std::cout << "   Inference Time     : " << std::setprecision(6) << time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "   Total FLOPs        : " << total_flops / 1e6 << " MFLOPs" << std::endl;
    std::cout << "   Total Memory I/O   : " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "   --------------------------------------" << std::endl;
    std::cout << "   Performance (GFLOPS): " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << "   Bandwidth (GB/s)    : " << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "   CGMA (FLOPs/Byte)   : " << cgma << " (Arithmetic Intensity)" << std::endl;
    std::cout << "   --------------------------------------" << std::endl;

    log_to_csv(file_version, B, num_threads, opt_status, run_id, 
               time_l1*1000.0, time_l2*1000.0, time_l3*1000.0, time_sec*1000.0, 
               gflops, bandwidth_gbps, cgma);

    return 0;
}