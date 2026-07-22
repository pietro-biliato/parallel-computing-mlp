#!/bin/bash

FILE_VERSION="omp_v5"
NUM_RUNS=10
CSV_OUT="./results_omp.csv"

# Array for Batch Sizes and Threads
BATCH_SIZES=(1 32 128 512 1024 4096 10000)
THREADS=(1 2 3 4 5 6 7 8 9 10 11 12 15 18 24 40)

echo "=== Automated Benchmarking Session Start ==="
echo "Saving results to: $CSV_OUT"

# Added the -fopenmp flag
echo "-> Compiling..."
g++ -O0 -std=c++17 -fopenmp omp_v5.cpp -o bench_no_opt
g++ -O3 -std=c++17 -fopenmp omp_v5.cpp -o bench_opt

for B in "${BATCH_SIZES[@]}"; do
    for T in "${THREADS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Test for Batch Size B = $B | Threads = $T"
        echo "------------------------------------------------"

        # No-Opt Variant
        echo "-> Running UNOPTIMIZED version (-O0)..."
        ./bench_no_opt $B $T $FILE_VERSION "no_opt" 0  # Run 0: Warm-up
        for ((run=1; run<=NUM_RUNS; run++)); do
            ./bench_no_opt $B $T $FILE_VERSION "no_opt" $run
        done

        # Opt Variant
        echo "-> Running OPTIMIZED version (-O3)..."
        ./bench_opt $B $T $FILE_VERSION "opt" 0        # Run 0: Warm-up
        for ((run=1; run<=NUM_RUNS; run++)); do
            ./bench_opt $B $T $FILE_VERSION "opt" $run
        done
        
    done
done

echo "===================================================="
echo "Benchmark completed! Results are in $CSV_OUT."
echo "===================================================="