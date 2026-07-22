#!/bin/bash

FILE_VERSION="serial_v4_relu"
NUM_RUNS=10
CSV_OUT="./benchmark_serial.csv"
BATCH_SIZES=(1 32 128 512 1024 4096 10000)

echo "=== Automated Benchmarking Session Start ==="
echo "Saving results to: $CSV_OUT"

echo "-> Compiling..."
g++ -O0 -std=c++17 serial_v4_relu.cpp -o bench_no_opt
g++ -O3 -std=c++17 serial_v4_relu.cpp -o bench_opt

for B in "${BATCH_SIZES[@]}"; do
    echo "------------------------------------------------"
    echo "Running Test for Batch Size B = $B"
    echo "------------------------------------------------"

    # No-Opt Variant
    echo "-> Running UNOPTIMIZED version (-O0)..."
    ./bench_no_opt $B $FILE_VERSION "no_opt" 0  # Run 0: Warm-up
    for ((run=1; run<=NUM_RUNS; run++)); do
        ./bench_no_opt $B $FILE_VERSION "no_opt" $run
    done

    # Opt Variant
    echo "-> Running OPTIMIZED version (-O3)..."
    ./bench_opt $B $FILE_VERSION "opt" 0        # Run 0: Warm-up
    for ((run=1; run<=NUM_RUNS; run++)); do
        ./bench_opt $B $FILE_VERSION "opt" $run
    done
done

echo "===================================================="
echo "Benchmark completed! The CSV now includes times per layer."
echo "===================================================="