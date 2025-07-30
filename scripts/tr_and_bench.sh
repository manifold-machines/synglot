#!/bin/bash

# Run the first command
echo "Running mmmu.py..."
python scripts/mmmu.py

# Check if the first command succeeded before running the second
if [ $? -eq 0 ]; then
    echo "First command completed successfully. Starting benchmark..."
    # Run the second command and log outputs to a file
    python benchmark_mmmu.py --models google/gemma-3-4b-it qwen/qwen-2.5-vl-7b-instruct mistralai/pixtral-12b google/gemma-3-12b-it google/gemma-3-27b-it meta-llama/llama-3.2-11b-vision-instruct --output overnight_run_results.json 2>&1 | tee benchmark_output.log
    echo "Benchmark completed. Output logged to benchmark_output.log"
else
    echo "First command failed. Stopping execution."
    exit 1
fi