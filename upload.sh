#!/bin/bash

# Since we have a large dataset (100GB+) with 300,000+ files, 
# we need to use the upload-large-folder command and specify the number of workers.
# This script tries different numbers of workers to upload the dataset, 
# starting with the highest power of 2 less than or equal to the number of physical CPU cores.

# Calculate the number of physical CPU cores (cores per socket * sockets)
cores_per_socket=$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')
sockets=$(lscpu | grep "Socket(s)" | awk '{print $NF}')
total_cores=$((cores_per_socket * sockets))

# Find highest power of 2 less than or equal to total_cores
max_workers=1
while [ $((max_workers * 2)) -le $total_cores ]; do
    max_workers=$((max_workers * 2))
done

echo "Cores per socket: $cores_per_socket"
echo "Sockets: $sockets"
echo "Total cores: $total_cores"
echo "Max workers (highest power of 2 ≤ $total_cores): $max_workers"

# Create worker array starting from max_workers and going down
workers=()
current=$max_workers
while [ $current -ge 1 ]; do
    workers+=($current)
    current=$((current / 2))
done

echo "Will try workers in this order: ${workers[@]}"


for num_workers in "${workers[@]}"; do
    echo "Trying with $num_workers workers..."
    
    if huggingface-cli upload-large-folder manifold-machines/lnqa-mk --repo-type=dataset /workspace --num-workers=$num_workers > upload.log 2>&1; then
        echo "Success with $num_workers workers!"
        break
    else
        echo "Failed with $num_workers workers (exit code: $?)"
        if [ "$num_workers" -eq 1 ]; then
            echo "All worker counts failed!"
            exit 1
        fi
    fi
done