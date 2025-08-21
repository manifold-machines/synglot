#!/bin/bash
# make sure to run this script from the root of the project, using tmux

# Function to cleanup caffeinate process
cleanup() {
    echo "Cleaning up caffeinate process..."
    killall caffeinate 2>/dev/null
}

# Set up trap to catch EXIT, INT (Ctrl+C), and TERM signals
trap cleanup EXIT INT TERM

# Start caffeinate in background
caffeinate -dimsu & # works on macos

# Run the Python script
python main.py translate --hf-dataset "ChongyanChen/VQAonline" --hf-split train --columns "question, context, answer" --streaming-mode --source-lang en --target-lang mk --backend openrouter --model-name google/gemini-flash-1.5-8b --output-dir ./translations_vqaonline

# The cleanup function will be called automatically due to the trap