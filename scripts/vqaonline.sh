#!/bin/bash
# make sure to run this script from the root of the project, using tmux

# Variable to store caffeinate PID
CAFFEINATE_PID=""

# Function to cleanup caffeinate process
cleanup() {
    if [ -n "$CAFFEINATE_PID" ] && kill -0 "$CAFFEINATE_PID" 2>/dev/null; then
        echo "Cleaning up caffeinate process (PID: $CAFFEINATE_PID)..."
        kill "$CAFFEINATE_PID" 2>/dev/null
    fi
}

# Set up trap to catch EXIT, INT (Ctrl+C), and TERM signals
trap cleanup EXIT INT TERM

# Start caffeinate in background and store its PID
caffeinate -dimsu & # works on macos
CAFFEINATE_PID=$!

echo "Started caffeinate with PID: $CAFFEINATE_PID"

# Run the Python script
python main.py translate --hf-dataset "ChongyanChen/VQAonline" --hf-split train --columns "question, context, answer" --streaming-mode --source-lang en --target-lang mk --backend openrouter --model-name google/gemini-flash-1.5-8b --output-dir ./translations_vqaonline

# The cleanup function will be called automatically due to the trap