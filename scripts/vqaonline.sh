# make sure to run this script from the root of the project, using tmux

caffeinate -dimsu &

python main.py translate --hf-dataset "ChongyanChen/VQAonline" --hf-split train --columns "question, context, answer" --streaming-mode --source-lang en --target-lang mk --backend openrouter --model-name google/gemini-flash-1.5-8b --output-dir ./translations_vqaonline

killall caffeinate