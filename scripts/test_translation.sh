#!/usr/bin/env bash

set -euo pipefail

# Minimal end-to-end smoke tests for translation CLI against HF datasets.
# Covers: sequential vs. batch, streaming vs. non-streaming, and marianmt/nllb/openai backends.

REPO_DIR="/home/ilijalichkovski/synglot"
MAIN_PY="$REPO_DIR/main.py"
OUT_DIR="$REPO_DIR/outputs/test_runs"
mkdir -p "$OUT_DIR"

PY="python3"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY not set. OpenAI tests will likely fail unless set." >&2
fi

set -x

# Dataset 1: ChongyanChen/VQAonline
# 1) Non-streaming, sequential, marianmt
$PY "$MAIN_PY" translate \
  --hf-dataset "ChongyanChen/VQAonline" \
  --source-lang en --target-lang de \
  --backend marianmt \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 2) Non-streaming, batching, openai
$PY "$MAIN_PY" translate \
  --hf-dataset "ChongyanChen/VQAonline" \
  --source-lang en --target-lang de \
  --backend openai \
  --use-batch \
  --batch-size 4 \
  --batch-job-description "test VQAonline openai batch" \
  --max-samples 5 \
  --output-dir "$OUT_DIR"

# 3) Streaming, sequential, nllb
$PY "$MAIN_PY" translate \
  --hf-dataset "ChongyanChen/VQAonline" \
  --source-lang en --target-lang de \
  --backend nllb \
  --device cpu \
  --streaming-mode \
  --batch-size 4 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 4) Non-streaming, batching, marianmt
$PY "$MAIN_PY" translate \
  --hf-dataset "ChongyanChen/VQAonline" \
  --source-lang en --target-lang de \
  --backend marianmt \
  --use-batch \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR"

# Dataset 2: princeton-nlp/CharXiv
# 5) Non-streaming, sequential, marianmt
$PY "$MAIN_PY" translate \
  --hf-dataset "princeton-nlp/CharXiv" \
  --source-lang en --target-lang es \
  --backend marianmt \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 6) Non-streaming, batching, openai
$PY "$MAIN_PY" translate \
  --hf-dataset "princeton-nlp/CharXiv" \
  --source-lang en --target-lang es \
  --backend openai \
  --use-batch \
  --batch-size 4 \
  --batch-job-description "test CharXiv openai batch" \
  --max-samples 5 \
  --output-dir "$OUT_DIR"

# 7) Streaming, sequential, nllb
$PY "$MAIN_PY" translate \
  --hf-dataset "princeton-nlp/CharXiv" \
  --source-lang en --target-lang es \
  --backend nllb \
  --device cpu \
  --streaming-mode \
  --batch-size 4 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 8) Non-streaming, batching, nllb
$PY "$MAIN_PY" translate \
  --hf-dataset "princeton-nlp/CharXiv" \
  --source-lang en --target-lang es \
  --backend nllb \
  --device cpu \
  --use-batch \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR"

# Dataset 3: FlagEval/ERQA
# 9) Non-streaming, sequential, marianmt
$PY "$MAIN_PY" translate \
  --hf-dataset "FlagEval/ERQA" \
  --source-lang en --target-lang fr \
  --backend marianmt \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 10) Non-streaming, batching, openai
$PY "$MAIN_PY" translate \
  --hf-dataset "FlagEval/ERQA" \
  --source-lang en --target-lang fr \
  --backend openai \
  --use-batch \
  --batch-size 4 \
  --batch-job-description "test ERQA openai batch" \
  --max-samples 5 \
  --output-dir "$OUT_DIR"

# 11) Streaming, sequential, nllb
$PY "$MAIN_PY" translate \
  --hf-dataset "FlagEval/ERQA" \
  --source-lang en --target-lang fr \
  --backend nllb \
  --device cpu \
  --streaming-mode \
  --batch-size 4 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1

# 12) Non-streaming, sequential, openai (extra coverage of sequential OpenAI)
$PY "$MAIN_PY" translate \
  --hf-dataset "FlagEval/ERQA" \
  --source-lang en --target-lang fr \
  --backend openai \
  --batch-size 4 \
  --max-samples 5 \
  --output-dir "$OUT_DIR" \
  --progress-interval 1 