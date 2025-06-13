# LNQA Dataset Translation with NLLB

Simple script to batch translate the LNQA dataset using the NLLB backend with automatic OOM error handling.

## Usage

```bash
cd scripts
python translate_lnqa_nllb.py
```

## Configuration

Edit the parameters at the bottom of `translate_lnqa_nllb.py`:

```python
batch_translate_dataset(
    source_lang='en',           # Source language (e.g., 'en' for English)
    target_lang='es',           # Target language (e.g., 'es' for Spanish) 
    initial_batch_size=16,      # Starting batch size
    min_batch_size=1            # Minimum batch size before giving up
)
```

## Supported Languages

The script supports all languages supported by NLLB, including:
- English (`en`), Spanish (`es`), French (`fr`), German (`de`)
- Chinese (`zh`), Japanese (`ja`), Korean (`ko`)
- Arabic (`ar`), Hindi (`hi`), Russian (`ru`)
- And many more...

## Features

- **Automatic OOM handling**: Reduces batch size when GPU memory runs out
- **Progress tracking**: Shows real-time progress and statistics
- **Error recovery**: Continues processing even if individual samples fail
- **Structured output**: Preserves original data alongside translations
- **GPU optimization**: Automatically uses GPU if available, falls back to CPU

## Output

Results are saved to `./outputs/lnqa_translated_{source_lang}_to_{target_lang}_nllb.jsonl`

Each line contains:
```json
{
    "image": "original_image_data",
    "original_qa": [{"question": "...", "answer": "..."}],
    "translated_qa": [{"question": "...", "answer": "...", "original_question": "...", "original_answer": "..."}]
}
```

## Requirements

- PyTorch
- Transformers
- Datasets library
- synglot library (from this repository) 