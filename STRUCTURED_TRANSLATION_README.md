# Structured Batch Translation with NLLB

This script enables batch translation of datasets containing structured data (lists of dictionaries with 'answer' and 'question' fields) using the NLLB backend while preserving the exact structure.

## Features

- ✅ **Preserves Structure**: Maintains the exact format of your structured data
- ✅ **Batch Processing**: Efficient batch translation using NLLB
- ✅ **Flexible Input**: Handles both JSON lists and JSON strings
- ✅ **Auto Device Detection**: Automatically uses GPU if available, falls back to CPU
- ✅ **Error Handling**: Robust error handling with detailed logging
- ✅ **Multiple Languages**: Supports 200+ languages via NLLB

## Input Format

Your dataset should be in JSONL format where each line contains a JSON object. The structured column should contain a list of dictionaries with 'question' and 'answer' fields:

```json
{
  "id": 1,
  "image_name": "example.jpg",
  "qa_pairs": [
    {
      "answer": "There are three people in the image.",
      "question": "How many people are present in the image?"
    },
    {
      "answer": "The three people are smiling.",
      "question": "What is the facial expression of the people in the image?"
    }
  ]
}
```

## Output Format

The script preserves your original data and adds a new column with translated content:

```json
{
  "id": 1,
  "image_name": "example.jpg",
  "qa_pairs": [
    {
      "answer": "There are three people in the image.",
      "question": "How many people are present in the image?"
    }
  ],
  "translated_qa_pairs_es": [
    {
      "answer": "Hay tres personas en la imagen.",
      "question": "¿Cuántas personas están presentes en la imagen?"
    }
  ]
}
```

## Installation

1. Make sure you have the required dependencies:
```bash
pip install torch transformers tqdm
```

2. Ensure you have the synglot package available (adjust the import path in the script if needed).

## Usage

### Command Line Usage

```bash
python structured_batch_translator.py \
  --input_file your_dataset.jsonl \
  --column qa_pairs \
  --source_lang en \
  --target_lang es \
  --batch_size 32 \
  --device auto
```

#### Command Line Arguments

- `--input_file` / `-i`: Path to input JSONL file (required)
- `--column` / `-c`: Column name containing structured data (required)
- `--source_lang` / `-s`: Source language code (required, e.g., 'en')
- `--target_lang` / `-t`: Target language code (required, e.g., 'es')
- `--output_file` / `-o`: Output file path (optional, auto-generated if not provided)
- `--batch_size` / `-b`: Batch size for translation (default: 32)
- `--device` / `-d`: Device for NLLB model ('auto', 'cpu', 'cuda', default: 'auto')
- `--model_name` / `-m`: NLLB model name (optional, defaults to facebook/nllb-200-distilled-600M)

### Python API Usage

```python
from structured_batch_translator import StructuredBatchTranslator

# Initialize translator
translator = StructuredBatchTranslator(
    source_lang='en',
    target_lang='es',
    device='auto'
)

# Translate entire dataset
output_path = translator.translate_dataset(
    input_file='your_dataset.jsonl',
    column_name='qa_pairs',
    batch_size=32
)

# Translate single item
translated_item = translator.translate_structured_column(
    dataset_item=your_data_item,
    column_name='qa_pairs'
)
```

## Supported Languages

The script supports 200+ languages through NLLB. Common language codes include:

- `en` - English
- `es` - Spanish  
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi

For a complete list, see the language mapping in the original `llm_translator.py` file.

## Performance Tips

1. **GPU Usage**: Use `--device cuda` for faster translation if you have a GPU
2. **Batch Size**: Increase batch size for better GPU utilization, decrease if you get memory errors
3. **Model Selection**: The default model `facebook/nllb-200-distilled-600M` is a good balance of speed and quality. For higher quality, use `facebook/nllb-200-1.3B` or `facebook/nllb-200-3.3B`

## Example

See `example_usage.py` for a complete working example that demonstrates:
- Creating sample data
- Translating a full dataset
- Translating individual items
- Different language pairs

Run the example:
```bash
python example_usage.py
```

## Error Handling

The script handles various error scenarios:
- Invalid JSON in input files
- Missing columns
- Empty or invalid structured data
- Translation failures (items with errors are saved with error information)

## Memory Management

For large datasets:
- The script processes items one by one to manage memory efficiently
- GPU memory is cleared after each batch for NLLB
- Consider using smaller batch sizes for very large structured data

## Output Files

Output files are automatically timestamped and named based on:
- Source language
- Target language
- Backend used
- Processing timestamp

Example: `dataset_translated_en_to_es_20241201_143022.jsonl` 