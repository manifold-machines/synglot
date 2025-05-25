# Synglot Cookbook Examples

This directory contains practical examples demonstrating how to use Synglot for various translation tasks.

## Enhanced StandardTranslator (Recommended)

New `StandardTranslator` functionalities include:

- **Automatic column validation** - Ensures specified columns exist in the dataset
- **Comprehensive error handling** - Catches and logs errors with detailed information
- **Progress tracking** - Real-time progress updates during translation
- **Automatic output generation** - Saves results to JSONL format with proper encoding
- **Batch processing support** - Efficient batch translation for large datasets
- **Summary statistics** - Detailed reports on translation success/failure rates

### Simple Usage Example

```python
from synglot.translate import StandardTranslator
from synglot.dataset import Dataset

# Load dataset
dataset = Dataset()
dataset.load_from_huggingface("gsm8k", config_name="main", split="train")

# Initialize translator
translator = StandardTranslator("en", "mk")

# Translate with one simple call - everything is handled automatically!
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=["question", "answer"],
    output_dir="./outputs"
)

print(f"Success rate: {summary['success_rate']:.2%}")
```

## Example Files

### 1. `translate_gsm8k_to_mk.py` 
**Updated example** - Shows how the GSM8K translation task becomes incredibly simple with the enhanced StandardTranslator.

**After (enhanced approach):**
- ~20 lines of code
- Automatic error handling
- Automatic progress tracking
- Automatic file I/O
- Automatic output formatting

## Key Benefits of the StandardTranslator

1. **Simplicity**: One method call handles the entire translation workflow
2. **Robustness**: Built-in error handling and recovery
3. **Performance**: Optimized batch processing for large datasets
4. **Monitoring**: Real-time progress tracking and detailed summaries
5. **Flexibility**: Configurable output paths, batch sizes, and error handling
6. **Reliability**: Automatic validation and comprehensive logging

## Choosing Between Methods

### Use `translate_dataset()` when:
- You need detailed error handling for each sample
- Working with smaller datasets (< 10k samples)
- You want to save error records for debugging
- Reliability is more important than speed

### Use `translate_dataset_batch()` when:
- Working with large datasets (> 10k samples)
- Performance is critical
- Your data is clean and errors are unlikely
- You want maximum throughput

## Output Format

Both methods generate JSONL files with the following structure:

```json
{
  "original_column": "original text",
  "translated_column_target_lang": "translated text",
  "other_columns": "preserved as-is"
}
```

Error records (when `save_errors=True`) include:
```json
{
  "sample_index": 123,
  "error": "Error message",
  "error_type": "ValueError",
  "original_data": {...}
}
```