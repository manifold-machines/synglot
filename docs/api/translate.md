# `synglot.translate`

This module provides classes for translating text between languages.

## `Translator` (Base Class)

```python
class Translator:
    def __init__(self, source_lang, target_lang)
    def translate(self, text)
    def translate_batch(self, texts, batch_size=32)
```

### `__init__(self, source_lang, target_lang)`
Initializes the base translator.

**Args:**
- `source_lang` (`str`): Source language code.
- `target_lang` (`str`): Target language code.

### `translate(self, text)`
Translate a single piece of text. This method must be implemented by subclasses.

**Args:**
- `text` (`str`): The text to translate.

### `translate_batch(self, texts, batch_size=32)`
Translate a batch of texts.

**Args:**
- `texts` (`list[str]`): A list of texts to translate.
- `batch_size` (`int`, optional): Batch size for translation. Defaults to `32`.

**Returns:**
- `list[str]`: A list of translated texts.

## `LLMTranslator`

```python
class LLMTranslator(Translator):
    def __init__(self, source_lang, target_lang, provider="openai", model_name=None, api_key=None)
    def translate(self, text)
```

### `__init__(self, source_lang, target_lang, provider="openai", model_name=None, api_key=None)`
Initializes the LLM API translator.

**Args:**
- `source_lang` (`str`): Source language code.
- `target_lang` (`str`): Target language code.
- `provider` (`str`, optional): API provider (e.g., "openai", "anthropic", "gemini"). Defaults to `"openai"`.
- `model_name` (`str`, optional): Model name to use for translation.
- `api_key` (`str`, optional): API key for the provider.

### `translate(self, text)`
Translate text using an LLM API.
*Note: This method is not yet fully implemented.*

**Args:**
- `text` (`str`): The text to translate.

**Returns:**
- `list[str]`: A list of translated texts.

## `StandardTranslator`

```python
class StandardTranslator(Translator):
    def __init__(self, source_lang, target_lang, backend="marianmt")
    def translate(self, text)
    def translate_batch(self, texts, batch_size=32)
    def translate_dataset(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=1, progress_interval=10, save_errors=True, append_mode=False)
    def translate_dataset_batch(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=32, progress_interval=100)
```

### `__init__(self, source_lang, target_lang, backend="marianmt")`
Initializes the standard translator, defaulting to MarianMT.

**Args:**
- `source_lang` (`str`): Source language code (e.g., 'en').
- `target_lang` (`str`): Target language code (e.g., 'fr').
- `backend` (`str`, optional): Backend translation system. Currently supports `"marianmt"`. Defaults to `"marianmt"`.

### `translate(self, text)`
Translate text using the configured standard translation system (MarianMT).

**Args:**
- `text` (`str`): The text to translate.

**Returns:**
- `str`: The translated text.

### `translate_batch(self, texts, batch_size=32)`
Translate a batch of texts using MarianMT.

**Args:**
- `texts` (`list[str]`): A list of texts to translate.
- `batch_size` (`int`, optional): Batch size for processing. Defaults to `32`.

**Returns:**
- `list[str]`: A list of translated texts.

### `translate_dataset(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=1, progress_interval=10, save_errors=True, append_mode=False)`
Translate specified columns in a dataset with comprehensive error handling and progress tracking.

**Args:**
- `dataset`: Dataset object to translate.
- `columns_to_translate` (`str` or `list[str]`): Column name(s) to translate.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated. Defaults to `None`.
- `output_dir` (`str`, optional): Directory to save output (used if `output_path` is `None`). Defaults to `"./outputs"`.
- `batch_size` (`int`, optional): Batch size for translation (currently processes one by one for error handling). Defaults to `1`.
- `progress_interval` (`int`, optional): Print progress every N samples. Defaults to `10`.
- `save_errors` (`bool`, optional): Whether to save error records to output. Defaults to `True`.
- `append_mode` (`bool`, optional): Whether to append to existing file or overwrite. Defaults to `False`.

**Returns:**
- `dict`: Summary statistics including success/error counts and output path.

**Example:**
```python
translator = StandardTranslator("en", "fr")
summary = translator.translate_dataset(
    dataset=my_dataset,
    columns_to_translate=["text", "description"],
    output_dir="./translations"
)
```

### `translate_dataset_batch(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=32, progress_interval=100)`
Translate specified columns in a dataset using batch processing for better performance.

**Note:** Batch processing provides better performance but less granular error handling.

**Args:**
- `dataset`: Dataset object to translate.
- `columns_to_translate` (`str` or `list[str]`): Column name(s) to translate.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated. Defaults to `None`.
- `output_dir` (`str`, optional): Directory to save output (used if `output_path` is `None`). Defaults to `"./outputs"`.
- `batch_size` (`int`, optional): Batch size for translation. Defaults to `32`.
- `progress_interval` (`int`, optional): Print progress every N samples. Defaults to `100`.

**Returns:**
- `dict`: Summary statistics including success/error counts and output path.

**Example:**
```python
translator = StandardTranslator("en", "es")
summary = translator.translate_dataset_batch(
    dataset=large_dataset,
    columns_to_translate="content",
    batch_size=64
)
```

## Usage Examples

### Simple Dataset Translation
```python
from synglot.translate import StandardTranslator
from synglot.dataset import Dataset

# Load your dataset
dataset = Dataset()
dataset.load_from_huggingface("your_dataset", split="train")

# Initialize translator
translator = StandardTranslator("en", "fr")

# Translate with one simple call - everything is handled automatically
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=["text", "description"]
)

print(f"Translated {summary['successful_translations']} samples")
print(f"Output saved to: {summary['output_path']}")
```

### Batch Processing for Large Datasets
```python
# For large datasets, use batch processing for better performance
summary = translator.translate_dataset_batch(
    dataset=large_dataset,
    columns_to_translate="content",
    batch_size=32
)
``` 