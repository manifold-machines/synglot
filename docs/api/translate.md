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
    def __init__(self, source_lang, target_lang, backend="marianmt", model_name=None, max_gen_tokens=1024, project_id=None)
    def translate(self, text)
    def translate_batch(self, texts, batch_size=32, batch_job_description="batch job")
    def retrieve_batch(self, batch_job_or_result, save_results=True)
    def translate_dataset(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=32, progress_interval=10, save_errors=True, append_mode=False, use_batch=False, batch_job_description="dataset translation", batch_request_limit=50000, batch_token_limit=1900000)
```

**Unified translator supporting standard ML models (MarianMT), LLM APIs (OpenAI), and Google Translate API.**

### `__init__(self, source_lang, target_lang, backend="marianmt", model_name=None, max_gen_tokens=1024, project_id=None)`
Initialize unified translator with support for multiple backends.

**Args:**
- `source_lang` (`str`): Source language code (e.g., 'en').
- `target_lang` (`str`): Target language code (e.g., 'fr').
- `backend` (`str`, optional): Backend translation system. Supports `"marianmt"`, `"openai"`, or `"google"`. Defaults to `"marianmt"`.
- `model_name` (`str`, optional): Model name. For OpenAI: e.g., 'gpt-4o-mini'; for MarianMT: auto-determined; ignored for Google.
- `max_gen_tokens` (`int`, optional): Maximum tokens for generation (used by OpenAI backend). Defaults to `1024`.
- `project_id` (`str`, optional): Google Cloud project ID (required for Google backend).

**Backend-specific Requirements:**
- **MarianMT**: Requires `transformers` library. Model downloaded automatically.
- **OpenAI**: Requires `OPENAI_API_KEY` environment variable.
- **Google**: Requires `google-cloud-translate` library and `GOOGLE_CLOUD_PROJECT_ID` environment variable or `project_id` parameter.

### `translate(self, text)`
Translate text using the configured backend (MarianMT, OpenAI, or Google Translate).

**Args:**
- `text` (`str`): The text to translate.

**Returns:**
- `str`: The translated text.

**Raises:**
- `RuntimeError`: If backend is not properly initialized or API errors occur.

### `translate_batch(self, texts, batch_size=32, batch_job_description="batch job")`
Translate a batch of texts using the configured backend.

**Args:**
- `texts` (`list[str]`): List of texts to translate.
- `batch_size` (`int`, optional): Batch size for MarianMT and Google Translate (ignored for OpenAI batch API). Defaults to `32`.
- `batch_job_description` (`str`, optional): Description for OpenAI batch jobs. Defaults to `"batch job"`.

**Returns:**
- **MarianMT and Google Translate**: `list[str]` - List of translated texts.
- **OpenAI**: Batch job object (use `retrieve_batch()` to get results when complete).

**Example:**
```python
# MarianMT or Google Translate - immediate results
translator = LLMTranslator("en", "fr", backend="marianmt")
translations = translator.translate_batch(["Hello", "World"])
print(translations)  # ['Bonjour', 'Monde']

# OpenAI - returns batch job for async processing
translator = LLMTranslator("en", "fr", backend="openai")
batch_job = translator.translate_batch(["Hello", "World"])
# Use retrieve_batch() later to get results
```

### `retrieve_batch(self, batch_job_or_result, save_results=True)`
Retrieve batch output content when the batch job is done. Only available for OpenAI backend.

**Args:**
- `batch_job_or_result`: Batch job object or result dict from `translate_batch`/`translate_dataset`.
- `save_results` (`bool`, optional): Whether to save results to file automatically (for dataset batches). Defaults to `True`.

**Returns:**
- File content, translations, or processing summary depending on input type.

**Raises:**
- `NotImplementedError`: If called with non-OpenAI backend.

### `translate_dataset(self, dataset, columns_to_translate, output_path=None, output_dir="./outputs", batch_size=32, progress_interval=10, save_errors=True, append_mode=False, use_batch=False, batch_job_description="dataset translation", batch_request_limit=50000, batch_token_limit=1900000)`
Translate specified columns in a dataset with comprehensive error handling and progress tracking.

**Args:**
- `dataset`: Dataset object to translate.
- `columns_to_translate` (`str`, `list[str]`, or `None`): Column name(s) to translate. If `None`, auto-detects translatable columns.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated. Defaults to `None`.
- `output_dir` (`str`, optional): Directory to save output (used if `output_path` is `None`). Defaults to `"./outputs"`.
- `batch_size` (`int`, optional): Batch size for translation. Defaults to `32`.
- `progress_interval` (`int`, optional): Print progress every N samples (used in non-batch mode). Defaults to `10`.
- `save_errors` (`bool`, optional): Whether to save error records to output (used in non-batch mode). Defaults to `True`.
- `append_mode` (`bool`, optional): Whether to append to existing file or overwrite (used in non-batch mode). Defaults to `False`.
- `use_batch` (`bool`, optional): Whether to use batch processing for better performance. Defaults to `False`.
- `batch_job_description` (`str`, optional): Description for batch jobs (OpenAI only). Defaults to `"dataset translation"`.
- `batch_request_limit` (`int`, optional): Maximum number of requests per batch for OpenAI backend. Defaults to `50000`.
- `batch_token_limit` (`int`, optional): Maximum number of tokens per batch for OpenAI backend. Defaults to `1900000`.

**Returns:**
- `dict`: Summary statistics including success/error counts and output path.
- For OpenAI batch mode: Returns batch job info that needs to be retrieved later with `retrieve_batch()`.

**Example:**
```python
# Basic dataset translation
translator = LLMTranslator("en", "fr", backend="marianmt")
summary = translator.translate_dataset(
    dataset=my_dataset,
    columns_to_translate=["text", "description"]
)

# OpenAI batch processing
translator = LLMTranslator("en", "fr", backend="openai")
batch_job = translator.translate_dataset(
    dataset=large_dataset,
    columns_to_translate="content",
    use_batch=True
)
# Later: results = translator.retrieve_batch(batch_job)
```

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

### Simple Translation with Different Backends

```python
from synglot.translate import LLMTranslator

# MarianMT (offline, fast)
translator = LLMTranslator("en", "fr", backend="marianmt")
result = translator.translate("Hello world")
print(result)  # "Bonjour le monde"

# OpenAI (requires API key)
translator = LLMTranslator("en", "fr", backend="openai", model_name="gpt-4o-mini")
result = translator.translate("Hello world")
print(result)  # "Bonjour le monde"

# Google Translate (requires project ID)
translator = LLMTranslator("en", "fr", backend="google", project_id="your-project-id")
result = translator.translate("Hello world")
print(result)  # "Bonjour le monde"
```

### Batch Translation

```python
texts = ["Hello", "World", "How are you?"]

# Synchronous batch translation (MarianMT/Google)
translator = LLMTranslator("en", "fr", backend="marianmt")
translations = translator.translate_batch(texts)
print(translations)  # ['Bonjour', 'Monde', 'Comment allez-vous?']

# Asynchronous batch translation (OpenAI)
translator = LLMTranslator("en", "fr", backend="openai")
batch_job = translator.translate_batch(texts, batch_job_description="My batch job")
# Later when job is complete:
results = translator.retrieve_batch(batch_job)
```

### Dataset Translation

```python
from synglot.translate import LLMTranslator
from synglot.dataset import Dataset

# Load your dataset
dataset = Dataset()
dataset.load_from_huggingface("your_dataset", split="train")

# Simple dataset translation with MarianMT
translator = LLMTranslator("en", "fr", backend="marianmt")
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=["text", "description"]
)
print(f"Translated {summary['successful_translations']} samples")

# Large dataset with OpenAI batch processing
translator = LLMTranslator("en", "fr", backend="openai")
batch_job = translator.translate_dataset(
    dataset=large_dataset,
    columns_to_translate="content",
    use_batch=True,
    batch_size=64
)
# Retrieve results when complete
results = translator.retrieve_batch(batch_job)
```

### Auto-detecting Translatable Columns

```python
# Auto-detect which columns contain translatable text
translator = LLMTranslator("en", "fr", backend="marianmt")
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=None  # Auto-detect
)
print(f"Auto-detected and translated columns: {summary['columns_translated']}")
``` 