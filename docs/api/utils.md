# `synglot.utils`

This module provides comprehensive utility classes and functions for the `synglot` library including configuration management, batch processing, and text manipulation utilities.

## `Config`

```python
class Config:
    def __init__(self, config_dict=None, config_file=None)
    def get(self, key_path, default=None)
    def set(self, key_path, value)
    def save(self, path)
    def load(self, path)
```

**Configuration manager with support for nested keys and YAML file handling.**

### `__init__(self, config_dict=None, config_file=None)`
Initializes the configuration manager. Loads defaults, then optionally updates from a dictionary and/or a YAML file.

**Args:**
- `config_dict` (`dict`, optional): Configuration dictionary to override defaults.
- `config_file` (`str`, optional): Path to a YAML configuration file.

**Default Configuration Structure:**
- `seed`: Random seed (default: 42)
- `generation_settings`: Settings for text generation
  - Default parameters for temperature, top_k, top_p, etc.
  - `pretraining`: Pretraining data generation settings
  - `conversation`: Conversation generation settings
- `translation_settings`: Translation-specific configurations
- `dataset_settings`: Dataset handling configurations

### `get(self, key_path, default=None)`
Get a configuration value using a dot-separated key path.

**Args:**
- `key_path` (`str`): Dot-separated path to the key (e.g., `"generation_settings.pretraining.topic_list"`).
- `default` (any, optional): Default value to return if the key is not found.

**Returns:**
- `any`: The configuration value or the default.

### `set(self, key_path, value)`
Set a configuration value using a dot-separated key path. Creates nested dictionaries if they don't exist.

**Args:**
- `key_path` (`str`): Dot-separated path to the key.
- `value` (any): Value to set.

### `save(self, path)`
Save the current configuration to a YAML file.

**Args:**
- `path` (`str`): Path to save the YAML file.

### `load(self, path)`
Load configuration from a YAML file, updating the current configuration.

**Args:**
- `path` (`str`): Path to the YAML configuration file to load.

**Example:**
```python
from synglot.utils import Config

# Create with custom settings
config = Config(config_dict={
    "generation_settings.default_temperature": 0.8,
    "translation_settings.default_model_name": "custom-model"
})

# Get nested values
temperature = config.get("generation_settings.default_temperature", 1.0)

# Set new values
config.set("custom_setting.new_param", "value")

# Save and load
config.save("my_config.yaml")
config.load("updated_config.yaml")
```

## Batch Processing Utilities

### `retrieve_batch(client, batch_job_or_result, save_results=True, batch_type="generation")`
Retrieve batch output content when the batch job is done. Handles both simple batch jobs and dataset batch jobs for both translation and generation.

**Args:**
- `client`: OpenAI client instance.
- `batch_job_or_result`: Batch job object or result dict from `translate_batch`/`translate_dataset`.
- `save_results` (`bool`, optional): Whether to save results to file automatically (for dataset batches). Defaults to `True`.
- `batch_type` (`str`, optional): Type of batch operation - `"generation"` or `"translation"`. Defaults to `"generation"`.

**Returns:**
- File content, translations, or processing summary depending on input type and batch_type.

**Example:**
```python
from synglot.utils import retrieve_batch
from synglot.translate import LLMTranslator

# Create batch job
translator = LLMTranslator("en", "fr", backend="openai")
batch_job = translator.translate_batch(["Hello", "World"])

# Retrieve results when complete
results = retrieve_batch(translator.client, batch_job, batch_type="translation")
```

## Text Processing Utilities

### `load_material_files(material_paths)`
Load and parse material files (.txt, .md) for content-based generation.

**Args:**
- `material_paths` (`str` | `list`): Path(s) to .txt or .md files containing source material.

**Returns:**
- `list[dict]`: List of dictionaries containing file content and metadata.

**Metadata includes:**
- `file_path`, `file_name`, `file_extension`
- `content`, `content_length`, `word_count`, `line_count`

**Example:**
```python
from synglot.utils import load_material_files

materials = load_material_files(["docs/readme.md", "research/paper.txt"])
for material in materials:
    print(f"Loaded {material['file_name']}: {material['word_count']} words")
```

### `chunk_text(text, chunk_size, overlap=0, preserve_words=True)`
Split text into overlapping chunks for processing large documents.

**Args:**
- `text` (`str`): Text to split into chunks.
- `chunk_size` (`int`): Size of each chunk in characters.
- `overlap` (`int`, optional): Number of characters to overlap between chunks. Defaults to `0`.
- `preserve_words` (`bool`, optional): Whether to preserve word boundaries. Defaults to `True`.

**Returns:**
- `list[dict]`: List of dictionaries containing chunk data and metadata.

**Chunk metadata includes:**
- `chunk_id`, `text`, `start_pos`, `end_pos`
- `length`, `word_count`, `overlap_start`, `overlap_end`

**Example:**
```python
from synglot.utils import chunk_text

chunks = chunk_text("Long document text...", chunk_size=500, overlap=50)
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['word_count']} words")
```

### `filter_generated_content(generated_texts, min_length=10, max_length=None, min_quality_score=0.5, remove_duplicates=True, similarity_threshold=0.9)`
Filter generated content based on quality metrics and remove duplicates.

**Args:**
- `generated_texts` (`list[str]`): List of generated text samples.
- `min_length` (`int`, optional): Minimum length in characters. Defaults to `10`.
- `max_length` (`int`, optional): Maximum length in characters (`None` for no limit). Defaults to `None`.
- `min_quality_score` (`float`, optional): Minimum quality score (0.0 to 1.0). Defaults to `0.5`.
- `remove_duplicates` (`bool`, optional): Whether to remove duplicate or near-duplicate texts. Defaults to `True`.
- `similarity_threshold` (`float`, optional): Threshold for considering texts as duplicates (0.0 to 1.0). Defaults to `0.9`.

**Returns:**
- `list[dict]`: List of dictionaries containing filtered texts and their quality metrics.

**Quality metrics include:**
- `word_count`, `sentence_count`, `word_diversity`
- `avg_sentence_length`, `avg_word_length`
- `readability_score`, `coherence_score`, `overall_score`

**Example:**
```python
from synglot.utils import filter_generated_content

generated = ["High quality text...", "Bad txt", "Another good text..."]
filtered = filter_generated_content(
    generated,
    min_length=20,
    min_quality_score=0.6,
    remove_duplicates=True
)

for result in filtered:
    print(f"Text: {result['text'][:50]}...")
    print(f"Quality: {result['quality_metrics']['overall_score']:.2f}")
```

### `extract_topics_from_text(text, max_topics=10)`
Extract potential topics from text content using keyword frequency analysis.

**Args:**
- `text` (`str`): Text to extract topics from.
- `max_topics` (`int`, optional): Maximum number of topics to return. Defaults to `10`.

**Returns:**
- `list[str]`: List of extracted topic strings.

**Example:**
```python
from synglot.utils import extract_topics_from_text

text = "Machine learning and artificial intelligence are transforming technology..."
topics = extract_topics_from_text(text, max_topics=5)
print(f"Extracted topics: {topics}")
```

## Usage Examples

### Configuration Management

```python
from synglot.utils import Config

# Create configuration with custom settings
config = Config({
    "generation_settings.default_temperature": 0.8,
    "generation_settings.pretraining.topic_prompt_template": "Write about {topic}:",
    "translation_settings.default_model_name": "custom-translator"
})

# Access nested configurations
temp = config.get("generation_settings.default_temperature")
topics = config.get("generation_settings.pretraining.general_topics_list", [])

# Update configuration
config.set("new_section.custom_param", "value")

# Save configuration
config.save("project_config.yaml")
```

### Text Processing Pipeline

```python
from synglot.utils import load_material_files, chunk_text, filter_generated_content

# Load source materials
materials = load_material_files(["docs/", "papers/research.txt"])

# Process each material
for material in materials:
    # Split into manageable chunks
    chunks = chunk_text(
        material["content"], 
        chunk_size=1000, 
        overlap=100,
        preserve_words=True
    )
    
    print(f"Split {material['file_name']} into {len(chunks)} chunks")
    
    # Process chunks with your generation pipeline
    generated_texts = []
    for chunk in chunks:
        # Your generation logic here
        generated_texts.extend(your_generator.generate_from_chunk(chunk["text"]))
    
    # Filter and clean generated content
    filtered_results = filter_generated_content(
        generated_texts,
        min_length=50,
        min_quality_score=0.6,
        remove_duplicates=True
    )
    
    print(f"Generated {len(generated_texts)} samples, kept {len(filtered_results)} after filtering")
```

### Batch Processing Workflow

```python
from synglot.utils import retrieve_batch
from synglot.translate import LLMTranslator
import time

# Setup OpenAI translator for batch processing
translator = LLMTranslator("en", "fr", backend="openai")

# Submit batch job
batch_job = translator.translate_dataset(
    dataset=large_dataset,
    columns_to_translate=["text", "description"],
    use_batch=True
)

print(f"Batch submitted: {batch_job['batch_id']}")

# Poll for completion
while True:
    results = retrieve_batch(translator.client, batch_job, batch_type="translation")
    
    if results and results.get("status") == "completed":
        print(f"Batch complete! Success rate: {results['success_rate']:.2%}")
        break
    elif results:
        print(f"Batch status: {results.get('status', 'unknown')}")
    
    time.sleep(30)  # Wait 30 seconds before checking again
``` 