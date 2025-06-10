# `synglot.generate`

This module provides classes for generating synthetic data using various backends.

## `Generator` (Base Class)

```python
class Generator:
    def __init__(self, target_lang, config=None)
    def generate(self, prompt=None, n_samples=1)
```

### `__init__(self, target_lang, config=None)`
Initializes the base generator.

**Args:**
- `target_lang` (`str`): Target language code.
- `config` (`dict` | `Config`, optional): Configuration parameters. If `None`, default `Config` is used.

### `generate(self, prompt=None, n_samples=1)`
Generate synthetic samples. This method must be implemented by subclasses.

**Args:**
- `prompt` (`str`, optional): Prompt to guide generation. Defaults to `None`.
- `n_samples` (`int`, optional): Number of samples to generate. Defaults to `1`.

## `LLMGenerator`

```python
class LLMGenerator(Generator):
    def __init__(self, target_lang, backend="huggingface", model_name=None, config=None, api_key=None, max_gen_tokens=1024, temperature=None, top_k=None, top_p=None, do_sample=None, max_new_tokens=None, min_length=None, return_full_text=None, seed=None)
    def generate(self, prompt=None, n_samples=1, **kwargs)
    def generate_from_material(self, material_paths, chunk_size=None, overlap=0, n_samples_per_chunk=3, output_path=None, output_dir="./outputs", save_to_file=False, **kwargs)
    def generate_pretraining(self, domain="general", n_samples=100, min_length=50, max_length=200, output_path=None, output_dir="./outputs", save_to_file=False)
    def generate_conversations(self, domain="general", n_samples=50, n_turns_min=2, n_turns_max=5, output_path=None, output_dir="./outputs", save_to_file=False)
    def generate_batch(self, prompts, batch_size=32, batch_job_description="batch generation")
    def retrieve_batch(self, saved_batch)
    @classmethod
    def from_preset(cls, preset_name, target_lang, backend="huggingface", **kwargs)
```

**Unified generator supporting HuggingFace models and OpenAI API with comprehensive synthetic data generation capabilities.**

### `__init__(self, target_lang, backend="huggingface", model_name=None, config=None, api_key=None, max_gen_tokens=1024, temperature=None, top_k=None, top_p=None, do_sample=None, max_new_tokens=None, min_length=None, return_full_text=None, seed=None)`
Initialize unified LLM generator with direct parameter access.

**Args:**
- `target_lang` (`str`): Target language code.
- `backend` (`str`, optional): Backend system. Supports `"huggingface"` or `"openai"`. Defaults to `"huggingface"`.
- `model_name` (`str`, optional): Model name. For HuggingFace: model identifier; For OpenAI: model name (e.g., 'gpt-4o'). Defaults to `"Qwen/Qwen2.5-1.5B-Instruct"` for HuggingFace, `"gpt-4.1-mini"` for OpenAI.
- `config` (`Config` | `dict`, optional): Configuration object or dictionary (fallback for advanced settings).
- `api_key` (`str`, optional): API key for OpenAI backend.
- `max_gen_tokens` (`int`, optional): Maximum tokens for generation (primary parameter for both backends). Defaults to `1024`.

**Direct Generation Parameters** (override config if specified):
- `temperature` (`float`, optional): Sampling temperature (0.0 to 2.0). Defaults to `1.0` for HuggingFace, `0.7` for OpenAI.
- `top_k` (`int`, optional): Top-k sampling parameter (HuggingFace only). Defaults to `50`.
- `top_p` (`float`, optional): Top-p (nucleus) sampling parameter. Defaults to `1.0`.
- `do_sample` (`bool`, optional): Whether to use sampling (HuggingFace only). Defaults to `True`.
- `max_new_tokens` (`int`, optional): Override max_gen_tokens for HuggingFace backend only.
- `min_length` (`int`, optional): Minimum total length (HuggingFace only).
- `return_full_text` (`bool`, optional): Whether to return full text including prompt (HuggingFace only). Defaults to `True`.
- `seed` (`int`, optional): Random seed for reproducible generation.

**Backend-specific Requirements:**
- **HuggingFace**: Requires `transformers` library. Model downloaded automatically.
- **OpenAI**: Requires `OPENAI_API_KEY` environment variable or `api_key` parameter.

### `generate(self, prompt=None, n_samples=1, **kwargs)`
Generate samples using the configured backend.

**Args:**
- `prompt` (`str`, optional): The prompt to generate from. Defaults to `None` (empty string).
- `n_samples` (`int`, optional): Number of samples to generate. Defaults to `1`.
- `**kwargs`: Additional generation parameters to override config.

**Returns:**
- `list[str]`: A list of generated text samples.

**Example:**
```python
from synglot.generate import LLMGenerator

# HuggingFace backend
generator = LLMGenerator("en", backend="huggingface")
samples = generator.generate("Write a story about", n_samples=3)

# OpenAI backend
generator = LLMGenerator("en", backend="openai", temperature=0.8)
samples = generator.generate("Explain quantum physics", n_samples=2)
```

### `generate_from_material(self, material_paths, chunk_size=None, overlap=0, n_samples_per_chunk=3, output_path=None, output_dir="./outputs", save_to_file=False, **kwargs)`
Generate synthetic data based on provided material files.

**Args:**
- `material_paths` (`str` | `list`): Path(s) to .txt or .md files containing source material.
- `chunk_size` (`int`, optional): Size of text chunks to process. If `None`, uses config default (1000).
- `overlap` (`int`, optional): Number of characters to overlap between chunks. Defaults to `0`.
- `n_samples_per_chunk` (`int`, optional): Number of samples to generate per chunk. Defaults to `3`.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated.
- `output_dir` (`str`, optional): Directory to save output. Defaults to `"./outputs"`.
- `save_to_file` (`bool`, optional): Whether to save results to file automatically. Defaults to `False`.
- `**kwargs`: Additional generation parameters.

**Returns:**
- `list[dict]`: List of generated samples with metadata (source_file, chunk_id, text, etc.).

**Example:**
```python
generator = LLMGenerator("en", backend="huggingface")
samples = generator.generate_from_material(
    material_paths=["docs/readme.md", "docs/guide.txt"],
    chunk_size=500,
    n_samples_per_chunk=2,
    save_to_file=True
)
```

### `generate_pretraining(self, domain="general", n_samples=100, min_length=50, max_length=200, output_path=None, output_dir="./outputs", save_to_file=False)`
Generate pretraining data using diversity settings from configuration.

**Args:**
- `domain` (`str`, optional): The domain for generation. `"general"` uses topic list. Defaults to `"general"`.
- `n_samples` (`int`, optional): Number of samples to generate. Defaults to `100`.
- `min_length` (`int`, optional): Minimum length of generated text. Defaults to `50`.
- `max_length` (`int`, optional): Maximum length of generated text. Defaults to `200`.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated.
- `output_dir` (`str`, optional): Directory to save output. Defaults to `"./outputs"`.
- `save_to_file` (`bool`, optional): Whether to save results to file automatically. Defaults to `False`.

**Returns:**
- `list[str]`: A list of generated pretraining texts.

**Example:**
```python
generator = LLMGenerator("en", backend="huggingface")
data = generator.generate_pretraining(
    domain="science",
    n_samples=50,
    min_length=100,
    save_to_file=True
)
```

### `generate_conversations(self, domain="general", n_samples=50, n_turns_min=2, n_turns_max=5, output_path=None, output_dir="./outputs", save_to_file=False)`
Generate multi-turn conversation data.

**Args:**
- `domain` (`str`, optional): Domain for conversations. Defaults to `"general"`.
- `n_samples` (`int`, optional): Number of conversations to generate. Defaults to `50`.
- `n_turns_min` (`int`, optional): Minimum number of turns per conversation. Defaults to `2`.
- `n_turns_max` (`int`, optional): Maximum number of turns per conversation. Defaults to `5`.
- `output_path` (`str`, optional): Full path for output file. If `None`, auto-generated.
- `output_dir` (`str`, optional): Directory to save output. Defaults to `"./outputs"`.
- `save_to_file` (`bool`, optional): Whether to save results to file automatically. Defaults to `False`.

**Returns:**
- `list[list[str]]`: List of conversations, each conversation is a list of turns.

**Example:**
```python
generator = LLMGenerator("en", backend="openai")
conversations = generator.generate_conversations(
    domain="customer_service",
    n_samples=20,
    n_turns_min=3,
    n_turns_max=7,
    save_to_file=True
)
```

### `generate_batch(self, prompts, batch_size=32, batch_job_description="batch generation")`
Generate samples for multiple prompts using batch processing.

**Args:**
- `prompts` (`list`): List of prompts to generate from.
- `batch_size` (`int`, optional): Batch size for processing. Defaults to `32`.
- `batch_job_description` (`str`, optional): Description for batch jobs (OpenAI). Defaults to `"batch generation"`.

**Returns:**
- **HuggingFace**: `list[str]` - List of generated texts (synchronous).
- **OpenAI**: Batch job object (use `retrieve_batch()` to get results when complete).

**Example:**
```python
prompts = ["Write about AI", "Explain climate change", "Describe quantum computing"]

# HuggingFace - immediate results
generator = LLMGenerator("en", backend="huggingface")
results = generator.generate_batch(prompts)

# OpenAI - async batch processing
generator = LLMGenerator("en", backend="openai")
batch_job = generator.generate_batch(prompts, batch_job_description="Tech topics")
# Later: results = generator.retrieve_batch(batch_job)
```

### `retrieve_batch(self, saved_batch)`
Retrieve batch generation results. Only available for OpenAI backend.

**Args:**
- `saved_batch`: Batch job object returned from `generate_batch`.

**Returns:**
- File content if completed, `None` if still in progress.

**Raises:**
- `NotImplementedError`: If called with non-OpenAI backend.

### `from_preset(cls, preset_name, target_lang, backend="huggingface", **kwargs)` (Class Method)
Create generator from a preset configuration.

**Args:**
- `preset_name` (`str`): Name of preset. Available: `"creative"`, `"precise"`, `"balanced"`, `"fast"`.
- `target_lang` (`str`): Target language code.
- `backend` (`str`, optional): Backend system. Defaults to `"huggingface"`.
- `**kwargs`: Override any preset parameters.

**Returns:**
- `LLMGenerator`: Configured generator instance.

**Preset Configurations:**
- **`"creative"`**: High temperature (1.2), diverse sampling for creative content.
- **`"precise"`**: Low temperature (0.3), focused sampling for factual content.
- **`"balanced"`**: Medium temperature (0.8), balanced approach.
- **`"fast"`**: Greedy decoding, optimized for speed.

**Example:**
```python
# Create a creative generator
generator = LLMGenerator.from_preset("creative", "en", backend="openai")

# Override preset parameters
generator = LLMGenerator.from_preset("precise", "en", temperature=0.1, max_gen_tokens=500)
```

## Usage Examples

### Basic Text Generation

```python
from synglot.generate import LLMGenerator

# HuggingFace model
generator = LLMGenerator("en", backend="huggingface", model_name="microsoft/DialoGPT-medium")
texts = generator.generate("Tell me about artificial intelligence", n_samples=3)

# OpenAI model with custom parameters
generator = LLMGenerator("en", backend="openai", model_name="gpt-4o", temperature=0.8)
texts = generator.generate("Write a poem about nature", n_samples=2)
```

### Material-Based Generation

```python
# Generate synthetic data from existing documents
generator = LLMGenerator("en", backend="huggingface")
samples = generator.generate_from_material(
    material_paths=["documentation/", "research_papers/"],
    chunk_size=800,
    n_samples_per_chunk=5,
    save_to_file=True,
    temperature=0.9
)
print(f"Generated {len(samples)} samples from source materials")
```

### Domain-Specific Data Generation

```python
# Generate pretraining data for a specific domain
generator = LLMGenerator("en", backend="openai")
pretraining_data = generator.generate_pretraining(
    domain="medical",
    n_samples=200,
    min_length=100,
    max_length=300,
    save_to_file=True
)

# Generate conversation data
conversations = generator.generate_conversations(
    domain="customer_support",
    n_samples=100,
    n_turns_min=4,
    n_turns_max=8,
    save_to_file=True
)
```

### Batch Processing

```python
# Prepare multiple prompts
prompts = [
    "Explain machine learning",
    "Describe renewable energy",
    "Write about space exploration",
    "Discuss climate change solutions"
]

# Batch generation with OpenAI
generator = LLMGenerator("en", backend="openai")
batch_job = generator.generate_batch(prompts, batch_job_description="Educational content")

# Check and retrieve results later
results = generator.retrieve_batch(batch_job)
if results:
    print(f"Generated {len(results)} responses")
```

### Using Presets

```python
# Creative writing
creative_gen = LLMGenerator.from_preset("creative", "en", backend="huggingface")
stories = creative_gen.generate("Once upon a time", n_samples=5)

# Factual content
precise_gen = LLMGenerator.from_preset("precise", "en", backend="openai")
facts = precise_gen.generate("Explain photosynthesis", n_samples=3)

# Fast generation for large-scale tasks
fast_gen = LLMGenerator.from_preset("fast", "en", max_gen_tokens=100)
quick_responses = fast_gen.generate_batch(many_prompts)
``` 