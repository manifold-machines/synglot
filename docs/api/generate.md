# `synglot.generate`

This module provides classes for generating synthetic data.

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

## `HFGenerator`

```python
class HFGenerator(Generator):
    def __init__(self, target_lang, model_name=None, config=None)
    def generate(self, prompt=None, n_samples=1, **kwargs)
    def generate_pretraining(self, domain="general", n_samples=100, min_length=50, max_length=200)
    def generate_conversations(self, domain="general", n_samples=50, n_turns_min=2, n_turns_max=5)
```

### `__init__(self, target_lang, model_name=None, config=None)`
Initializes the HuggingFace generator.

**Args:**
- `target_lang` (`str`): Target language code.
- `model_name` (`str`, optional): HuggingFace model name. Can also be set in config. Defaults to a Qwen model.
- `config` (`Config` | `dict`, optional): Configuration object or dictionary.

### `generate(self, prompt=None, n_samples=1, **kwargs)`
Generate samples using a HuggingFace model.

**Args:**
- `prompt` (`str`, optional): The prompt to generate from. Defaults to `None` (empty string).
- `n_samples` (`int`, optional): Number of samples to generate. Defaults to `1`.
- `**kwargs`: Additional generation parameters to override config or pass to the pipeline.

**Returns:**
- `list[str]`: A list of generated text samples.

### `generate_pretraining(self, domain="general", n_samples=100, min_length=50, max_length=200)`
Generate pretraining data using diversity settings from the configuration.

**Args:**
- `domain` (`str`, optional): The domain for generation. `"general"` uses a topic list. Defaults to `"general"`.
- `n_samples` (`int`, optional): Number of samples to generate. Defaults to `100`.
- `min_length` (`int`, optional): Minimum length of generated text. Defaults to `50`.
- `max_length` (`int`, optional): Maximum length of generated text. Defaults to `200`.

**Returns:**
- `list[str]`: A list of generated pretraining texts.

### `generate_conversations(self, domain="general", n_samples=50, n_turns_min=2, n_turns_max=5)`
Generate multi-turn conversation data.

**Args:**
- `domain` (`str`, optional): The domain for the conversation. Defaults to `"general"`.
- `n_samples` (`int`, optional): Number of conversations to generate. Defaults to `50`.
- `n_turns_min` (`int`, optional): Minimum number of turns per conversation. Defaults to `2`.
- `n_turns_max` (`int`, optional): Maximum number of turns per conversation. Defaults to `5`.

**Returns:**
- `list[list[str]]`: A list of conversations, where each conversation is a list of utterances.

## `OpenAIGenerator`

*Note: Implementation details for `OpenAIGenerator` are not yet available in the provided code.*

```python
class OpenAIGenerator(Generator):
    def __init__(self, target_lang, api_key=None, model_name="gpt-4", config=None)
    # Methods (generate, generate_pretraining, generate_conversations) are defined but not implemented.
``` 