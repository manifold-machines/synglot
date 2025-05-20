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