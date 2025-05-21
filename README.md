# Synglot: Synthetic Data Generation and Translation Toolkit

[![Build Status](https://img.shields.io/travis/com/manifold-intelligence/synglot.svg)](https://travis-ci.com/manifold-intelligence/synglot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

`Synglot` is a Python library designed to empower researchers and developers in the field of Natural Language Processing (NLP) by providing a flexible and extensible toolkit for:

- **Synthetic Data Generation**: Create diverse datasets for pretraining, instruction tuning, or specific NLP tasks using various generation strategies and models.
- **Machine Translation**: Easily translate text between multiple languages using both open-source Hugging Face models and commercial LLM APIs.
- **Dataset Management**: Handle and manipulate multilingual datasets efficiently, with support for loading from Hugging Face Hub, local files, and saving in common formats.
- **Text Analysis**: Analyze your datasets to understand their characteristics, including length distributions, token counts, and vocabulary.

## Roadmap

- **Implement `LLMTranslator`**: Integrate support for various commercial LLM APIs for translation.
- **Implement `OpenAIGenerator`**: Add a generator backend for API models (e.g., GPT-4o, Claude, Gemini).
- **More Cookbooks**: Develop and document more practical implementations of and use-cases for `synglot`.

## Key Features

- **Modular Design**: Easily extendable components for generation, translation, and data handling.
- **Configurable Generation**: Fine-tune data generation with a flexible configuration system (`synglot.utils.Config`), supporting various parameters like temperature, top-k, and custom prompts.
- **Hugging Face Integration**: Seamlessly use models from the Hugging Face Hub for both generation (`HFGenerator`) and translation (`StandardTranslator`, `HFTranslator`).
- **LLM API Support (Planned)**: Designed to incorporate translators (`LLMTranslator`) and generators (`OpenAIGenerator`, `DeepSeekGenerator`) that leverage commercial LLM APIs.
- **Versatile Dataset Handling**: `synglot.Dataset` allows loading data from Hugging Face Hub (including specific configurations), local JSON/CSV files, and saving datasets.
- **Built-in Analysis**: The `synglot.analyze.Analyzer` provides tools to get statistics, length distributions, and vocabulary information from your datasets.
- **Continuous Output**: Example scripts demonstrate how to save processed data continuously, ideal for long-running tasks.

## Installation

`Synglot` uses `uv` for package management. If you don't have `uv` installed, you can install it by following the official instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

**From Source (Recommended for development):**
```bash
git clone https://github.com/manifold-intelligence/synglot.git
cd synglot
# To install uv on macOS and Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start: Translating a Dataset

Here's a brief example of how to use `synglot` to translate the "gsm8k" dataset from English to Macedonian and save the results. For the full script, see `docs/examples/translate_gsm8k_to_mk.py`.

```python
from synglot.dataset import Dataset
from synglot.translate import StandardTranslator
import os
import json

def run_translation_example():
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "gsm8k_mk_readme_example.jsonl")

    source_lang = "en"
    target_lang = "mk"

    print(f"Initializing translator for {source_lang} to {target_lang}...")
    translator = StandardTranslator(source_lang=source_lang, target_lang=target_lang)
    print("Translator initialized.")

    print("Loading gsm8k dataset...")
    dataset = Dataset()
    # Using a small subset for the README example
    dataset.load_from_huggingface("gsm8k", config_name="main", split="train[:1%]") 
    print(f"Dataset loaded with {len(dataset)} samples.")

    print(f"Starting translation, saving to {output_file_path}...")
    with open(output_file_path, 'a', encoding='utf-8') as f_out:
        for i, item in enumerate(dataset):
            original_question = item.get("question")
            original_answer = item.get("answer")

            if not original_question:
                continue
            
            print(f"Translating sample {i+1}/{len(dataset)}...")
            translated_question = translator.translate(original_question)
            
            output_record = {
                "original_question": original_question,
                "translated_question_mk": translated_question,
                "original_answer": original_answer
            }
            f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    print("Translation example finished.")

if __name__ == "__main__":
    run_translation_example()
```

To run the full example script which includes more robust error handling and processing of the entire dataset:
```bash
python docs/examples/translate_gsm8k_to_mk.py
```

## API Documentation

Detailed API documentation for each module can be found in the `docs/api` directory:

- **[Main API Docs](./docs/api/README.md)**
- [Dataset Module](./docs/api/dataset.md)
- [Translate Module](./docs/api/translate.md)
- [Generate Module](./docs/api/generate.md)
- [Analyze Module](./docs/api/analyze.md)
- [Utils Module (Config)](./docs/api/utils.md)

## Configuration

`Synglot` uses a `Config` class (`synglot.utils.Config`) for managing settings, especially for data generation. This allows for detailed control over parameters like model choice, generation hyperparameters (temperature, top_k, top_p), prompt templates, and more.

Configurations can be loaded from YAML files or Python dictionaries. The default configuration includes settings for:
- Seed for reproducibility
- General generation settings (temperature, token limits, etc.)
- Pretraining data generation strategies (e.g., topic-based prompting)
- Conversational data generation (speaker prefixes, turn length)

See `synglot/utils/config.py` for the default structure and `docs/api/utils.md` for API details.

## Supported Backends

### Translation
- **`StandardTranslator`**: Primarily uses MarianMT models from Hugging Face (e.g., `Helsinki-NLP/opus-mt-{src}-{tgt}`).
- **`HFTranslator`**: Generic Hugging Face model translator (implementation in progress).

### Generation
- **`HFGenerator`**: Uses text-generation models from Hugging Face (e.g. Qwen, Llama).
- **`OpenAIGenerator`**: For OpenAI models (implementation in progress).

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to the existing style and includes tests where appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Hugging Face for their `transformers` and `datasets` libraries.
- The NLP group at the University of Helsinki for their work on the MarianMT models, which are used as the default for the `StandardTranslator`.
- The open-source community for the models and tools that make projects like this possible.