# Synglot: Comprehensive Synthetic Data Generation and Translation Toolkit

[![Build Status](https://img.shields.io/travis/com/manifold-intelligence/synglot.svg)](https://travis-ci.com/manifold-intelligence/synglot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

> **Note: This project is currently under development. Many features may break and the generation functionality is not fully implemented. Features and APIs may change without notice. Use with caution. You're welcome to contribute!**


`Synglot` is a Python NLP toolkit for bulk machine translation, synthetic data generation, and multilingual dataset management. The goal is to make it extremely simple to bulk translate datasets in new languages, as well as synthetically generate new ones, in order to empower NLP research in low-resource languages.


## Roadmap
- implement further throttling to the OpenAI translator to avoid enqueued token limits, as well as TPD limits.
- add sentence splitting to the NLLB backend for higher quality.
- integrate the PDF scripts into the core Synglot library
- refactor and implement real generation functionality
- add OpenRouter support for more providers

## Key Features

- **Unified Translation**: Multi-backend translation supporting MarianMT, NLLB, OpenAI, and Google Translate APIs
- **Generation**: Synthetic data generation with HuggingFace and OpenAI backends
- **Powerful Dataset Management**: Comprehensive dataset handling with pandas-like operations
- **Batch Processing**: Process many examples at once with optimized performance (especially NLLB)
- **CLI Interface**: Full command-line interface for all operations

## Installation

### Basic Installation
```bash
git clone https://github.com/manifold-intelligence/synglot.git
cd synglot
# Install uv package manager if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

### Backend Dependencies

#### For OpenAI Support
```bash
pip install openai
export OPENAI_API_KEY="your-openai-api-key"
```

#### For Google Translate Support
```bash
pip install google-cloud-translate
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
# Set up authentication (see CLI guide for details)
```

#### For HuggingFace Models (MarianMT and NLLB)
```bash
uv pip install transformers torch
```

## Quick Start

### Translation Example
```python
from synglot.translate import LLMTranslator
from synglot.dataset import Dataset

# Load dataset
dataset = Dataset()
dataset.load_from_huggingface("squad", split="train[:100]", columns=["question", "context"])

# Initialize translator (supports 'marianmt', 'nllb', 'openai', 'google')
translator = LLMTranslator("en", "es", backend="nllb")

# Translate dataset with comprehensive error handling
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=["question", "context"],
    output_dir="./translations"
)

print(f"Translated {summary['successful_translations']} samples")
print(f"Success rate: {summary['success_rate']:.2%}")
```

## Command Line Interface

Synglot provides a comprehensive CLI for all operations:

### Translation Commands

#### Basic Translation
```bash
# Translate local dataset
python main.py translate \
  --dataset-path data.json \
  --columns text,title \
  --source-lang en \
  --target-lang es \
  --backend nllb

# Translate HuggingFace dataset
python main.py translate \
  --hf-dataset squad \
  --columns question,context \
  --source-lang en \
  --target-lang fr \
  --backend nllb \
  --max-samples 1000
```

#### Batch Translation (Optimized performance with NLLB, 50% cost reduction with OpenAI)
```bash
python main.py translate \
  --dataset-path large_dataset.json \
  --columns content \
  --source-lang en \
  --target-lang zh \
  --backend nllb \
  --use-batch
```

## Architecture

### Translation Backends
- **MarianMT**: Fast, offline translation (free, limited language pairs)
- **NLLB**: High-performance, 200+ languages, optimized batch processing (free, offline)
- **OpenAI**: High-quality translation with batch processing support
- **Google Translate**: 100+ languages, cost-effective for large volumes

### Generation Backends
- **HuggingFace**: Local models (Qwen, Llama, etc.) with full parameter control
- **OpenAI**: GPT models with async batch processing

### Dataset Management
- **Loading**: HuggingFace Hub, local JSON/CSV files
- **Operations**: Filter, map, sort, group, concatenate
- **Analysis**: Statistics, quality metrics, vocabulary analysis
- **Advanced Indexing**: Pandas-like row/column access

## API Documentation

Comprehensive API documentation is available:

- **[Main API Docs](./docs/api/README.md)**
- **[Translation Module](./docs/api/translate.md)** - Multi-backend translation with batch processing
- **[Generation Module](./docs/api/generate.md)** - Synthetic data generation with presets
- **[Dataset Module](./docs/api/dataset.md)** - Comprehensive data manipulation
- **[Utils Module](./docs/api/utils.md)** - Configuration, batch processing, text utilities
- **[Analysis Module](./docs/api/analyze.md)** - Dataset analysis and metrics

## Configuration

Synglot uses a powerful configuration system for fine-grained control:

```python
from synglot.utils import Config

# Create custom configuration
config = Config({
    "generation_settings.default_temperature": 0.8,
    "generation_settings.pretraining.topic_prompt_template": "Write about {topic}:",
    "translation_settings.default_model_name": "custom-model"
})

# Use presets for quick setup
from synglot.generate import LLMGenerator

creative_gen = LLMGenerator.from_preset("creative", "en", backend="openai")
precise_gen = LLMGenerator.from_preset("precise", "en", temperature=0.1)
```

## Backend Comparison

| Feature | MarianMT | NLLB | OpenAI | Google Translate |
|---------|----------|------|--------|------------------|
| **Setup** | Medium | Easy | Easy | Medium |
| **Cost** | Free | Free | Pay per token | Pay per character |
| **Quality** | Good | Excellent | Excellent | Excellent |
| **Speed** | Fast (local) | Very Fast (local) | Medium | Fast |
| **Languages** | Limited pairs | 200+ | Many | 100+ |
| **Batch** | Yes | Optimized | Yes (async) | Yes |
| **Offline** | Yes | Yes | No | No |
| **Best For** | Development, Privacy | High performance, Many languages | High quality, Complex | Cost-effective |

## Advanced Features

### Batch Processing with Cost Optimization (for OpenAI translations) and Performance Optimization (for NLLB)
```python
# NLLB batch processing (optimized performance, free)
translator = LLMTranslator("en", "zh", backend="nllb")
batch_job = translator.translate_dataset(
    dataset=large_dataset,
    columns_to_translate=["content"],
    use_batch=True,
    batch_size=1000  # Optimized for NLLB performance
)

# OpenAI batch processing (50% cost reduction)
translator = LLMTranslator("en", "zh", backend="openai")
batch_job = translator.translate_dataset(
    dataset=large_dataset,
    columns_to_translate=["content"],
    use_batch=True,
    batch_request_limit=40000,  # Optimize batch size
    batch_token_limit=1900000
)

# Monitor progress
results = translator.retrieve_batch(batch_job)
```



### Configuration-Driven Workflows
```python
# Load configuration from file
config = Config(config_file="project_config.yaml")

# Use configuration across components
generator = LLMGenerator("en", config=config)
translator = LLMTranslator("en", "fr", config=config)
```

## Performance Tips

1. **Use batch processing** for large datasets (optimized performance with NLLB, 50% cost reduction with OpenAI)
2. **Start with small samples** using `max_samples` parameter for testing
3. **Choose the right backend** based on your needs (quality vs cost vs speed vs language coverage)
4. **Use NLLB for maximum language coverage** and batch performance optimization
5. **Filter generated content** to maintain quality
6. **Use configuration presets** for consistent parameters

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Hugging Face** for their transformers and datasets libraries
- **Meta AI** for their No Language Left Behind (NLLB) models and research
- **OpenAI** for their powerful language models and batch API
- **Google Cloud** for their Translation API
- **University of Helsinki NLP group** for MarianMT models
- The open-source community for making projects like this possible

## Support

- **Documentation**: [API Docs](./docs/api/README.md)
- **Issues**: [GitHub Issues](https://github.com/manifold-intelligence/synglot/issues)
- **Examples**: Check the `docs/examples/` directory

---

**Ready to transform your NLP workflows?** Start with the quick examples above and explore the comprehensive API documentation for advanced usage.