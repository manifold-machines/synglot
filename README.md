# Synglot: Comprehensive Synthetic Data Generation and Translation Toolkit

[![Build Status](https://img.shields.io/travis/com/manifold-intelligence/synglot.svg)](https://travis-ci.com/manifold-intelligence/synglot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

`Synglot` is a powerful Python toolkit designed to empower researchers and developers in Natural Language Processing (NLP) with comprehensive capabilities for synthetic data generation, machine translation, and multilingual dataset management.

## 🚀 Key Features

- **🔄 Unified Translation**: Multi-backend translation supporting MarianMT, OpenAI, and Google Translate APIs
- **✨ Advanced Generation**: Sophisticated synthetic data generation with HuggingFace and OpenAI backends
- **📊 Powerful Dataset Management**: Comprehensive dataset handling with pandas-like operations
- **⚡ Batch Processing**: Efficient batch processing with OpenAI's async batch API (50% cost reduction)
- **🛠️ CLI Interface**: Full command-line interface for all operations
- **🔧 Flexible Configuration**: Rich configuration system with YAML support
- **🔍 Built-in Analysis**: Dataset analysis and quality metrics
- **🌍 Multilingual**: Support for 100+ languages through various backends

## 📦 Installation

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

#### For HuggingFace Models
```bash
pip install transformers torch
```

## 🎯 Quick Start

### Translation Example
```python
from synglot.translate import LLMTranslator
from synglot.dataset import Dataset

# Load dataset
dataset = Dataset()
dataset.load_from_huggingface("squad", split="train[:100]", columns=["question", "context"])

# Initialize translator (supports 'marianmt', 'openai', 'google')
translator = LLMTranslator("en", "es", backend="openai")

# Translate dataset with comprehensive error handling
summary = translator.translate_dataset(
    dataset=dataset,
    columns_to_translate=["question", "context"],
    output_dir="./translations"
)

print(f"Translated {summary['successful_translations']} samples")
print(f"Success rate: {summary['success_rate']:.2%}")
```

### Generation Example
```python
from synglot.generate import LLMGenerator

# Initialize generator (supports 'huggingface', 'openai')
generator = LLMGenerator("es", backend="openai", temperature=0.8)

# Generate diverse training data
pretraining_data = generator.generate_pretraining(
    domain="science",
    n_samples=100,
    min_length=200,
    max_length=500,
    save_to_file=True
)

# Generate conversations
conversations = generator.generate_conversations(
    domain="customer_service",
    n_samples=50,
    n_turns_min=3,
    n_turns_max=7,
    save_to_file=True
)

# Generate from existing materials
samples = generator.generate_from_material(
    material_paths=["docs/*.md", "papers/*.txt"],
    chunk_size=1000,
    n_samples_per_chunk=3,
    save_to_file=True
)
```

### Dataset Operations
```python
from synglot.dataset import Dataset

# Load and manipulate data
dataset = Dataset()
dataset.load_from_huggingface("imdb", columns=["text", "label"])

# Powerful data operations
positive_reviews = dataset.filter(lambda row: row['label'] == 1)
clean_data = dataset.map(lambda text: text.lower(), columns=['text'])
sample_data = dataset.sample(100, random_state=42)

# Advanced indexing
first_10_texts = dataset[0:10, 'text']
specific_columns = dataset[['text', 'label']]

# Analysis
dataset.info()
dataset.describe()
label_counts = dataset.value_counts('label')
```

## 💻 Command Line Interface

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
  --backend openai

# Translate HuggingFace dataset
python main.py translate \
  --hf-dataset squad \
  --columns question,context \
  --source-lang en \
  --target-lang fr \
  --backend google \
  --max-samples 1000
```

#### Batch Translation (50% cost reduction with OpenAI)
```bash
python main.py translate \
  --dataset-path large_dataset.json \
  --columns content \
  --source-lang en \
  --target-lang zh \
  --backend openai \
  --use-batch
```

### Generation Commands

#### Pretraining Data Generation
```bash
python main.py generate \
  --target-lang es \
  --mode pretraining \
  --domain science \
  --n-samples 1000 \
  --min-length 200 \
  --max-length 500
```

#### Conversation Generation
```bash
python main.py generate \
  --target-lang fr \
  --mode conversation \
  --domain technology \
  --n-samples 100 \
  --n-turns-min 3 \
  --n-turns-max 7
```

#### Material-Based Generation
```bash
python main.py generate \
  --target-lang de \
  --mode material \
  --material-path ./docs/*.txt \
  --chunk-size 1500 \
  --n-samples-per-chunk 4
```

## 🏗️ Architecture

### Translation Backends
- **MarianMT**: Fast, offline translation (free, limited language pairs)
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

## 📚 API Documentation

Comprehensive API documentation is available:

- **[Main API Docs](./docs/api/README.md)**
- **[Translation Module](./docs/api/translate.md)** - Multi-backend translation with batch processing
- **[Generation Module](./docs/api/generate.md)** - Synthetic data generation with presets
- **[Dataset Module](./docs/api/dataset.md)** - Comprehensive data manipulation
- **[Utils Module](./docs/api/utils.md)** - Configuration, batch processing, text utilities
- **[Analysis Module](./docs/api/analyze.md)** - Dataset analysis and metrics

## 🔧 Configuration

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

## 🎛️ Backend Comparison

| Feature | MarianMT | OpenAI | Google Translate |
|---------|----------|--------|------------------|
| **Setup** | Medium | Easy | Medium |
| **Cost** | Free | Pay per token | Pay per character |
| **Quality** | Good | Excellent | Excellent |
| **Speed** | Fast (local) | Medium | Fast |
| **Languages** | Limited pairs | Many | 100+ |
| **Batch** | Yes | Yes (async) | Yes |
| **Offline** | Yes | No | No |
| **Best For** | Development, Privacy | High quality, Complex | Many languages, Cost-effective |

## 📋 Examples

### Complete Translation Workflow
```python
from synglot.dataset import Dataset
from synglot.translate import LLMTranslator

# Load dataset
dataset = Dataset()
dataset.load_from_huggingface("gsm8k", split="train", columns=["question", "answer"])

# Multi-step translation workflow
translator = LLMTranslator("en", "es", backend="openai")

# 1. Test with small sample
sample = dataset.sample(10)
test_summary = translator.translate_dataset(
    dataset=sample,
    columns_to_translate=["question"]
)

# 2. Scale up with batch processing
if test_summary['success_rate'] > 0.9:
    batch_job = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=["question", "answer"],
        use_batch=True  # 50% cost reduction
    )
    
    # 3. Retrieve results when complete
    results = translator.retrieve_batch(batch_job)
```

### Advanced Generation Pipeline
```python
from synglot.generate import LLMGenerator
from synglot.utils import load_material_files, chunk_text, filter_generated_content

# Material-based generation with quality filtering
generator = LLMGenerator("fr", backend="huggingface", model_name="microsoft/DialoGPT-medium")

# Load and process materials
materials = load_material_files(["docs/*.md", "papers/*.txt"])

all_generated = []
for material in materials:
    # Split into chunks
    chunks = chunk_text(material["content"], chunk_size=1000, overlap=100)
    
    # Generate from each chunk
    for chunk in chunks:
        generated = generator.generate(
            prompt=f"Based on this text: {chunk['text'][:200]}...",
            n_samples=3,
            temperature=0.8
        )
        all_generated.extend(generated)

# Filter for quality
filtered = filter_generated_content(
    all_generated,
    min_length=50,
    min_quality_score=0.7,
    remove_duplicates=True
)

print(f"Generated {len(all_generated)} samples, kept {len(filtered)} after filtering")
```

## 🔍 Advanced Features

### Batch Processing with Cost Optimization
```python
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

### Quality-Aware Generation
```python
from synglot.utils import filter_generated_content

# Generate with quality filtering
generator = LLMGenerator("es", backend="openai")
raw_generated = generator.generate("Write about AI", n_samples=100)

# Filter for high-quality content
filtered = filter_generated_content(
    raw_generated,
    min_quality_score=0.8,
    remove_duplicates=True,
    similarity_threshold=0.9
)
```

### Configuration-Driven Workflows
```python
# Load configuration from file
config = Config(config_file="project_config.yaml")

# Use configuration across components
generator = LLMGenerator("en", config=config)
translator = LLMTranslator("en", "fr", config=config)
```

## 🚀 Performance Tips

1. **Use batch processing** for large datasets (50% cost reduction with OpenAI)
2. **Start with small samples** using `max_samples` parameter for testing
3. **Choose the right backend** based on your needs (quality vs cost vs speed)
4. **Filter generated content** to maintain quality
5. **Use configuration presets** for consistent parameters

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **Hugging Face** for their transformers and datasets libraries
- **OpenAI** for their powerful language models and batch API
- **Google Cloud** for their Translation API
- **University of Helsinki NLP group** for MarianMT models
- The open-source community for making projects like this possible

## 📞 Support

- **Documentation**: [API Docs](./docs/api/README.md)
- **Issues**: [GitHub Issues](https://github.com/manifold-intelligence/synglot/issues)
- **Examples**: Check the `docs/examples/` directory

---

**Ready to transform your NLP workflows?** Start with the quick examples above and explore the comprehensive API documentation for advanced usage.