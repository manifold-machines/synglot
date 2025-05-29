# Synglot CLI Usage Guide

The Synglot CLI (`main.py`) provides a command-line interface to access all the functionality of the Synglot library for translation and synthetic data generation.

## Installation

Make sure you have the synglot library installed and its dependencies:

```bash
pip install -e .
```

### Additional Dependencies for Backends

#### Google Translate Backend
```bash
pip install google-cloud-translate
```

#### OpenAI Backend  
```bash
pip install openai
```

#### MarianMT Backend (HuggingFace)
```bash
pip install transformers torch
```

## Basic Usage

```bash
python main.py <command> [options]
```

Available commands:
- `translate` - Translate datasets between languages
- `generate` - Generate synthetic data

## Backend Setup

### Google Translate Setup

You have several authentication options:

#### Option A: Application Default Credentials (Recommended for Development)
```bash
# Install Google Cloud CLI if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth application-default login

# Enable the Translation API
gcloud services enable translate.googleapis.com

# Set your project ID
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

#### Option B: Service Account Key
1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

### OpenAI Setup
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### MarianMT Setup
No additional setup required - models are downloaded automatically.

## Translation Command

### Basic Translation Examples

#### 1. Translate a local JSON file (using default OpenAI backend)
```bash
python main.py translate \
  --dataset-path data.json \
  --columns text,title \
  --source-lang en \
  --target-lang es
```

#### 2. Translate using Google Translate
```bash
python main.py translate \
  --dataset-path data.csv \
  --columns description \
  --source-lang en \
  --target-lang fr \
  --backend google \
  --output-path translations_fr.jsonl
```

#### 3. Translate using MarianMT (offline)
```bash
python main.py translate \
  --dataset-path data.json \
  --columns text \
  --source-lang en \
  --target-lang de \
  --backend marianmt \
  --batch-size 16
```

#### 4. Translate from HuggingFace datasets
```bash
python main.py translate \
  --hf-dataset wmt16 \
  --hf-config de-en \
  --hf-split validation \
  --columns translation.en \
  --source-lang en \
  --target-lang de \
  --max-samples 100
```

#### 5. Use OpenAI batch translation for large datasets
```bash
python main.py translate \
  --dataset-path data.json \
  --columns text \
  --source-lang en \
  --target-lang zh \
  --backend openai \
  --use-batch
```

### Translation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset-path` | Path to local dataset file (JSON/CSV) | - |
| `--hf-dataset` | HuggingFace dataset name | - |
| `--hf-config` | HF dataset configuration/subset | - |
| `--hf-split` | HF dataset split | train |
| `--hf-columns` | Columns to load from HF dataset | all |
| `--columns` | Columns to translate (required) | - |
| `--source-lang` | Source language code (required) | - |
| `--target-lang` | Target language code (required) | - |
| `--backend` | Translation backend | openai |
| `--model-name` | Specific model name | auto |
| `--output-path` | Output file path | auto-generated |
| `--output-dir` | Output directory | ./outputs |
| `--batch-size` | Batch size for translation | 32 |
| `--use-batch` | Use batch translation | false |
| `--max-samples` | Limit samples for testing | unlimited |

## Generation Command

### Basic Generation Examples

#### 1. General text generation (using default OpenAI backend)
```bash
python main.py generate \
  --target-lang es \
  --prompt "Write about artificial intelligence" \
  --n-samples 50 \
  --output ai_texts_es.json
```

#### 2. Pretraining data generation
```bash
python main.py generate \
  --target-lang fr \
  --mode pretraining \
  --domain science \
  --n-samples 100 \
  --min-length 100 \
  --max-length 500
```

#### 3. Conversation generation
```bash
python main.py generate \
  --target-lang de \
  --mode conversation \
  --domain technology \
  --n-samples 20 \
  --n-turns-min 3 \
  --n-turns-max 7
```

#### 4. Material-based generation
```bash
python main.py generate \
  --target-lang ja \
  --mode material \
  --material-path ./docs/*.txt \
  --n-samples-per-chunk 3 \
  --chunk-size 1000
```

#### 5. Use HuggingFace for generation
```bash
python main.py generate \
  --target-lang es \
  --backend huggingface \
  --prompt "Explain quantum computing" \
  --n-samples 10 \
  --temperature 0.7
```

### Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-lang` | Target language code (required) | - |
| `--mode` | Generation mode | general |
| `--prompt` | Prompt for generation | - |
| `--n-samples` | Number of samples | 10 |
| `--material-path` | Path to material files (supports wildcards) | - |
| `--chunk-size` | Size of text chunks for material processing | - |
| `--n-samples-per-chunk` | Samples per chunk for material mode | 3 |
| `--domain` | Domain for pretraining/conversation | general |
| `--min-length` | Min length for pretraining | 50 |
| `--max-length` | Max length for pretraining | 200 |
| `--n-turns-min` | Min conversation turns | 2 |
| `--n-turns-max` | Max conversation turns | 5 |
| `--backend` | Generation backend | openai |
| `--model-name` | Specific model name | auto |
| `--temperature` | Sampling temperature | 1.0 |
| `--max-new-tokens` | Max new tokens to generate | 100 |
| `--output` | Output file path | auto-generated |
| `--output-dir` | Output directory | ./outputs |
| `--format` | Output format | jsonl |
| `--use-batch` | Use batch generation | false |
| `--batch-size` | Batch size for HuggingFace | 32 |
| `--batch-job-description` | Description for OpenAI batch jobs | CLI batch generation |

## Generation Modes

### 1. General Mode (`--mode general`)
Generate text based on a prompt or without prompts for general-purpose text.

```bash
python main.py generate \
  --target-lang en \
  --mode general \
  --prompt "The future of renewable energy" \
  --n-samples 20
```

### 2. Pretraining Mode (`--mode pretraining`)
Generate diverse text suitable for language model pretraining.

```bash
python main.py generate \
  --target-lang es \
  --mode pretraining \
  --domain literature \
  --n-samples 1000 \
  --min-length 200 \
  --max-length 800
```

### 3. Conversation Mode (`--mode conversation`)
Generate multi-turn conversations.

```bash
python main.py generate \
  --target-lang fr \
  --mode conversation \
  --domain education \
  --n-samples 50 \
  --n-turns-min 4 \
  --n-turns-max 8
```

### 4. Material Mode (`--mode material`)
Generate content based on existing material files.

```bash
python main.py generate \
  --target-lang de \
  --mode material \
  --material-path ./knowledge_base/*.txt \
  --chunk-size 1500 \
  --n-samples-per-chunk 4 \
  --format jsonl
```

## Backend Comparison

| Feature | MarianMT | OpenAI | Google Translate |
|---------|----------|--------|------------------|
| **Setup Complexity** | Medium | Easy | Medium |
| **Cost** | Free (local) | Pay per token | Pay per character |
| **Quality** | Good | Excellent | Excellent |
| **Speed** | Fast (local) | Medium | Fast |
| **Language Support** | Limited pairs | Many | 100+ languages |
| **Batch Processing** | Yes | Yes (async) | Yes |
| **Internet Required** | No | Yes | Yes |
| **Best For** | Development, Privacy | High quality, Complex tasks | Many languages, Cost-effective |

### When to Use Each Backend

- **MarianMT**: Development, privacy-sensitive data, offline usage
- **OpenAI**: Highest quality translations, complex texts, batch processing
- **Google Translate**: Wide language support, cost-effective for large volumes

## Output Formats

### JSON (`--format json`)
Default format, saves all data as a JSON array:
```json
[
  {
    "text": "Generated text...",
    "prompt": "Original prompt",
    "domain": "science"
  }
]
```

### JSONL (`--format jsonl`)
One JSON object per line:
```json
{"text": "Generated text 1...", "prompt": "Original prompt"}
{"text": "Generated text 2...", "prompt": "Original prompt"}
```

### TXT (`--format txt`)
Plain text format with double newlines between samples:
```
Generated text 1...

Generated text 2...
```

## Language Codes

Use standard ISO 639-1 language codes:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | Chinese (Simplified) | zh |
| Spanish | es | Japanese | ja |
| French | fr | Korean | ko |
| German | de | Arabic | ar |
| Portuguese | pt | Russian | ru |
| Italian | it | Hindi | hi |

For Google Translate, see the [complete list of supported languages](https://cloud.google.com/translate/docs/languages).

## Global Options

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Enable verbose output |
| `--config CONFIG` | Path to configuration file |

## Examples Workflow

### Complete Translation Workflow

1. **Test with a small sample using Google Translate:**
```bash
python main.py translate \
  --hf-dataset squad \
  --columns question,context \
  --source-lang en \
  --target-lang es \
  --backend google \
  --max-samples 10 \
  --output-path sample_translation.jsonl
```

2. **Scale up with OpenAI batch processing:**
```bash
python main.py translate \
  --dataset-path my_data.json \
  --columns title,description \
  --source-lang en \
  --target-lang fr \
  --backend openai \
  --use-batch \
  --output-dir ./french_translations/
```

3. **Use MarianMT for offline processing:**
```bash
python main.py translate \
  --dataset-path sensitive_data.json \
  --columns content \
  --source-lang en \
  --target-lang de \
  --backend marianmt \
  --batch-size 16
```

### Complete Generation Workflow

1. **Generate training data:**
```bash
python main.py generate \
  --target-lang es \
  --mode pretraining \
  --domain medical \
  --n-samples 500 \
  --min-length 150 \
  --max-length 600 \
  --format jsonl \
  --output medical_pretraining_es.jsonl
```

2. **Generate conversations:**
```bash
python main.py generate \
  --target-lang fr \
  --mode conversation \
  --domain customer_service \
  --n-samples 100 \
  --n-turns-min 3 \
  --n-turns-max 6 \
  --format json \
  --output customer_conversations_fr.json
```

3. **Generate from existing materials:**
```bash
python main.py generate \
  --target-lang de \
  --mode material \
  --material-path ./knowledge_base/*.txt \
  --chunk-size 1500 \
  --n-samples-per-chunk 4 \
  --format jsonl
```

## Cost Optimization Tips

### For Google Translate
- Use batch processing to minimize API calls
- Monitor character usage in Google Cloud Console
- Consider the free tier limits for development

### For OpenAI
- Use batch API (`--use-batch`) for 50% cost reduction on large datasets
- Monitor token usage and set appropriate `max-new-tokens`
- Use cheaper models for development/testing

### For MarianMT
- Free to use but requires local compute resources
- Consider GPU acceleration for faster processing
- Best for privacy-sensitive or offline scenarios

## Tips

1. **Testing with small samples:** Use `--max-samples` to test with a small subset before running on large datasets.

2. **Batch processing:** Use `--use-batch` for faster translation and generation when processing many samples.

3. **Output organization:** Use `--output-dir` to organize outputs by language or task.

4. **Memory management:** For large datasets, consider processing in chunks by using `--max-samples` with multiple runs.

5. **Quality vs Speed vs Cost:** 
   - MarianMT: Fast and free but limited quality in certain languages
   - OpenAI: High quality but costs per token
   - Google: Good quality, wide language support, but steeper costs than OpenAI

6. **Authentication:** Set up environment variables for backends to avoid passing credentials in commands.

## Error Handling

The CLI includes comprehensive error handling:
- Invalid datasets or missing columns are reported clearly
- Translation errors are logged and saved separately
- Generation failures include detailed error messages
- Progress tracking shows success/failure rates
- Authentication issues provide helpful guidance

## Troubleshooting

### Google Translate Issues
- **Permission Denied**: Ensure your account has the `Cloud Translation API User` role
- **Project Not Found**: Verify project ID and API enablement
- **Quota Exceeded**: Check API quotas in Google Cloud Console

### OpenAI Issues
- **API Key**: Verify `OPENAI_API_KEY` environment variable
- **Rate Limits**: Use `--use-batch` for large datasets
- **Model Access**: Ensure your account has access to the specified model

### MarianMT Issues
- **Memory**: Reduce `--batch-size` if running out of memory
- **Model Download**: Ensure internet connection for initial model downloads
- **Language Pairs**: Check if the language pair is supported

For more advanced usage and programmatic access, refer to the main Synglot library documentation. 