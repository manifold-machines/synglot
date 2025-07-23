#!/usr/bin/env python3
"""
Synglot CLI - Command Line Interface for Synthetic Data Generation and Translation Toolkit

Usage examples:

Translation:
  
# Translate all columns automatically
python main.py translate --hf-dataset wmt16 --target-lang mk --source-lang en --backend openai
  
# Use batch translation for better performance
python main.py translate --hf-dataset wmt16 --target-lang mk --source-lang en --backend openai --use-batch --batch-size 50

# For large datasets exceeding 50,000 requests, the system automatically splits into multiple batch jobs
python main.py translate --hf-dataset opus100 --target-lang fr --source-lang en --backend openai --use-batch

# Customize the batch request limit (useful if OpenAI changes their limits)
python main.py translate --hf-dataset large_dataset --target-lang es --source-lang en --backend openai --use-batch --batch-request-limit 30000

# Use streaming mode for very large datasets to reduce memory usage
python main.py translate --hf-dataset very_large_dataset --target-lang fr --source-lang en --backend nllb --streaming-mode --batch-size 16

# Translate nested fields like Q&A datasets
python main.py translate --hf-dataset squad --columns qa.question,qa.answer --target-lang es --source-lang en --backend openai

# Use NLLB with GPU and auto batch size reduction for out-of-memory protection
python main.py translate --hf-dataset large_dataset --target-lang zh --source-lang en --backend nllb --device cuda --auto-reduce-batch-size --min-batch-size 4

# Google Translate with custom project and media handling
python main.py translate --dataset-path dataset.json --target-lang ja --source-lang en --backend google --project-id my-project --media-output-dir ./media_files

# Append translations to existing file with custom progress reporting
python main.py translate --dataset-path new_data.csv --target-lang de --source-lang en --backend marianmt --append-mode --progress-interval 5

Generation:
  
# Generate pretraining data
python main.py generate --target-lang fr --mode pretraining --domain science --n-samples 5 --min-length 100 --max-length 300

# Generate large-scale pretraining data using OpenAI batch API for cost efficiency  
python main.py generate --target-lang es --mode pretraining --domain medical --backend openai --n-samples 1000 --min-length 150 --max-length 500 --use-batch --batch-job-description "Medical pretraining data generation"

# Use generation presets for quick setup
python main.py generate --target-lang en --preset creative --n-samples 20 --prompt "Write about artificial intelligence"

# Advanced parameter control
python main.py generate --target-lang de --backend huggingface --temperature 0.8 --top-p 0.9 --top-k 40 --max-gen-tokens 150

# Use HuggingFace-specific max_new_tokens override
python main.py generate --target-lang es --backend huggingface --max-gen-tokens 200 --max-new-tokens 100

# OpenAI with custom max tokens
python main.py generate --target-lang fr --backend openai --max-gen-tokens 300 --temperature 0.7


"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import random
from datetime import datetime

# Add the current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset, Dataset
import pandas as pd
from synglot import LLMTranslator, LLMGenerator
from synglot.utils import config


def setup_translate_parser(subparsers):
    """Set up the translate command parser."""
    translate_parser = subparsers.add_parser(
        'translate', 
        help='Translate datasets between languages',
        description='Translate text columns in datasets from source to target language'
    )
    
    # Dataset source options (mutually exclusive)
    dataset_group = translate_parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--dataset-path', 
        type=str,
        help='Path to local dataset file (JSON or CSV)'
    )
    dataset_group.add_argument(
        '--hf-dataset',
        type=str, 
        help='HuggingFace dataset name (e.g., wmt16, opus100)'
    )
    
    # HuggingFace dataset options
    translate_parser.add_argument(
        '--hf-config',
        type=str,
        help='HuggingFace dataset configuration/subset name'
    )
    translate_parser.add_argument(
        '--hf-split',
        type=str,
        default='train',
        help='HuggingFace dataset split (default: train)'
    )
    translate_parser.add_argument(
        '--hf-columns',
        type=str,
        help='Comma-separated list of columns to load from HF dataset'
    )
    
    # Translation options
    translate_parser.add_argument(
        '--columns',
        type=str,
        required=False,
        help='Comma-separated list of column names to translate (if not provided, all columns will be translated). Supports nested fields like "qa.question"'
    )
    translate_parser.add_argument(
        '--source-lang',
        type=str,
        required=True,
        help='Source language code (e.g., en, es, fr, de)'
    )
    translate_parser.add_argument(
        '--target-lang',
        type=str,
        required=True,
        help='Target language code (e.g., en, es, fr, de)'
    )
    
    # Translation model options
    translate_parser.add_argument(
        '--backend',
        type=str,
        choices=['marianmt', 'openai', 'google', 'nllb'],
        default='openai',
        help='Translation backend to use (default: openai)'
    )
    translate_parser.add_argument(
        '--model-name',
        type=str,
        help='Specific model name (optional, backend will choose default)'
    )
    translate_parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device for NLLB model (auto, cpu, cuda, or specific device like cuda:0). Default: auto'
    )
    translate_parser.add_argument(
        '--project-id',
        type=str,
        help='Google Cloud project ID (required for Google backend, can also use GOOGLE_CLOUD_PROJECT_ID env var)'
    )
    
    # Output options
    translate_parser.add_argument(
        '--output-path',
        type=str,
        help='Output file path (default: auto-generated in ./outputs/)'
    )
    translate_parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    translate_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for translation (default: 32)'
    )
    
    # Batch processing options
    translate_parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Use batch translation for better performance'
    )
    translate_parser.add_argument(
        '--batch-request-limit',
        type=int,
        default=50000,
        help='Maximum requests per OpenAI batch job (default: 50000). Only applies to OpenAI backend with --use-batch.'
    )
    translate_parser.add_argument(
        '--batch-token-limit',
        type=int,
        default=1900000,
        help='Maximum tokens per OpenAI batch job (default: 1900000). Only applies to OpenAI backend with --use-batch.'
    )
    translate_parser.add_argument(
        '--batch-job-description',
        type=str,
        default='CLI dataset translation',
        help='Description for OpenAI batch jobs (default: CLI dataset translation)'
    )
    
    # Advanced processing options
    translate_parser.add_argument(
        '--streaming-mode',
        action='store_true',
        help='Process dataset in streaming mode for very large datasets (reduces memory usage)'
    )
    translate_parser.add_argument(
        '--auto-reduce-batch-size',
        action='store_true',
        default=True,
        help='Automatically reduce batch size on out-of-memory errors (default: enabled)'
    )
    translate_parser.add_argument(
        '--no-auto-reduce-batch-size',
        action='store_false',
        dest='auto_reduce_batch_size',
        help='Disable automatic batch size reduction on OOM errors'
    )
    translate_parser.add_argument(
        '--min-batch-size',
        type=int,
        default=1,
        help='Minimum batch size when auto-reducing (default: 1)'
    )
    
    # Nested field and data structure options
    translate_parser.add_argument(
        '--nested-field-separator',
        type=str,
        default='.',
        help='Separator for nested field names like "qa.question" (default: .)'
    )
    
    # Media and file handling options
    translate_parser.add_argument(
        '--media-output-dir',
        type=str,
        help='Directory to save media files (images, etc.). If not specified, uses {output_dir}/{dataset_name}_media'
    )
    translate_parser.add_argument(
        '--media-field-name',
        type=str,
        default='image',
        help='Name of the field containing media data (default: image)'
    )
    
    # Progress and error handling options
    translate_parser.add_argument(
        '--progress-interval',
        type=int,
        default=10,
        help='Print progress every N samples in sequential mode (default: 10)'
    )
    translate_parser.add_argument(
        '--no-save-errors',
        action='store_false',
        dest='save_errors',
        default=True,
        help='Do not save error records to output file (default: errors are saved)'
    )
    translate_parser.add_argument(
        '--append-mode',
        action='store_true',
        help='Append to existing output file instead of overwriting'
    )
    
    # Testing and debugging options
    translate_parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to translate (for testing)'
    )


def setup_generate_parser(subparsers):
    """Set up the generate command parser."""
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate synthetic data',
        description='Generate synthetic text data in specified languages'
    )
    
    # Required arguments
    generate_parser.add_argument(
        '--target-lang',
        type=str,
        required=True,
        help='Target language code for generation (e.g., en, es, fr, de)'
    )
    
    # Generation mode
    generate_parser.add_argument(
        '--mode',
        type=str,
        choices=['general', 'pretraining', 'conversation', 'material'],
        default='general',
        help='Generation mode (default: general)'
    )
    
    # Preset configuration (enhanced)
    generate_parser.add_argument(
        '--preset',
        type=str,
        choices=['creative', 'precise', 'balanced', 'fast'],
        help='Use a preset configuration for generation parameters (overrides individual parameter settings)'
    )
    
    # General generation options
    generate_parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt for generation (for general mode)'
    )
    generate_parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to generate (default: 10)'
    )
    
    # Material generation options
    generate_parser.add_argument(
        '--material-path',
        type=str,
        help='Path to material files for material mode (supports wildcards like ./docs/*.txt)'
    )
    generate_parser.add_argument(
        '--chunk-size',
        type=int,
        help='Size of text chunks for material processing'
    )
    generate_parser.add_argument(
        '--n-samples-per-chunk',
        type=int,
        default=3,
        help='Number of samples to generate per chunk (default: 3)'
    )
    
    # Pretraining specific options
    generate_parser.add_argument(
        '--domain',
        type=str,
        default='general',
        help='Domain for pretraining generation (default: general)'
    )
    generate_parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum length for pretraining texts (default: 50)'
    )
    generate_parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Maximum length for pretraining texts (default: 200)'
    )
    
    # Conversation specific options
    generate_parser.add_argument(
        '--n-turns-min',
        type=int,
        default=2,
        help='Minimum turns in conversation (default: 2)'
    )
    generate_parser.add_argument(
        '--n-turns-max',
        type=int,
        default=5,
        help='Maximum turns in conversation (default: 5)'
    )
    
    # Model options
    generate_parser.add_argument(
        '--backend',
        type=str,
        choices=['huggingface', 'openai'],
        default='openai',
        help='Generation backend (default: openai)'
    )
    generate_parser.add_argument(
        '--model-name',
        type=str,
        help='Specific model name for generation'
    )
    
    # Enhanced generation parameters (updated)
    generate_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature (0.0-2.0). Default varies by backend.'
    )
    generate_parser.add_argument(
        '--max-new-tokens',
        type=int,
        help='Maximum new tokens to generate (HuggingFace only, overrides --max-gen-tokens if specified)'
    )
    generate_parser.add_argument(
        '--max-gen-tokens',
        type=int,
        default=1024,
        help='Maximum tokens for generation (primary parameter for both backends, default: 1024)'
    )
    generate_parser.add_argument(
        '--top-k',
        type=int,
        help='Top-k sampling parameter (HuggingFace only)'
    )
    generate_parser.add_argument(
        '--top-p',
        type=float,
        help='Top-p (nucleus) sampling parameter (0.0-1.0)'
    )
    generate_parser.add_argument(
        '--do-sample',
        action='store_true',
        help='Enable sampling for generation (HuggingFace only)'
    )
    generate_parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Disable sampling for generation (HuggingFace only)'
    )
    generate_parser.add_argument(
        '--min-gen-length',
        type=int,
        help='Minimum total length for generation (HuggingFace only)'
    )
    generate_parser.add_argument(
        '--return-full-text',
        action='store_true',
        help='Return full text including prompt (HuggingFace only)'
    )
    generate_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible generation'
    )
    
    # Output options
    generate_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated based on params)'
    )
    generate_parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    generate_parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl', 'txt'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    # Batch generation options
    generate_parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Use batch generation for better performance (OpenAI) or efficiency (HuggingFace)'
    )
    generate_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for HuggingFace batch processing (default: 32)'
    )
    generate_parser.add_argument(
        '--batch-job-description',
        type=str,
        default='CLI batch generation',
        help='Description for OpenAI batch jobs (default: CLI batch generation)'
    )


def load_dataset_from_args(args) -> Dataset:
    """Load dataset based on CLI arguments using HuggingFace datasets."""
    
    if args.dataset_path:
        # Load from local file
        print(f"Loading dataset from {args.dataset_path}")
        
        if args.dataset_path.endswith('.csv'):
            # Load CSV using pandas first for easier handling
            df = pd.read_csv(args.dataset_path)
            dataset = Dataset.from_pandas(df)
        elif args.dataset_path.endswith('.json'):
            # Load JSON file
            with open(args.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            else:
                raise ValueError("JSON file must contain a list of records")
        elif args.dataset_path.endswith('.jsonl'):
            # Load JSONL file
            data = []
            with open(args.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported file format. Supported formats: .csv, .json, .jsonl")
        
        print(f"Loaded dataset from {args.dataset_path}")
        
    elif args.hf_dataset:
        # Load from HuggingFace
        print(f"Loading HuggingFace dataset: {args.hf_dataset}")
        
        streaming_mode = getattr(args, 'streaming_mode', False)
        
        try:
            dataset = load_dataset(
                args.hf_dataset,
                name=args.hf_config,
                split=args.hf_split,
                streaming=streaming_mode
            )
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            raise
        
        # Select specific columns if requested
        if args.hf_columns:
            hf_columns = [col.strip() for col in args.hf_columns.split(',')]
            if not streaming_mode:
                # For non-streaming datasets, we can validate columns exist
                available_columns = dataset.column_names
                missing_columns = [col for col in hf_columns if col not in available_columns]
                if missing_columns:
                    print(f"Warning: Columns {missing_columns} not found in dataset")
                    print(f"Available columns: {available_columns}")
                    # Keep only columns that exist
                    hf_columns = [col for col in hf_columns if col in available_columns]
            
            if hf_columns:
                dataset = dataset.select_columns(hf_columns)
        
        if args.hf_config:
            print(f"Config: {args.hf_config}")
        print(f"Split: {args.hf_split}")
    
    else:
        raise ValueError("Either --dataset-path or --hf-dataset must be provided")
    
    # Print dataset info
    if not getattr(args, 'streaming_mode', False):
        print(f"Dataset loaded with {len(dataset)} samples")
        print(f"Available columns: {dataset.column_names}")
    else:
        print("Dataset loaded in streaming mode")
        # For streaming datasets, we can't easily get the length, but we can show columns
        try:
            # Get first item to see column structure
            first_item = next(iter(dataset))
            print(f"Sample columns: {list(first_item.keys())}")
        except:
            print("Could not determine columns for streaming dataset")
    
    # Limit samples if requested (only for non-streaming datasets)
    if hasattr(args, 'max_samples') and args.max_samples and not getattr(args, 'streaming_mode', False):
        if len(dataset) > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
            print(f"Limited to {args.max_samples} samples for testing")
    
    return dataset


def run_translate(args):
    """Execute translation command."""
    print(f"Starting translation from {args.source_lang} to {args.target_lang}")
    print(f"Backend: {args.backend}")
    
    # Print additional options when enabled
    if args.streaming_mode:
        print("Streaming mode: ENABLED (for large datasets)")
    if args.auto_reduce_batch_size:
        print(f"Auto batch size reduction: ENABLED (min: {args.min_batch_size})")
    if args.nested_field_separator != '.':
        print(f"Nested field separator: '{args.nested_field_separator}'")
    if args.media_output_dir:
        print(f"Media output directory: {args.media_output_dir}")
    if args.append_mode:
        print("Append mode: ENABLED")
    
    # Load dataset
    dataset = load_dataset_from_args(args)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty")
        return
    
    # Parse columns to translate
    if args.columns:
        columns_to_translate = [col.strip() for col in args.columns.split(',')]
    else:
        # If no columns specified, translate all available columns
        if hasattr(dataset, 'column_names') and dataset.column_names:
            columns_to_translate = list(dataset.column_names)
        else:
            print("Error: Dataset has no columns and no columns were specified for translation")
            return
    
    print(f"Columns to translate: {columns_to_translate}")
    
    # Initialize translator with new parameters
    translator = LLMTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        project_id=args.project_id
    )
    
    # Run translation
    try:
        if args.use_batch:
            print("Using batch translation for better performance...")
        elif args.streaming_mode:
            print("Using streaming mode for large dataset processing...")
        else:
            print("Using sequential translation...")
            
        result = translator.translate_dataset(
            dataset=dataset,
            columns_to_translate=columns_to_translate,
            output_path=args.output_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            save_errors=args.save_errors,
            append_mode=args.append_mode,
            use_batch=args.use_batch,
            batch_job_description=args.batch_job_description,
            batch_request_limit=args.batch_request_limit,
            batch_token_limit=args.batch_token_limit,
            streaming_mode=args.streaming_mode,
            auto_reduce_batch_size=args.auto_reduce_batch_size,
            min_batch_size=args.min_batch_size,
            nested_field_separator=args.nested_field_separator,
            media_output_dir=args.media_output_dir,
            media_field_name=args.media_field_name
        )
        
        print("\n" + "="*50)
        print("TRANSLATION SUMMARY")
        print("="*50)
        
        # Handle different result types (immediate results vs batch jobs)
        if isinstance(result, dict) and 'batch_job' in result:
            # This is an OpenAI batch job result
            print("OpenAI batch job created successfully!")
            print(f"Batch ID: {result['batch_id']}")
            print(f"Status: {result['status']}")
            print(f"Total requests: {result['total_requests']}")
            print(f"Output will be saved to: {result['output_path']}")
            print("\nNote: This is an asynchronous batch job.")
            print("Use translator.retrieve_batch() to check status and get results when complete.")
        elif isinstance(result, dict) and 'multiple_batches' in result:
            # This is multiple OpenAI batch jobs result
            print("Multiple OpenAI batch jobs created successfully!")
            print(f"Total batches created: {result['total_batches']}")
            print(f"Total requests: {result['total_requests']}")
            print(f"Each batch respects OpenAI's {result.get('batch_request_limit', 50000)} request limit")
            print("\nBatch details:")
            for i, batch_job in enumerate(result['batch_jobs'], 1):
                print(f"  Batch {i}: ID {batch_job['batch_id']}, {batch_job['total_requests']} requests")
                print(f"    Output: {batch_job['output_path']}")
            print("\nNote: These are asynchronous batch jobs.")
            print("Use translator.retrieve_batch() with each batch_job individually to get results when complete.")
            print("You can access individual batch jobs via result['batch_jobs'][index]")
        else:
            # This is immediate results
            for key, value in result.items():
                if key not in ['batch_job']:  # Don't print the actual batch job object
                    print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return


def run_generate(args):
    """Execute generation command."""
    print(f"Starting generation in {args.target_lang}")
    print(f"Mode: {args.mode}")
    print(f"Backend: {args.backend}")
    if args.use_batch:
        print("Using batch generation for better performance...")
    
    # Validate arguments based on mode
    if args.mode == 'material' and not args.material_path:
        print("Error: --material-path is required for material mode")
        return
    
    # Prepare generation parameters from CLI arguments
    generation_params = {}
    
    # Handle do_sample argument logic (mutually exclusive flags)
    if args.do_sample and args.no_sample:
        print("Error: --do-sample and --no-sample are mutually exclusive")
        return
    elif args.do_sample:
        generation_params['do_sample'] = True
    elif args.no_sample:
        generation_params['do_sample'] = False
    
    # Add other parameters if provided
    if args.temperature is not None:
        generation_params['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        generation_params['max_new_tokens'] = args.max_new_tokens
    if args.top_k is not None:
        generation_params['top_k'] = args.top_k
    if args.top_p is not None:
        generation_params['top_p'] = args.top_p
    if args.min_gen_length is not None:
        generation_params['min_length'] = args.min_gen_length
    if args.return_full_text:
        generation_params['return_full_text'] = True
    if args.seed is not None:
        generation_params['seed'] = args.seed
    
    # Initialize generator - use preset if specified, otherwise use individual parameters
    try:
        if args.preset:
            print(f"Using preset configuration: {args.preset}")
            generator = LLMGenerator.from_preset(
                preset_name=args.preset,
                target_lang=args.target_lang,
                backend=args.backend,
                model_name=args.model_name,
                max_gen_tokens=args.max_gen_tokens,
                **generation_params  # Preset parameters can be overridden by CLI args
            )
        else:
            generator = LLMGenerator(
                target_lang=args.target_lang,
                backend=args.backend,
                model_name=args.model_name,
                max_gen_tokens=args.max_gen_tokens,
                **generation_params
            )
            
        print(f"Generator initialized successfully")
        if generation_params:
            print(f"Active generation parameters: {generation_params}")
            
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return
    
    # Determine if we should save to file
    save_to_file = True  # Always save to file in CLI mode
    output_dir = args.output_dir  # Use output directory from arguments
    
    # Generate data based on mode
    generated_data = []
    
    try:
        if args.mode == 'general':
            print(f"Generating {args.n_samples} general samples...")
            if args.prompt:
                print(f"Using prompt: {args.prompt}")
            
            if args.use_batch:
                # Prepare prompts for batch generation
                prompts = [args.prompt or ""] * args.n_samples
                
                if args.backend == "openai":
                    # OpenAI batch generation
                    batch_job = generator.generate_batch(
                        prompts=prompts,
                        batch_job_description=args.batch_job_description
                    )
                    
                    print("\n" + "="*50)
                    print("BATCH GENERATION SUMMARY")
                    print("="*50)
                    print("OpenAI batch job created successfully!")
                    print(f"Batch ID: {batch_job.id}")
                    print(f"Status: {batch_job.status}")
                    print(f"Total requests: {len(prompts)}")
                    print("\nNote: This is an asynchronous batch job.")
                    print("Use generator.retrieve_batch() to check status and get results when complete.")
                    return
                
                elif args.backend == "huggingface":
                    # HuggingFace batch generation
                    generated_texts = generator.generate_batch(
                        prompts=prompts,
                        batch_size=args.batch_size
                    )
                    generated_data = [{"text": text, "prompt": args.prompt} for text in generated_texts]
            else:
                # Regular generation - pass additional kwargs for runtime parameter override
                runtime_kwargs = {}
                # max_new_tokens is already passed to constructor if specified, so no need to pass again
                
                generated_texts = generator.generate(
                    prompt=args.prompt,
                    n_samples=args.n_samples,
                    **runtime_kwargs
                )
                generated_data = [{"text": text, "prompt": args.prompt} for text in generated_texts]
            
            # Manual save for general mode (if not OpenAI batch)
            if generated_data:
                if args.output:
                    output_path = args.output
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    batch_suffix = "_batch" if args.use_batch else ""
                    preset_suffix = f"_{args.preset}" if args.preset else ""
                    filename = f"generated_general_{args.target_lang}_{args.backend}{preset_suffix}{batch_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
                    output_path = os.path.join(output_dir, filename)
                
                # Save data
                if args.format == 'json':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(generated_data, f, ensure_ascii=False, indent=2)
                elif args.format == 'jsonl':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for item in generated_data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                elif args.format == 'txt':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for item in generated_data:
                            if 'text' in item:
                                f.write(item['text'] + '\n\n')
                
                print(f"Results saved to: {output_path}")
            
        elif args.mode == 'pretraining':
            print(f"Generating {args.n_samples} pretraining samples...")
            print(f"Domain: {args.domain}")
            print(f"Length range: {args.min_length}-{args.max_length}")
            
            if args.use_batch:
                # For pretraining batch generation, create prompts based on domain/topics
                if args.domain.lower() == "general":
                    topics = generator.config.get("generation_settings.pretraining.general_topics_list", [])
                    prompt_template = generator.config.get("generation_settings.pretraining.topic_prompt_template", "Write a short text about {topic}.")
                    if topics:
                        prompts = [prompt_template.format(topic=random.choice(topics)) for _ in range(args.n_samples)]
                    else:
                        prompts = [prompt_template.format(topic="general knowledge") for _ in range(args.n_samples)]
                else:
                    prompt_template = generator.config.get("generation_settings.pretraining.topic_prompt_template", "Write a short text about {topic}.")
                    prompts = [prompt_template.format(topic=args.domain) for _ in range(args.n_samples)]
                
                if args.backend == "openai":
                    # OpenAI batch generation for pretraining
                    batch_job = generator.generate_batch(
                        prompts=prompts,
                        batch_job_description=f"{args.batch_job_description} - pretraining {args.domain}"
                    )
                    
                    print("\n" + "="*50)
                    print("BATCH GENERATION SUMMARY")
                    print("="*50)
                    print("OpenAI batch job created successfully!")
                    print(f"Batch ID: {batch_job.id}")
                    print(f"Status: {batch_job.status}")
                    print(f"Total requests: {len(prompts)}")
                    print("\nNote: This is an asynchronous batch job.")
                    print("Use generator.retrieve_batch() to check status and get results when complete.")
                    return
                
                elif args.backend == "huggingface":
                    # HuggingFace batch generation for pretraining
                    generated_texts = generator.generate_batch(
                        prompts=prompts,
                        batch_size=args.batch_size
                    )
                    
                    # Save with the same format as regular pretraining
                    if args.output:
                        output_path = args.output
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        batch_suffix = "_batch"
                        preset_suffix = f"_{args.preset}" if args.preset else ""
                        filename = f"generated_pretraining_{args.domain}_{args.target_lang}_{args.backend}{preset_suffix}{batch_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                        output_path = os.path.join(output_dir, filename)
                    
                    output_dirname = os.path.dirname(output_path)
                    if output_dirname:
                        os.makedirs(output_dirname, exist_ok=True)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for text in generated_texts:
                            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                    
                    print(f"Pretraining data saved to: {output_path}")
                    generated_data = [{"text": text, "domain": args.domain, "mode": "pretraining"} for text in generated_texts]
            else:
                # Regular pretraining generation
                generated_texts = generator.generate_pretraining(
                    domain=args.domain,
                    n_samples=args.n_samples,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    output_path=args.output,
                    output_dir=output_dir,
                    save_to_file=save_to_file
                )
                generated_data = [{"text": text, "domain": args.domain, "mode": "pretraining"} for text in generated_texts]
            
        elif args.mode == 'conversation':
            print(f"Generating {args.n_samples} conversations...")
            print(f"Domain: {args.domain}")
            print(f"Turns range: {args.n_turns_min}-{args.n_turns_max}")
            
            if args.use_batch:
                print("Note: Batch generation for conversations uses regular generation internally due to the multi-turn nature.")
            
            # Conversation generation doesn't benefit from batch processing due to its sequential nature
            # Use regular conversation generation
            conversations = generator.generate_conversations(
                domain=args.domain,
                n_samples=args.n_samples,
                n_turns_min=args.n_turns_min,
                n_turns_max=args.n_turns_max,
                output_path=args.output,
                output_dir=output_dir,
                save_to_file=save_to_file
            )
            generated_data = [{"conversation": conv, "domain": args.domain, "mode": "conversation"} for conv in conversations]
        
        elif args.mode == 'material':
            print(f"Generating samples from material...")
            print(f"Material path: {args.material_path}")
            if args.chunk_size:
                print(f"Chunk size: {args.chunk_size}")
            print(f"Samples per chunk: {args.n_samples_per_chunk}")
            
            if args.use_batch:
                print("Note: Batch generation for material mode uses regular generation internally due to chunk-based processing.")
            
            # Use glob to expand wildcards in material path
            import glob
            material_paths = glob.glob(args.material_path)
            if not material_paths:
                raise ValueError(f"No files found matching pattern: {args.material_path}")
            
            # Material generation doesn't benefit from batch processing due to its chunk-based nature
            # Use regular material generation
            generated_samples = generator.generate_from_material(
                material_paths=material_paths,
                chunk_size=args.chunk_size,
                n_samples_per_chunk=args.n_samples_per_chunk,
                output_path=args.output,
                output_dir=output_dir,
                save_to_file=save_to_file
            )
            
            # Convert the structured samples to the expected format for summary
            generated_data = []
            for sample in generated_samples:
                generated_data.append({
                    "text": sample["generated_text"],
                    "source_file": sample["source_file_name"],
                    "chunk_id": sample["chunk_id"],
                    "mode": "material"
                })
        
        # Only print summary if we have immediate results (not for OpenAI batch jobs)
        if generated_data:
            print(f"\nGenerated {len(generated_data)} samples")
            
            # Show sample output
            print("\nSample output:")
            print("-" * 40)
            if 'text' in generated_data[0]:
                sample_text = generated_data[0]['text']
                print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
            elif 'conversation' in generated_data[0]:
                sample_conv = str(generated_data[0]['conversation'])
                print(sample_conv[:200] + "..." if len(sample_conv) > 200 else sample_conv)
                
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Synglot CLI - Synthetic Data Generation and Translation Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command parsers
    setup_translate_parser(subparsers)
    setup_generate_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up configuration if provided
    if args.config:
        # Load configuration file if provided
        # This would require implementing config file loading
        print(f"Loading configuration from: {args.config}")
    
    # Execute commands
    if args.command == 'translate':
        run_translate(args)
    elif args.command == 'generate':
        run_generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 