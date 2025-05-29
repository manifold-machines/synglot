#!/usr/bin/env python3
"""
Synglot CLI - Command Line Interface for Synthetic Data Generation and Translation Toolkit

Usage examples:

Translation:
  
# Translate all columns automatically
python main.py translate --hf-dataset wmt16 --target-lang mk --source-lang en --backend openai
  
# Use batch translation for better performance
python main.py translate --hf-dataset wmt16 --target-lang mk --source-lang en --backend openai --use-batch --batch-size 50

Generation:
  
# Generate pretraining data
python main.py generate --target-lang fr --mode pretraining --domain science --n-samples 5 --min-length 100 --max-length 300

# Generate large-scale pretraining data using OpenAI batch API for cost efficiency  
python main.py generate --target-lang es --mode pretraining --domain medical --backend openai --n-samples 1000 --min-length 150 --max-length 500 --use-batch --batch-job-description "Medical pretraining data generation"


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

from synglot import Dataset, LLMTranslator, LLMGenerator
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
        help='Comma-separated list of column names to translate (if not provided, all columns will be translated)'
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
        choices=['marianmt', 'openai', 'google'],
        default='openai',
        help='Translation backend to use (default: openai)'
    )
    translate_parser.add_argument(
        '--model-name',
        type=str,
        help='Specific model name (optional, backend will choose default)'
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
    translate_parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Use batch translation for better performance'
    )
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
    
    # Generation parameters
    generate_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    generate_parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum new tokens to generate (default: 100)'
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


def load_dataset(args) -> Dataset:
    """Load dataset based on CLI arguments."""
    dataset = Dataset(source_lang=args.source_lang, target_lang=args.target_lang)
    
    if args.dataset_path:
        # Load from local file
        file_format = 'csv' if args.dataset_path.endswith('.csv') else 'json'
        dataset.load_from_file(args.dataset_path, format=file_format)
        print(f"Loaded dataset from {args.dataset_path}")
        
    elif args.hf_dataset:
        # Load from HuggingFace
        hf_columns = args.hf_columns.split(',') if args.hf_columns else None
        dataset.load_from_huggingface(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            columns=hf_columns,
            config_name=args.hf_config
        )
        print(f"Loaded HuggingFace dataset: {args.hf_dataset}")
        if args.hf_config:
            print(f"Config: {args.hf_config}")
        print(f"Split: {args.hf_split}")
    
    print(f"Dataset loaded with {len(dataset)} samples")
    if hasattr(dataset, 'columns') and dataset.columns:
        print(f"Available columns: {list(dataset.columns)}")
    
    # Limit samples if requested
    if hasattr(args, 'max_samples') and args.max_samples:
        if len(dataset) > args.max_samples:
            dataset = dataset.head(args.max_samples)
            print(f"Limited to {args.max_samples} samples for testing")
    
    return dataset


def run_translate(args):
    """Execute translation command."""
    print(f"Starting translation from {args.source_lang} to {args.target_lang}")
    print(f"Backend: {args.backend}")
    
    # Load dataset
    dataset = load_dataset(args)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty")
        return
    
    # Parse columns to translate
    if args.columns:
        columns_to_translate = [col.strip() for col in args.columns.split(',')]
    else:
        # If no columns specified, translate all available columns
        if hasattr(dataset, 'columns') and dataset.columns:
            columns_to_translate = list(dataset.columns)
        else:
            print("Error: Dataset has no columns and no columns were specified for translation")
            return
    
    print(f"Columns to translate: {columns_to_translate}")
    
    # Initialize translator
    translator = LLMTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        backend=args.backend,
        model_name=args.model_name
    )
    
    # Run translation
    try:
        if args.use_batch:
            print("Using batch translation for better performance...")
        else:
            print("Using sequential translation...")
            
        result = translator.translate_dataset(
            dataset=dataset,
            columns_to_translate=columns_to_translate,
            output_path=args.output_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            use_batch=args.use_batch
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
    
    # Initialize generator
    generator = LLMGenerator(
        target_lang=args.target_lang,
        backend=args.backend,
        model_name=args.model_name
    )
    
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
                # Regular generation
                generated_texts = generator.generate(
                    prompt=args.prompt,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens
                )
                generated_data = [{"text": text, "prompt": args.prompt} for text in generated_texts]
            
            # Manual save for general mode (if not OpenAI batch)
            if generated_data:
                if args.output:
                    output_path = args.output
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    batch_suffix = "_batch" if args.use_batch else ""
                    filename = f"generated_general_{args.target_lang}_{args.backend}{batch_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                        filename = f"generated_pretraining_{args.domain}_{args.target_lang}_{args.backend}{batch_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
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