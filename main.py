#!/usr/bin/env python3
"""
Synglot CLI - Command Line Interface for Synthetic Data Generation and Translation Toolkit

Usage examples:
  python main.py translate --dataset-path data.json --columns text,title --target-lang es --source-lang en
  python main.py translate --dataset-path data.json --target-lang es --source-lang en  # Translates all columns
  python main.py translate --hf-dataset wmt16 --hf-config de-en --columns translation.en --target-lang de --source-lang en  
  python main.py translate --hf-dataset wmt16 --hf-config de-en --target-lang de --source-lang en  # Translates all columns
  python main.py generate --target-lang es --n-samples 100 --prompt "Write about technology" --output generated_data.json
  python main.py generate --target-lang fr --mode pretraining --domain science --n-samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add the current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from synglot import Dataset, LLMTranslator, HFGenerator, OpenAIGenerator
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
        choices=['marianmt', 'openai'],
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
        choices=['general', 'pretraining', 'conversation'],
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
        choices=['hf', 'openai'],
        default='hf',
        help='Generation backend (default: hf for HuggingFace)'
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
        '--format',
        type=str,
        choices=['json', 'jsonl', 'txt'],
        default='json',
        help='Output format (default: json)'
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
            print("Using batch translation...")
            result = translator.translate_dataset_batch(
                dataset=dataset,
                columns_to_translate=columns_to_translate,
                output_path=args.output_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size
            )
        else:
            print("Using individual translation...")
            result = translator.translate_dataset(
                dataset=dataset,
                columns_to_translate=columns_to_translate,
                output_path=args.output_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size
            )
        
        print("\n" + "="*50)
        print("TRANSLATION SUMMARY")
        print("="*50)
        for key, value in result.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return


def run_generate(args):
    """Execute generation command."""
    print(f"Starting generation in {args.target_lang}")
    print(f"Mode: {args.mode}")
    print(f"Backend: {args.backend}")
    
    # Initialize generator
    if args.backend == 'openai':
        generator = OpenAIGenerator(target_lang=args.target_lang)
    else:  # hf
        generator = HFGenerator(
            target_lang=args.target_lang,
            model_name=args.model_name
        )
    
    # Generate data based on mode
    generated_data = []
    
    try:
        if args.mode == 'general':
            print(f"Generating {args.n_samples} general samples...")
            if args.prompt:
                print(f"Using prompt: {args.prompt}")
            
            generated_texts = generator.generate(
                prompt=args.prompt,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens
            )
            generated_data = [{"text": text, "prompt": args.prompt} for text in generated_texts]
            
        elif args.mode == 'pretraining':
            print(f"Generating {args.n_samples} pretraining samples...")
            print(f"Domain: {args.domain}")
            print(f"Length range: {args.min_length}-{args.max_length}")
            
            generated_texts = generator.generate_pretraining(
                domain=args.domain,
                n_samples=args.n_samples,
                min_length=args.min_length,
                max_length=args.max_length
            )
            generated_data = [{"text": text, "domain": args.domain, "mode": "pretraining"} for text in generated_texts]
            
        elif args.mode == 'conversation':
            print(f"Generating {args.n_samples} conversations...")
            print(f"Domain: {args.domain}")
            print(f"Turns range: {args.n_turns_min}-{args.n_turns_max}")
            
            conversations = generator.generate_conversations(
                domain=args.domain,
                n_samples=args.n_samples,
                n_turns_min=args.n_turns_min,
                n_turns_max=args.n_turns_max
            )
            generated_data = [{"conversation": conv, "domain": args.domain, "mode": "conversation"} for conv in conversations]
        
        # Save generated data
        if not args.output:
            # Auto-generate output filename
            output_filename = f"generated_{args.mode}_{args.target_lang}_{args.n_samples}samples"
            if args.format == 'jsonl':
                output_filename += '.jsonl'
            elif args.format == 'txt':
                output_filename += '.txt'
            else:
                output_filename += '.json'
            args.output = output_filename
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        if args.format == 'json':
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=2)
        elif args.format == 'jsonl':
            with open(args.output, 'w', encoding='utf-8') as f:
                for item in generated_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif args.format == 'txt':
            with open(args.output, 'w', encoding='utf-8') as f:
                for item in generated_data:
                    if 'text' in item:
                        f.write(item['text'] + '\n\n')
                    elif 'conversation' in item:
                        f.write(str(item['conversation']) + '\n\n')
        
        print(f"\nGenerated {len(generated_data)} samples")
        print(f"Saved to: {args.output}")
        
        # Show sample
        if generated_data:
            print("\nSample output:")
            print("-" * 40)
            if 'text' in generated_data[0]:
                print(generated_data[0]['text'][:200] + "..." if len(generated_data[0]['text']) > 200 else generated_data[0]['text'])
            elif 'conversation' in generated_data[0]:
                print(str(generated_data[0]['conversation'])[:200] + "..." if len(str(generated_data[0]['conversation'])) > 200 else str(generated_data[0]['conversation']))
                
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