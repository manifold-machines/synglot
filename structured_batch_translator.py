#!/usr/bin/env python3
"""
Structured Batch Translator Script

This script handles batch translation of datasets containing structured data 
(lists of dictionaries with 'answer' and 'question' fields) using the NLLB backend.
It preserves the exact structure while translating the content.

Usage:
    python structured_batch_translator.py --input_file dataset.jsonl --column qa_pairs --source_lang en --target_lang es
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any, Union
from datetime import datetime
import copy

# Add the synglot package to the path (adjust as needed)
# sys.path.append('/path/to/synglot')  # Uncomment and adjust if needed

from synglot.translate.llm_translator import LLMTranslator


class StructuredBatchTranslator:
    """Handles batch translation of structured data while preserving format."""
    
    def __init__(self, source_lang: str, target_lang: str, device: str = "auto", model_name: str = None):
        """
        Initialize the structured translator.
        
        Args:
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'es') 
            device: Device for NLLB model ('auto', 'cpu', 'cuda')
            model_name: NLLB model name (defaults to facebook/nllb-200-distilled-600M)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Initialize NLLB translator
        print(f"Initializing NLLB translator: {source_lang} -> {target_lang}")
        self.translator = LLMTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            backend="nllb",
            model_name=model_name,
            device=device
        )
        print("NLLB translator initialized successfully!")
    
    def extract_texts_from_structured_data(self, structured_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract all text content from structured data for batch translation.
        
        Args:
            structured_data: List of dictionaries containing 'answer' and 'question' fields
            
        Returns:
            Dictionary containing texts to translate and mapping information
        """
        texts_to_translate = []
        text_mapping = []  # Maps each text back to its location
        
        for item_idx, item in enumerate(structured_data):
            if isinstance(item, dict):
                # Extract question
                if 'question' in item and item['question']:
                    texts_to_translate.append(str(item['question']))
                    text_mapping.append({
                        'item_idx': item_idx,
                        'field': 'question',
                        'text_idx': len(texts_to_translate) - 1
                    })
                
                # Extract answer
                if 'answer' in item and item['answer']:
                    texts_to_translate.append(str(item['answer']))
                    text_mapping.append({
                        'item_idx': item_idx,
                        'field': 'answer',
                        'text_idx': len(texts_to_translate) - 1
                    })
        
        return {
            'texts': texts_to_translate,
            'mapping': text_mapping
        }
    
    def reconstruct_structured_data(self, original_data: List[Dict[str, Any]], 
                                  translations: List[str], 
                                  mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reconstruct the original structure with translated content.
        
        Args:
            original_data: Original structured data
            translations: List of translated texts
            mapping: Mapping information from extract_texts_from_structured_data
            
        Returns:
            Structured data with translated content
        """
        # Create deep copy to avoid modifying original
        translated_data = copy.deepcopy(original_data)
        
        # Apply translations using the mapping
        for map_item in mapping:
            item_idx = map_item['item_idx']
            field = map_item['field']
            text_idx = map_item['text_idx']
            
            # Set the translated text
            translated_data[item_idx][field] = translations[text_idx]
        
        return translated_data
    
    def translate_structured_column(self, dataset_item: Dict[str, Any], 
                                  column_name: str, 
                                  batch_size: int = 32) -> Dict[str, Any]:
        """
        Translate a structured column in a dataset item.
        
        Args:
            dataset_item: Single item from dataset
            column_name: Name of column containing structured data
            batch_size: Batch size for translation
            
        Returns:
            Dataset item with translated structured data
        """
        if column_name not in dataset_item:
            raise ValueError(f"Column '{column_name}' not found in dataset item")
        
        structured_data = dataset_item[column_name]
        
        if not isinstance(structured_data, list):
            try:
                # Try to parse as JSON if it's a string
                if isinstance(structured_data, str):
                    structured_data = json.loads(structured_data)
                else:
                    raise ValueError(f"Column '{column_name}' is not a list or JSON string")
            except json.JSONDecodeError as e:
                raise ValueError(f"Column '{column_name}' contains invalid JSON: {e}")
        
        # Extract texts for translation
        extraction_result = self.extract_texts_from_structured_data(structured_data)
        
        if not extraction_result['texts']:
            print(f"Warning: No translatable text found in column '{column_name}'")
            return dataset_item
        
        print(f"Extracted {len(extraction_result['texts'])} text segments for translation")
        
        # Perform batch translation
        print("Performing batch translation...")
        translations = self.translator.translate_batch(
            extraction_result['texts'], 
            batch_size=batch_size
        )
        
        # Reconstruct with translations
        translated_structured_data = self.reconstruct_structured_data(
            structured_data,
            translations,
            extraction_result['mapping']
        )
        
        # Create result with translated column
        result = copy.deepcopy(dataset_item)
        translated_column_name = f"translated_{column_name}_{self.target_lang}"
        result[translated_column_name] = translated_structured_data
        
        return result
    
    def translate_dataset(self, input_file: str, column_name: str, 
                         output_file: str = None, batch_size: int = 32) -> str:
        """
        Translate an entire dataset with structured columns.
        
        Args:
            input_file: Path to input JSONL file
            column_name: Name of column containing structured data
            output_file: Path to output file (auto-generated if None)
            batch_size: Batch size for translation
            
        Returns:
            Path to output file
        """
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{base_name}_translated_{self.source_lang}_to_{self.target_lang}_{timestamp}.jsonl"
        
        print(f"Reading dataset from: {input_file}")
        print(f"Translating column: {column_name}")
        print(f"Output will be saved to: {output_file}")
        
        # Read dataset
        dataset = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    dataset.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
        
        if not dataset:
            raise ValueError("No valid data found in input file")
        
        print(f"Loaded {len(dataset)} items from dataset")
        
        # Process each item in the dataset
        total_items = len(dataset)
        translated_dataset = []
        
        for i, item in enumerate(dataset, 1):
            print(f"\nProcessing item {i}/{total_items}")
            
            try:
                translated_item = self.translate_structured_column(
                    item, column_name, batch_size
                )
                translated_dataset.append(translated_item)
                print(f"Successfully translated item {i}")
                
            except Exception as e:
                print(f"Error translating item {i}: {e}")
                # Add original item with error info
                error_item = copy.deepcopy(item)
                error_item['translation_error'] = str(e)
                translated_dataset.append(error_item)
        
        # Save results
        print(f"\nSaving results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in translated_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Translation complete! Results saved to: {output_file}")
        return output_file


def main():
    """Main function to run the structured batch translator."""
    parser = argparse.ArgumentParser(description='Batch translate structured data using NLLB')
    parser.add_argument('--input_file', '-i', required=True, 
                       help='Input JSONL file path')
    parser.add_argument('--column', '-c', required=True,
                       help='Column name containing structured data to translate')
    parser.add_argument('--source_lang', '-s', required=True,
                       help='Source language code (e.g., en)')
    parser.add_argument('--target_lang', '-t', required=True,
                       help='Target language code (e.g., es)')
    parser.add_argument('--output_file', '-o', 
                       help='Output file path (auto-generated if not provided)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                       help='Batch size for translation (default: 32)')
    parser.add_argument('--device', '-d', default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for NLLB model (default: auto)')
    parser.add_argument('--model_name', '-m',
                       help='NLLB model name (default: facebook/nllb-200-distilled-600M)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    try:
        # Initialize translator
        translator = StructuredBatchTranslator(
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            device=args.device,
            model_name=args.model_name
        )
        
        # Perform translation
        output_path = translator.translate_dataset(
            input_file=args.input_file,
            column_name=args.column,
            output_file=args.output_file,
            batch_size=args.batch_size
        )
        
        print(f"\n✅ Translation completed successfully!")
        print(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 