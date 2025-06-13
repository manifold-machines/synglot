#!/usr/bin/env python3
"""
Simple script to batch translate the LNQA dataset using NLLB backend.
Handles the nested "qa" column structure and includes OOM error handling.
Uses streaming mode for memory efficiency with proper batch translation.
Saves images to separate directory for HuggingFace dataset optimization.
"""

import json
import os
from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator
import torch
from itertools import islice
from PIL import Image

def translate_batch_texts(texts, translator):
    """Translate a batch of texts using the translator's batch capabilities."""
    if not texts:
        return []
    
    # Check if translator has batch translation capability
    if hasattr(translator, 'translate_batch'):
        return translator.translate_batch(texts)
    else:
        # Fallback to individual translation if no batch method
        return [translator.translate(text) for text in texts]

def batch_translate_dataset(source_lang='en', target_lang='es', initial_batch_size=32, min_batch_size=1):
    """
    Translate LNQA dataset with proper batch translation and automatic batch size reduction on OOM errors.
    Uses streaming mode for memory efficiency.
    Saves images to separate directory for HuggingFace optimization.
    
    Args:
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'es') 
        initial_batch_size: Starting batch size (number of text pieces, not samples)
        min_batch_size: Minimum batch size before giving up
    """
    
    print(f"Loading LNQA dataset in streaming mode...")
    # Load the dataset in streaming mode
    try:
        ds = load_dataset("vikhyatk/lnqa", streaming=True)
        
        # Use train split (adjust if needed)
        dataset = ds['train']
        print(f"Dataset loaded in streaming mode")
        
        # Test if we can get at least one sample
        test_iter = iter(dataset)
        first_sample = next(test_iter, None)
        if first_sample is None:
            print("Warning: No samples found in streaming dataset")
            return
        else:
            print(f"Successfully accessed streaming dataset. First sample keys: {list(first_sample.keys())}")
    except Exception as e:
        print(f"Error loading streaming dataset: {e}")
        return
    
    # Initialize NLLB translator
    print(f"Initializing NLLB translator ({source_lang} -> {target_lang})...")
    translator = LLMTranslator(
        source_lang=source_lang,
        target_lang=target_lang, 
        backend="nllb",
        device="auto"  # Will use GPU if available
    )
    
    # Prepare output directories
    output_dir = "./outputs"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"lnqa_translated_{source_lang}_to_{target_lang}_nllb.jsonl")
    
    current_batch_size = initial_batch_size
    processed_count = 0
    batch_number = 0
    
    print(f"Starting translation with batch size {current_batch_size} text pieces...")
    print(f"Output will be saved to: {output_file}")
    print(f"Images will be saved to: {images_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Create a fresh iterator from the dataset
        dataset_iter = iter(dataset)
        
        while True:
            try:
                # Collect batch of text pieces from multiple samples
                batch_texts = []
                batch_samples = []
                sample_indices = []  # Maps each text to its sample index in batch_samples
                text_types = []  # 'question' or 'answer'
                qa_indices = []  # Which QA pair within the sample
                
                current_text_count = 0
                
                while True:
                    try:
                        sample = next(dataset_iter)
                        
                        # Add sample to batch
                        sample_idx = len(batch_samples)
                        batch_samples.append(sample)
                        
                        # Add all QA pairs and track which sample they belong to
                        # Always process ALL QA pairs from this sample to avoid incomplete translations
                        for qa_idx, qa_pair in enumerate(sample['qa']):
                            # Add question
                            if 'question' in qa_pair and qa_pair['question']:
                                batch_texts.append(qa_pair['question'])
                                sample_indices.append(sample_idx)
                                text_types.append('question')
                                qa_indices.append(qa_idx)
                                current_text_count += 1
                            
                            # Add answer
                            if 'answer' in qa_pair and qa_pair['answer']:
                                batch_texts.append(qa_pair['answer'])
                                sample_indices.append(sample_idx)
                                text_types.append('answer')
                                qa_indices.append(qa_idx)
                                current_text_count += 1
                        
                        # Only check batch size after completing the entire sample
                        # This ensures no sample has incomplete QA pairs
                        if current_text_count >= current_batch_size:
                            break
                            
                    except StopIteration:
                        break
                
                # If no texts collected, we've reached the end
                if not batch_texts:
                    print("Reached end of dataset")
                    break
                
                batch_number += 1
                print(f"Processing batch {batch_number}: {len(batch_texts)} text pieces from {len(batch_samples)} samples")
                
                # Translate all texts in batch
                print(f"  Translating batch of {len(batch_texts)} texts...")
                translated_texts = translate_batch_texts(batch_texts, translator)
                
                # Reconstruct original structure and save images
                print(f"  Reconstructing {len(batch_samples)} samples and saving images...")
                for sample_idx, sample in enumerate(batch_samples):
                    try:
                        # Save the image and get its path
                        image_filename = f"image_{processed_count:06d}.jpg"
                        image_path = os.path.join(images_dir, image_filename)
                        relative_image_path = f"images/{image_filename}"
                        
                        # Save the PIL image
                        if hasattr(sample['image'], 'save'):
                            sample['image'].save(image_path, 'JPEG', quality=100)
                        else:
                            print(f"  Warning: Image for sample {processed_count} is not a valid PIL image")
                            relative_image_path = None
                        
                        # Create translated QA pairs for this sample
                        translated_qa = []
                        for qa_idx, qa_pair in enumerate(sample['qa']):
                            translated_pair = {}
                            
                            # Find translated question for this sample and QA index
                            for i, (s_idx, t_type, q_idx) in enumerate(zip(sample_indices, text_types, qa_indices)):
                                if s_idx == sample_idx and q_idx == qa_idx and t_type == 'question':
                                    translated_pair['question'] = translated_texts[i]
                                    break
                            
                            # Find translated answer for this sample and QA index
                            for i, (s_idx, t_type, q_idx) in enumerate(zip(sample_indices, text_types, qa_indices)):
                                if s_idx == sample_idx and q_idx == qa_idx and t_type == 'answer':
                                    translated_pair['answer'] = translated_texts[i]
                                    break
                            
                            translated_qa.append(translated_pair)
                        
                        # Create output record with image path reference
                        output_record = {
                            'image': relative_image_path,  # Path to saved image file
                            # 'original_qa': sample['qa'],
                            'translated_qa': translated_qa
                        }
                        
                        # Write to file
                        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure data is written
                        
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            print(f"  Processed {processed_count} samples")
                            
                    except Exception as e:
                        print(f"  Error processing sample {processed_count + 1}: {e}")
                        # Write error record
                        error_record = {
                            'sample_index': processed_count + 1,
                            'error': str(e),
                            'image': None,
                            'original_sample': {
                                'qa': sample.get('qa', [])
                            }
                        }
                        f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                        f.flush()
                        processed_count += 1
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    print(f"  OOM Error with batch size {current_batch_size}: {e}")
                    
                    # Reduce batch size
                    current_batch_size = max(current_batch_size // 2, min_batch_size)
                    
                    if current_batch_size < min_batch_size:
                        print(f"  Batch size reduced to minimum ({min_batch_size}), but still getting OOM. Stopping.")
                        break
                    
                    print(f"  Reducing batch size to {current_batch_size} and retrying...")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Continue with next iteration - the iterator will continue from where it left off
                    continue
                else:
                    print(f"  Unexpected error: {e}")
                    break
            
            except Exception as e:
                print(f"  Unexpected error processing batch: {e}")
                # Skip this batch and continue
                continue
    
    print(f"\nTranslation complete!")
    print(f"Processed {processed_count} samples")
    print(f"Final batch size: {current_batch_size}")
    print(f"Output saved to: {output_file}")
    print(f"Images saved to: {images_dir}")

if __name__ == "__main__":
    # Example: Translate from English to Macedonian
    # Change these parameters as needed
    batch_translate_dataset(
        source_lang='en',
        target_lang='mk', 
        initial_batch_size=100,  # Number of text pieces (questions + answers) per batch
        min_batch_size=1
    ) 