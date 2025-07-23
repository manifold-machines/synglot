#!/usr/bin/env python3
import json
import os
from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator
import torch
from itertools import islice
from PIL import Image
from huggingface_hub import login
from dotenv import load_dotenv

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
    Translate ShareGPT-4o dataset with proper batch translation and automatic batch size reduction on OOM errors.
    Uses streaming mode for memory efficiency.
    Saves images to separate directory for HuggingFace optimization.
    
    Args:
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'es') 
        initial_batch_size: Starting batch size (number of text pieces, not samples)
        min_batch_size: Minimum batch size before giving up
    """
    
    print(f"Loading ShareGPT-4o image caption dataset...")
    dataset = load_dataset(
        "OpenGVLab/ShareGPT-4o",
        "image_caption",
        split="images"
    )
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
    images_dir = os.path.join(output_dir, "sharegpt_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sharegpt_translated_{source_lang}_to_{target_lang}_nllb.jsonl")
    
    batch_size = initial_batch_size
    processed_count = 0
    batch = []
    image_data = []
    
    print(f"Starting translation with batch size {batch_size} text pieces...")
    print(f"Output will be saved to: {output_file}")
    print(f"Images will be saved to: {images_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Create a fresh iterator from the dataset
        for item in dataset:
            caption = item.get('caption')
            if caption:
                batch.append(caption)
                image_data.append(item['image'])
            
            if len(batch) >= batch_size:
                try:
                    print(f"Translating batch of {len(batch)} captions...")
                    translated_batch = translate_batch_texts(batch, translator)
                    for original, translated, image in zip(batch, translated_batch, image_data):
                        img_filename = f"sharegpt_image_{processed_count:06d}.jpg"
                        img_path = os.path.join(images_dir, img_filename)

                        if isinstance(image, Image.Image):
                            image.save(img_path)
                        else:
                            print(f"Warning: Image for sample {processed_count} is not a valid image")
                            img_path = None
                        output_record = {
                            'image': os.path.join("sharegpt_images", img_filename),
                            'original_caption': original,
                            'translated_caption': translated
                        }
                        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        processed_count += 1
                    batch = []
                    image_data = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM Error with batch size {batch_size}: {e}")
                        batch_size = max(batch_size // 2, min_batch_size)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if batch:
            print(f"Translating final batch of {len(batch)} captions...")
            translated_batch = translate_batch_texts(batch, translator)
            for original, translated, image in zip(batch, translated_batch, image_data):
                img_filename = f"sharegpt_image_{processed_count:06d}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                relative_path = os.path.join("sharegpt_images", img_filename)

                if isinstance(image, Image.Image):
                    image.save(img_path)
                else:
                    relative_path = None
                output_record = {
                    'image': relative_path,
                    'original_caption': original,
                    'translated_caption': translated
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1

        
    
    print(f"\nTranslation complete!")
    print(f"Processed {processed_count} samples")
    print(f"Final batch size: {batch_size}")
    print(f"Output saved to: {output_file}")
    print(f"Images saved to: {images_dir}")

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    print(hf_token)
    login(token=hf_token)
    print("Logged in to HuggingFace")
    # Example: Translate from English to Macedonian
    # Change these parameters as needed
    batch_translate_dataset(
        source_lang='en',
        target_lang='mk', 
        initial_batch_size=100,  # Number of text pieces (questions + answers) per batch
        min_batch_size=1
    ) 