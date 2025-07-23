#!/usr/bin/env python3
import os
from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator
from huggingface_hub import login
from dotenv import load_dotenv

def translate_sharegpt_dataset(source_lang='en', target_lang='es', initial_batch_size=32, min_batch_size=1):
    """
    Translate ShareGPT-4o dataset using the LLMTranslator's built-in dataset translation capabilities.
    
    Args:
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'es') 
        initial_batch_size: Starting batch size
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
        device="auto",  # Will use GPU if available
    )
    
    # Use the built-in translate_dataset method with streaming mode for memory efficiency
    result = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=['conversations.value'],  # Translate the 'value' field within conversations
        output_dir="./outputs",
        batch_size=initial_batch_size,
        streaming_mode=True,  # Memory efficient for large datasets
        auto_reduce_batch_size=True,  # Automatically handle OOM errors
        min_batch_size=min_batch_size,
        media_field_name='image',  # Handle image saving automatically
        save_errors=True
    )
    
    print(f"\nTranslation complete!")
    print(f"Results: {result}")
    return result

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace")
    
    # Translate from English to Macedonian
    translate_sharegpt_dataset(
        source_lang='en',
        target_lang='mk', 
        initial_batch_size=128,
        min_batch_size=1
    ) 