#!/usr/bin/env python3
"""
Simplified script to batch translate the LNQA dataset using the enhanced LLMTranslator.
This demonstrates how the core library now handles all the complex functionality
that was previously implemented in the original script.
"""

from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator

def translate_lnqa_simplified(source_lang='en', target_lang='mk', initial_batch_size=100):
    """
    Translate LNQA dataset using the enhanced core library functionality.
    All the complex features from the original script are now handled automatically:
    - Streaming mode for memory efficiency
    - OOM handling with automatic batch size reduction  
    - Nested QA structure handling
    - Image file saving
    - Progress tracking
    - Error handling
    
    Args:
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'mk') 
        initial_batch_size: Starting batch size for text pieces
    """
    
    print(f"Loading LNQA dataset in streaming mode...")
    # Load the dataset in streaming mode
    ds = load_dataset("vikhyatk/lnqa", streaming=True)
    dataset = ds['train']
    
    print(f"Initializing NLLB translator ({source_lang} -> {target_lang})...")
    translator = LLMTranslator(
        source_lang=source_lang,
        target_lang=target_lang, 
        backend="nllb",
        device="auto"  # Will use GPU if available
    )
    
    # Use the enhanced translate_dataset method with all the new features
    result = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=["qa.question", "qa.answer"],  # Nested field support
        streaming_mode=True,                                # Streaming dataset support
        auto_reduce_batch_size=True,                       # OOM handling
        min_batch_size=1,                                  # Minimum batch size
        batch_size=initial_batch_size,                     # Starting batch size
        nested_field_separator=".",                        # Handle nested structures
        media_field_name="image",                          # Handle image files
        save_errors=True,                                  # Save error records
        output_dir="./outputs"                             # Output directory
    )
    
    print("\nTranslation Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    # Example: Translate from English to Macedonian
    # The core library now handles all the complexity automatically
    translate_lnqa_simplified(
        source_lang='en',
        target_lang='mk', 
        initial_batch_size=100
    ) 