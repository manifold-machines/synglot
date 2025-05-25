import os
from synglot.translate import StandardTranslator
from synglot.dataset import Dataset

def main():
    # Initialize translator
    source_lang = "en"
    target_lang = "mk"
    
    print(f"Initializing translator for {source_lang} to {target_lang}...")
    try:
        translator = StandardTranslator(source_lang=source_lang, target_lang=target_lang)
        print("Translator initialized successfully.")
    except Exception as e:
        print(f"Error initializing translator: {e}")
        print("Please ensure you have a working internet connection and the Helsinki-NLP model for en-mk is available.")
        print("You might need to install additional libraries like sacremoses: pip install sacremoses")
        return

    print("Loading gsm8k dataset from Hugging Face (main config, train split)...")
    try:
        # Initialize and load dataset
        dataset = Dataset()
        dataset.load_from_huggingface("gsm8k", config_name="main", split="train")
        print(f"Dataset loaded: {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use the enhanced translator to handle everything automatically
    print("Starting translation using enhanced StandardTranslator...")
    
    # Option 1: Translate with detailed error handling (recommended for production)
    summary = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=["question", "answer"],  # Specify columns to translate
        output_dir="./outputs",  # Output directory
        progress_interval=10,    # Progress updates every 10 samples
        save_errors=True        # Save error records for debugging
    )
    
    # Option 2: For better performance with large datasets, use batch processing
    # Note: Uncomment the following lines to use batch processing instead
    # summary = translator.translate_dataset_batch(
    #     dataset=dataset,
    #     columns_to_translate=["question", "answer"],
    #     batch_size=32,
    #     progress_interval=100
    # )
    
    # Print final summary
    print("\n" + "="*50)
    print("TRANSLATION SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 