import os
import re
from synglot.translate import LLMTranslator
from synglot.dataset import Dataset

def preprocess_text(text):
    """
    Remove all text enclosed in << and >> brackets.
    
    Args:
        text (str): Input text that may contain <<...>> patterns
        
    Returns:
        str: Text with all <<...>> patterns removed
    """
    if not isinstance(text, str):
        return text
    
    # Remove all occurrences of <<...>> using regex
    # The pattern <<.*?>> uses non-greedy matching to handle multiple occurrences
    cleaned_text = re.sub(r'<<.*?>>', '', text)
    return cleaned_text

def preprocess_dataset(dataset, columns_to_clean):
    """
    Apply preprocessing to specified columns in the dataset.
    
    Args:
        dataset: The dataset object to preprocess
        columns_to_clean (list): List of column names to clean
    """
    print(f"Preprocessing dataset to remove <<...>> patterns from columns: {columns_to_clean}")
    
    # Get the underlying data (list of dictionaries)
    data = dataset._data
    
    if not data:
        print("  Dataset is empty, nothing to preprocess")
        return
    
    # Check if we have dictionary-based data
    if not isinstance(data[0], dict):
        print("  Warning: Dataset does not contain dictionary-based data, skipping preprocessing")
        return
    
    # Get available columns from the first row
    available_columns = list(data[0].keys())
    
    # Apply preprocessing to each specified column
    for column in columns_to_clean:
        if column in available_columns:
            print(f"  Cleaning column: {column}")
            # Apply preprocessing to each row
            for row in data:
                if column in row:
                    row[column] = preprocess_text(row[column])
        else:
            print(f"  Warning: Column '{column}' not found in dataset. Available columns: {available_columns}")
    
    print("Preprocessing completed.")

def main():
    # Initialize translator with OpenAI backend
    source_lang = "en"
    target_lang = "mk"
    
    print(f"Initializing OpenAI translator for {source_lang} to {target_lang}...")
    try:
        translator = LLMTranslator(
            source_lang=source_lang, 
            target_lang=target_lang,
            backend="openai",
            model_name="gpt-4.1-mini"  # You can change this to other models like "gpt-4o" if needed
        )
        print("OpenAI translator initialized successfully.")
    except Exception as e:
        print(f"Error initializing translator: {e}")
        print("Please ensure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. A working internet connection")
        print("3. Sufficient OpenAI API credits")
        print("You can set the API key by creating a .env file with: OPENAI_API_KEY=your_key_here")
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

    # Preprocess the dataset to remove <<...>> patterns
    columns_to_translate = ["question", "answer"]
    preprocess_dataset(dataset, columns_to_translate)

    # Use the enhanced translator to handle everything automatically
    print("Starting translation using OpenAI LLM translator...")
    
    # Option 1: Translate with detailed error handling (recommended for production)
    summary = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=columns_to_translate,  # Use the same columns variable
        output_dir="./outputs",  # Output directory
        progress_interval=10,    # Progress updates every 10 samples
        save_errors=True        # Save error records for debugging
    )
    
    # Option 2: For better performance with large datasets, use batch processing
    # Note: For OpenAI backend, batch processing creates async jobs that need to be retrieved later
    # Uncomment the following lines to use batch processing instead
    # summary = translator.translate_dataset_batch(
    #     dataset=dataset,
    #     columns_to_translate=columns_to_translate,
    #     batch_size=32,  # This parameter is used differently for OpenAI batch API
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