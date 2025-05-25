import os
import json
from synglot.translate import LLMTranslator
from synglot.dataset import Dataset # Assuming Dataset can be initialized with a list of dicts

def main():
    # Ensure output directory exists
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "gsm8k_mk.jsonl")

    source_lang = "en"
    target_lang = "mk"

    print(f"Initializing translator for {source_lang} to {target_lang}...")
    try:
        translator = LLMTranslator(source_lang=source_lang, target_lang=target_lang)
        print("Translator initialized successfully.")
    except Exception as e:
        print(f"Error initializing translator: {e}")
        print("Please ensure you have a working internet connection and the Helsinki-NLP model for en-mk is available.")
        print("You might need to install additional libraries like sacremoses: pip install sacremoses")
        return

    print("Loading gsm8k dataset from Hugging Face (main config, train split)...")
    try:
        # Initialize an empty synglot Dataset object
        synglot_gsm8k_dataset = Dataset()
        # Load data using the synglot Dataset's method, specifying the config name
        synglot_gsm8k_dataset.load_from_huggingface("gsm8k", config_name="main", split="train")
        print(f"Dataset loaded: {len(synglot_gsm8k_dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Submit a test batch of 20 examples
    test_data = synglot_gsm8k_dataset[:20]
    test_questions = test_data['question']
    batch = translator.translate_batch(test_questions)

    # Write the retrieved batch translations to a file
    #print(f"Starting translation and saving to {output_file_path}...")
    #with open(output_file_path, 'a', encoding='utf-8') as f:   
        #retrieved_content = translator.retrieve_batch(batch)
        #f.write(json.dumps(retrieved_content))

if __name__ == "__main__":
    main() 