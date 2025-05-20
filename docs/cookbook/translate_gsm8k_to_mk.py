import os
import json
from synglot.translate import StandardTranslator
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
        translator = StandardTranslator(source_lang=source_lang, target_lang=target_lang)
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

    print(f"Starting translation and saving to {output_file_path}...")
    # Open the file in append mode to save outputs continuously
    with open(output_file_path, 'a', encoding='utf-8') as f:
        for i, item in enumerate(synglot_gsm8k_dataset):
            try:
                original_question = item.get("question")
                original_answer = item.get("answer")

                if not original_question:
                    print(f"Skipping item {i+1} due to missing 'question' field.")
                    continue

                print(f"Translating sample {i+1}/{len(synglot_gsm8k_dataset)}...")
                translated_question = translator.translate(original_question)
                
                output_record = {
                    "original_question": original_question,
                    "translated_question_mk": translated_question,
                    "original_answer": original_answer
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                
                if (i + 1) % 10 == 0: # Log progress every 10 samples
                    print(f"Processed and saved {i+1} samples.")

            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                # Optionally, save error information or skip the sample
                error_record = {
                    "sample_index": i+1,
                    "error": str(e),
                    "original_question": item.get("question", "N/A")
                }
                f.write(json.dumps(error_record, ensure_ascii=False) + '\n')

    print(f"Translation complete. Output saved to {output_file_path}")

if __name__ == "__main__":
    main() 