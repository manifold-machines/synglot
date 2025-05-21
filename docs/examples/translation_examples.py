# Example workflow for dataset translation
from synglot import Dataset, StandardTranslator

# Load dataset from HuggingFace
dataset = Dataset()
dataset.load_from_huggingface("wikitext", split="train", columns=["text"])

# Initialize translator
translator = StandardTranslator(source_lang="en", target_lang="sw", 
                          model_name="Helsinki-NLP/opus-mt-en-sw")

# Create translation function to apply to dataset
def translate_text(sample):
    sample["translated_text"] = translator.translate(sample["text"])
    return sample

# Apply translation to dataset
translated_dataset = dataset.map(translate_text)

# Save translated dataset
translated_dataset.save("translated_wikitext_sw.json") 