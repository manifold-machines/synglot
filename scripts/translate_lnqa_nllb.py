from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator

def translate_lnqa_simplified(source_lang='en', target_lang='mk', initial_batch_size=100):
    ds = load_dataset("vikhyatk/lnqa", streaming=True)
    dataset = ds['train']
    
    print(f"Initializing NLLB translator ({source_lang} -> {target_lang})...")
    translator = LLMTranslator(
        source_lang=source_lang,
        target_lang=target_lang, 
        backend="nllb",
        device="auto"
    )
    
    result = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=["qa.question", "qa.answer"], # nested
        streaming_mode=True,                               
        auto_reduce_batch_size=True, # OOM handling
        min_batch_size=1,
        batch_size=initial_batch_size,
        nested_field_separator=".",
        media_field_name="image",
        save_errors=True,
        output_dir="./outputs"
    )
    
    print("\nTranslation Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    translate_lnqa_simplified(
        source_lang='en',
        target_lang='mk', 
        initial_batch_size=100
    ) 