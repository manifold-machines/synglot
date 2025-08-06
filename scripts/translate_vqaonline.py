from datasets import load_dataset
from synglot.translate.llm_translator import LLMTranslator

def translate_VQAonline(source_lang='en', target_lang='mk'):
    ds = load_dataset("ChongyanChen/VQAonline", streaming=True)
    dataset = ds['train']
    
    print(f"Initializing translator ({source_lang} -> {target_lang})...")
    translator = LLMTranslator(
        source_lang=source_lang,
        target_lang=target_lang, 
        backend="openai",
        device="auto"
    )
    
    result = translator.translate_dataset(
        dataset=dataset,
        columns_to_translate=["qa.question", "qa.answer"], # nested
        streaming_mode=True,                               
        auto_reduce_batch_size=True, # OOM handling
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
    translate_VQAonline(
        source_lang='en',
        target_lang='mk'
    ) 