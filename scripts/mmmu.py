from synglot.translate.llm_translator import LLMTranslator
from datasets import load_dataset


def translate_mmmu():
    mmmu_configs = ['Accounting', 'Agriculture'] #'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    for config in mmmu_configs:
        translator = LLMTranslator(
            source_lang="en",
            target_lang="mk",
            backend="nllb",
            model_name="facebook/nllb-200-3.3B",
            max_gen_tokens=1024,
            project_id="synglot-441912",
            device="cuda"
        )
        dataset = load_dataset("MMMU/MMMU", config, split="validation")
        result = translator.translate_dataset(
            dataset=dataset,
            columns_to_translate=["question", "options"],
            streaming_mode=True,
            auto_reduce_batch_size=True,
            min_batch_size=1,
            batch_size=100,
            media_field_name="image",
            output_dir=f"eval_datasets/mmmu_{config}"
        )
        for key, value in result.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    translate_mmmu()