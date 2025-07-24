from synglot.translate.llm_translator import LLMTranslator
from datasets import load_dataset
import wandb


def translate_mmmu():
    mmmu_configs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    translator = LLMTranslator(
    source_lang="en",
    target_lang="mk",
    backend="nllb",
    model_name="facebook/nllb-200-3.3B",
    max_gen_tokens=1024,
    project_id="synglot-441912",
    device="cuda"
    )
    
    # Log total number of configs to process
    wandb.log({"total_configs": len(mmmu_configs)})
    
    for i, config in enumerate(mmmu_configs):
        print(f"Processing config {i+1}/{len(mmmu_configs)}: {config}")
        wandb.log({"current_config": config, "config_progress": i+1})
        
        dataset = load_dataset("MMMU/MMMU", config, split="test")
        result = translator.translate_dataset(
            dataset=dataset,
            columns_to_translate=["question", "options"],
            streaming_mode=True,
            auto_reduce_batch_size=True,
            min_batch_size=1,
            batch_size=100,
            media_field_name="image",
            output_dir=f"evaluation/mmmu_{config}"
        )
        
        # Log translation results for this config
        log_data = {f"{config}_{key}": value for key, value in result.items()}
        log_data["completed_configs"] = i + 1
        wandb.log(log_data)
        
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    # Log completion
    wandb.log({"translation_complete": True})
    print("All MMMU configs translated successfully!")

if __name__ == "__main__":
    wandb.init(project="mmmu-translation", entity="manifold-multimodal")
    translate_mmmu()