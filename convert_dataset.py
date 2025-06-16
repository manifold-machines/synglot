import json
import os
from datasets import Dataset, Image, Features, Value, Sequence
from tqdm import tqdm

def convert_jsonl_images_to_parquet(jsonl_path, images_folder, output_path, dataset_name=None):
    """
    Convert JSONL file + images folder to HuggingFace Parquet dataset
    
    Args:
        jsonl_path: Path to the .jsonl file
        images_folder: Path to the images folder
        output_path: Where to save the parquet dataset locally
        dataset_name: Optional name for pushing to hub (format: "username/dataset-name")
    """
    
    print("Loading JSONL data...")
    data = []
    
    # Read JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} entries")
    
    # Prepare data for dataset creation
    dataset_data = {
        "image": [],
        "translated_qa": []
    }
    
    print("Processing entries...")
    missing_images = []
    
    for entry in tqdm(data, desc="Processing entries"):
        image_path = entry["image"]
        full_image_path = os.path.join(images_folder, os.path.basename(image_path))
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            missing_images.append(image_path)
            continue
            
        dataset_data["image"].append(full_image_path)
        dataset_data["translated_qa"].append(entry["translated_qa"])
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found. First few:")
        for img in missing_images[:5]:
            print(f"  - {img}")
    
    print(f"Creating dataset with {len(dataset_data['image'])} valid entries...")
    
    # Define features for better type handling
    features = Features({
        "image": Image(),
        "translated_qa": [{"question": Value("string"), "answer": Value("string")}]
    })
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_data, features=features)
    
    print("Saving dataset to disk...")
    dataset.save_to_disk(output_path)
    
    # Also save as parquet files directly
    parquet_path = output_path + "_parquet"
    dataset.to_parquet(parquet_path + "/data.parquet")
    
    print(f"Dataset saved to: {output_path}")
    print(f"Parquet file saved to: {parquet_path}/data.parquet")
    print(f"Dataset info:")
    print(f"  - Total entries: {len(dataset)}")
    print(f"  - Features: {list(dataset.features.keys())}")
    print(f"  - Sample entry keys: {list(dataset[0].keys())}")
    
    # Upload to hub if requested
    if dataset_name:
        print(f"Uploading to HuggingFace Hub as {dataset_name}...")
        dataset.push_to_hub(dataset_name)
        print("Upload complete!")
    
    return dataset

# Example usage
if __name__ == "__main__":
    # Update these paths to match your setup
    jsonl_file = "/workspace/outputs/lnqa_translated_en_to_mk_nllb.jsonl"  # Path to your JSONL file
    images_dir = "/workspace/outputs/images"           # Path to your images folder
    output_dir = "/workspace/converted_dataset"  # Where to save locally
    
    # Optional: set this to upload directly to hub
    # hub_name = "your-username/your-dataset-name"
    hub_name = "manifold-machines/lnqa-mk"
    
    dataset = convert_jsonl_images_to_parquet(
        jsonl_path=jsonl_file,
        images_folder=images_dir,
        output_path=output_dir,
        dataset_name=hub_name
    )
    
    # Test the dataset
    print("\nTesting dataset:")
    print(f"First entry image shape: {dataset[0]['image'].size}")
    print(f"First QA pair: {dataset[0]['translated_qa'][0]}")