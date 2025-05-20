# Example workflow for synthetic data generation
from synglot import Dataset, HFGenerator, Config

# Create generator configuration
config = Config({
    "temperature": 0.7,
    "domain": "science",
    "distribution": "uniform",
    "max_length": 500
})

# Initialize generator
generator = HFGenerator(target_lang="am", config=config, api_key="YOUR_API_KEY")

# Generate pretraining dataset
pretraining_data = generator.generate_pretraining(
    domain="science", 
    n_samples=1000,
    min_length=100,
    max_length=500
)

# Create dataset from generated data
dataset = Dataset(data=pretraining_data, target_lang="am")

# Save dataset
dataset.save("amharic_science_pretraining.json") 