# Example workflow for dataset analysis
from synglot import Dataset, Analyzer

# Load dataset
dataset = Dataset()
dataset.load_from_file("swahili_conversations.json")

# Create analyzer
analyzer = Analyzer(dataset)

# Get basic statistics
stats = analyzer.get_stats()
print(f"Dataset size: {stats['size']}")
print(f"Average length: {stats['avg_length']}")
print(f"Vocabulary size: {stats['vocab_size']}")

# Get random samples
samples = analyzer.random_sample(n=3)
for i, sample in enumerate(samples):
    print(f"Sample {i+1}: {sample['text'][:100]}...") 