from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import Counter
import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset  # HuggingFace datasets

class Analyzer:
    """Dataset analysis tools."""
    
    def __init__(self, dataset: 'Dataset'):
        """
        Initialize analyzer.
        
        Args:
            dataset (Dataset): HuggingFace Dataset to analyze
        """
        self.dataset = dataset

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'size': Number of samples in the dataset.
                - 'avg_length': Average length of samples (character-based).
                - 'vocab_size': Size of the vocabulary (based on default tokenization).
                - 'total_tokens': Total number of tokens (based on default tokenization).
        """
        size = len(self.dataset)
        if size == 0:
            return {
                'size': 0,
                'avg_length': 0,
                'vocab_size': 0,
                'total_tokens': 0,
            }

        lengths = [len(sample.get('text', '')) for sample in self.dataset]
        avg_length = np.mean(lengths) if lengths else 0
        
        # For vocab_size and total_tokens, use the vocabulary and token_count methods
        # Using default space-based tokenizer for these stats if no specific tokenizer is implied
        vocab = self.vocabulary() # Uses default tokenizer
        token_counts = self.token_count() # Uses default tokenizer

        return {
            'size': size,
            'avg_length': avg_length,
            'vocab_size': len(vocab),
            'total_tokens': token_counts,
        }

    def length_distribution(self) -> Dict[int, int]:
        """Get distribution of sample lengths (character-based).

        Returns:
            Dict[int, int]: A dictionary where keys are lengths and values are counts.
        """
        if not self.dataset:
            return {}
        lengths = [len(sample.get('text', '')) for sample in self.dataset]
        return dict(Counter(lengths))

    def token_count(self, tokenizer: Optional[callable] = None) -> int:
        """
        Count total tokens in the dataset.
        If no tokenizer is provided, splits by whitespace.
        Assumes samples are dictionaries with a 'text' key.

        Args:
            tokenizer (Optional[callable]): A function to tokenize text. 
                                            Defaults to splitting by whitespace.

        Returns:
            int: Total number of tokens in the dataset.
        """
        if not self.dataset:
            return 0

        total_tokens = 0
        
        if tokenizer is None:
            # Default to splitting by whitespace
            tokenizer = lambda text: text.split()

        for sample in self.dataset:
            text = sample.get('text', '')
            if text: # Ensure text is not empty
                 tokens = tokenizer(text)
                 total_tokens += len(tokens)
        return total_tokens

    def vocabulary(self, tokenizer: Optional[callable] = None) -> List[str]:
        """
        Extract vocabulary from the dataset.
        If no tokenizer is provided, splits by whitespace.
        Assumes samples are dictionaries with a 'text' key.

        Args:
            tokenizer (Optional[callable]): A function to tokenize text. 
                                            Defaults to splitting by whitespace.

        Returns:
            List[str]: A sorted list of unique tokens in the dataset.
        """
        if not self.dataset:
            return []

        all_tokens = Counter()
        
        if tokenizer is None:
            # Default to splitting by whitespace
            tokenizer = lambda text: text.split()
            
        for sample in self.dataset:
            text = sample.get('text', '')
            if text: # Ensure text is not empty
                tokens = tokenizer(text)
                all_tokens.update(tokens)
        
        return sorted(list(all_tokens.keys()))

    def random_sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get random samples from dataset.

        Args:
            n (int): Number of random samples to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of random samples.
        """
        # HuggingFace datasets have a shuffle method and can be sliced
        import random
        
        if len(self.dataset) == 0:
            return []
        
        if len(self.dataset) <= n:
            return list(self.dataset)
        
        # Create indices for random sampling
        indices = random.sample(range(len(self.dataset)), n)
        return [self.dataset[i] for i in indices] 