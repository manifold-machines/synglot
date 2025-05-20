# `synglot.analyze`

This module provides tools for analyzing datasets.

## `Analyzer`

```python
class Analyzer:
    def __init__(self, dataset: Dataset)
    def get_stats(self) -> Dict[str, Any]
    def length_distribution(self) -> Dict[int, int]
    def token_count(self, tokenizer: Optional[callable] = None) -> int
    def vocabulary(self, tokenizer: Optional[callable] = None) -> List[str]
    def random_sample(self, n: int = 5) -> List[Dict[str, Any]]
```

### `__init__(self, dataset: Dataset)`
Initializes the analyzer with a dataset.

**Args:**
- `dataset` (`Dataset`): The dataset to analyze.

### `get_stats(self) -> Dict[str, Any]`
Get basic statistics about the dataset.

**Returns:**
- `Dict[str, Any]`: A dictionary containing:
    - `'size'`: Number of samples in the dataset.
    - `'avg_length'`: Average length of samples (character-based).
    - `'vocab_size'`: Size of the vocabulary (based on default tokenization).
    - `'total_tokens'`: Total number of tokens (based on default tokenization).

### `length_distribution(self) -> Dict[int, int]`
Get distribution of sample lengths (character-based).

**Returns:**
- `Dict[int, int]`: A dictionary where keys are lengths and values are counts.

### `token_count(self, tokenizer: Optional[callable] = None) -> int`
Count total tokens in the dataset. If no tokenizer is provided, splits by whitespace.

**Args:**
- `tokenizer` (`Optional[callable]`): A function to tokenize text. Defaults to splitting by whitespace.

**Returns:**
- `int`: Total number of tokens in the dataset.

### `vocabulary(self, tokenizer: Optional[callable] = None) -> List[str]`
Extract vocabulary from the dataset. If no tokenizer is provided, splits by whitespace.

**Args:**
- `tokenizer` (`Optional[callable]`): A function to tokenize text. Defaults to splitting by whitespace.

**Returns:**
- `List[str]`: A sorted list of unique tokens in the dataset.

### `random_sample(self, n: int = 5) -> List[Dict[str, Any]]`
Get random samples from dataset.

**Args:**
- `n` (`int`): Number of random samples to retrieve.

**Returns:**
- `List[Dict[str, Any]]`: A list of random samples. 