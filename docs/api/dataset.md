# `synglot.dataset`

This module provides the core dataset handling capabilities.

## `Dataset`

```python
class Dataset:
    def __init__(self, data=None, source_lang=None, target_lang=None)
    def load_from_huggingface(self, dataset_name, split="train", columns=None)
    def load_from_file(self, file_path, format="json")
    def save(self, path, format="json")
    def sample(self, n=5, random_state=None)
    def head(self, n=5)
    def __len__(self)
    def __getitem__(self, idx)
```

### `__init__(self, data=None, source_lang=None, target_lang=None)`
Initializes the dataset.

**Args:**
- `data` (`list`|`dict`|`str`, optional): Input data, which can be a list of samples, a dictionary, or a path to a data file. Defaults to `None`.
- `source_lang` (`str`, optional): Source language code. Defaults to `None`.
- `target_lang` (`str`, optional): Target language code. Defaults to `None`.

### `load_from_huggingface(self, dataset_name, split="train", columns=None)`
Load data from HuggingFace datasets.

**Args:**
- `dataset_name` (`str`): Name of the dataset on HuggingFace Hub.
- `split` (`str`, optional): Dataset split to load (e.g., "train", "test"). Defaults to `"train"`.
- `columns` (`list`, optional): List of column names to load. If `None`, all columns are loaded. Defaults to `None`.

### `load_from_file(self, file_path, format="json")`
Load data from a local file.

**Args:**
- `file_path` (`str`): Path to the data file.
- `format` (`str`, optional): Format of the file ("json", "csv"). Defaults to `"json"`.

### `save(self, path, format="json")`
Save the dataset to a file.

**Args:**
- `path` (`str`): Path to save the file.
- `format` (`str`, optional): Format to save the file in ("json", "csv"). Defaults to `"json"`.

### `sample(self, n=5, random_state=None)`
Get `n` random samples from the dataset.

**Args:**
- `n` (`int`, optional): Number of samples to retrieve. Defaults to `5`.
- `random_state` (`int`, optional): Seed for random number generator for reproducibility. Defaults to `None`.

**Returns:**
- `list`: A list of random samples.

### `head(self, n=5)`
Get the first `n` samples from the dataset.

**Args:**
- `n` (`int`, optional): Number of samples to retrieve. Defaults to `5`.

**Returns:**
- `list`: A list of the first `n` samples.

### `__len__(self)`
Returns the number of samples in the dataset.

### `__getitem__(self, idx)`
Access dataset samples by index or slice. 