# `synglot.dataset`

This module provides comprehensive dataset handling capabilities for multilingual data processing.

## `Dataset`

```python
class Dataset:
    def __init__(self, data=None, source_lang=None, target_lang=None)
    def load_from_huggingface(self, dataset_name, split="train", columns=None, config_name=None)
    def load_from_file(self, file_path, format="json")
    def save(self, path, format="json")
    def sample(self, n=5, random_state=None)
    def head(self, n=5)
    def tail(self, n=5)
    def shuffle(self, random_state=None)
    def copy(self)
    def filter(self, condition)
    def map(self, transform_fn, columns=None)
    def select_columns(self, columns)
    def drop_columns(self, columns)
    def rename_columns(self, column_mapping)
    def add_column(self, column_name, values=None, default_value=None)
    def sort(self, key_column, reverse=False)
    def unique_values(self, column)
    def value_counts(self, column)
    def group_by(self, column)
    def concat(self, other_dataset)
    def info(self)
    def describe(self, column=None)
    @property
    def columns(self)
    @property
    def shape(self)
    def __len__(self)
    def __getitem__(self, idx)
    def __setitem__(self, idx, value)
    def __iter__(self)
```

**Core dataset class for handling multilingual data with comprehensive data manipulation capabilities.**

### `__init__(self, data=None, source_lang=None, target_lang=None)`
Initialize dataset with optional data.

**Args:**
- `data` (`list`|`dict`|`str`, optional): Input data, which can be a list of samples, a dictionary, or a path to a data file. Defaults to `None`.
- `source_lang` (`str`, optional): Source language code. Defaults to `None`.
- `target_lang` (`str`, optional): Target language code. Defaults to `None`.

## Data Loading Methods

### `load_from_huggingface(self, dataset_name, split="train", columns=None, config_name=None)`
Load data from HuggingFace datasets.

**Args:**
- `dataset_name` (`str`): Name of the dataset on HuggingFace Hub.
- `split` (`str`, optional): Dataset split to load (e.g., "train", "test"). Defaults to `"train"`.
- `columns` (`list`, optional): List of column names to load. If `None`, all columns are loaded. Defaults to `None`.
- `config_name` (`str`, optional): Configuration name for datasets with multiple configs. Defaults to `None`.

**Example:**
```python
from synglot.dataset import Dataset

dataset = Dataset()
dataset.load_from_huggingface("squad", split="train", columns=["question", "context"])
```

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

## Data Sampling and Access Methods

### `sample(self, n=5, random_state=None)`
Get `n` random samples from the dataset.

**Args:**
- `n` (`int`, optional): Number of samples to retrieve. Defaults to `5`.
- `random_state` (`int`, optional): Seed for random number generator for reproducibility. Defaults to `None`.

**Returns:**
- `Dataset`: A new dataset containing random samples.

### `head(self, n=5)`
Get the first `n` samples from the dataset.

**Args:**
- `n` (`int`, optional): Number of samples to retrieve. Defaults to `5`.

**Returns:**
- `Dataset`: A new dataset containing the first `n` samples.

### `tail(self, n=5)`
Get the last `n` samples from the dataset.

**Args:**
- `n` (`int`, optional): Number of samples to retrieve. Defaults to `5`.

**Returns:**
- `Dataset`: A new dataset containing the last `n` samples.

### `shuffle(self, random_state=None)`
Shuffle the dataset in place.

**Args:**
- `random_state` (`int`, optional): Seed for random number generator. Defaults to `None`.

**Returns:**
- `Dataset`: The same dataset (shuffled in place) for method chaining.

### `copy(self)`
Create a deep copy of the dataset.

**Returns:**
- `Dataset`: A new dataset that is a deep copy of the original.

## Data Filtering and Transformation Methods

### `filter(self, condition)`
Filter dataset based on a condition function.

**Args:**
- `condition` (`callable`): Function that takes a row and returns `True`/`False`.

**Returns:**
- `Dataset`: New filtered dataset.

**Example:**
```python
# Filter rows where 'score' > 0.8
filtered = dataset.filter(lambda row: row.get('score', 0) > 0.8)
```

### `map(self, transform_fn, columns=None)`
Apply a transformation function to the dataset.

**Args:**
- `transform_fn` (`callable`): Function to apply to each row or column value.
- `columns` (`list`, optional): Specific columns to apply transformation to. If `None`, applies to entire row.

**Returns:**
- `Dataset`: New transformed dataset.

**Example:**
```python
# Transform text to lowercase
transformed = dataset.map(lambda text: text.lower(), columns=['text'])
```

## Column Manipulation Methods

### `select_columns(self, columns)`
Select specific columns from the dataset.

**Args:**
- `columns` (`list`): List of column names to select.

**Returns:**
- `Dataset`: New dataset with only selected columns.

### `drop_columns(self, columns)`
Drop specific columns from the dataset.

**Args:**
- `columns` (`list`): List of column names to drop.

**Returns:**
- `Dataset`: New dataset without dropped columns.

### `rename_columns(self, column_mapping)`
Rename columns in the dataset.

**Args:**
- `column_mapping` (`dict`): Mapping of old_name -> new_name.

**Returns:**
- `Dataset`: New dataset with renamed columns.

**Example:**
```python
renamed = dataset.rename_columns({'old_name': 'new_name', 'text': 'content'})
```

### `add_column(self, column_name, values=None, default_value=None)`
Add a new column to the dataset.

**Args:**
- `column_name` (`str`): Name of the new column.
- `values` (`list`, optional): List of values for the new column.
- `default_value`: Default value if values list is shorter than dataset.

**Returns:**
- `Dataset`: New dataset with added column.

## Data Analysis Methods

### `sort(self, key_column, reverse=False)`
Sort dataset by a column.

**Args:**
- `key_column` (`str`): Column name to sort by.
- `reverse` (`bool`, optional): Sort in descending order if `True`. Defaults to `False`.

**Returns:**
- `Dataset`: New sorted dataset.

### `unique_values(self, column)`
Get unique values in a column.

**Args:**
- `column` (`str`): Column name.

**Returns:**
- `list`: List of unique values.

### `value_counts(self, column)`
Count occurrences of each value in a column.

**Args:**
- `column` (`str`): Column name.

**Returns:**
- `dict`: Dictionary with value -> count mapping.

### `group_by(self, column)`
Group dataset by a column.

**Args:**
- `column` (`str`): Column name to group by.

**Returns:**
- `dict`: Dictionary with group_value -> Dataset mapping.

### `concat(self, other_dataset)`
Concatenate with another dataset.

**Args:**
- `other_dataset` (`Dataset`): Another dataset to concatenate.

**Returns:**
- `Dataset`: New concatenated dataset.

## Information and Statistics Methods

### `info(self)`
Print comprehensive information about the dataset including column types and non-null counts.

### `describe(self, column=None)`
Get descriptive statistics for numeric columns.

**Args:**
- `column` (`str`, optional): Specific column to describe, or `None` for all numeric columns.

**Example:**
```python
dataset.info()  # Print dataset overview
dataset.describe('score')  # Statistics for 'score' column
dataset.describe()  # Statistics for all numeric columns
```

## Properties

### `columns`
Get column names.

**Returns:**
- `list`: List of column names.

### `shape`
Get dataset shape (rows, columns).

**Returns:**
- `tuple`: (number_of_rows, number_of_columns).

## Magic Methods and Advanced Indexing

### `__len__(self)`
Returns the number of samples in the dataset.

### `__getitem__(self, idx)`
Access dataset samples with advanced indexing support.

**Supported Index Types:**
- **Integer**: Access single row by index
- **Slice**: Access multiple rows
- **String**: Access all values in a column
- **List of strings**: Select multiple columns
- **Tuple (row, col)**: Advanced indexing for specific rows and columns

**Examples:**
```python
# Single row access
row = dataset[0]

# Slice access
subset = dataset[10:20]

# Column access
texts = dataset['text']

# Multiple columns
subset = dataset[['text', 'label']]

# Advanced indexing (rows 0-5, columns 'text' and 'label')
subset = dataset[0:5, ['text', 'label']]
```

### `__setitem__(self, idx, value)`
Set values in the dataset.

**Examples:**
```python
# Set entire column
dataset['new_column'] = [1, 2, 3, ...]

# Set single value
dataset['column'] = "same_value_for_all"

# Set single row
dataset[0] = {"text": "new text", "label": "new label"}
```

### `__iter__(self)`
Make dataset iterable.

**Example:**
```python
for row in dataset:
    print(row)
```

## Usage Examples

### Basic Dataset Operations

```python
from synglot.dataset import Dataset

# Create and load data
dataset = Dataset()
dataset.load_from_huggingface("imdb", split="train", columns=["text", "label"])

# Basic info
print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {dataset.columns}")
dataset.info()
```

### Data Manipulation

```python
# Filter positive reviews
positive_reviews = dataset.filter(lambda row: row['label'] == 1)

# Transform text to lowercase
dataset_lower = dataset.map(lambda text: text.lower(), columns=['text'])

# Add new column
dataset_with_length = dataset.add_column('text_length', 
                                        values=[len(text) for text in dataset['text']])

# Sort by text length
sorted_dataset = dataset_with_length.sort('text_length', reverse=True)
```

### Data Analysis

```python
# Get statistics
label_counts = dataset.value_counts('label')
unique_labels = dataset.unique_values('label')

# Group by label
grouped = dataset.group_by('label')

# Describe numeric columns
dataset.describe('text_length')
```

### Advanced Indexing

```python
# Sample data for quick inspection
sample_data = dataset.sample(10, random_state=42)

# Access specific rows and columns
first_10_texts = dataset[0:10, 'text']
specific_columns = dataset[['text', 'label']]

# Iterate through dataset
for i, row in enumerate(dataset):
    if i >= 5:  # Only first 5 rows
        break
    print(f"Text: {row['text'][:50]}... Label: {row['label']}")
```

### Dataset Combination

```python
# Split dataset
train_data = dataset.head(8000)
test_data = dataset.tail(2000)

# Combine datasets
combined = train_data.concat(test_data)

# Create shuffled copy
shuffled_dataset = dataset.copy().shuffle(random_state=42)
``` 