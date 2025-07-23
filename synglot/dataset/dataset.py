class Dataset:
    """Core dataset class for handling multilingual data."""
    
    def __init__(self, data=None, source_lang=None, target_lang=None, name=None):
        """
        Initialize dataset with optional data.
        
        Args:
            data (list/dict/str): Input data or path to data
            source_lang (str): Source language code
            target_lang (str): Target language code
            name (str): Optional name for the dataset
        """
        self._data = []
        if isinstance(data, str):
            # Assuming data is a path to a file
            # We'll implement file loading in load_from_file
            # For now, this will be a placeholder
            print(f"File path provided: {data}. Implement loading in load_from_file.")
            pass  # Placeholder for file loading
        elif isinstance(data, (list, dict)):
            self._data = data
        elif data is not None:
            raise ValueError("Unsupported data type. Provide a list, dict, or file path string.")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.name = name
        
    def load_from_huggingface(self, dataset_name, split="train", columns=None, config_name=None):
        """Load data from HuggingFace datasets."""
        from datasets import load_dataset
        
        try:
            # Pass config_name to hf_load_dataset if provided
            hf_dataset = load_dataset(dataset_name, name=config_name, split=split)
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            return

        if columns:
            if not all(col in hf_dataset.column_names for col in columns):
                print(f"One or more specified columns not in dataset. Available columns: {hf_dataset.column_names}")
                # Decide how to handle this: raise error, or load all, or load available?
                # For now, let's load all if columns are mismatched to be safe, or one could raise an error.
                print("Loading all available columns instead.")
                self._data = [row for row in hf_dataset]
            else:
                self._data = [{col: row[col] for col in columns} for row in hf_dataset]
        else:
            # If no columns specified, load all columns
            self._data = [row for row in hf_dataset]

    def load_from_file(self, file_path, format="json"):
        """Load data from local file."""
        try:
            if format == "json":
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            elif format == "csv":
                import csv
                self._data = []
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._data.append(row)
            # elif format == "txt": # Example for a plain text file, one entry per line
            #     with open(file_path, 'r', encoding='utf-8') as f:
            #         self._data = [line.strip() for line in f.readlines()]
            else:
                print(f"Unsupported file format: {format}. Please use 'json' or 'csv'.")
                # Or raise ValueError(f"Unsupported file format: {format}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            # Or raise FileNotFoundError(f"File not found at {file_path}")
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            # Or raise

    def save(self, path, format="json"):
        """Save dataset to file."""
        if not self._data:
            print("Dataset is empty. Nothing to save.")
            return

        try:
            if format == "json":
                import json
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=4)
            elif format == "csv":
                import csv
                if not self._data or not isinstance(self._data[0], dict):
                    print("CSV format requires a list of dictionaries. Cannot save.")
                    return
                
                fieldnames = self._data[0].keys()
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self._data)
            else:
                print(f"Unsupported file format: {format}. Please use 'json' or 'csv'.")
        except Exception as e:
            print(f"Error saving dataset to {path}: {e}")

    def sample(self, n=5, random_state=None):
        """Get n random samples from the dataset."""
        import random

        if not self._data:
            print("Dataset is empty. Cannot sample.")
            return Dataset([], self.source_lang, self.target_lang, self.name)

        if n > len(self._data):
            print(f"Requested sample size {n} is larger than dataset size {len(self._data)}. Returning all data.")
            n = len(self._data)
        
        if random_state is not None:
            random.seed(random_state)
            
        sampled_data = random.sample(self._data, n)
        return Dataset(sampled_data, self.source_lang, self.target_lang, self.name)

    def head(self, n=5):
        """Get first n samples from the dataset."""
        if not self._data:
            print("Dataset is empty. Cannot get head.")
            return Dataset([], self.source_lang, self.target_lang, self.name)
        
        return Dataset(self._data[:n], self.source_lang, self.target_lang, self.name)

    def tail(self, n=5):
        """Get last n samples from the dataset."""
        if not self._data:
            print("Dataset is empty. Cannot get tail.")
            return Dataset([], self.source_lang, self.target_lang, self.name)
        
        return Dataset(self._data[-n:], self.source_lang, self.target_lang, self.name)

    def shuffle(self, random_state=None):
        """Shuffle the dataset in place."""
        import random
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(self._data)
        return self

    def copy(self):
        """Create a deep copy of the dataset."""
        import copy
        new_dataset = Dataset()
        new_dataset._data = copy.deepcopy(self._data)
        new_dataset.source_lang = self.source_lang
        new_dataset.target_lang = self.target_lang
        new_dataset.name = self.name
        return new_dataset

    def filter(self, condition):
        """
        Filter dataset based on a condition function.
        
        Args:
            condition (callable): Function that takes a row and returns True/False
            
        Returns:
            Dataset: New filtered dataset
        """
        filtered_data = [row for row in self._data if condition(row)]
        return Dataset(filtered_data, self.source_lang, self.target_lang, self.name)

    def map(self, transform_fn, columns=None):
        """
        Apply a transformation function to the dataset.
        
        Args:
            transform_fn (callable): Function to apply to each row
            columns (list): Specific columns to apply transformation to
            
        Returns:
            Dataset: New transformed dataset
        """
        if columns is None:
            # Apply to entire row
            transformed_data = [transform_fn(row) for row in self._data]
        else:
            # Apply to specific columns
            transformed_data = []
            for row in self._data:
                new_row = row.copy()
                for col in columns:
                    if col in row:
                        new_row[col] = transform_fn(row[col])
                transformed_data.append(new_row)
        
        return Dataset(transformed_data, self.source_lang, self.target_lang, self.name)

    def select_columns(self, columns):
        """
        Select specific columns from the dataset.
        
        Args:
            columns (list): List of column names to select
            
        Returns:
            Dataset: New dataset with only selected columns
        """
        if not self._data:
            return Dataset([], self.source_lang, self.target_lang, self.name)
        
        selected_data = []
        for row in self._data:
            selected_row = {col: row.get(col) for col in columns if col in row}
            selected_data.append(selected_row)
        
        return Dataset(selected_data, self.source_lang, self.target_lang, self.name)

    def drop_columns(self, columns):
        """
        Drop specific columns from the dataset.
        
        Args:
            columns (list): List of column names to drop
            
        Returns:
            Dataset: New dataset without dropped columns
        """
        if not self._data:
            return Dataset([])
        
        dropped_data = []
        for row in self._data:
            new_row = {k: v for k, v in row.items() if k not in columns}
            dropped_data.append(new_row)
        
        return Dataset(dropped_data, self.source_lang, self.target_lang, self.name)

    def rename_columns(self, column_mapping):
        """
        Rename columns in the dataset.
        
        Args:
            column_mapping (dict): Mapping of old_name -> new_name
            
        Returns:
            Dataset: New dataset with renamed columns
        """
        if not self._data:
            return Dataset([])
        
        renamed_data = []
        for row in self._data:
            new_row = {}
            for k, v in row.items():
                new_key = column_mapping.get(k, k)
                new_row[new_key] = v
            renamed_data.append(new_row)
        
        return Dataset(renamed_data, self.source_lang, self.target_lang, self.name)

    def add_column(self, column_name, values=None, default_value=None):
        """
        Add a new column to the dataset.
        
        Args:
            column_name (str): Name of the new column
            values (list): List of values for the new column
            default_value: Default value if values list is shorter than dataset
            
        Returns:
            Dataset: New dataset with added column
        """
        if not self._data:
            return Dataset([])
        
        new_data = []
        for i, row in enumerate(self._data):
            new_row = row.copy()
            if values is not None and i < len(values):
                new_row[column_name] = values[i]
            else:
                new_row[column_name] = default_value
            new_data.append(new_row)
        
        return Dataset(new_data, self.source_lang, self.target_lang, self.name)

    def sort(self, key_column, reverse=False):
        """
        Sort dataset by a column.
        
        Args:
            key_column (str): Column name to sort by
            reverse (bool): Sort in descending order if True
            
        Returns:
            Dataset: New sorted dataset
        """
        if not self._data:
            return Dataset([])
        
        try:
            sorted_data = sorted(self._data, key=lambda x: x.get(key_column), reverse=reverse)
            return Dataset(sorted_data, self.source_lang, self.target_lang, self.name)
        except Exception as e:
            print(f"Error sorting by column '{key_column}': {e}")
            return self.copy()

    def unique_values(self, column):
        """
        Get unique values in a column.
        
        Args:
            column (str): Column name
            
        Returns:
            list: List of unique values
        """
        if not self._data:
            return []
        
        values = [row.get(column) for row in self._data if column in row]
        return list(set(values))

    def value_counts(self, column):
        """
        Count occurrences of each value in a column.
        
        Args:
            column (str): Column name
            
        Returns:
            dict: Dictionary with value -> count mapping
        """
        if not self._data:
            return {}
        
        from collections import Counter
        values = [row.get(column) for row in self._data if column in row]
        return dict(Counter(values))

    def group_by(self, column):
        """
        Group dataset by a column.
        
        Args:
            column (str): Column name to group by
            
        Returns:
            dict: Dictionary with group_value -> Dataset mapping
        """
        if not self._data:
            return {}
        
        groups = {}
        for row in self._data:
            group_key = row.get(column)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(row)
        
        # Convert lists to Dataset objects
        return {k: Dataset(v, self.source_lang, self.target_lang, self.name) for k, v in groups.items()}

    def concat(self, other_dataset):
        """
        Concatenate with another dataset.
        
        Args:
            other_dataset (Dataset): Another dataset to concatenate
            
        Returns:
            Dataset: New concatenated dataset
        """
        if not isinstance(other_dataset, Dataset):
            raise TypeError("Can only concatenate with another Dataset object")
        
        combined_data = self._data + other_dataset._data
        return Dataset(combined_data, self.source_lang, self.target_lang, self.name)

    def info(self):
        """Print information about the dataset."""
        if not self._data:
            print("Dataset is empty")
            return
        
        print(f"Dataset Info:")
        print(f"  Number of rows: {len(self._data)}")
        print(f"  Source language: {self.source_lang}")
        print(f"  Target language: {self.target_lang}")
        print(f"  Name: {self.name}")
        
        if isinstance(self._data[0], dict):
            columns = list(self._data[0].keys())
            print(f"  Columns ({len(columns)}): {columns}")
            
            # Show data types and non-null counts
            for col in columns:
                values = [row.get(col) for row in self._data if col in row]
                non_null_count = sum(1 for v in values if v is not None)
                data_types = set(type(v).__name__ for v in values if v is not None)
                print(f"    {col}: {non_null_count}/{len(self._data)} non-null, types: {list(data_types)}")

    def describe(self, column=None):
        """
        Get descriptive statistics for numeric columns.
        
        Args:
            column (str): Specific column to describe, or None for all numeric columns
        """
        if not self._data:
            print("Dataset is empty")
            return
        
        if not isinstance(self._data[0], dict):
            print("Describe only works with dictionary-based data")
            return
        
        columns_to_describe = [column] if column else list(self._data[0].keys())
        
        for col in columns_to_describe:
            values = [row.get(col) for row in self._data if col in row and isinstance(row.get(col), (int, float))]
            
            if not values:
                print(f"Column '{col}': No numeric data")
                continue
            
            print(f"Column '{col}':")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {sum(values) / len(values):.2f}")
            print(f"  Min: {min(values)}")
            print(f"  Max: {max(values)}")
            if len(values) > 1:
                import statistics
                print(f"  Std: {statistics.stdev(values):.2f}")
                print(f"  Median: {statistics.median(values):.2f}")

    @property
    def columns(self):
        """Get column names."""
        if not self._data or not isinstance(self._data[0], dict):
            return []
        return list(self._data[0].keys())

    @property
    def shape(self):
        """Get dataset shape (rows, columns)."""
        if not self._data:
            return (0, 0)
        if isinstance(self._data[0], dict):
            return (len(self._data), len(self._data[0]))
        return (len(self._data), 1)

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self._data)

    def __getitem__(self, idx):
        """Access dataset by index or column name."""
        if not self._data:
            raise IndexError("Dataset is empty.")
        
        # Handle column access by string
        if isinstance(idx, str):
            # Return all values for this column
            if not isinstance(self._data[0], dict):
                raise TypeError("Column access only supported for dictionary-based data")
            if idx not in self._data[0]:
                raise KeyError(f"Column '{idx}' not found in dataset")
            return [row.get(idx) for row in self._data]
        
        # Handle multiple column access
        elif isinstance(idx, list) and all(isinstance(i, str) for i in idx):
            if not isinstance(self._data[0], dict):
                raise TypeError("Column access only supported for dictionary-based data")
            return self.select_columns(idx)
        
        # Handle slicing
        elif isinstance(idx, slice):
            sliced_data = self._data[idx]
            return Dataset(sliced_data, self.source_lang, self.target_lang, self.name)
        
        # Handle single integer index
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._data)
            if idx < 0 or idx >= len(self._data):
                raise IndexError("Dataset index out of range.")
            return self._data[idx]
        
        # Handle tuple for advanced indexing (rows, columns)
        elif isinstance(idx, tuple) and len(idx) == 2:
            row_idx, col_idx = idx
            
            # Get rows first
            if isinstance(row_idx, slice):
                rows = self._data[row_idx]
            elif isinstance(row_idx, int):
                if row_idx < 0:
                    row_idx += len(self._data)
                if row_idx < 0 or row_idx >= len(self._data):
                    raise IndexError("Dataset index out of range.")
                rows = [self._data[row_idx]]
            else:
                raise TypeError("Row index must be integer or slice")
            
            # Handle column selection
            if isinstance(col_idx, str):
                # Single column
                return [row.get(col_idx) for row in rows]
            elif isinstance(col_idx, list) and all(isinstance(i, str) for i in col_idx):
                # Multiple columns
                result_data = []
                for row in rows:
                    selected_row = {col: row.get(col) for col in col_idx if col in row}
                    result_data.append(selected_row)
                return Dataset(result_data, self.source_lang, self.target_lang, self.name)
            else:
                raise TypeError("Column index must be string or list of strings")
        
        else:
            raise TypeError("Dataset indices must be integers, slices, strings, or tuples.")

    def __setitem__(self, idx, value):
        """Set values in the dataset."""
        if not self._data:
            raise IndexError("Dataset is empty.")
        
        # Handle column assignment
        if isinstance(idx, str):
            if not isinstance(self._data[0], dict):
                raise TypeError("Column assignment only supported for dictionary-based data")
            
            if isinstance(value, list):
                if len(value) != len(self._data):
                    raise ValueError(f"Value list length ({len(value)}) must match dataset length ({len(self._data)})")
                for i, row in enumerate(self._data):
                    row[idx] = value[i]
            else:
                # Broadcast single value to all rows
                for row in self._data:
                    row[idx] = value
        
        # Handle single row assignment
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._data)
            if idx < 0 or idx >= len(self._data):
                raise IndexError("Dataset index out of range.")
            self._data[idx] = value
        
        else:
            raise TypeError("Assignment index must be string (column) or integer (row).")

    def __iter__(self):
        """Make dataset iterable."""
        return iter(self._data)

    def __repr__(self):
        """String representation of the dataset."""
        if not self._data:
            return "Dataset(empty)"
        
        n_rows = len(self._data)
        if isinstance(self._data[0], dict):
            n_cols = len(self._data[0])
            return f"Dataset({n_rows} rows, {n_cols} columns)"
        else:
            return f"Dataset({n_rows} rows)"

    def __str__(self):
        """Pretty print the dataset."""
        if not self._data:
            return "Empty Dataset"
        
        # Show first few rows
        n_show = min(5, len(self._data))
        result = f"Dataset with {len(self._data)} rows:\n"
        
        for i in range(n_show):
            result += f"  [{i}]: {self._data[i]}\n"
        
        if len(self._data) > n_show:
            result += f"  ... and {len(self._data) - n_show} more rows"
        
        return result