class Dataset:
    """Core dataset class for handling multilingual data."""
    
    def __init__(self, data=None, source_lang=None, target_lang=None):
        """
        Initialize dataset with optional data.
        
        Args:
            data (list/dict/str): Input data or path to data
            source_lang (str): Source language code
            target_lang (str): Target language code
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
            return []

        if n > len(self._data):
            print(f"Requested sample size {n} is larger than dataset size {len(self._data)}. Returning all data.")
            n = len(self._data)
        
        if random_state is not None:
            random.seed(random_state)
            
        return random.sample(self._data, n)

    def head(self, n=5):
        """Get first n samples from the dataset."""
        if not self._data:
            print("Dataset is empty. Cannot get head.")
            return []
        
        return self._data[:n]

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self._data)

    def __getitem__(self, idx):
        """Access dataset by index."""
        if not self._data:
            raise IndexError("Dataset is empty.")
            
        if isinstance(idx, slice):
            # Handle slicing
            return self._data[idx]
        elif isinstance(idx, int):
            # Handle single integer index
            if idx < 0:
                 idx += len(self._data)
            if idx < 0 or idx >= len(self._data):
                raise IndexError("Dataset index out of range.")
            return self._data[idx]
        else:
            raise TypeError("Dataset indices must be integers or slices.")