import unittest
import os
import json
import csv
from unittest.mock import patch, mock_open, MagicMock

# Adjust the path to import the Dataset class correctly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synglot.dataset.dataset import Dataset

class TestDataset(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.sample_data_list = [
            {"id": 1, "text": "Hello world"},
            {"id": 2, "text": "This is a test"},
            {"id": 3, "text": "Another sentence"}
        ]
        self.source_lang = "en"
        self.target_lang = "fr"

    def test_init_empty(self):
        """Test initializing an empty dataset."""
        dataset = Dataset()
        self.assertEqual(len(dataset._data), 0)
        self.assertIsNone(dataset.source_lang)
        self.assertIsNone(dataset.target_lang)

    def test_init_with_list_data(self):
        """Test initializing with a list of data."""
        dataset = Dataset(data=self.sample_data_list, source_lang=self.source_lang, target_lang=self.target_lang)
        self.assertEqual(dataset._data, self.sample_data_list)
        self.assertEqual(dataset.source_lang, self.source_lang)
        self.assertEqual(dataset.target_lang, self.target_lang)

    def test_init_with_file_path_placeholder(self):
        """Test initializing with a file path (current placeholder behavior)."""
        # This test reflects the current placeholder behavior in __init__ for file paths
        # It might need adjustment if __init__'s file handling changes
        with patch('builtins.print') as mocked_print:
            dataset = Dataset(data="some/file/path.json")
            mocked_print.assert_called_with("File path provided: some/file/path.json. Implement loading in load_from_file.")
            self.assertEqual(len(dataset._data), 0) # Expecting _data to be empty as loading is deferred

    def test_init_with_unsupported_data_type(self):
        """Test initializing with an unsupported data type."""
        with self.assertRaises(ValueError):
            Dataset(data=12345) # Integer is not supported

    # --- __len__ tests ---
    def test_len_empty(self):
        """Test __len__ on an empty dataset."""
        dataset = Dataset()
        self.assertEqual(len(dataset), 0)

    def test_len_with_data(self):
        """Test __len__ on a dataset with data."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(len(dataset), len(self.sample_data_list))

    # --- __getitem__ tests ---
    def test_getitem_single_integer(self):
        """Test __getitem__ with a single integer index."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(dataset[0], self.sample_data_list[0])
        self.assertEqual(dataset[1], self.sample_data_list[1])

    def test_getitem_negative_index(self):
        """Test __getitem__ with a negative integer index."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(dataset[-1], self.sample_data_list[-1])
        self.assertEqual(dataset[-2], self.sample_data_list[-2])
    
    def test_getitem_slice(self):
        """Test __getitem__ with a slice."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(dataset[0:2], self.sample_data_list[0:2])
        self.assertEqual(dataset[:2], self.sample_data_list[:2])
        self.assertEqual(dataset[1:], self.sample_data_list[1:])

    def test_getitem_out_of_range(self):
        """Test __getitem__ with an out-of-range index."""
        dataset = Dataset(data=self.sample_data_list)
        with self.assertRaises(IndexError):
            _ = dataset[len(self.sample_data_list)]
        with self.assertRaises(IndexError):
            _ = dataset[-len(self.sample_data_list) - 1]

    def test_getitem_empty_dataset(self):
        """Test __getitem__ on an empty dataset."""
        dataset = Dataset()
        with self.assertRaises(IndexError):
            _ = dataset[0]

    def test_getitem_invalid_type(self):
        """Test __getitem__ with an invalid index type."""
        dataset = Dataset(data=self.sample_data_list)
        with self.assertRaises(TypeError):
            _ = dataset["invalid_index"]

    # --- head tests ---
    def test_head_basic(self):
        """Test head method basic functionality."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(dataset.head(2), self.sample_data_list[:2])

    def test_head_n_larger_than_data(self):
        """Test head method when n is larger than dataset size."""
        dataset = Dataset(data=self.sample_data_list)
        self.assertEqual(dataset.head(10), self.sample_data_list)

    def test_head_empty_dataset(self):
        """Test head method on an empty dataset."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            self.assertEqual(dataset.head(), [])
            mocked_print.assert_called_with("Dataset is empty. Cannot get head.")
            
    def test_head_default_n(self):
        """Test head method with default n."""
        dataset = Dataset(data=self.sample_data_list * 3) # Make data longer than default n=5
        self.assertEqual(dataset.head(), (self.sample_data_list * 3)[:5])


    # --- sample tests ---
    def test_sample_basic(self):
        """Test sample method basic functionality."""
        dataset = Dataset(data=self.sample_data_list)
        sampled_data = dataset.sample(2)
        self.assertEqual(len(sampled_data), 2)
        for item in sampled_data:
            self.assertIn(item, self.sample_data_list)

    def test_sample_with_random_state(self):
        """Test sample method with random_state for reproducibility."""
        dataset = Dataset(data=self.sample_data_list * 10) # Larger data for better sampling test
        sample1 = dataset.sample(5, random_state=42)
        sample2 = dataset.sample(5, random_state=42)
        self.assertEqual(sample1, sample2)
    
    def test_sample_n_larger_than_data(self):
        """Test sample method when n is larger than dataset size."""
        dataset = Dataset(data=self.sample_data_list)
        with patch('builtins.print') as mocked_print:
            sampled_data = dataset.sample(10)
            self.assertEqual(len(sampled_data), len(self.sample_data_list))
            mocked_print.assert_called_with(f"Requested sample size 10 is larger than dataset size {len(self.sample_data_list)}. Returning all data.")
            # Check if all original items are present, order might change due to sample
            for item in self.sample_data_list:
                self.assertIn(item, sampled_data)


    def test_sample_empty_dataset(self):
        """Test sample method on an empty dataset."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            self.assertEqual(dataset.sample(), [])
            mocked_print.assert_called_with("Dataset is empty. Cannot sample.")

    # --- load_from_file tests ---
    @patch("builtins.open", new_callable=mock_open)
    def test_load_from_file_json(self, mocked_file):
        """Test loading from a JSON file."""
        json_data = json.dumps(self.sample_data_list)
        mocked_file.return_value.read.return_value = json_data
        
        dataset = Dataset()
        dataset.load_from_file("dummy.json", format="json")
        
        mocked_file.assert_called_once_with("dummy.json", 'r', encoding='utf-8')
        self.assertEqual(dataset._data, self.sample_data_list)

    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.DictReader")
    def test_load_from_file_csv(self, mock_dict_reader, mocked_file):
        """Test loading from a CSV file."""
        # Mock DictReader to return sample data
        mock_dict_reader.return_value = iter(self.sample_data_list)

        dataset = Dataset()
        dataset.load_from_file("dummy.csv", format="csv")

        mocked_file.assert_called_once_with("dummy.csv", 'r', encoding='utf-8', newline='')
        self.assertEqual(dataset._data, self.sample_data_list)

    def test_load_from_file_not_found(self):
        """Test loading a non-existent file."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.load_from_file("non_existent_file.json", format="json")
            mocked_print.assert_any_call("Error: File not found at non_existent_file.json")

    def test_load_from_file_unsupported_format(self):
        """Test loading an unsupported file format."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.load_from_file("dummy.txt", format="txt")
            mocked_print.assert_called_with("Unsupported file format: txt. Please use 'json' or 'csv'.")

    @patch("builtins.open", new_callable=mock_open)
    def test_load_from_file_json_decode_error(self, mocked_file):
        """Test loading a malformed JSON file."""
        mocked_file.return_value.read.return_value = "this is not valid json"
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.load_from_file("malformed.json", format="json")
            # We check if 'Error loading data' is part of any call, as the exact error message might vary
            self.assertTrue(any("Error loading data from malformed.json" in call_args[0][0] for call_args in mocked_print.call_args_list))

    # --- save tests ---
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_json(self, mock_json_dump, mocked_file):
        """Test saving to a JSON file."""
        dataset = Dataset(data=self.sample_data_list)
        dataset.save("output.json", format="json")

        mocked_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(self.sample_data_list, mocked_file.return_value, ensure_ascii=False, indent=4)

    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.DictWriter")
    def test_save_csv(self, mock_csv_writer, mocked_file):
        """Test saving to a CSV file."""
        dataset = Dataset(data=self.sample_data_list)
        # Mock the DictWriter instance and its methods
        mock_writer_instance = MagicMock()
        mock_csv_writer.return_value = mock_writer_instance

        dataset.save("output.csv", format="csv")

        mocked_file.assert_called_once_with("output.csv", 'w', encoding='utf-8', newline='')
        mock_csv_writer.assert_called_once_with(mocked_file.return_value, fieldnames=self.sample_data_list[0].keys())
        mock_writer_instance.writeheader.assert_called_once()
        mock_writer_instance.writerows.assert_called_once_with(self.sample_data_list)
        
    def test_save_empty_dataset(self):
        """Test saving an empty dataset."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.save("empty.json")
            mocked_print.assert_called_with("Dataset is empty. Nothing to save.")

    def test_save_unsupported_format(self):
        """Test saving with an unsupported format."""
        dataset = Dataset(data=self.sample_data_list)
        with patch('builtins.print') as mocked_print:
            dataset.save("output.txt", format="txt")
            mocked_print.assert_called_with("Unsupported file format: txt. Please use 'json' or 'csv'.")

    def test_save_csv_invalid_data_type(self):
        """Test saving to CSV when data is not a list of dicts."""
        dataset = Dataset(data=["string1", "string2"]) # Data is list of strings
        with patch('builtins.print') as mocked_print:
             dataset.save("output.csv", format="csv")
             mocked_print.assert_any_call("CSV format requires a list of dictionaries. Cannot save.")
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump", side_effect=IOError("Disk full"))
    def test_save_json_io_error(self, mock_json_dump, mocked_file):
        """Test saving to JSON with an IO error."""
        dataset = Dataset(data=self.sample_data_list)
        with patch('builtins.print') as mocked_print:
            dataset.save("output.json", format="json")
            self.assertTrue(any("Error saving dataset to output.json: Disk full" in call_args[0][0] for call_args in mocked_print.call_args_list))


    # --- load_from_huggingface tests ---
    @patch("datasets.load_dataset")
    def test_load_from_huggingface_basic(self, mock_load_dataset):
        """Test basic loading from HuggingFace."""
        hf_mock_data = [{"text": "sample 1"}, {"text": "sample 2"}]
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.column_names = ["text"]
        # Make the mock_dataset_obj iterable
        mock_dataset_obj.__iter__.return_value = iter(hf_mock_data)
        
        mock_load_dataset.return_value = mock_dataset_obj
        
        dataset = Dataset()
        dataset.load_from_huggingface("test_dataset", split="train")
        
        mock_load_dataset.assert_called_once_with("test_dataset", split="train")
        self.assertEqual(dataset._data, hf_mock_data)

    @patch("datasets.load_dataset")
    def test_load_from_huggingface_with_columns(self, mock_load_dataset):
        """Test loading from HuggingFace with specified columns."""
        hf_mock_data_full = [
            {"text": "sample 1", "label": 0, "id": "a"}, 
            {"text": "sample 2", "label": 1, "id": "b"}
        ]
        expected_data_subset = [
            {"text": "sample 1", "id": "a"},
            {"text": "sample 2", "id": "b"}
        ]
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.column_names = ["text", "label", "id"]
        mock_dataset_obj.__iter__.return_value = iter(hf_mock_data_full)
        
        mock_load_dataset.return_value = mock_dataset_obj
        
        dataset = Dataset()
        dataset.load_from_huggingface("test_dataset", columns=["text", "id"])
        
        self.assertEqual(dataset._data, expected_data_subset)

    @patch("datasets.load_dataset")
    def test_load_from_huggingface_mismatched_columns(self, mock_load_dataset):
        """Test loading from HuggingFace when specified columns don't exist (should load all)."""
        hf_mock_data_full = [
            {"text": "sample 1", "label": 0}, 
            {"text": "sample 2", "label": 1}
        ]
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.column_names = ["text", "label"]
        mock_dataset_obj.__iter__.return_value = iter(hf_mock_data_full)
        
        mock_load_dataset.return_value = mock_dataset_obj
        
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.load_from_huggingface("test_dataset", columns=["text", "non_existent_col"])
            mocked_print.assert_any_call("One or more specified columns not in dataset. Available columns: ['text', 'label']")
            mocked_print.assert_any_call("Loading all available columns instead.")
            self.assertEqual(dataset._data, hf_mock_data_full) # Expect all data

    @patch("datasets.load_dataset", side_effect=Exception("HF Load Error"))
    def test_load_from_huggingface_error(self, mock_load_dataset):
        """Test error handling when loading from HuggingFace fails."""
        dataset = Dataset()
        with patch('builtins.print') as mocked_print:
            dataset.load_from_huggingface("non_existent_dataset")
            mocked_print.assert_called_with("Error loading dataset from HuggingFace: HF Load Error")
            self.assertEqual(dataset._data, []) # Data should remain empty or unchanged

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
