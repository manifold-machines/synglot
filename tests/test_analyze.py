import unittest
import os
import sys
from collections import Counter
import numpy as np
from unittest.mock import MagicMock, PropertyMock

# Adjust the path to import the Analyzer and Dataset classes correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synglot.analyze.analyzer import Analyzer
# A mock or simplified Dataset class for testing purposes
# We'll use MagicMock for the dataset instance, but if Dataset class is simple,
# we could define a minimal one here or import the actual one.
# For now, let's assume we will mock its relevant behavior.
# from synglot.dataset.dataset import Dataset # If using the actual Dataset

class MockDataset:
    def __init__(self, data=None):
        self._data = data if data is not None else []

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def sample(self, n=5, random_state=None):
        import random
        if random_state is not None:
            random.seed(random_state)
        if n >= len(self._data):
            return list(self._data) # Return a copy
        return random.sample(self._data, n)


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.empty_dataset = MockDataset([])
        self.sample_data_list = [
            {"text": "Hello world"},
            {"text": "This is a test"},
            {"text": "Another sentence here"},
            {"text": "Hello world again"} # Duplicate for vocab testing
        ]
        self.dataset_with_data = MockDataset(self.sample_data_list)
        
        self.data_for_length_dist = [
            {"text": "Short"}, # 5
            {"text": "Medium length"}, # 13
            {"text": "Short"}, # 5
            {"text": "Very very long sentence example"} # 31
        ]
        self.dataset_for_length_dist = MockDataset(self.data_for_length_dist)

        self.dataset_with_empty_text = MockDataset([
            {"text": "Valid"},
            {"text": ""}, # Empty string
            {}, # No 'text' key
            {"other_key": "value"} # No 'text' key
        ])


    def test_init(self):
        """Test Analyzer initialization."""
        analyzer = Analyzer(self.dataset_with_data)
        self.assertIsNotNone(analyzer.dataset)
        self.assertEqual(analyzer.dataset, self.dataset_with_data)

    def test_get_stats_empty_dataset(self):
        """Test get_stats on an empty dataset."""
        analyzer = Analyzer(self.empty_dataset)
        stats = analyzer.get_stats()
        expected_stats = {
            'size': 0,
            'avg_length': 0,
            'vocab_size': 0,
            'total_tokens': 0,
        }
        self.assertEqual(stats, expected_stats)

    def test_get_stats_with_data(self):
        """Test get_stats on a dataset with data."""
        analyzer = Analyzer(self.dataset_with_data)
        stats = analyzer.get_stats()
        
        self.assertEqual(stats['size'], 4)
        
        # Expected lengths: "Hello world" (11), "This is a test" (14), "Another sentence here" (21), "Hello world again" (17)
        # Total length = 11 + 14 + 21 + 17 = 63
        # Avg length = 63 / 4 = 15.75
        self.assertAlmostEqual(stats['avg_length'], 15.75)

        # Expected vocabulary (default whitespace split):
        # "Hello", "world", "This", "is", "a", "test", "Another", "sentence", "here", "again"
        # Vocab size = 10
        self.assertEqual(stats['vocab_size'], 10)
        
        # Total tokens:
        # "Hello world" (2) + "This is a test" (4) + "Another sentence here" (3) + "Hello world again" (3) = 12
        self.assertEqual(stats['total_tokens'], 12)

    def test_length_distribution_empty_dataset(self):
        """Test length_distribution on an empty dataset."""
        analyzer = Analyzer(self.empty_dataset)
        dist = analyzer.length_distribution()
        self.assertEqual(dist, {})

    def test_length_distribution_with_data(self):
        """Test length_distribution on a dataset with data."""
        analyzer = Analyzer(self.dataset_for_length_dist)
        dist = analyzer.length_distribution()
        # lengths: 5, 13, 5, 31
        expected_dist = {
            5: 2,
            13: 1,
            31: 1
        }
        self.assertEqual(dist, expected_dist)

    def test_token_count_empty_dataset(self):
        """Test token_count on an empty dataset."""
        analyzer = Analyzer(self.empty_dataset)
        self.assertEqual(analyzer.token_count(), 0)

    def test_token_count_with_data_default_tokenizer(self):
        """Test token_count with default whitespace tokenizer."""
        analyzer = Analyzer(self.dataset_with_data)
        # "Hello world" (2) + "This is a test" (4) + "Another sentence here" (3) + "Hello world again" (3) = 12
        self.assertEqual(analyzer.token_count(), 12)

    def test_token_count_with_custom_tokenizer(self):
        """Test token_count with a custom tokenizer."""
        analyzer = Analyzer(self.dataset_with_data)
        custom_tokenizer = lambda text: text.split('o') # Splits by 'o'
        # H"ello" w"o"rld -> ["Hell", " w", "rld"] (3)
        # This is a test -> ["This is a test"] (1)
        # An"o"ther sentence here -> ["An", "ther sentence here"] (2)
        # Hell"o" w"o"rld again -> ["Hell", " w", "rld again"] (3)
        # Total = 3 + 1 + 2 + 3 = 9
        self.assertEqual(analyzer.token_count(tokenizer=custom_tokenizer), 9)
    
    def test_token_count_with_empty_or_missing_text(self):
        """Test token_count when samples have empty or missing 'text' keys."""
        analyzer = Analyzer(self.dataset_with_empty_text)
        # {"text": "Valid"} -> 1 token (default split)
        # {"text": ""} -> 0 tokens
        # {} -> 0 tokens
        # {"other_key": "value"} -> 0 tokens
        self.assertEqual(analyzer.token_count(), 1)


    def test_vocabulary_empty_dataset(self):
        """Test vocabulary on an empty dataset."""
        analyzer = Analyzer(self.empty_dataset)
        self.assertEqual(analyzer.vocabulary(), [])

    def test_vocabulary_with_data_default_tokenizer(self):
        """Test vocabulary with default whitespace tokenizer."""
        analyzer = Analyzer(self.dataset_with_data)
        vocab = analyzer.vocabulary()
        expected_vocab = sorted(["Hello", "world", "This", "is", "a", "test", "Another", "sentence", "here", "again"])
        self.assertEqual(vocab, expected_vocab)

    def test_vocabulary_with_custom_tokenizer(self):
        """Test vocabulary with a custom tokenizer."""
        analyzer = Analyzer(self.dataset_with_data)
        custom_tokenizer = lambda text: list(text) # Character tokenizer
        vocab = analyzer.vocabulary(tokenizer=custom_tokenizer)
        
        all_chars = set()
        for item in self.sample_data_list:
            all_chars.update(list(item["text"]))
        expected_vocab = sorted(list(all_chars))
        self.assertEqual(vocab, expected_vocab)

    def test_vocabulary_with_empty_or_missing_text(self):
        """Test vocabulary when samples have empty or missing 'text' keys."""
        analyzer = Analyzer(self.dataset_with_empty_text)
        # From {"text": "Valid"} -> ["Valid"]
        expected_vocab = sorted(["Valid"])
        self.assertEqual(analyzer.vocabulary(), expected_vocab)


    def test_random_sample_basic(self):
        """Test random_sample basic functionality."""
        # Using MagicMock for dataset to control its 'sample' method directly
        mock_dataset = MagicMock(spec=MockDataset) # Use spec to ensure it mimics MockDataset
        type(mock_dataset)._data = PropertyMock(return_value=self.sample_data_list) # Mock _data for len
        
        # Configure the mock 'sample' method
        expected_samples = [self.sample_data_list[0], self.sample_data_list[2]]
        mock_dataset.sample = MagicMock(return_value=expected_samples)

        # Configure __len__ for the mock_dataset
        mock_dataset.__len__.return_value = len(self.sample_data_list)
        
        analyzer = Analyzer(mock_dataset)
        samples = analyzer.random_sample(n=2)
        
        self.assertEqual(samples, expected_samples)
        mock_dataset.sample.assert_called_once_with(n=2)

    def test_random_sample_n_larger_than_data(self):
        """Test random_sample when n is larger than dataset size."""
        # Here, we test the fallback behavior of Analyzer's random_sample
        # if dataset.sample is not robust or if we want to specifically test Analyzer's logic.
        # However, the current Analyzer.random_sample defers to dataset.sample
        # So, we rely on the mock_dataset.sample to handle this logic as per the Dataset spec.
        
        small_data = [self.sample_data_list[0]]
        mock_small_dataset = MockDataset(small_data) # Use our actual MockDataset
        
        analyzer = Analyzer(mock_small_dataset)
        # MockDataset's sample method returns all data if n is too large
        samples = analyzer.random_sample(n=5)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples, small_data)

    def test_random_sample_empty_dataset(self):
        """Test random_sample on an empty dataset."""
        analyzer = Analyzer(self.empty_dataset) # Uses MockDataset
        samples = analyzer.random_sample(n=5)
        self.assertEqual(samples, [])

    def test_random_sample_fallback_behavior(self):
        """Test random_sample fallback if dataset has no 'sample' method."""
        # Create a dataset object that does *not* have a .sample() method
        class DatasetWithoutSampleMethod:
            def __init__(self, data):
                self._data = data
            def __len__(self):
                return len(self._data)
            def __iter__(self):
                return iter(self._data)
            def __getitem__(self, item): # Needed for list(self.dataset) in fallback
                return self._data[item]

        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        dataset_no_sample = DatasetWithoutSampleMethod(data)
        
        analyzer = Analyzer(dataset_no_sample)
        
        with unittest.mock.patch('random.sample') as mock_random_sample:
            mock_random_sample.return_value = [data[0]] # Mock the output of random.sample
            result = analyzer.random_sample(n=1)
            
            # Check if our mock random.sample was called with the correct arguments
            # The first argument to random.sample should be a list representation of the dataset
            # The second argument is n
            mock_random_sample.assert_called_once()
            # Check args: random.sample(list(dataset_no_sample), 1)
            # Since list(dataset_no_sample) would be `data`, we check that.
            # However, the call is random.sample(population, k)
            # So, call_args[0][0] is the population, call_args[0][1] is k.
            self.assertEqual(mock_random_sample.call_args[0][0], data)
            self.assertEqual(mock_random_sample.call_args[0][1], 1)
            self.assertEqual(result, [data[0]])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 