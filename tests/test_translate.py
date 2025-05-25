import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust the path to import the LLMTranslator class correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synglot.translate.llm_translator import LLMTranslator

# Mock the Hugging Face classes
class MockAutoTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text, return_tensors, padding, truncation, max_length=None):
        # Simulate tokenization
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]} # Dummy tokenized output

    def decode(self, tokens, skip_special_tokens):
        # Simulate decoding
        if tokens[0] == [1, 2, 3, 4, 5]: # Expected for single translation
            return "Bonjour monde"
        elif tokens[0] == [6,7,8,9,10]: # Expected for batch translation item 1
             return "Bonjour."
        elif tokens[0] == [11,12,13,14,15]: # Expected for batch translation item 2
             return "Au revoir"
        return "mocked translation"

    @classmethod
    def from_pretrained(cls, model_name):
        if "unsupported" in model_name:
            raise Exception("Simulated model loading failure")
        return cls()

class MockAutoModelForSeq2SeqLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, **inputs):
        # Simulate model generation
        # Check the input_ids to return different outputs for different inputs
        input_ids = inputs["input_ids"]
        if len(input_ids) == 1 : # single input
             return [[1,2,3,4,5]] # Dummy generated tokens for single translation
        elif len(input_ids) > 1: # batch input
            return [[6,7,8,9,10], [11,12,13,14,15]] # Dummy generated tokens for batch
        return [[0]] # default

    @classmethod
    def from_pretrained(cls, model_name):
        if "unsupported" in model_name:
            raise Exception("Simulated model loading failure")
        return cls()

@patch('transformers.AutoTokenizer', MockAutoTokenizer)
@patch('transformers.AutoModelForSeq2SeqLM', MockAutoModelForSeq2SeqLM)
class TestLLMTranslator(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.source_lang = "en"
        self.target_lang = "fr"
        # Initialize with a supported language pair for most tests
        self.translator = LLMTranslator(source_lang=self.source_lang, target_lang=self.target_lang)

    def test_init_successful(self):
        """Test successful initialization of LLMTranslator with MarianMT backend."""
        self.assertIsNotNone(self.translator.tokenizer)
        self.assertIsNotNone(self.translator.model)
        self.assertEqual(self.translator.source_lang, self.source_lang)
        self.assertEqual(self.translator.target_lang, self.target_lang)
        self.assertEqual(self.translator.backend, "marianmt")

    def test_init_unsupported_language_pair(self):
        """Test initialization with an unsupported language pair for MarianMT."""
        with self.assertRaises(ValueError) as context:
            LLMTranslator(source_lang="unsupported", target_lang="lang")
        self.assertIn("Failed to load MarianMT model for unsupported-lang", str(context.exception))

    def test_init_unsupported_backend(self):
        """Test initialization with an unsupported backend."""
        with self.assertRaises(NotImplementedError) as context:
            LLMTranslator(source_lang=self.source_lang, target_lang=self.target_lang, backend="unsupported_backend")
        self.assertIn("Backend 'unsupported_backend' is not supported", str(context.exception))

    def test_translate_successful(self):
        """Test successful translation."""
        text_to_translate = "Hello world"
        expected_translation = "Bonjour monde"
        translated_text = self.translator.translate(text_to_translate)
        self.assertEqual(translated_text, expected_translation)

    def test_translate_model_not_initialized(self):
        """Test translation when model is not initialized."""
        self.translator.model = None
        with self.assertRaises(RuntimeError) as context:
            self.translator.translate("Some text")
        self.assertIn("MarianMT model and tokenizer not initialized properly.", str(context.exception))

    def test_translate_tokenizer_not_initialized(self):
        """Test translation when tokenizer is not initialized."""
        self.translator.tokenizer = None
        with self.assertRaises(RuntimeError) as context:
            self.translator.translate("Some text")
        self.assertIn("MarianMT model and tokenizer not initialized properly.", str(context.exception))

    # We need to re-patch the AutoModelForSeq2SeqLM for this specific test
    # to simulate a different backend, or lack of one for the translate method.
    @patch('transformers.AutoTokenizer', MockAutoTokenizer)
    @patch('transformers.AutoModelForSeq2SeqLM', MockAutoModelForSeq2SeqLM)
    def test_translate_unsupported_backend_in_translate(self):
        """Test translate method with an unsupported backend (should be caught in init, but good to check)."""
        # This scenario is technically prevented by __init__,
        # but this tests the else branch in translate() directly.
        translator = LLMTranslator(self.source_lang, self.target_lang) # Re-init to ensure it's marianmt
        translator.backend = "other_backend" # Manually override backend after init
        with self.assertRaises(NotImplementedError) as context:
            translator.translate("Some text")
        self.assertIn("Translation for backend 'other_backend' is not implemented.", str(context.exception))

    def test_translate_batch_successful(self):
        """Test successful batch translation."""
        texts_to_translate = ["Hi", "Bye"] 
        expected_translations = ["Bonjour.", "Au revoir."]
        translated_texts = self.translator.translate_batch(texts_to_translate, batch_size=2)
        self.assertEqual(translated_texts, expected_translations)

    def test_translate_batch_different_batch_size(self):
        """Test successful batch translation with a batch size smaller than total texts."""
        # Test with batch_size=2 (tests one call to generate with multiple inputs)
        translated_texts_b2 = self.translator.translate_batch(["Hi", "Bye"], batch_size=2)
        self.assertEqual(translated_texts_b2, ["Bonjour.", "Au revoir."])

        # Test with batch_size=1 (tests multiple calls to generate, each with single input)
        # For this to pass with current global mocks, each individual translation of "Hi", "Bye", "Hello world"
        # should result in "Bonjour monde" because model.generate with single input always gives [[1,2,3,4,5]]
        # and the decode for that is now "Bonjour monde".
        # The translation for "Bye" (which also becomes a single translation) will also be "Bonjour monde"
        # as it will also go through the same single input path in generate -> decode.
        translated_texts_b1 = self.translator.translate_batch(["Hi", "Bye", "Hello world"], batch_size=1)
        self.assertEqual(translated_texts_b1, ["Bonjour monde", "Bonjour monde", "Bonjour monde"])

    def test_translate_batch_model_not_initialized(self):
        """Test batch translation when model is not initialized."""
        self.translator.model = None
        with self.assertRaises(RuntimeError) as context:
            self.translator.translate_batch(["Some text", "Another text"])
        self.assertIn("MarianMT model and tokenizer not initialized properly.", str(context.exception))

    def test_translate_batch_tokenizer_not_initialized(self):
        """Test batch translation when tokenizer is not initialized."""
        self.translator.tokenizer = None
        with self.assertRaises(RuntimeError) as context:
            self.translator.translate_batch(["Some text", "Another text"])
        self.assertIn("MarianMT model and tokenizer not initialized properly.", str(context.exception))

    @patch('synglot.translate.base.Translator.translate_batch') # Mock the base class method
    @patch('transformers.AutoTokenizer', MockAutoTokenizer) # Keep other mocks for init
    @patch('transformers.AutoModelForSeq2SeqLM', MockAutoModelForSeq2SeqLM)
    def test_translate_batch_fallback_to_superclass(self, mock_base_translate_batch):
        """Test that translate_batch falls back to superclass for non-marianmt backends."""
        # Initialize a new translator for this test to avoid altering self.translator state
        translator_other_backend = LLMTranslator(self.source_lang, self.target_lang)
        translator_other_backend.backend = "other_backend" # Manually set backend

        texts = ["text1", "text2"]
        batch_size = 10
        translator_other_backend.translate_batch(texts, batch_size=batch_size)
        mock_base_translate_batch.assert_called_once_with(texts, batch_size)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 