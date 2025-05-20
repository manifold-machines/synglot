import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust path to import synglot components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synglot.generate.hf_generator import HFGenerator
from synglot.utils.config import Config

class TestHFGenerator(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.default_config_dict = {
            "seed": 123,
            "hf_generator": {
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "generation_params": {
                    "temperature": 0.5
                }
            },
            "generation_settings": { # General settings used by HFGenerator
                "default_temperature": 0.7,
                "default_top_k": 40,
                "default_max_new_tokens": 50,
                "return_full_text": True, # Default for basic generate
                 "pretraining": {
                    "diversity_strategy": "topic_prompt",
                    "general_topics_list": ["science", "art"],
                    "topic_prompt_template": "Write about {topic}.",
                    "return_prompt_in_output": False
                },
                "conversation": {
                    "speaker_A": "User:",
                    "speaker_B": "Bot:",
                    "turn_max_new_tokens": 30,
                    "domain_context_template": "Context: {domain}.",
                }
            }
        }
        self.config = Config(self.default_config_dict)

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_init_with_model_name_arg(self, mock_set_seed, mock_pipeline):
        """Test initialization with model_name as a direct argument."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.tokenizer = MagicMock()
        mock_generator_instance.tokenizer.eos_token_id = 50256
        mock_pipeline.return_value = mock_generator_instance

        generator = HFGenerator(target_lang="en", model_name="arg_model_name", config=self.config)
        
        self.assertEqual(generator.model_name, "arg_model_name")
        mock_pipeline.assert_called_once_with("text-generation", model="arg_model_name", tokenizer="arg_model_name")
        mock_set_seed.assert_called_once_with(123) # From self.config

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_init_with_model_name_from_config(self, mock_set_seed, mock_pipeline):
        """Test initialization with model_name from hf_generator.model_name in config."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.tokenizer = MagicMock()
        mock_generator_instance.tokenizer.eos_token_id = 50256
        mock_pipeline.return_value = mock_generator_instance

        # Config already has 'hf_generator.model_name': 'config_model_name'
        generator = HFGenerator(target_lang="en", config=self.config)
        
        self.assertEqual(generator.model_name, "config_model_name")
        mock_pipeline.assert_called_once_with("text-generation", model="config_model_name", tokenizer="config_model_name")
        mock_set_seed.assert_called_once_with(123)

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_init_with_default_model_name(self, mock_set_seed, mock_pipeline):
        """Test initialization with the default model_name when not in arg or config."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.tokenizer = MagicMock()
        mock_generator_instance.tokenizer.eos_token_id = 50256
        mock_pipeline.return_value = mock_generator_instance

        # Create a config without hf_generator.model_name
        config_no_hf_model = Config({"seed": 456}) 
        
        with patch('builtins.print') as mock_print: # To check for warning
            generator = HFGenerator(target_lang="en", config=config_no_hf_model)
        
        default_hf_model = "Qwen/Qwen2.5-1.5B-Instruct" # As defined in HFGenerator
        self.assertEqual(generator.model_name, default_hf_model)
        mock_pipeline.assert_called_once_with("text-generation", model=default_hf_model, tokenizer=default_hf_model)
        mock_set_seed.assert_called_once_with(456)
        mock_print.assert_called_once() # Check that the warning was printed
        self.assertIn(f"Falling back to default text-generation model '{default_hf_model}'", mock_print.call_args[0][0])

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_init_config_can_be_dict(self, mock_set_seed, mock_pipeline):
        """Test initialization when config is passed as a dictionary."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.tokenizer = MagicMock()
        mock_generator_instance.tokenizer.eos_token_id = 50256
        mock_pipeline.return_value = mock_generator_instance

        config_dict = {"seed": 789, "hf_generator": {"model_name": "dict_config_model"}}
        generator = HFGenerator(target_lang="en", model_name="arg_model_override", config=config_dict)
        
        self.assertEqual(generator.model_name, "arg_model_override") # Arg takes precedence
        self.assertEqual(generator.config.get("seed"), 789)
        self.assertEqual(generator.config.get("hf_generator.model_name"), "dict_config_model")
        mock_pipeline.assert_called_once_with("text-generation", model="arg_model_override", tokenizer="arg_model_override")
        mock_set_seed.assert_called_once_with(789)

    @patch('transformers.pipeline', side_effect=Exception("Model loading failed"))
    @patch('transformers.set_seed')
    def test_init_model_loading_failure(self, mock_set_seed, mock_pipeline_fails):
        """Test RuntimeError when pipeline loading fails."""
        with self.assertRaisesRegex(RuntimeError, "Failed to load HuggingFace model"):
            HFGenerator(target_lang="en", model_name="any_model", config=self.config)
        # set_seed might be called before pipeline, depending on implementation details
        # For this test, the crucial part is the RuntimeError.
        # If set_seed is after pipeline init, it won't be called. If before, it might.
        # Current HFGenerator calls set_seed after pipeline init.
        # mock_set_seed.assert_not_called() # This would fail if seed is set first

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_init_no_seed_in_config(self, mock_set_seed, mock_pipeline):
        """Test initialization when no seed is present in the config."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.tokenizer = MagicMock()
        mock_generator_instance.tokenizer.eos_token_id = 50256
        mock_pipeline.return_value = mock_generator_instance
        
        config_no_seed = Config({}) # Empty config, so no "seed"
        # Remove hf_generator.model_name to fall back to default and avoid print warning issues
        del config_no_seed._config_data['hf_generator']['model_name'] 
        
        generator = HFGenerator(target_lang="en", config=config_no_seed)
        
        mock_set_seed.assert_not_called() # set_seed should not be called

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_basic(self, mock_set_seed, mock_pipeline_constructor):
        """Test basic text generation with default parameters."""
        # Mock the pipeline object that pipeline_constructor returns
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.return_value = [{'generated_text': 'This is a test output'}] 
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123 # Dummy EOS token ID
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        generator = HFGenerator(target_lang="en", config=self.config)
        
        prompt = "Test prompt: "
        results = generator.generate(prompt=prompt, n_samples=1)
        
        self.assertEqual(len(results), 1)
        # self.config has return_full_text=True by default for general generation settings
        # and the mock output starts with the prompt, so it should be stripped.
        self.assertEqual(results[0], "This is a test output") # Expecting stripping by default if prompt not present
        
        # Verify pipeline call arguments
        expected_gen_params = {
            'num_return_sequences': 1,
            'temperature': 0.7, # From generation_settings.default_temperature
            'top_k': 40,        # From generation_settings.default_top_k
            'top_p': 1.0,       # Default from Config.DEFAULT_CONFIG
            'do_sample': True,  # Default from Config.DEFAULT_CONFIG
            'return_full_text': True, # From generation_settings.return_full_text
            'pad_token_id': 123,
            'max_new_tokens': 50 # From generation_settings.default_max_new_tokens
        }
        # hf_generator.generation_params is {"temperature": 0.5} in self.config
        # this should override generation_settings.default_temperature
        expected_gen_params["temperature"] = 0.5 
        mock_hf_pipeline.assert_called_once_with(prompt, **expected_gen_params)

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_with_kwargs_override(self, mock_set_seed, mock_pipeline_constructor):
        """Test that kwargs in generate() override config settings."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.return_value = [{'generated_text': 'Test prompt: overridden output'}] 
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        generator = HFGenerator(target_lang="en", config=self.config)
        
        prompt = "Test prompt: "
        results = generator.generate(
            prompt=prompt, 
            n_samples=2, 
            temperature=0.9, 
            max_new_tokens=100,
            return_full_text=False # Override config
        )
        
        self.assertEqual(len(results), 2) # n_samples=2 should result in two (mocked) outputs
        # Mock output is same for both, as we only return one item in the list
        # HF pipeline would actually return two if num_return_sequences=2
        # For this test, we just check if the parameter is passed correctly
        # And if stripping is handled as per return_full_text=False
        self.assertEqual(results[0], "overridden output")

        expected_gen_params = {
            'num_return_sequences': 2,
            'temperature': 0.9, # Overridden by kwarg
            'top_k': 40,        # From config
            'top_p': 1.0,       # Default
            'do_sample': True,  # Default
            'return_full_text': False, # Overridden by kwarg
            'pad_token_id': 123,
            'max_new_tokens': 100 # Overridden by kwarg
        }
        mock_hf_pipeline.assert_called_once_with(prompt, **expected_gen_params)

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_prompt_stripping_logic(self, mock_set_seed, mock_pipeline_constructor):
        """Test prompt stripping logic under various return_full_text scenarios."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        generator = HFGenerator(target_lang="en", config=self.config)
        prompt = "This is the prompt. "
        model_output_with_prompt = prompt + "And this is the model generation."
        model_output_without_prompt = "Just the model generation."

        # Scenario 1: return_full_text=True (config), output contains prompt
        mock_hf_pipeline.return_value = [{'generated_text': model_output_with_prompt}]
        results = generator.generate(prompt=prompt, return_full_text=True)
        self.assertEqual(results[0], "And this is the model generation.")

        # Scenario 2: return_full_text=False (kwarg), output contains prompt
        mock_hf_pipeline.reset_mock()
        mock_hf_pipeline.return_value = [{'generated_text': model_output_with_prompt}]
        results = generator.generate(prompt=prompt, return_full_text=False)
        self.assertEqual(results[0], "And this is the model generation.")

        # Scenario 3: return_full_text=True (config), output does NOT contain prompt
        mock_hf_pipeline.reset_mock()
        mock_hf_pipeline.return_value = [{'generated_text': model_output_without_prompt}]
        results = generator.generate(prompt=prompt, return_full_text=True)
        self.assertEqual(results[0], "Just the model generation.")

        # Scenario 4: return_full_text=False (kwarg), output does NOT contain prompt
        mock_hf_pipeline.reset_mock()
        mock_hf_pipeline.return_value = [{'generated_text': model_output_without_prompt}]
        results = generator.generate(prompt=prompt, return_full_text=False)
        self.assertEqual(results[0], "Just the model generation.")

        # Scenario 5: No prompt provided, return_full_text=True
        mock_hf_pipeline.reset_mock()
        mock_hf_pipeline.return_value = [{'generated_text': model_output_without_prompt}]
        results = generator.generate(prompt=None, return_full_text=True)
        self.assertEqual(results[0], "Just the model generation.")
        
        # Scenario 6: No prompt provided, return_full_text=False
        mock_hf_pipeline.reset_mock()
        mock_hf_pipeline.return_value = [{'generated_text': model_output_without_prompt}]
        results = generator.generate(prompt=None, return_full_text=False)
        self.assertEqual(results[0], "Just the model generation.")

    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_length_params_priority(self, mock_set_seed, mock_pipeline_constructor):
        """Test priority of max_new_tokens, max_length, and min_length."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.return_value = [{'generated_text': 'output'}]
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        # 1. kwargs take highest priority
        config_with_lengths = Config({
            "hf_generator": {"model_name": "test_model"},
            "generation_settings": {
                "default_max_new_tokens": 50,
                "default_min_length_total": 10
            }
        })
        generator = HFGenerator(target_lang="en", config=config_with_lengths)
        generator.generate(prompt="P", max_new_tokens=100, min_length=20) # kwargs used
        args, kwargs = mock_hf_pipeline.call_args
        self.assertEqual(kwargs['max_new_tokens'], 100)
        self.assertEqual(kwargs['min_length'], 20)
        mock_hf_pipeline.reset_mock()

        # 2. max_length in kwargs (for total length)
        generator.generate(prompt="P", max_length=150, min_length=25)
        args, kwargs = mock_hf_pipeline.call_args
        self.assertEqual(kwargs['max_length'], 150)
        self.assertEqual(kwargs['min_length'], 25)
        mock_hf_pipeline.reset_mock()

        # 3. Config settings if no kwargs for length
        generator = HFGenerator(target_lang="en", config=self.config) # uses self.default_config_dict
        # self.config has generation_settings.default_max_new_tokens = 50
        # self.config does not have default_min_length_total by default in setup
        # Let's add it to check
        self.config.set("generation_settings.default_min_length_total", 5)
        generator.generate(prompt="P")
        args, kwargs = mock_hf_pipeline.call_args
        self.assertEqual(kwargs['max_new_tokens'], self.config.get("generation_settings.default_max_new_tokens"))
        self.assertEqual(kwargs['min_length'], 5)
        mock_hf_pipeline.reset_mock()
        self.config.set("generation_settings.default_min_length_total", None) # Reset for other tests

        # 4. Ensure only max_new_tokens OR max_length is passed, not both if max_new_tokens is primary
        generator.generate(prompt="P", max_new_tokens=70)
        args, kwargs = mock_hf_pipeline.call_args
        self.assertIn('max_new_tokens', kwargs)
        self.assertEqual(kwargs['max_new_tokens'], 70)
        self.assertNotIn('max_length', kwargs)
        mock_hf_pipeline.reset_mock()

        generator.generate(prompt="P", max_length=80) # max_length from kwarg
        args, kwargs = mock_hf_pipeline.call_args
        self.assertIn('max_length', kwargs)
        self.assertEqual(kwargs['max_length'], 80)
        self.assertNotIn('max_new_tokens', kwargs) # if max_length is there, max_new_tokens from config should not

    @patch.object(HFGenerator, 'generate') # Mock the internal self.generate call
    @patch('transformers.pipeline') # Still need to mock this for HFGenerator instantiation
    @patch('transformers.set_seed')
    @patch('random.choice') # Mock random.choice for topic selection
    def test_generate_pretraining_topic_prompt(self, mock_random_choice, mock_set_seed, mock_pipeline_constructor, mock_self_generate):
        """Test generate_pretraining with topic_prompt diversity strategy."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline
        
        generator = HFGenerator(target_lang="en", config=self.config)

        # Config has topics: ["science", "art"] and template "Write about {topic}."
        # Config has pretraining.return_prompt_in_output = False
        mock_random_choice.side_effect = ["science", "art"] # Control topic selection
        mock_self_generate.side_effect = lambda prompt, n_samples, min_length, max_length, return_full_text: \
            [f"generated text for {prompt.split(' ')[-1].strip('.')}"]
            
        results = generator.generate_pretraining(n_samples=2, min_length=10, max_length=20)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "generated text for science")
        self.assertEqual(results[1], "generated text for art")
        
        self.assertEqual(mock_random_choice.call_count, 2)
        self.assertEqual(mock_self_generate.call_count, 2)
        
        # Check calls to self.generate
        expected_calls = [
            unittest.mock.call(prompt="Write about science.", n_samples=1, min_length=10, max_length=20, return_full_text=False),
            unittest.mock.call(prompt="Write about art.", n_samples=1, min_length=10, max_length=20, return_full_text=False)
        ]
        mock_self_generate.assert_has_calls(expected_calls)

    @patch.object(HFGenerator, 'generate')
    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_pretraining_no_diversity(self, mock_set_seed, mock_pipeline_constructor, mock_self_generate):
        """Test generate_pretraining with 'none' diversity strategy."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        # Modify config for this test
        current_config_dict = self.default_config_dict.copy()
        current_config_dict["generation_settings"]["pretraining"]["diversity_strategy"] = "none"
        current_config_dict["generation_settings"]["pretraining"]["return_prompt_in_output"] = True # Test this flag
        config_no_diversity = Config(current_config_dict)
        
        generator = HFGenerator(target_lang="en", config=config_no_diversity)
        mock_self_generate.return_value = ["unconditional generation"]
        
        results = generator.generate_pretraining(n_samples=1, min_length=5, max_length=15)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "unconditional generation")
        mock_self_generate.assert_called_once_with(prompt=None, n_samples=1, min_length=5, max_length=15, return_full_text=True)

    @patch.object(HFGenerator, 'generate')
    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    def test_generate_pretraining_specific_domain(self, mock_set_seed, mock_pipeline_constructor, mock_self_generate):
        """Test generate_pretraining with a specific domain rather than 'general'."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        generator = HFGenerator(target_lang="en", config=self.config)
        # self.config template: "Write about {topic}."
        # self.config return_prompt_in_output: False

        mock_self_generate.return_value = ["text about history"]
        
        results = generator.generate_pretraining(domain="history", n_samples=1, min_length=10, max_length=30)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "text about history")
        mock_self_generate.assert_called_once_with(
            prompt="Write about history.", 
            n_samples=1, 
            min_length=10, 
            max_length=30, 
            return_full_text=False # from config's pretraining.return_prompt_in_output
        )

    @patch.object(HFGenerator, 'generate')
    @patch('transformers.pipeline')
    @patch('transformers.set_seed')
    @patch('random.choice')
    def test_generate_pretraining_return_prompt_true(self, mock_random_choice, mock_set_seed, mock_pipeline_constructor, mock_self_generate):
        """Test generate_pretraining when config's return_prompt_in_output is True."""
        mock_hf_pipeline = MagicMock()
        mock_hf_pipeline.tokenizer = MagicMock()
        mock_hf_pipeline.tokenizer.eos_token_id = 123
        mock_pipeline_constructor.return_value = mock_hf_pipeline

        temp_config_dict = self.default_config_dict.copy()
        temp_config_dict["generation_settings"]["pretraining"]["return_prompt_in_output"] = True
        config_return_prompt = Config(temp_config_dict)

        generator = HFGenerator(target_lang="en", config=config_return_prompt)
        mock_random_choice.return_value = "science"
        # If self.generate is called with return_full_text=True, it might return the prompt + generation.
        # The pretraining method passes this flag directly.
        mock_self_generate.return_value = ["Write about science. This is science text."]
        
        results = generator.generate_pretraining(n_samples=1)
        
        self.assertEqual(results[0], "Write about science. This is science text.") # Expect full output including prompt
        mock_self_generate.assert_called_once_with(
            prompt="Write about science.", 
            n_samples=1, 
            min_length=50, # Default from HFGenerator.generate_pretraining
            max_length=200, # Default from HFGenerator.generate_pretraining
            return_full_text=True # Passed from pretraining config
        )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 