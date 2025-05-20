import copy
import yaml

class Config:
    """
    Configuration for synglot components.
    Manages settings using a dictionary-like interface with support for nested keys.
    """
    
    DEFAULT_CONFIG = {
        "seed": 42,
        "generation_settings": {
            "default_temperature": 1.0,
            "default_top_k": 50,
            "default_top_p": 1.0,
            "default_do_sample": True,
            "default_max_new_tokens": 150, # Increased default
            "return_full_text": True,      # General default for the 'generate' method

            "pretraining": {
                "diversity_strategy": "topic_prompt", # Options: "none", "topic_prompt"
                "general_topics_list": [
                    "everyday life and routines", "food and cooking recipes", "travel and adventure stories", 
                    "science and technology breakthroughs", "music genres and history", "art and painting techniques", 
                    "basics of contract law", "classic literature summaries", "current global events",
                    "sports and recreational activities", "nature and wildlife conservation", "historical events and figures",
                    "philosophy and ethical dilemmas", "personal finance and budgeting", "health and wellness tips",
                    "different career paths", "learning new languages", "environmental issues", "space exploration",
                    "mental health awareness", "the future of artificial intelligence"
                ],
                "topic_prompt_template": "Write a short, informative text about {topic}.",
                "return_prompt_in_output": False # Specific to pretraining outputs via generate_pretraining
            },
            "conversation": {
                 "speaker_A": "User:",
                 "speaker_B": "Assistant:",
                 "turn_max_new_tokens": 80,
                 "domain_context_template": "This is a conversation about {domain}.",
                 "ensure_alternating_speakers": True 
            }
        },
        "translation_settings": {
            # Placeholder for translation specific configs
            "default_model_name": "Helsinki-NLP/opus-mt-en-fr" 
        },
        "dataset_settings": {
            # Placeholder for dataset specific configs
            "default_batch_size": 32
        }
        # Add other top-level categories as needed (e.g., analysis_settings)
    }

    def __init__(self, config_dict=None, config_file=None):
        """
        Initialize configuration.
        Loads defaults, then optionally updates from a dictionary and/or a file.
        
        Args:
            config_dict (dict, optional): Configuration dictionary to override defaults.
            config_file (str, optional): Path to configuration file (e.g., YAML). Not fully implemented for loading.
        """
        self._config_data = copy.deepcopy(self.DEFAULT_CONFIG)

        if config_dict:
            self._deep_update(self._config_data, config_dict)

        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_configs = yaml.safe_load(f)
                if file_configs:
                    self._deep_update(self._config_data, file_configs)
            except FileNotFoundError:
                print(f"Warning: Config file '{config_file}' not found.")
            except Exception as e:
                print(f"Warning: Error loading config file '{config_file}': {e}")
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get(self, key_path, default=None):
        """
        Get a configuration value using a dot-separated key path.
        
        Args:
            key_path (str): Dot-separated path to the key (e.g., "generation_settings.pretraining.topic_list").
            default (any, optional): Default value to return if key is not found.
            
        Returns:
            any: The configuration value or the default.
        """
        keys = key_path.split('.')
        current_level = self._config_data
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return default
        return current_level

    def set(self, key_path, value):
        """
        Set a configuration value using a dot-separated key path.
        Creates nested dictionaries if they don't exist.
        
        Args:
            key_path (str): Dot-separated path to the key.
            value (any): Value to set.
        """
        keys = key_path.split('.')
        current_level = self._config_data
        for i, key in enumerate(keys[:-1]):
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {} # Create dict if not exists
            current_level = current_level[key]
        current_level[keys[-1]] = value
        
    def save(self, path):
        """Save current configuration to a YAML file."""
        try:
            with open(path, 'w') as f:
                yaml.dump(self._config_data, f, indent=4, sort_keys=False)
            print(f"Configuration saved to '{path}'")
        except Exception as e:
            print(f"Error saving configuration to '{path}': {e}")
            
    def load(self, path):
        """
        Load configuration from a YAML file, updating the current config.
        Note: __init__ already handles initial loading from file. This can be used to reload or merge.
        """
        try:
            with open(path, 'r') as f:
                file_configs = yaml.safe_load(f)
            if file_configs:
                self._deep_update(self._config_data, file_configs)
                print(f"Configuration loaded and merged from '{path}'")
            else:
                print(f"Warning: Config file '{path}' is empty or invalid.")
        except FileNotFoundError:
            print(f"Warning: Config file '{path}' not found for loading.")
        except Exception as e:
            print(f"Warning: Error loading config file '{path}': {e}") 