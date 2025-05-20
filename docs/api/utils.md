# `synglot.utils`

This module provides utility classes and functions for the `synglot` library.

## `Config`

```python
class Config:
    def __init__(self, config_dict=None, config_file=None)
    def get(self, key_path, default=None)
    def set(self, key_path, value)
    def save(self, path)
    def load(self, path)
```

### `__init__(self, config_dict=None, config_file=None)`
Initializes the configuration manager. Loads defaults, then optionally updates from a dictionary and/or a YAML file.

**Args:**
- `config_dict` (`dict`, optional): Configuration dictionary to override defaults.
- `config_file` (`str`, optional): Path to a YAML configuration file.

### `get(self, key_path, default=None)`
Get a configuration value using a dot-separated key path.

**Args:**
- `key_path` (`str`): Dot-separated path to the key (e.g., `"generation_settings.pretraining.topic_list"`).
- `default` (any, optional): Default value to return if the key is not found.

**Returns:**
- `any`: The configuration value or the default.

### `set(self, key_path, value)`
Set a configuration value using a dot-separated key path. Creates nested dictionaries if they don't exist.

**Args:**
- `key_path` (`str`): Dot-separated path to the key.
- `value` (any): Value to set.

### `save(self, path)`
Save the current configuration to a YAML file.

**Args:**
- `path` (`str`): Path to save the YAML file.

### `load(self, path)`
Load configuration from a YAML file, updating the current configuration.

**Args:**
- `path` (`str`): Path to the YAML configuration file to load. 