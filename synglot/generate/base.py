from synglot.utils.config import Config

class Generator:
    """Base class for data generators."""
    
    def __init__(self, target_lang, config=None):
        """
        Initialize generator.
        
        Args:
            target_lang (str): Target language code
            config (dict | Config, optional): Configuration parameters as a dict or a Config object.
                                            If None, default Config will be used.
        """
        self.target_lang = target_lang
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = Config(config_dict=config)
        else: # Handles None or other unexpected types
            self.config = Config() # Initialize with default configuration
        
    def generate(self, prompt=None, n_samples=1):
        """
        Generate synthetic samples.
        Must be implemented by subclasses.
        """
        raise NotImplementedError 