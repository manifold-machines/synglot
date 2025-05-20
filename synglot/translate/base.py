class Translator:
    """Base class for all translator implementations."""
    
    def __init__(self, source_lang, target_lang):
        """
        Initialize translator.
        
        Args:
            source_lang (str): Source language code
            target_lang (str): Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text):
        """
        Translate a single piece of text.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def translate_batch(self, texts, batch_size=32):
        """Translate a batch of texts."""
        # Default implementation: translate one by one.
        # Subclasses can override for more efficient batching.
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # This could be optimized in subclasses by calling a batch API
            for text_item in batch:
                translations.append(self.translate(text_item))
        return translations 