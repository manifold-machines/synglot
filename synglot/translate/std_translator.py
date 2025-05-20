from .base import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class StandardTranslator(Translator):
    """Translator using standard machine translation workflows, defaulting to MarianMT."""
    
    def __init__(self, source_lang, target_lang, backend="marianmt"):
        """
        Initialize standard translator.
        
        Args:
            source_lang (str): Source language code (e.g., 'en')
            target_lang (str): Target language code (e.g., 'fr')
            backend (str): Backend translation system. Currently supports 'marianmt'.
        """
        super().__init__(source_lang, target_lang)
        self.backend = backend

        if self.backend == "marianmt":
            try:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MarianMT model for {source_lang}-{target_lang}. "
                    f"Ensure the language pair is supported and transformers library is correctly installed. Error: {e}"
                )
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not supported by StandardTranslator. Currently, only 'marianmt' is supported."
            )

    def translate(self, text):
        """Translate using the configured standard translation system (MarianMT)."""
        if self.backend == "marianmt":
            if not self.model or not self.tokenizer:
                raise RuntimeError("MarianMT model and tokenizer not initialized properly.")
            
            # For some models, a prefix is needed for the source language if the model is multilingual.
            # For opus-mt-{src}-{tgt} models, this is usually not required.
            # However, if you were using a multilingual model like 'mbart-large-50-many-to-many-mmt',
            # you would set: self.tokenizer.src_lang = source_lang
            # And then: encoded_text = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # For Helsinki-NLP models, direct tokenization is usually fine.

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated_tokens = self.model.generate(**inputs)
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
        else:
            # This case should ideally be caught in __init__
            raise NotImplementedError(f"Translation for backend '{self.backend}' is not implemented.")

    # translate_batch can be inherited. For MarianMT, batching is handled efficiently by the model.generate() 
    # when a list of strings is passed to the tokenizer. However, the base class implementation
    # calls translate one-by-one. For better performance, this could be overridden.
    def translate_batch(self, texts, batch_size=32):
        """Translate a batch of texts using MarianMT, processing in chunks defined by batch_size."""
        if self.backend == "marianmt":
            if not self.model or not self.tokenizer:
                raise RuntimeError("MarianMT model and tokenizer not initialized properly.")
            
            translations = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                # Move inputs to the same device as the model, if using GPU
                # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                translated_tokens = self.model.generate(**inputs)
                batch_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
                translations.extend(batch_translations)
            return translations
        else:
            # Fallback to base class implementation if backend is not marianmt
            # This ensures that if other backends are added to StandardTranslator which don't have
            # a custom batching strategy, they still work (though potentially less efficiently).
            return super().translate_batch(texts, batch_size) 