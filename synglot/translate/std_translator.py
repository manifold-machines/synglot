from .base import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from typing import List, Union, Optional, Dict, Any

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

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
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

    def translate_dataset(self, 
                         dataset, 
                         columns_to_translate: Union[str, List[str]], 
                         output_path: Optional[str] = None,
                         output_dir: str = "./outputs",
                         batch_size: int = 1,
                         progress_interval: int = 10,
                         save_errors: bool = True,
                         append_mode: bool = False) -> Dict[str, Any]:
        """
        Translate specified columns in a dataset with comprehensive error handling and progress tracking.
        
        Args:
            dataset: Dataset object to translate
            columns_to_translate (str or list): Column name(s) to translate
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None)
            batch_size (int): Batch size for translation (currently processes one by one for error handling)
            progress_interval (int): Print progress every N samples
            save_errors (bool): Whether to save error records to output
            append_mode (bool): Whether to append to existing file or overwrite
            
        Returns:
            dict: Summary statistics including success/error counts and output path
        """
        # Ensure columns_to_translate is a list
        if isinstance(columns_to_translate, str):
            columns_to_translate = [columns_to_translate]
        
        # Validate columns exist in dataset
        if hasattr(dataset, 'columns') and dataset.columns:
            missing_columns = [col for col in columns_to_translate if col not in dataset.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in dataset. Available columns: {dataset.columns}")
        
        # Setup output path
        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"translated_{self.source_lang}_to_{self.target_lang}.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize counters
        total_samples = len(dataset)
        success_count = 0
        error_count = 0
        
        print(f"Starting translation of {total_samples} samples...")
        print(f"Translating columns: {columns_to_translate}")
        print(f"Output will be saved to: {output_path}")
        
        # Open file for writing
        file_mode = 'a' if append_mode else 'w'
        with open(output_path, file_mode, encoding='utf-8') as f:
            for i, item in enumerate(dataset):
                try:
                    # Create output record starting with original data
                    output_record = dict(item) if isinstance(item, dict) else {"original_data": item}
                    
                    # Translate each specified column
                    for column in columns_to_translate:
                        original_text = item.get(column) if isinstance(item, dict) else str(item)
                        
                        if original_text is None:
                            print(f"Warning: Column '{column}' is None in sample {i+1}, skipping translation for this column.")
                            continue
                        
                        if not isinstance(original_text, str):
                            original_text = str(original_text)
                        
                        if not original_text.strip():
                            print(f"Warning: Column '{column}' is empty in sample {i+1}, skipping translation for this column.")
                            continue
                        
                        # Perform translation
                        translated_text = self.translate(original_text)
                        
                        # Add translated column to output
                        translated_column_name = f"translated_{column}_{self.target_lang}"
                        output_record[translated_column_name] = translated_text
                    
                    # Write successful record
                    f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    success_count += 1
                    
                    # Progress reporting
                    if (i + 1) % progress_interval == 0:
                        print(f"Processed {i+1}/{total_samples} samples. Success: {success_count}, Errors: {error_count}")
                
                except Exception as e:
                    error_count += 1
                    print(f"Error processing sample {i+1}: {e}")
                    
                    if save_errors:
                        # Create error record
                        error_record = {
                            "sample_index": i+1,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "original_data": dict(item) if isinstance(item, dict) else item
                        }
                        f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
        
        # Final summary
        summary = {
            "total_samples": total_samples,
            "successful_translations": success_count,
            "errors": error_count,
            "success_rate": success_count / total_samples if total_samples > 0 else 0,
            "output_path": output_path,
            "columns_translated": columns_to_translate,
            "source_language": self.source_lang,
            "target_language": self.target_lang
        }
        
        print(f"\nTranslation complete!")
        print(f"Total samples: {total_samples}")
        print(f"Successful translations: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Output saved to: {output_path}")
        
        return summary

    def translate_dataset_batch(self, 
                               dataset, 
                               columns_to_translate: Union[str, List[str]], 
                               output_path: Optional[str] = None,
                               output_dir: str = "./outputs",
                               batch_size: int = 32,
                               progress_interval: int = 100) -> Dict[str, Any]:
        """
        Translate specified columns in a dataset using batch processing for better performance.
        Note: Batch processing provides better performance but less granular error handling.
        
        Args:
            dataset: Dataset object to translate
            columns_to_translate (str or list): Column name(s) to translate
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None)
            batch_size (int): Batch size for translation
            progress_interval (int): Print progress every N samples
            
        Returns:
            dict: Summary statistics including success/error counts and output path
        """
        # Ensure columns_to_translate is a list
        if isinstance(columns_to_translate, str):
            columns_to_translate = [columns_to_translate]
        
        # Validate columns exist in dataset
        if hasattr(dataset, 'columns') and dataset.columns:
            missing_columns = [col for col in columns_to_translate if col not in dataset.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in dataset. Available columns: {dataset.columns}")
        
        # Setup output path
        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"translated_{self.source_lang}_to_{self.target_lang}_batch.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        total_samples = len(dataset)
        print(f"Starting batch translation of {total_samples} samples...")
        print(f"Translating columns: {columns_to_translate}")
        print(f"Batch size: {batch_size}")
        print(f"Output will be saved to: {output_path}")
        
        # Process each column separately for batch translation
        translated_data = []
        
        # Convert dataset to list for easier batch processing
        dataset_list = list(dataset)
        
        for column in columns_to_translate:
            print(f"Processing column: {column}")
            
            # Extract texts for this column
            texts_to_translate = []
            valid_indices = []
            
            for i, item in enumerate(dataset_list):
                text = item.get(column) if isinstance(item, dict) else str(item)
                if text is not None and str(text).strip():
                    texts_to_translate.append(str(text))
                    valid_indices.append(i)
            
            if not texts_to_translate:
                print(f"No valid texts found for column '{column}', skipping.")
                continue
            
            # Perform batch translation
            print(f"Translating {len(texts_to_translate)} texts for column '{column}'...")
            translated_texts = self.translate_batch(texts_to_translate, batch_size)
            
            # Map translations back to original indices
            translation_map = dict(zip(valid_indices, translated_texts))
            
            # Add translations to dataset
            translated_column_name = f"translated_{column}_{self.target_lang}"
            for i, item in enumerate(dataset_list):
                if i == 0:
                    # Initialize translated_data with original data
                    if not translated_data:
                        translated_data = [dict(item) if isinstance(item, dict) else {"original_data": item} 
                                         for item in dataset_list]
                
                if i in translation_map:
                    translated_data[i][translated_column_name] = translation_map[i]
                else:
                    translated_data[i][translated_column_name] = None
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in translated_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Summary
        summary = {
            "total_samples": total_samples,
            "successful_translations": total_samples,  # Batch mode assumes all succeed
            "errors": 0,
            "success_rate": 1.0,
            "output_path": output_path,
            "columns_translated": columns_to_translate,
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "batch_size": batch_size
        }
        
        print(f"\nBatch translation complete!")
        print(f"Total samples: {total_samples}")
        print(f"Output saved to: {output_path}")
        
        return summary 