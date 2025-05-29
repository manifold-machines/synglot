from .base import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import json
import tempfile
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Union, Optional, Dict, Any

class LLMTranslator(Translator):
    """Unified translator supporting both standard ML models (MarianMT) and LLM APIs (OpenAI)."""
    
    def __init__(self, source_lang, target_lang, backend="marianmt", model_name=None, max_gen_tokens=1024):
        """
        Initialize unified translator.
        
        Args:
            source_lang (str): Source language code (e.g., 'en')
            target_lang (str): Target language code (e.g., 'fr')
            backend (str): Backend translation system. Supports 'marianmt' or 'openai'.
            model_name (str): Model name (for OpenAI: e.g., 'gpt-4o-mini'; for MarianMT: auto-determined)
        """
        super().__init__(source_lang, target_lang)
        self.backend = backend
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens

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
        elif self.backend == "openai":
            # Load environment variables from .env file
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI backend.")
            
            self.model_name = model_name if model_name else "gpt-4o-mini"
            self.client = openai.OpenAI(api_key=self.api_key)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not supported. Currently supports 'marianmt' and 'openai'."
            )

    def translate(self, text):
        """Translate using the configured backend (MarianMT or OpenAI)."""
        if self.backend == "marianmt":
            if not self.model or not self.tokenizer:
                raise RuntimeError("MarianMT model and tokenizer not initialized properly.")
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = self.model.generate(**inputs)
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
            
        elif self.backend == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"You are a translator. Translate the following text from {self.source_lang} to {self.target_lang}."},
                        {"role": "user", "content": text}
                    ],
                    max_completion_tokens=self.max_gen_tokens,
                    temperature=0.4
                )
                translated_text = response.choices[0].message.content.strip()
                return translated_text
            except openai.APIError as e:
                raise RuntimeError(f"OpenAI API error: {e}")
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred during translation: {e}")
        else:
            raise NotImplementedError(f"Translation for backend '{self.backend}' is not implemented.")

    def translate_batch(self, texts, batch_size=32, batch_job_description="batch job"):
        """
        Translate a batch of texts using the configured backend.
        
        Args:
            texts (list): List of texts to translate
            batch_size (int): Batch size for MarianMT (ignored for OpenAI batch API)
            batch_job_description (str): Description for OpenAI batch jobs
            
        Returns:
            For MarianMT: List of translated texts
            For OpenAI: Batch job object (use retrieve_batch to get results)
        """
        if self.backend == "marianmt":
            if not self.model or not self.tokenizer:
                raise RuntimeError("MarianMT model and tokenizer not initialized properly.")
            
            translations = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated_tokens = self.model.generate(**inputs)
                batch_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
                translations.extend(batch_translations)
            return translations
            
        elif self.backend == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            temp_file_path = None
            try:
                # Create a temporary file to store the batch requests in the required format
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as batchfile:
                    temp_file_path = batchfile.name
                    for i, text in enumerate(texts):
                        request = {
                            "custom_id": f"request-{i+1}",
                            "method": "POST", 
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model_name,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": f"You are a translator. Translate the following text from {self.source_lang} to {self.target_lang}."
                                    },
                                    {
                                        "role": "user",
                                        "content": text
                                    }
                                ],
                                "max_completion_tokens": self.max_gen_tokens,
                                "temperature": 0.4
                            }
                        }
                        batchfile.write(json.dumps(request) + '\n')
                    
                    # Ensure the entire file is written
                    batchfile.flush()
                
                # Upload file to openai
                with open(temp_file_path, "rb") as file_to_upload:
                    batch_input_file = self.client.files.create(
                        file=file_to_upload,
                        purpose="batch"
                    )

                # Create batch
                batch_input_file_id = batch_input_file.id
                created_batch = self.client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": batch_job_description
                    }
                )

                return created_batch
                
            except openai.APIError as e:
                raise RuntimeError(f"OpenAI API error during batch translation: {e}")
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred during batch translation: {e}")
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except OSError as e:
                        logging.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
        else:
            # Fallback to base class implementation
            return super().translate_batch(texts, batch_size)

    def retrieve_batch(self, saved_batch):
        """
        Retrieve the batch output content when the OpenAI batch job is done.
        Only applicable for OpenAI backend.
        
        Args:
            saved_batch: Batch job object returned from translate_batch
            
        Returns:
            File content if completed, None if still in progress
        """
        if self.backend == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            try:
                batch_id = saved_batch.id
                batch_status = self.client.batches.retrieve(batch_id).status
                
                if batch_status == "completed":
                    output_file_id = self.client.batches.retrieve(batch_id).output_file_id
                    file_content = self.client.files.content(output_file_id)
                    return file_content
                elif batch_status == "failed":
                    raise RuntimeError(f"Batch job {batch_id} failed")
                elif batch_status == "cancelled":
                    raise RuntimeError(f"Batch job {batch_id} was cancelled")
                else:
                    # Batch is still in progress (validating, in_progress, finalizing)
                    logging.info(f"Batch not completed; current status is {batch_status}.")
                    return None
                    
            except openai.APIError as e:
                raise RuntimeError(f"OpenAI API error during batch retrieval: {e}")
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred during batch retrieval: {e}")
        else:
            raise NotImplementedError("retrieve_batch is only available for OpenAI backend.")

    def translate_dataset(self, 
                         dataset, 
                         columns_to_translate: Union[str, List[str], None], 
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
        elif columns_to_translate is None:
            # If no columns specified, automatically detect all available columns
            if hasattr(dataset, 'columns') and dataset.columns:
                all_columns = list(dataset.columns)
                columns_to_translate = self._filter_translatable_columns(dataset, all_columns)
                print(f"No columns specified, auto-detecting translatable columns: {columns_to_translate}")
                if not columns_to_translate:
                    raise ValueError("No translatable text columns found in dataset")
            else:
                raise ValueError("No columns specified and dataset has no detectable columns")
        
        # Validate columns exist in dataset
        if hasattr(dataset, 'columns') and dataset.columns:
            missing_columns = [col for col in columns_to_translate if col not in dataset.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in dataset. Available columns: {dataset.columns}")
        
        # Setup output path
        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"translated_{self.source_lang}_to_{self.target_lang}_{self.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize counters
        total_samples = len(dataset)
        success_count = 0
        error_count = 0
        
        print(f"Starting translation of {total_samples} samples using {self.backend} backend...")
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
            "target_language": self.target_lang,
            "backend": self.backend
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
                               columns_to_translate: Union[str, List[str], None], 
                               output_path: Optional[str] = None,
                               output_dir: str = "./outputs",
                               batch_size: int = 32,
                               progress_interval: int = 100) -> Dict[str, Any]:
        """
        Translate specified columns in a dataset using batch processing for better performance.
        Note: Batch processing provides better performance but less granular error handling.
        For OpenAI backend, this creates a batch job that needs to be retrieved later.
        
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
        elif columns_to_translate is None:
            # If no columns specified, automatically detect all available columns
            if hasattr(dataset, 'columns') and dataset.columns:
                all_columns = list(dataset.columns)
                columns_to_translate = self._filter_translatable_columns(dataset, all_columns)
                print(f"No columns specified, auto-detecting translatable columns: {columns_to_translate}")
                if not columns_to_translate:
                    raise ValueError("No translatable text columns found in dataset")
            else:
                raise ValueError("No columns specified and dataset has no detectable columns")
        
        # Validate columns exist in dataset
        if hasattr(dataset, 'columns') and dataset.columns:
            missing_columns = [col for col in columns_to_translate if col not in dataset.columns]
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} not found in dataset. Available columns: {dataset.columns}")
        
        # Setup output path
        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"translated_{self.source_lang}_to_{self.target_lang}_{self.backend}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        total_samples = len(dataset)
        print(f"Starting batch translation of {total_samples} samples using {self.backend} backend...")
        print(f"Translating columns: {columns_to_translate}")
        print(f"Batch size: {batch_size}")
        print(f"Output will be saved to: {output_path}")
        
        if self.backend == "openai":
            # For OpenAI, we need to handle batch processing differently
            # This is a simplified version - you might want to implement more sophisticated batch handling
            print("Note: OpenAI batch processing creates async jobs. Use retrieve_batch() to get results when complete.")
            
            # For now, fall back to individual processing for OpenAI in dataset context
            # You could extend this to create proper batch jobs per column
            return self.translate_dataset(dataset, columns_to_translate, output_path, output_dir, 
                                        batch_size=1, progress_interval=progress_interval)
        
        # Process each column separately for batch translation (MarianMT)
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
            "backend": self.backend,
            "batch_size": batch_size
        }
        
        print(f"\nBatch translation complete!")
        print(f"Total samples: {total_samples}")
        print(f"Output saved to: {output_path}")
        
        return summary