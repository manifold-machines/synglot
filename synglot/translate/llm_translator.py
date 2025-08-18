from .base import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration
import openai
import json
import tempfile
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Union, Optional, Dict, Any
from tqdm import tqdm
import tiktoken
import torch
import requests

from synglot.utils.batch_utils import split_requests_by_limits, create_single_batch_job_with_tokens
from synglot.utils.language_mappings import get_nllb_language_code
from synglot.utils.nested_utils import (
    extract_texts_from_field,
    set_translated_nested_value
)
from synglot.utils.dataset_utils import auto_detect_translatable_columns, column_exists, save_media_file
from synglot.utils.text_utils import num_tokens_consumed_from_request

class LLMTranslator(Translator):
    """Unified translator supporting standard ML models (MarianMT), LLM APIs (OpenAI), Google Translate API, and NLLB."""
    
    def __init__(self, source_lang, target_lang, backend="marianmt", model_name=None, max_gen_tokens=1024, project_id=None, device="auto"):
        """
        Initialize unified translator.
        
        Args:
            source_lang (str): Source language code (e.g., 'en')
            target_lang (str): Target language code (e.g., 'fr')
            backend (str): Backend translation system. Supports 'marianmt', 'openai', 'google', or 'nllb'.
            model_name (str): Model name (for OpenAI: e.g., 'gpt-4o-mini'; for MarianMT: auto-determined; for NLLB: defaults to facebook/nllb-200-3.3B; ignored for Google)
            max_gen_tokens (int): Maximum tokens for generation (used by OpenAI backend)
            project_id (str): Google Cloud project ID (required for Google backend)
            device (str): Device for NLLB model ('auto', 'cpu', 'cuda', or specific device)
        """
        super().__init__(source_lang, target_lang)
        self.backend = backend
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens
        self.project_id = project_id
        self.device = device
        load_dotenv()

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
        elif self.backend == "nllb":
            try:
                self.model_name = model_name if model_name else "facebook/nllb-200-3.3B"
                
                if device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                if self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device):
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name, torch_dtype=torch.float16 # use float16 for efficiency if on CUDA
                    ).to(self.device).eval()
                else:
                    self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name).to(self.device).eval()
                
                # map language codes to NLLB format
                self.source_lang_nllb = get_nllb_language_code(source_lang)
                self.target_lang_nllb = get_nllb_language_code(target_lang)
                
                print(f"NLLB model loaded on {self.device}")
                print(f"Source language: {source_lang} -> {self.source_lang_nllb}")
                print(f"Target language: {target_lang} -> {self.target_lang_nllb}")
                
            except Exception as e:
                raise ValueError(
                    f"Failed to load NLLB model {self.model_name}. "
                    f"Ensure the model is available and transformers/torch libraries are correctly installed. Error: {e}"
                )
        elif self.backend == "openai":
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI backend.")
            
            self.model_name = model_name if model_name else "gpt-4.1-mini"
            self.client = openai.OpenAI(api_key=self.api_key)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            
            try:
                self.encoding = tiktoken.encoding_for_model(self.model_name) # using for counting tokens
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base") # for unknown models
        elif self.backend == "google":
            try:
                from google.cloud import translate_v3
            except ImportError:
                raise ImportError(
                    "Google Cloud Translate library not found. Install it with: pip install google-cloud-translate"
                )
            
            self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT_ID')
            if not self.project_id:
                raise ValueError(
                    "Google Cloud project ID is required for Google backend. "
                    "Provide it via project_id parameter or GOOGLE_CLOUD_PROJECT_ID environment variable."
                )
            
            self.client = translate_v3.TranslationServiceClient()
            self.parent = f"projects/{self.project_id}"
        elif self.backend == "openrouter":
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            }
            self.model_name = model_name if model_name else "moonshotai/kimi-k2"
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not supported. Currently supports 'marianmt', 'openai', 'google', 'nllb', and 'openrouter'."
            )



    def translate(self, text):
        """Translate using the configured backend (MarianMT, OpenAI, Google Translate, or NLLB)."""
        if self.backend == "marianmt":
            if not self.model or not self.tokenizer:
                raise RuntimeError("MarianMT model and tokenizer not initialized properly.")
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = self.model.generate(**inputs)
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
            
        elif self.backend == "nllb":
            if not self.model or not self.tokenizer:
                raise RuntimeError("NLLB model and tokenizer not initialized properly.")
            
            self.tokenizer.src_lang = self.source_lang_nllb
            
            encoded_input = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb)
            translated_tokens = self.model.generate(
                **encoded_input, 
                forced_bos_token_id=target_token_id
            )
            
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            return translated_text
            
        elif self.backend == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"You are a professional translator. Your ONLY task is to translate text from {self.source_lang} to {self.target_lang}. Do NOT answer questions, provide explanations, or add any commentary. Simply return the translation of the given text and nothing else. If the input appears to be a question, translate the question itself, do not answer it."},
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
        
        elif self.backend == "google":
            if not self.client:
                raise RuntimeError("Google Translate client not initialized properly.")
            
            try:
                response = self.client.translate_text(
                    parent=self.parent,
                    contents=[text],
                    target_language_code=self.target_lang,
                    source_language_code=self.source_lang
                )
                translated_text = response.translations[0].translated_text
                return translated_text
            except Exception as e:
                raise RuntimeError(f"Google Translate API error: {e}")
        
        elif self.backend == "openrouter":
            payload = {
            "model": self.model_name,
            "messages": [
                {
                "role": "system",
                "content": f"You are a professional translator. Your ONLY task is to translate text from {self.source_lang} to {self.target_lang}. Do NOT answer questions, provide explanations, or add any commentary. Simply return the translation of the given text and nothing else. If the input appears to be a question, translate the question itself, do not answer it."
                },
                {
                "role": "user",
                "content": text
                }
            ],
            "temperature": 0.4,
            "max_tokens": self.max_gen_tokens  # Increased from 2000 to allow longer responses
            }
            response = requests.post(self.url, headers=self.headers, json=payload)
            response_json = response.json()
            return response_json['choices'][0]['message']['content']

        else:
            raise NotImplementedError(f"Translation for backend '{self.backend}' is not implemented.")

    def translate_batch(self, texts, batch_size=32, batch_job_description="batch job"):
        """
        Translate a batch of texts using the configured backend.
        
        Args:
            texts (list): List of texts to translate
            batch_size (int): Batch size for MarianMT, NLLB, and Google Translate (ignored for OpenAI batch API)
            batch_job_description (str): Description for OpenAI batch jobs
            
        Returns:
            For MarianMT, NLLB, and Google Translate: List of translated texts
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
            
        elif self.backend == "nllb":
            if not self.model or not self.tokenizer:
                raise RuntimeError("NLLB model and tokenizer not initialized properly.")
            
            self.tokenizer.src_lang = self.source_lang_nllb
            target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb)
            
            translations = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                encoded_input = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                translated_tokens = self.model.generate(
                    **encoded_input, 
                    forced_bos_token_id=target_token_id
                )
                
                batch_translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translations.extend(batch_translations)
                
                if self.device != "cpu":
                    del encoded_input, translated_tokens
                    torch.cuda.empty_cache()
            
            return translations
            
        elif self.backend == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            temp_file_path = None
            try:
                # temporary file to store the batch requests in the required format
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
                                        "content": f"You are a professional translator. Your ONLY task is to translate text from {self.source_lang} to {self.target_lang}. Do NOT answer questions, provide explanations, or add any commentary. Simply return the translation of the given text and nothing else. If the input appears to be a question, translate the question itself, do not answer it."
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
                    
                    batchfile.flush()
                
                with open(temp_file_path, "rb") as file_to_upload:
                    batch_input_file = self.client.files.create(
                        file=file_to_upload,
                        purpose="batch"
                    )

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
                # clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except OSError as e:
                        logging.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
        
        elif self.backend == "google":
            if not self.client:
                raise RuntimeError("Google Translate client not initialized properly.")
            
            try:
                translations = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    response = self.client.translate_text(
                        parent=self.parent,
                        contents=batch,
                        target_language_code=self.target_lang,
                        source_language_code=self.source_lang
                    )
                    
                    batch_translations = [translation.translated_text for translation in response.translations]
                    translations.extend(batch_translations)
                
                return translations
                
            except Exception as e:
                raise RuntimeError(f"Google Translate API error during batch translation: {e}")
        
        elif self.backend == "openrouter":
            raise NotImplementedError("OpenRouter batch translation is not supported.")
        else:
            # Fallback to base class implementation for unsupported backends
            return super().translate_batch(texts, batch_size)

    def translate_dataset(self, 
                         dataset, 
                         columns_to_translate: Union[str, List[str], None], 
                         output_path: Optional[str] = None,
                         output_dir: str = "./outputs",
                         batch_size: int = 32,
                         progress_interval: int = 10,
                         save_errors: bool = True,
                         append_mode: bool = False,
                         use_batch: bool = False,
                         batch_job_description: str = "dataset translation",
                         batch_request_limit: int = 50000,
                         batch_token_limit: int = 1900000,
                         streaming_mode: bool = False,
                         auto_reduce_batch_size: bool = True,
                         min_batch_size: int = 1,
                         nested_field_separator: str = ".",
                         media_output_dir: Optional[str] = None,
                         media_field_name: str = "image") -> Dict[str, Any]:
        """
        Translate specified columns in a dataset with comprehensive error handling and progress tracking.
        
        Args:
            dataset: Dataset object to translate
            columns_to_translate (str or list): Column name(s) to translate. Supports nested fields like "qa.question"
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None)
            batch_size (int): Batch size for translation
            progress_interval (int): Print progress every N samples (used in non-batch mode)
            save_errors (bool): Whether to save error records to output (used in non-batch mode)
            append_mode (bool): Whether to append to existing file or overwrite (used in non-batch mode)
            use_batch (bool): Whether to use batch processing for better performance
            batch_job_description (str): Description for batch jobs (OpenAI only)
            batch_request_limit (int): Maximum number of requests per batch for OpenAI backend
            batch_token_limit (int): Maximum number of tokens per batch for OpenAI backend
            streaming_mode (bool): Whether to process dataset in streaming mode (for large datasets)
            auto_reduce_batch_size (bool): Whether to automatically reduce batch size on OOM errors
            min_batch_size (int): Minimum batch size when auto-reducing
            nested_field_separator (str): Separator for nested field names (e.g., "qa.question")
            media_output_dir (str, optional): Directory to save media files (images, etc.). If None, uses output_dir/media
            media_field_name (str): Name of the field containing media data
            
        Returns:
            dict: Summary statistics including success/error counts and output path
            For OpenAI batch mode: Returns batch job info that needs to be retrieved later with retrieve_batch()
        """
        # get the columns to translate, auto-translate if needed
        if isinstance(columns_to_translate, str):
            columns_to_translate = [columns_to_translate]
        elif columns_to_translate is None:
            columns_to_translate = auto_detect_translatable_columns(dataset, streaming_mode, nested_field_separator)
        
        # sanity check for missing columns
        if not streaming_mode and hasattr(dataset, 'columns') and dataset.columns:
            missing_columns = [col for col in columns_to_translate if not column_exists(dataset, col, nested_field_separator)]
            if missing_columns:
                available = list(dataset.columns) if hasattr(dataset, 'columns') else "unknown"
                raise ValueError(f"Columns {missing_columns} not found in dataset. Available columns: {available}")
        
        # output paths for text and media

        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            batch_suffix = "_batch" if use_batch else ""
            streaming_suffix = "_streaming" if streaming_mode else ""
            filename = f"translated_{self.source_lang}_to_{self.target_lang}_{self.backend}{batch_suffix}{streaming_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            output_dirname = os.path.dirname(output_path)
            if output_dirname:
                os.makedirs(output_dirname, exist_ok=True)
        
        if media_output_dir is None:
            media_output_dir = os.path.join(output_dir, "media")
        os.makedirs(media_output_dir, exist_ok=True)
        
        # Determine total samples (only if not streaming)
        if streaming_mode:
            total_samples = None
            print(f"Starting {'batch ' if use_batch else ''}translation in streaming mode using {self.backend} backend...")
        else:
            total_samples = len(dataset)
            print(f"Starting {'batch ' if use_batch else ''}translation of {total_samples} samples using {self.backend} backend...")
        
        print(f"Translating columns: {columns_to_translate}")
        if use_batch:
            print(f"Batch size: {batch_size}")
            if self.backend == "openai":
                print(f"Request limit per batch: {batch_request_limit}")
                print(f"Token limit per batch: {batch_token_limit:,}")
        if auto_reduce_batch_size and self.backend in ["nllb", "marianmt"]:
            print(f"Auto batch size reduction enabled (min: {min_batch_size})")
        print(f"Output will be saved to: {output_path}")
        if media_output_dir:
            print(f"Media files will be saved to: {media_output_dir}")
        
        if streaming_mode:
            return self._translate_dataset_streaming_mode(
                dataset, columns_to_translate, output_path, batch_size, 
                auto_reduce_batch_size, min_batch_size, nested_field_separator, 
                media_output_dir, media_field_name, save_errors
            )
        elif use_batch:
            return self._translate_dataset_batch_mode(dataset, columns_to_translate, output_path, 
                                                    batch_size, batch_job_description, total_samples, batch_request_limit, batch_token_limit)
        else:
            return self._translate_dataset_sequential_mode(dataset, columns_to_translate, output_path,
                                                         progress_interval, save_errors, append_mode, total_samples)

    def _translate_dataset_streaming_mode(self, dataset, columns_to_translate, output_path, 
                                        initial_batch_size, auto_reduce_batch_size, min_batch_size, 
                                        nested_field_separator, media_output_dir, media_field_name, save_errors):
        """Handle streaming mode translation with OOM protection and nested structure support."""
        current_batch_size = initial_batch_size
        processed_count = 0
        success_count = 0
        error_count = 0
        batch_number = 0
        
        print(f"Starting streaming translation with batch size {current_batch_size}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            dataset_iter = iter(dataset)
            
            while True:
                try:
                    batch_samples = []
                    batch_texts = []
                    text_metadata = []  # Tracks which text belongs to which sample/field
                    
                    for _ in range(current_batch_size):
                        try:
                            sample = next(dataset_iter)
                            sample_idx = len(batch_samples)
                            batch_samples.append(sample)
                            
                            for column in columns_to_translate:
                                texts = extract_texts_from_field(sample, column, nested_field_separator)
                                for text_info in texts:
                                    batch_texts.append(text_info['text'])
                                    text_metadata.append({
                                        'sample_idx': sample_idx,
                                        'column': column,
                                        'path': text_info['path'],
                                        'index': text_info.get('index', None)
                                    })
                        except StopIteration:
                            break
                    
                    if not batch_samples:
                        print("Reached end of dataset")
                        break
                    
                    batch_number += 1
                    print(f"Processing batch {batch_number}: {len(batch_texts)} text pieces from {len(batch_samples)} samples")
                    
                    if batch_texts:
                        print(f"  Translating batch of {len(batch_texts)} texts...")
                        if hasattr(self, 'translate_batch'):
                            translated_texts = self.translate_batch(batch_texts)
                        else:
                            translated_texts = [self.translate(text) for text in batch_texts]
                        
                        print(f"  Reconstructing {len(batch_samples)} samples...")
                        for sample_idx, sample in enumerate(batch_samples):
                            try:
                                output_record = dict(sample) if isinstance(sample, dict) else {"original_data": sample}
                                
                                if media_field_name in output_record:
                                    media_path = save_media_file(
                                        output_record[media_field_name], 
                                        media_output_dir, 
                                        processed_count, 
                                        media_field_name
                                    )
                                    if media_path:
                                        output_record[media_field_name] = media_path
                                
                                self._add_translations_to_sample(
                                    output_record, translated_texts, text_metadata, 
                                    sample_idx, nested_field_separator
                                )
                                
                                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                                f.flush()
                                
                                success_count += 1
                                processed_count += 1
                                
                                if processed_count % 10 == 0:
                                    print(f"  Processed {processed_count} samples")
                                    
                            except Exception as e:
                                error_count += 1
                                print(f"  Error processing sample {processed_count + 1}: {e}")
                                
                                if save_errors:
                                    error_record = {
                                        'sample_index': processed_count + 1,
                                        'error': str(e),
                                        'error_type': type(e).__name__,
                                        'original_data': dict(sample) if isinstance(sample, dict) else sample
                                    }
                                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                                    f.flush()
                                
                                processed_count += 1
                    else:
                        for sample in batch_samples:
                            output_record = dict(sample) if isinstance(sample, dict) else {"original_data": sample}
                            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                            processed_count += 1
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if auto_reduce_batch_size and ("out of memory" in str(e).lower() or "oom" in str(e).lower()):
                        print(f"  OOM Error with batch size {current_batch_size}: {e}")
                        
                        current_batch_size = max(current_batch_size // 2, min_batch_size)
                        
                        if current_batch_size < min_batch_size:
                            print(f"  Batch size reduced to minimum ({min_batch_size}), but still getting OOM. Stopping.")
                            break
                        
                        print(f"  Reducing batch size to {current_batch_size} and retrying...")
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        continue
                    else:
                        print(f"  Unexpected error: {e}")
                        break
                
                except Exception as e:
                    print(f"  Unexpected error processing batch: {e}")
                    error_count += len(batch_samples) if batch_samples else 1
                    continue
        
        # Final summary
        summary = {
            "total_samples": processed_count,
            "successful_translations": success_count,
            "errors": error_count,
            "success_rate": success_count / processed_count if processed_count > 0 else 0,
            "output_path": output_path,
            "columns_translated": columns_to_translate,
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "backend": self.backend,
            "final_batch_size": current_batch_size,
            "streaming_mode": True,
            "status": "completed"
        }
        
        print(f"\nStreaming translation complete!")
        print(f"Total samples processed: {processed_count}")
        print(f"Successful translations: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Final batch size: {current_batch_size}")
        print(f"Output saved to: {output_path}")
        
        return summary

    def _add_translations_to_sample(self, output_record, translated_texts, text_metadata, 
                                  sample_idx, nested_field_separator):
        """Add translated texts back to the sample record."""
        for i, metadata in enumerate(text_metadata):
            if metadata['sample_idx'] == sample_idx:
                translation = translated_texts[i]
                field_path = metadata['path']
                
                if nested_field_separator in field_path:
                    # For nested fields like "qa.question", create "translated_qa.question"
                    parts = field_path.split(nested_field_separator)
                    translated_field = f"translated_{parts[0]}{nested_field_separator}{nested_field_separator.join(parts[1:])}"
                else:
                    translated_field = f"translated_{field_path}_{self.target_lang}"
                
                if nested_field_separator in translated_field:
                    # Handle nested structure
                    set_translated_nested_value(
                        output_record, translated_field, translation, 
                        nested_field_separator, metadata.get('index')
                    )
                else:
                    # Simple field
                    output_record[translated_field] = translation



    def _translate_dataset_batch_mode(self, dataset, columns_to_translate, output_path, 
                                    batch_size, batch_job_description, total_samples, batch_request_limit=40000, batch_token_limit=1900000):
        """Handle batch mode translation for datasets."""
        if self.backend == "openai":
            print("Creating OpenAI batch jobs for dataset translation... Use retrieve_batch() to get results when complete.")
            
            dataset_list = list(dataset)
            
            all_texts_with_metadata = []
            
            request_id = 0
            for column in columns_to_translate:
                print(f"Preparing batch requests for column: {column}")
                
                for i, item in enumerate(dataset_list):
                    text = item.get(column) if isinstance(item, dict) else str(item)
                    if text is not None and str(text).strip():
                        request_json = {
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": f"You are a professional translator. Your ONLY task is to translate text from {self.source_lang} to {self.target_lang}. Do NOT answer questions, provide explanations, or add any commentary. Simply return the translation of the given text and nothing else. If the input appears to be a question, translate the question itself, do not answer it."
                                },
                                {
                                    "role": "user",
                                    "content": str(text)
                                }
                            ],
                            "max_completion_tokens": self.max_gen_tokens,
                            "temperature": 0.4
                        }
                        
                        token_count = num_tokens_consumed_from_request(request_json, self.encoding, self.max_gen_tokens) if self.backend == "openai" else 0
                        
                        all_texts_with_metadata.append({
                            "text": str(text),
                            "column": column,
                            "sample_index": i,
                            "request_id": request_id,
                            "token_count": token_count,
                            "request_json": request_json
                        })
                        request_id += 1
            
            if not all_texts_with_metadata:
                raise ValueError("No valid texts found for translation")
            
            total_requests = len(all_texts_with_metadata)
            total_tokens = sum(item["token_count"] for item in all_texts_with_metadata)
            print(f"Total translation requests: {total_requests}")
            print(f"Total estimated tokens: {total_tokens:,}")
            
            needs_split = total_requests > batch_request_limit or total_tokens > batch_token_limit
            
            if not needs_split: # single batch job
                print(f"Creating single batch job with {total_requests} requests and {total_tokens:,} tokens...")
                batch_job = create_single_batch_job_with_tokens(
                    self.client, all_texts_with_metadata, batch_job_description, output_path, 
                    columns_to_translate, total_samples, self.source_lang, self.target_lang, self.backend
                )
                return batch_job
            else: # split respecting both limits
                print(f"Splitting into multiple batches due to limits:")
                print(f"  Request limit: {batch_request_limit}")
                print(f"  Token limit: {batch_token_limit:,}")
                
                batches = split_requests_by_limits(all_texts_with_metadata, batch_request_limit, batch_token_limit)
                
                print(f"Created {len(batches)} batches")
                
                batch_jobs = []
                for i, batch_chunk in enumerate(batches):
                    chunk_tokens = sum(item["token_count"] for item in batch_chunk)
                    chunk_description = f"{batch_job_description} - batch {i+1}/{len(batches)}"
                    print(f"Creating batch {i+1}/{len(batches)} with {len(batch_chunk)} requests and {chunk_tokens:,} tokens...")
                    
                    # Modify output path for each batch
                    base_path, ext = os.path.splitext(output_path)
                    chunk_output_path = f"{base_path}_batch_{i+1}{ext}"
                    
                    batch_job = create_single_batch_job_with_tokens(
                        self.client, batch_chunk, chunk_description, chunk_output_path, 
                        columns_to_translate, total_samples, self.source_lang, self.target_lang, self.backend
                    )
                    batch_jobs.append(batch_job)
                
                print(f"All {len(batches)} batch jobs created successfully!")
                print("Each batch will need to be retrieved separately using retrieve_batch()")
                
                return {
                    "multiple_batches": True,
                    "batch_jobs": batch_jobs,
                    "total_batches": len(batches),
                    "total_requests": total_requests,
                    "total_tokens": total_tokens,
                    "columns_translated": columns_to_translate,
                    "source_language": self.source_lang,
                    "target_language": self.target_lang,
                    "backend": self.backend,
                    "output_paths": [job["output_path"] for job in batch_jobs],
                    "dataset_size": total_samples,
                    "status": "batches_submitted",
                    "instructions": "Use retrieve_batch() with each batch_job individually to get results when complete. Results will be saved to separate files."
                }
                
        else:
            # synchronous batch processing for MarianMT, NLLB, and Google Translate
            print(f"Using {self.backend} batch processing...")
            
            dataset_list = list(dataset)
            translated_data = []
            for item in dataset_list:
                result_item = dict(item) if isinstance(item, dict) else {"original_data": item}
                
                # Handle all media files (detect PIL Image objects automatically)
                for field_name, field_value in list(result_item.items()):
                    if hasattr(field_value, 'save') and hasattr(field_value, 'format'):  # PIL Image detection
                        # For batch mode, we'll remove image fields to prevent serialization issues
                        # Images can be processed separately if needed
                        del result_item[field_name]
                
                translated_data.append(result_item)
            
            success_count = 0
            error_count = 0
            
            for column in columns_to_translate:
                print(f"Processing column: {column}")
                
                texts_to_translate = []
                text_indices = []
                
                for i, item in enumerate(dataset_list):
                    texts = extract_texts_from_field(item, column, ".")
                    for text_info in texts:
                        texts_to_translate.append(text_info['text'])
                        text_indices.append({'sample_idx': i, 'text_info': text_info})
                
                if not texts_to_translate:
                    print(f"No valid texts found for column '{column}', skipping.")
                    continue
                
                print(f"Translating {len(texts_to_translate)} texts for column '{column}'...")
                
                try:
                    translated_texts = self.translate_batch(texts_to_translate, batch_size)
                    
                    for idx_info, translation in zip(text_indices, translated_texts):
                        sample_idx = idx_info['sample_idx']
                        text_info = idx_info['text_info']
                        
                        if "." in text_info['path']:
                            # if nested as "qa.question", create "translated_qa.question"
                            parts = text_info['path'].split(".")
                            translated_field = f"translated_{parts[0]}.{'.'.join(parts[1:])}"
                            set_translated_nested_value(
                                translated_data[sample_idx], translated_field, translation, 
                                ".", text_info.get('index')
                            )
                        else:
                            translated_column_name = f"translated_{column}_{self.target_lang}"
                            translated_data[sample_idx][translated_column_name] = translation
                        
                        success_count += 1
                        
                    print(f"Successfully translated {len(translated_texts)} texts for column '{column}'")
                    
                except Exception as e:
                    print(f"Error translating column '{column}': {e}")
                    error_count += len(texts_to_translate)
                    
                    # add None if failed
                    for idx_info in text_indices:
                        sample_idx = idx_info['sample_idx']
                        text_info = idx_info['text_info']
                        
                        if "." in text_info['path']:
                            parts = text_info['path'].split(".")
                            translated_field = f"translated_{parts[0]}.{'.'.join(parts[1:])}"
                            set_translated_nested_value(
                                translated_data[sample_idx], translated_field, None, 
                                ".", text_info.get('index')
                            )
                        else:
                            translated_column_name = f"translated_{column}_{self.target_lang}"
                            translated_data[sample_idx][translated_column_name] = None
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in translated_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            summary = {
                "total_samples": total_samples,
                "successful_translations": success_count,
                "errors": error_count,
                "success_rate": success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0,
                "output_path": output_path,
                "columns_translated": columns_to_translate,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "backend": self.backend,
                "batch_size": batch_size,
                "status": "completed"
            }
            
            print(f"\n{self.backend} batch translation complete!")
            print(f"Total samples: {total_samples}")
            print(f"Successful translations: {success_count}")
            print(f"Errors: {error_count}")
            print(f"Success rate: {summary['success_rate']:.2%}")
            print(f"Output saved to: {output_path}")
            
            return summary

    def _translate_dataset_sequential_mode(self, dataset, columns_to_translate, output_path,
                                         progress_interval, save_errors, append_mode, total_samples):
        """Handle sequential mode translation for datasets."""
        # counters
        success_count = 0
        error_count = 0
        
        file_mode = 'a' if append_mode else 'w'
        with open(output_path, file_mode, encoding='utf-8') as f:
            with tqdm(total=total_samples, desc="Translating", unit="samples") as pbar:
                for i, item in enumerate(dataset):
                    try:
                        output_record = dict(item) if isinstance(item, dict) else {"original_data": item}
                        
                        for column in columns_to_translate:
                            texts = extract_texts_from_field(item, column, ".")
                            
                            if not texts:
                                print(f"Warning: No translatable text found in column '{column}' for sample {i+1}")
                                continue
                            
                            for text_info in texts:
                                try:
                                    translated_text = self.translate(text_info['text'])
                                    
                                    if "." in text_info['path']:
                                        parts = text_info['path'].split(".")
                                        translated_field = f"translated_{parts[0]}.{'.'.join(parts[1:])}"
                                        set_translated_nested_value(
                                            output_record, translated_field, translated_text, 
                                            ".", text_info.get('index')
                                        )
                                    else:
                                        translated_column_name = f"translated_{column}_{self.target_lang}"
                                        output_record[translated_column_name] = translated_text
                                
                                except Exception as e:
                                    print(f"Warning: Failed to translate text in column '{column}' for sample {i+1}: {e}")
                                    continue
                        
                        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        success_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        print(f"Error processing sample {i+1}: {e}")
                        
                        if save_errors:
                            error_record = {
                                "sample_index": i+1,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "original_data": dict(item) if isinstance(item, dict) else item
                            }
                            f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': success_count,
                        'Errors': error_count,
                        'Rate': f"{success_count/(success_count+error_count)*100:.1f}%" if (success_count + error_count) > 0 else "0.0%"
                    })
        
        summary = {
            "total_samples": total_samples,
            "successful_translations": success_count,
            "errors": error_count,
            "success_rate": success_count / total_samples if total_samples > 0 else 0,
            "output_path": output_path,
            "columns_translated": columns_to_translate,
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "backend": self.backend,
            "status": "completed"
        }
        
        print(f"\nTranslation complete!")
        print(f"Total samples: {total_samples}")
        print(f"Successful translations: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Output saved to: {output_path}")
        
        return summary