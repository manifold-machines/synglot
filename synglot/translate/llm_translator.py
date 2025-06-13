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
from synglot.utils import retrieve_batch
from tqdm import tqdm
import tiktoken
import torch

class LLMTranslator(Translator):
    """Unified translator supporting standard ML models (MarianMT), LLM APIs (OpenAI), Google Translate API, and NLLB."""
    
    def __init__(self, source_lang, target_lang, backend="marianmt", model_name=None, max_gen_tokens=1024, project_id=None, device="auto"):
        """
        Initialize unified translator.
        
        Args:
            source_lang (str): Source language code (e.g., 'en')
            target_lang (str): Target language code (e.g., 'fr')
            backend (str): Backend translation system. Supports 'marianmt', 'openai', 'google', or 'nllb'.
            model_name (str): Model name (for OpenAI: e.g., 'gpt-4o-mini'; for MarianMT: auto-determined; for NLLB: defaults to facebook/nllb-200-distilled-600M; ignored for Google)
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
                self.model_name = model_name if model_name else "facebook/nllb-200-distilled-600M"
                
                # Determine device
                if device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device
                
                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Use float16 for efficiency if on CUDA
                if self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device):
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name, torch_dtype=torch.float16
                    ).to(self.device).eval()
                else:
                    self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name).to(self.device).eval()
                
                # Map language codes to NLLB format
                self.source_lang_nllb = self._map_lang_to_nllb(source_lang)
                self.target_lang_nllb = self._map_lang_to_nllb(target_lang)
                
                print(f"NLLB model loaded on {self.device}")
                print(f"Source language: {source_lang} -> {self.source_lang_nllb}")
                print(f"Target language: {target_lang} -> {self.target_lang_nllb}")
                
            except Exception as e:
                raise ValueError(
                    f"Failed to load NLLB model {self.model_name}. "
                    f"Ensure the model is available and transformers/torch libraries are correctly installed. Error: {e}"
                )
        elif self.backend == "openai":
            # Load environment variables from .env file
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI backend.")
            
            self.model_name = model_name if model_name else "gpt-4.1-mini"
            self.client = openai.OpenAI(api_key=self.api_key)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Initialize tokenizer for token counting
            try:
                self.encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        elif self.backend == "google":
            try:
                from google.cloud import translate_v3
            except ImportError:
                raise ImportError(
                    "Google Cloud Translate library not found. Install it with: pip install google-cloud-translate"
                )
            
            # Load environment variables from .env file
            load_dotenv()
            
            # Get project ID from parameter, environment variable, or raise error
            self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT_ID')
            if not self.project_id:
                raise ValueError(
                    "Google Cloud project ID is required for Google backend. "
                    "Provide it via project_id parameter or GOOGLE_CLOUD_PROJECT_ID environment variable."
                )
            
            self.client = translate_v3.TranslationServiceClient()
            self.parent = f"projects/{self.project_id}"
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not supported. Currently supports 'marianmt', 'openai', 'google', and 'nllb'."
            )

    def _map_lang_to_nllb(self, lang_code):
        """
        Map standard language codes to NLLB format.
        This is a basic mapping - extend as needed for more languages.
        """
        # Common language mappings to NLLB format
        lang_mapping = {
            'en': 'eng_Latn',
            'es': 'spa_Latn', 
            'fr': 'fra_Latn',
            'de': 'deu_Latn',
            'it': 'ita_Latn',
            'pt': 'por_Latn',
            'ru': 'rus_Cyrl',
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'ko': 'kor_Hang',
            'ar': 'arb_Arab',
            'hi': 'hin_Deva',
            'tr': 'tur_Latn',
            'pl': 'pol_Latn',
            'nl': 'nld_Latn',
            'sv': 'swe_Latn',
            'da': 'dan_Latn',
            'no': 'nob_Latn',
            'fi': 'fin_Latn',
            'el': 'ell_Grek',
            'he': 'heb_Hebr',
            'th': 'tha_Thai',
            'vi': 'vie_Latn',
            'uk': 'ukr_Cyrl',
            'cs': 'ces_Latn',
            'hu': 'hun_Latn',
            'ro': 'ron_Latn',
            'bg': 'bul_Cyrl',
            'hr': 'hrv_Latn',
            'sk': 'slk_Latn',
            'sl': 'slv_Latn',
            'et': 'est_Latn',
            'lv': 'lav_Latn',
            'lt': 'lit_Latn',
            'mk': 'mkd_Cyrl',
            'id': 'ind_Latn',
            'ms': 'zsm_Latn',
            'bn': 'ben_Beng',
            'ta': 'tam_Taml',
            'te': 'tel_Telu',
            'ml': 'mal_Mlym',
            'kn': 'kan_Knda',
            'gu': 'guj_Gujr',
            'pa': 'pan_Guru',
            'ur': 'urd_Arab',
            'fa': 'pes_Arab',
            'sw': 'swh_Latn',
            'am': 'amh_Ethi',
            'ig': 'ibo_Latn',
            'yo': 'yor_Latn',
            'ha': 'hau_Latn',
            'zu': 'zul_Latn',
            'af': 'afr_Latn',
            'eu': 'eus_Latn',
            'ca': 'cat_Latn',
            'gl': 'glg_Latn',
            'cy': 'cym_Latn',
            'ga': 'gle_Latn',
            'is': 'isl_Latn',
            'mt': 'mlt_Latn',
            'sq': 'als_Latn',
            'be': 'bel_Cyrl',
            'az': 'azj_Latn',
            'ka': 'kat_Geor',
            'hy': 'hye_Armn',
            'kk': 'kaz_Cyrl',
            'ky': 'kir_Cyrl',
            'uz': 'uzn_Latn',
            'tg': 'tgk_Cyrl',
            'mn': 'khk_Cyrl',
            'ne': 'npi_Deva',
            'si': 'sin_Sinh',
            'my': 'mya_Mymr',
            'km': 'khm_Khmr',
            'lo': 'lao_Laoo'
        }
        
        mapped = lang_mapping.get(lang_code)
        if mapped:
            return mapped
        else:
            # If no mapping found, try to construct one with Latin script as default
            print(f"Warning: No NLLB mapping found for '{lang_code}', using '{lang_code}_Latn' as fallback")
            return f"{lang_code}_Latn"

    def _num_tokens_consumed_from_request(self, request_json: dict) -> int:
        """Count the number of tokens in a chat completion request."""
        if self.backend != "openai":
            return 0
            
        num_tokens = 0
        messages = request_json.get("messages", [])
        
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        
        # Add completion tokens
        max_completion_tokens = request_json.get("max_completion_tokens", self.max_gen_tokens)
        num_tokens += max_completion_tokens
        
        return num_tokens

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
            
            # Set source language for tokenizer
            self.tokenizer.src_lang = self.source_lang_nllb
            
            # Tokenize input
            encoded_input = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Generate translation with forced target language
            target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb)
            translated_tokens = self.model.generate(
                **encoded_input, 
                forced_bos_token_id=target_token_id
            )
            
            # Decode and return translation
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
            
            # Set source language for tokenizer
            self.tokenizer.src_lang = self.source_lang_nllb
            target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb)
            
            translations = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                # Generate translations
                translated_tokens = self.model.generate(
                    **encoded_input, 
                    forced_bos_token_id=target_token_id
                )
                
                # Decode batch translations
                batch_translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translations.extend(batch_translations)
                
                # Clean up GPU memory if using CUDA
                if self.device != "cpu":
                    del encoded_input, translated_tokens
                    torch.cuda.empty_cache()
            
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
        
        elif self.backend == "google":
            if not self.client:
                raise RuntimeError("Google Translate client not initialized properly.")
            
            try:
                translations = []
                # Process texts in batches to avoid hitting API limits
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    # Google Translate can handle multiple texts in a single request
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
        else:
            # Fallback to base class implementation for unsupported backends
            return super().translate_batch(texts, batch_size)

    def retrieve_batch(self, batch_job_or_result, save_results=True):
        """
        Retrieve batch output content when the batch job is done.
        Handles both simple batch jobs and dataset batch jobs.
        
        Args:
            batch_job_or_result: Batch job object or result dict from translate_batch/translate_dataset
            save_results (bool): Whether to save results to file automatically (for dataset batches)
            
        Returns:
            File content, translations, or processing summary depending on input type
        """
        if self.backend != "openai":
            raise NotImplementedError("retrieve_batch is only available for OpenAI backend.")
        
        return retrieve_batch(self.client, batch_job_or_result, save_results, batch_type="translation")

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
                         batch_token_limit: int = 1900000) -> Dict[str, Any]:
        """
        Translate specified columns in a dataset with comprehensive error handling and progress tracking.
        
        Args:
            dataset: Dataset object to translate
            columns_to_translate (str or list): Column name(s) to translate
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
            
        Returns:
            dict: Summary statistics including success/error counts and output path
            For OpenAI batch mode: Returns batch job info that needs to be retrieved later with retrieve_batch()
        """
        # Ensure columns_to_translate is a list
        if isinstance(columns_to_translate, str):
            columns_to_translate = [columns_to_translate]
        elif columns_to_translate is None:
            # Auto-detect translatable columns
            if hasattr(dataset, 'columns') and dataset.columns:
                all_columns = list(dataset.columns)
                # Inline logic for finding translatable columns
                columns_to_translate = []
                sample_size = min(10, len(dataset))
                sample_data = list(dataset)[:sample_size]
                
                for column in all_columns:
                    is_translatable = False
                    for item in sample_data:
                        value = item.get(column) if isinstance(item, dict) else str(item)
                        if isinstance(value, str) and len(value.strip()) > 0:
                            if any(c.isalpha() for c in value):
                                is_translatable = True
                                break
                    if is_translatable:
                        columns_to_translate.append(column)
                
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
            batch_suffix = "_batch" if use_batch else ""
            filename = f"translated_{self.source_lang}_to_{self.target_lang}_{self.backend}{batch_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            output_path = os.path.join(output_dir, filename)
        else:
            # Ensure output directory exists only if there's actually a directory in the path
            output_dirname = os.path.dirname(output_path)
            if output_dirname:  # Only create directory if path contains a directory
                os.makedirs(output_dirname, exist_ok=True)
        
        total_samples = len(dataset)
        print(f"Starting {'batch ' if use_batch else ''}translation of {total_samples} samples using {self.backend} backend...")
        print(f"Translating columns: {columns_to_translate}")
        if use_batch:
            print(f"Batch size: {batch_size}")
            if self.backend == "openai":
                print(f"Request limit per batch: {batch_request_limit}")
                print(f"Token limit per batch: {batch_token_limit:,}")
        print(f"Output will be saved to: {output_path}")
        
        if use_batch:
            return self._translate_dataset_batch_mode(dataset, columns_to_translate, output_path, 
                                                    batch_size, batch_job_description, total_samples, batch_request_limit, batch_token_limit)
        else:
            return self._translate_dataset_sequential_mode(dataset, columns_to_translate, output_path,
                                                         progress_interval, save_errors, append_mode, total_samples)

    def _translate_dataset_batch_mode(self, dataset, columns_to_translate, output_path, 
                                    batch_size, batch_job_description, total_samples, batch_request_limit=40000, batch_token_limit=1900000):
        """Handle batch mode translation for datasets."""
        if self.backend == "openai":
            # OpenAI batch processing creates batch jobs for each column
            print("Creating OpenAI batch jobs for dataset translation...")
            print("Note: OpenAI batch processing creates async jobs. Use retrieve_batch() to get results when complete.")
            
            # Convert dataset to list for easier processing
            dataset_list = list(dataset)
            
            all_texts_with_metadata = []
            
            # Collect all texts from all columns with metadata and calculate tokens
            request_id = 0
            for column in columns_to_translate:
                print(f"Preparing batch requests for column: {column}")
                
                for i, item in enumerate(dataset_list):
                    text = item.get(column) if isinstance(item, dict) else str(item)
                    if text is not None and str(text).strip():
                        # Create request to calculate token consumption
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
                        
                        token_count = self._num_tokens_consumed_from_request(request_json)
                        
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
            
            # Determine if we need to split based on either limit
            needs_split = total_requests > batch_request_limit or total_tokens > batch_token_limit
            
            if not needs_split:
                # Single batch job
                print(f"Creating single batch job with {total_requests} requests and {total_tokens:,} tokens...")
                batch_job = self._create_single_batch_job_with_tokens(all_texts_with_metadata, batch_job_description, output_path, columns_to_translate, total_samples)
                return batch_job
            else:
                # Multiple batch jobs needed - split respecting both limits
                print(f"Splitting into multiple batches due to limits:")
                print(f"  Request limit: {batch_request_limit}")
                print(f"  Token limit: {batch_token_limit:,}")
                
                batches = self._split_requests_by_limits(all_texts_with_metadata, batch_request_limit, batch_token_limit)
                
                print(f"Created {len(batches)} batches")
                
                batch_jobs = []
                for i, batch_chunk in enumerate(batches):
                    chunk_tokens = sum(item["token_count"] for item in batch_chunk)
                    chunk_description = f"{batch_job_description} - batch {i+1}/{len(batches)}"
                    print(f"Creating batch {i+1}/{len(batches)} with {len(batch_chunk)} requests and {chunk_tokens:,} tokens...")
                    
                    # Modify output path for each batch
                    base_path, ext = os.path.splitext(output_path)
                    chunk_output_path = f"{base_path}_batch_{i+1}{ext}"
                    
                    batch_job = self._create_single_batch_job_with_tokens(batch_chunk, chunk_description, chunk_output_path, columns_to_translate, total_samples)
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
            # MarianMT, NLLB, and Google Translate batch processing - synchronous
            print(f"Using {self.backend} batch processing...")
            
            # Convert dataset to list for easier processing
            dataset_list = list(dataset)
            
            # Initialize result data structure
            translated_data = []
            for item in dataset_list:
                result_item = dict(item) if isinstance(item, dict) else {"original_data": item}
                translated_data.append(result_item)
            
            success_count = 0
            error_count = 0
            
            # Process each column
            for column in columns_to_translate:
                print(f"Processing column: {column}")
                
                # Extract all texts for this column
                texts_to_translate = []
                text_indices = []
                
                for i, item in enumerate(dataset_list):
                    text = item.get(column) if isinstance(item, dict) else str(item)
                    if text is not None and str(text).strip():
                        texts_to_translate.append(str(text))
                        text_indices.append(i)
                
                if not texts_to_translate:
                    print(f"No valid texts found for column '{column}', skipping.")
                    continue
                
                print(f"Translating {len(texts_to_translate)} texts for column '{column}'...")
                
                try:
                    # Perform batch translation
                    translated_texts = self.translate_batch(texts_to_translate, batch_size)
                    
                    # Add translations back to the dataset
                    translated_column_name = f"translated_{column}_{self.target_lang}"
                    for idx, translation in zip(text_indices, translated_texts):
                        translated_data[idx][translated_column_name] = translation
                        success_count += 1
                        
                    print(f"Successfully translated {len(translated_texts)} texts for column '{column}'")
                    
                except Exception as e:
                    print(f"Error translating column '{column}': {e}")
                    error_count += len(texts_to_translate)
                    
                    # Add None values for failed translations
                    translated_column_name = f"translated_{column}_{self.target_lang}"
                    for idx in text_indices:
                        translated_data[idx][translated_column_name] = None
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in translated_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Summary
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

    def _split_requests_by_limits(self, requests_with_metadata, request_limit, token_limit):
        """Split requests into batches respecting both request and token limits."""
        batches = []
        current_batch = []
        current_request_count = 0
        current_token_count = 0
        
        for request in requests_with_metadata:
            request_tokens = request["token_count"]
            
            # Check if adding this request would exceed either limit
            would_exceed_requests = current_request_count + 1 > request_limit
            would_exceed_tokens = current_token_count + request_tokens > token_limit
            
            if (would_exceed_requests or would_exceed_tokens) and current_batch:
                # Start a new batch
                batches.append(current_batch)
                current_batch = []
                current_request_count = 0
                current_token_count = 0
            
            # Add request to current batch
            current_batch.append(request)
            current_request_count += 1
            current_token_count += request_tokens
        
        # Add the last batch if it has any requests
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _create_single_batch_job_with_tokens(self, texts_with_metadata, batch_job_description, output_path, columns_to_translate, total_samples):
        """Create a single OpenAI batch job from a list of texts with metadata and token counts."""
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as batchfile:
                temp_file_path = batchfile.name
                for item in texts_with_metadata:
                    request = {
                        "custom_id": f"req-{item['request_id']}-{item['column']}-{item['sample_index']}",
                        "method": "POST", 
                        "url": "/v1/chat/completions",
                        "body": item["request_json"]
                    }
                    batchfile.write(json.dumps(request) + '\n')
                
                batchfile.flush()
            
            # Upload file to OpenAI
            with open(temp_file_path, "rb") as file_to_upload:
                batch_input_file = self.client.files.create(
                    file=file_to_upload,
                    purpose="batch"
                )

            # Create batch
            batch_input_file_id = batch_input_file.id
            total_tokens = sum(item["token_count"] for item in texts_with_metadata)
            created_batch = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"{batch_job_description} - {len(texts_with_metadata)} translations, ~{total_tokens:,} tokens"
                }
            )

            print(f"Batch job created successfully!")
            print(f"Batch ID: {created_batch.id}")
            print(f"Status: {created_batch.status}")
            print(f"Total requests: {len(texts_with_metadata)}")
            print(f"Estimated tokens: {total_tokens:,}")
            
            return {
                "batch_job": created_batch,
                "batch_id": created_batch.id,
                "total_requests": len(texts_with_metadata),
                "total_tokens": total_tokens,
                "columns_translated": columns_to_translate,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "backend": self.backend,
                "output_path": output_path,
                "dataset_size": total_samples,
                "status": "batch_submitted",
                "instructions": "Use retrieve_batch() with the batch_job to get results when complete"
            }
            
        except Exception as e:
            raise RuntimeError(f"OpenAI batch creation failed: {e}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    logging.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

    def _translate_dataset_sequential_mode(self, dataset, columns_to_translate, output_path,
                                         progress_interval, save_errors, append_mode, total_samples):
        """Handle sequential mode translation for datasets."""
        # Initialize counters
        success_count = 0
        error_count = 0
        
        # Open file for writing
        file_mode = 'a' if append_mode else 'w'
        with open(output_path, file_mode, encoding='utf-8') as f:
            # Create progress bar
            with tqdm(total=total_samples, desc="Translating", unit="samples") as pbar:
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
                    
                    # Update progress bar with current stats
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': success_count,
                        'Errors': error_count,
                        'Rate': f"{success_count/(success_count+error_count)*100:.1f}%" if (success_count + error_count) > 0 else "0.0%"
                    })
        
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