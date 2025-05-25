import openai
import json
import tempfile
import os
import logging
from .base import Translator

class LLMTranslator(Translator):
    """Translator using commercial LLM APIs."""
    
    def __init__(self, source_lang, target_lang, provider="openai", 
                 model_name=None, api_key=None):
        """
        Initialize LLM API translator.
        
        Args:
            source_lang (str): Source language code
            target_lang (str): Target language code
            provider (str): API provider (openai, anthropic, gemini)
            model_name (str): Model name
            api_key (str): API key
        """
        super().__init__(source_lang, target_lang)
        self.provider = provider
        self.model_name = model_name if model_name else "gpt-4.1-mini"
        self.api_key = api_key

        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider.")
            self.client = openai.OpenAI(api_key=self.api_key)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        # TODO: Initialize API client based on other providers
        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not supported yet.")

    def translate(self, text):
        """Translate using LLM API."""
        if self.provider == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"You are a translator. Translate the following text from {self.source_lang} to {self.target_lang}."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=1024, # Adjust as needed
                    temperature=0.4  # Adjust for creativity vs. precision
                )
                translated_text = response.choices[0].message.content.strip()
                return translated_text
            except openai.APIError as e:
                # Handle API errors (e.g., rate limits, server errors)
                raise RuntimeError(f"OpenAI API error: {e}")
            except Exception as e:
                # Handle other potential errors
                raise RuntimeError(f"An unexpected error occurred during translation: {e}")
        # TODO: Implement translation logic for other specific providers
        raise NotImplementedError(f"LLM API translation for provider '{self.provider}' not yet implemented.")


    def translate_batch(self, text_array, batch_job_description="batch job"):
        """Translate in batches using an API."""
        if self.provider == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not initialized properly.")
            
            temp_file_path = None
            try:
                # Create a temporary file to store the batch requests in the required format
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as batchfile:
                    temp_file_path = batchfile.name
                    for i, text in enumerate(text_array):
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
                                "max_tokens": 1024,
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
            raise NotImplementedError(f"Batch translation with other API providers than OpenAI is not yet implemented.")
        
    def retrieve_batch(self, saved_batch):
        """Retrieve the batch output content when the job is done."""
        if self.provider == "openai":
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
            raise NotImplementedError(f"Batch cannot be retrieved, as batch processing is not available with other API providers than OpenAI.")