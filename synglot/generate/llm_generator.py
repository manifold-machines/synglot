import random
import os
import tempfile
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

from .base import Generator
from synglot.utils.config import Config
from synglot.utils import retrieve_batch, load_material_files, chunk_text, filter_generated_content

class LLMGenerator(Generator):
    """Unified generator supporting HuggingFace models and OpenAI API."""
    
    def __init__(self, target_lang, backend="huggingface", model_name=None, config=None, api_key=None, max_gen_tokens=1024):
        """
        Initialize unified LLM generator.
        
        Args:
            target_lang (str): Target language code
            backend (str): Backend system. Supports 'huggingface' or 'openai'.
            model_name (str, optional): Model name. For HF: model identifier; For OpenAI: model name (e.g., 'gpt-4o')
            config (Config | dict, optional): Configuration object or dictionary
            api_key (str, optional): API key for OpenAI backend
            max_gen_tokens (int): Maximum tokens for generation (used by OpenAI backend)
        """
        super().__init__(target_lang, config)
        self.backend = backend
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens
        
        if self.backend == "huggingface":
            self._init_huggingface()
        elif self.backend == "openai":
            self._init_openai(api_key)
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not supported. Currently supports 'huggingface' and 'openai'."
            )
    
    def _init_huggingface(self):
        """Initialize HuggingFace backend."""
        try:
            from transformers import pipeline, set_seed
        except ImportError:
            raise ImportError(
                "transformers library not found. Install it with: pip install transformers"
            )
        
        # Determine model name with priority: argument > config > default
        resolved_model_name = self.model_name
        
        if resolved_model_name is None:
            resolved_model_name = self.config.get("llm_generator.model_name")
        
        if resolved_model_name is None:
            # Default text generation model
            resolved_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            print(
                f"Warning: No model_name provided for LLMGenerator via argument or 'llm_generator.model_name' in config. "
                f"Falling back to default text-generation model '{resolved_model_name}'."
            )
        
        self.model_name = resolved_model_name
        
        try:
            self.generator_pipeline = pipeline("text-generation", model=self.model_name, tokenizer=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{self.model_name}': {e}")
        
        # Set seed from config if present
        seed = self.config.get("seed")
        if seed is not None:
            set_seed(seed)
    
    def _init_openai(self, api_key=None):
        """Initialize OpenAI backend."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai library not found. Install it with: pip install openai"
            )
        
        # Load environment variables
        load_dotenv()
        
        # Get API key with priority: parameter > config > environment
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = self.config.get("llm_generator.api_key")
        if resolved_api_key is None:
            resolved_api_key = os.getenv('OPENAI_API_KEY')
        
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI backend. Provide it via api_key parameter, config, or environment variable.")
        
        # Set default model name if not provided
        if self.model_name is None:
            self.model_name = self.config.get("llm_generator.model_name", "gpt-4.1-mini")
        
        try:
            self.client = openai.OpenAI(api_key=resolved_api_key)
            self.async_client = openai.AsyncOpenAI(api_key=resolved_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def generate(self, prompt=None, n_samples=1, **kwargs):
        """
        Generate samples using the configured backend.

        Args:
            prompt (str, optional): The prompt to generate from. Defaults to None (empty string).
            n_samples (int, optional): Number of samples to generate. Defaults to 1.
            **kwargs: Additional generation parameters to override config.

        Returns:
            list[str]: A list of generated text samples.
        """
        if self.backend == "huggingface":
            return self._generate_huggingface(prompt, n_samples, **kwargs)
        elif self.backend == "openai":
            return self._generate_openai(prompt, n_samples, **kwargs)
        else:
            raise NotImplementedError(f"Generation for backend '{self.backend}' is not implemented.")
    
    def _generate_huggingface(self, prompt, n_samples, **kwargs):
        """Generate using HuggingFace model."""
        prompt_text = prompt if prompt is not None else ""

        # Default generation parameters
        gen_params = {
            "num_return_sequences": n_samples,
            "temperature": self.config.get("generation_settings.default_temperature", 1.0),
            "top_k": self.config.get("generation_settings.default_top_k", 50),
            "top_p": self.config.get("generation_settings.default_top_p", 1.0),
            "do_sample": self.config.get("generation_settings.default_do_sample", True),
            "return_full_text": self.config.get("generation_settings.return_full_text", True),
            "pad_token_id": self.generator_pipeline.tokenizer.eos_token_id if self.generator_pipeline.tokenizer and hasattr(self.generator_pipeline.tokenizer, 'eos_token_id') else None
        }
        
        # Apply specific generation params from config
        hf_specific_params = self.config.get("llm_generator.generation_params", {})
        gen_params.update(hf_specific_params)
        
        # Handle length parameters with priority: kwargs > config > defaults
        default_max_new_tokens = self.config.get("generation_settings.default_max_new_tokens", 100)
        default_min_length_total = self.config.get("generation_settings.default_min_length_total", None)

        final_gen_params = {}
        if "max_new_tokens" in kwargs:
            final_gen_params["max_new_tokens"] = kwargs.pop("max_new_tokens")
        elif "max_length" in kwargs:
            final_gen_params["max_length"] = kwargs.pop("max_length")
        else:
            final_gen_params["max_new_tokens"] = default_max_new_tokens
        
        if "min_length" in kwargs:
            final_gen_params["min_length"] = kwargs.pop("min_length")
        elif default_min_length_total is not None:
            final_gen_params["min_length"] = default_min_length_total

        # Merge parameters: kwargs override config defaults
        gen_params.update(kwargs) 
        final_gen_params.update(gen_params)
        
        # Determine if prompt should be stripped based on final return_full_text setting
        should_return_full_text = final_gen_params.get("return_full_text")

        # Generate
        outputs_raw = self.generator_pipeline(prompt_text, **final_gen_params)
        
        # Post-process outputs
        generated_texts = []
        for output in outputs_raw:
            text = output['generated_text']
            if should_return_full_text and prompt_text and text.startswith(prompt_text):
                # Strip prompt if return_full_text is true and prompt was part of output
                generated_texts.append(text[len(prompt_text):].strip())
            elif not should_return_full_text and prompt_text and text.startswith(prompt_text):
                # If we don't want full text, ensure prompt is removed
                generated_texts.append(text[len(prompt_text):].strip())
            elif not should_return_full_text and not (prompt_text and text.startswith(prompt_text)):
                # if not returning full text and prompt wasn't prepended by pipeline (e.g. because prompt was empty)
                generated_texts.append(text.strip()) # just strip whitespace
            else: # return_full_text is False and prompt was not in output, or return_full_text is True and prompt was not in output
                generated_texts.append(text.strip())
        
        return generated_texts
    
    def _generate_openai(self, prompt, n_samples, **kwargs):
        """Generate using OpenAI API."""
        prompt_text = prompt if prompt is not None else ""
        
        # Default generation parameters
        temperature = kwargs.get("temperature", self.config.get("generation_settings.default_temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.max_gen_tokens)
        
        generated_texts = []
        
        try:
            for _ in range(n_samples):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt_text}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                generated_text = response.choices[0].message.content.strip()
                generated_texts.append(generated_text)
                
        except Exception as e:
            raise RuntimeError(f"OpenAI API error during generation: {e}")
        
        return generated_texts
    
    def generate_from_topics(self, topics=None, n_samples_per_topic=5, **kwargs):
        """
        Generate synthetic data from a list of topics.
        
        Args:
            topics (list, optional): List of topics. If None, generates topics automatically.
            n_samples_per_topic (int): Number of samples to generate per topic.
            **kwargs: Additional generation parameters.
            
        Returns:
            list[dict]: List of generated samples with metadata (topic, text, etc.).
        """
        # TODO: Implement topic-based generation
        # - Use provided topics or generate them with generate_topics()
        # - Create prompts for each topic using templates from config
        # - Generate samples for each topic
        # - Return structured data with topic metadata
        pass
    
    def generate_from_material(self, material_paths, chunk_size=None, overlap=0, n_samples_per_chunk=3, 
                              output_path=None, output_dir="./outputs", save_to_file=False, **kwargs):
        """
        Generate synthetic data based on provided material files.
        
        Args:
            material_paths (str | list): Path(s) to .txt or .md files containing source material.
            chunk_size (int, optional): Size of text chunks to process. If None, uses config default.
            overlap (int): Number of characters to overlap between chunks.
            n_samples_per_chunk (int): Number of samples to generate per chunk.
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None and save_to_file is True).
            save_to_file (bool): Whether to save results to file automatically.
            **kwargs: Additional generation parameters.
            
        Returns:
            list[dict]: List of generated samples with metadata (source_file, chunk_id, text, etc.).
        """
        # Load material files
        materials = load_material_files(material_paths)
        
        # Get chunk size from config if not provided
        if chunk_size is None:
            chunk_size = self.config.get("generation_settings.material_generation.default_chunk_size", 1000)
        
        generated_samples = []
        
        for material in materials:
            # Split material into chunks
            chunks = chunk_text(material["content"], chunk_size, overlap)
            
            print(f"Processing {len(chunks)} chunks from {material['file_name']}")
            
            for chunk in chunks:
                # Create context-aware prompt
                prompt_template = self.config.get(
                    "generation_settings.material_generation.chunk_prompt_template",
                    "Based on this text: '{chunk_text}', generate related content:"
                )
                
                prompt = prompt_template.format(chunk_text=chunk["text"][:200] + "...")
                
                # Generate samples for this chunk
                try:
                    generated_texts = self.generate(prompt, n_samples_per_chunk, **kwargs)
                    
                    for i, text in enumerate(generated_texts):
                        sample = {
                            "generated_text": text,
                            "source_file": material["file_path"],
                            "source_file_name": material["file_name"],
                            "chunk_id": chunk["chunk_id"],
                            "chunk_start": chunk["start_pos"],
                            "chunk_end": chunk["end_pos"],
                            "sample_index": i,
                            "prompt_used": prompt,
                            "generation_params": kwargs
                        }
                        generated_samples.append(sample)
                        
                except Exception as e:
                    print(f"Error generating from chunk {chunk['chunk_id']} in {material['file_name']}: {e}")
                    continue
        
        print(f"Generated {len(generated_samples)} samples from {len(materials)} material files")
        
        # Save to file if requested
        if save_to_file:
            if output_path is None:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"generated_from_material_{self.target_lang}_{self.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                output_path = os.path.join(output_dir, filename)
            else:
                # Ensure output directory exists only if there's actually a directory in the path
                output_dirname = os.path.dirname(output_path)
                if output_dirname:  # Only create directory if path contains a directory
                    os.makedirs(output_dirname, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in generated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"Results saved to: {output_path}")
        
        return generated_samples
    
    @classmethod
    def generate_topics(cls, domain="general", n_topics=20, topic_type="general", **kwargs):
        """
        Generate a list of topics for synthetic data generation.
        
        Args:
            domain (str): Domain for topic generation (e.g., "science", "technology", "general").
            n_topics (int): Number of topics to generate.
            topic_type (str): Type of topics ("general", "questions", "tasks", etc.).
            **kwargs: Additional parameters for topic generation.
            
        Returns:
            list[str]: List of generated topics.
        """
        # TODO: Implement topic generation
        # - Use predefined topic lists from config for different domains
        # - Or generate topics dynamically using LLM
        # - Support different topic types (general topics, questions, tasks, etc.)
        # - Return list of topics as strings
        pass
    
    def generate_pretraining(self, domain="general", n_samples=100, min_length=50, max_length=200,
                           output_path=None, output_dir="./outputs", save_to_file=False):
        """
        Generate pretraining data using diversity settings from configuration.
        
        Args:
            domain (str): The domain for generation. "general" uses topic list.
            n_samples (int): Number of samples to generate.
            min_length (int): Minimum length of generated text.
            max_length (int): Maximum length of generated text.
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None and save_to_file is True).
            save_to_file (bool): Whether to save results to file automatically.
            
        Returns:
            list[str]: A list of generated pretraining texts.
        """
        pretraining_data = []
        
        diversity_strategy = self.config.get("generation_settings.pretraining.diversity_strategy", "none")
        topics = self.config.get("generation_settings.pretraining.general_topics_list", [])
        prompt_template = self.config.get("generation_settings.pretraining.topic_prompt_template", "Write a short text about {topic}.")
        # Specific for pretraining: config drives whether prompt is included in output
        return_prompt = self.config.get("generation_settings.pretraining.return_prompt_in_output", False)

        for _ in range(n_samples):
            current_prompt = None
            if domain.lower() == "general" and diversity_strategy == "topic_prompt" and topics:
                selected_topic = random.choice(topics)
                current_prompt = prompt_template.format(topic=selected_topic)
            elif domain and domain.lower() != "general": # Specific domain given
                # Could also use a template for specific domains if desired
                current_prompt = prompt_template.format(topic=domain) 
            # Else, current_prompt remains None (unconditional generation or strategy is "none")

            # Generate with backend-appropriate parameters
            if self.backend == "huggingface":
                generated_texts = self.generate(
                    prompt=current_prompt, 
                    n_samples=1, 
                    min_length=min_length, 
                    max_length=max_length,
                    return_full_text=return_prompt # Use config setting for pretrain output
                )
            elif self.backend == "openai":
                # For OpenAI, we control length via max_tokens and handle prompt inclusion manually
                generated_texts = self.generate(
                    prompt=current_prompt, 
                    n_samples=1, 
                    max_tokens=max_length
                )
                # For OpenAI, manually handle prompt inclusion based on return_prompt setting
                if generated_texts and current_prompt and not return_prompt:
                    # OpenAI typically doesn't include the prompt in response, so this is usually not needed
                    # but we keep it for consistency
                    pass
            
            if generated_texts:
                # If current_prompt was used and return_prompt is False, already handled appropriately by backend
                # If return_prompt is True, it's included (for HF) or we add it manually (for OpenAI)
                text = generated_texts[0]
                if self.backend == "openai" and return_prompt and current_prompt:
                    text = f"{current_prompt} {text}"
                pretraining_data.append(text)
        
        # Save to file if requested
        if save_to_file:
            if output_path is None:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"generated_pretraining_{domain}_{self.target_lang}_{self.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                output_path = os.path.join(output_dir, filename)
            else:
                # Ensure output directory exists only if there's actually a directory in the path
                output_dirname = os.path.dirname(output_path)
                if output_dirname:  # Only create directory if path contains a directory
                    os.makedirs(output_dirname, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in pretraining_data:
                    # Save as simple text format for pretraining data
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            
            print(f"Pretraining data saved to: {output_path}")
        
        return pretraining_data
        
    def generate_conversations(self, domain="general", n_samples=50, n_turns_min=2, n_turns_max=5,
                              output_path=None, output_dir="./outputs", save_to_file=False):
        """
        Generate multi-turn conversation data.
        
        Args:
            domain (str): Domain for conversations.
            n_samples (int): Number of conversations to generate.
            n_turns_min (int): Minimum number of turns per conversation.
            n_turns_max (int): Maximum number of turns per conversation.
            output_path (str, optional): Full path for output file. If None, auto-generated.
            output_dir (str): Directory to save output (used if output_path is None and save_to_file is True).
            save_to_file (bool): Whether to save results to file automatically.
            
        Returns:
            list[list[str]]: List of conversations, each conversation is a list of turns.
        """
        conversations = []
        
        speaker_A = self.config.get("generation_settings.conversation.speaker_A", "User:")
        speaker_B = self.config.get("generation_settings.conversation.speaker_B", "Assistant:")
        turn_max_new_tokens = self.config.get("generation_settings.conversation.turn_max_new_tokens", 80)
        domain_template = self.config.get("generation_settings.conversation.domain_context_template", "This is a conversation about {domain}.")

        for _ in range(n_samples):
            current_conversation_turns = []
            full_conversation_history_prompt = ""

            num_turns = random.randint(n_turns_min, n_turns_max)

            initial_context = ""
            if domain and domain.lower() != "general":
                initial_context = domain_template.format(domain=domain) + "\n"
            
            full_conversation_history_prompt = initial_context

            for i in range(num_turns):
                current_speaker_prefix = speaker_A if i % 2 == 0 else speaker_B
                
                prompt_for_model_this_turn = full_conversation_history_prompt.strip()
                # Add current speaker only if history is not empty, or if it is the very first turn
                if full_conversation_history_prompt or i == 0:
                     prompt_for_model_this_turn += f"\n{current_speaker_prefix}"
                else: # Should not happen if logic is correct, but as a fallback
                    prompt_for_model_this_turn = f"{current_speaker_prefix}"

                # Generate with backend-appropriate parameters
                if self.backend == "huggingface":
                    generated_utterances = self.generate(
                        prompt=prompt_for_model_this_turn.strip(),
                        n_samples=1,
                        max_new_tokens=turn_max_new_tokens,
                        return_full_text=False # We don't want the full history + current prompt in the utterance
                    )
                elif self.backend == "openai":
                    generated_utterances = self.generate(
                        prompt=prompt_for_model_this_turn.strip(),
                        n_samples=1,
                        max_tokens=turn_max_new_tokens
                    )

                if not generated_utterances:
                    break 

                actual_utterance = generated_utterances[0].strip()
                
                # More robustly remove the *next* speaker prefix if model over-generates
                # (and also current speaker prefix if it was somehow re-added by model)
                next_speaker_prefix_to_check = speaker_B if i % 2 == 0 else speaker_A
                
                prefixes_to_strip = [next_speaker_prefix_to_check.strip(), current_speaker_prefix.strip()]
                for prefix in prefixes_to_strip:
                    if actual_utterance.startswith(prefix):
                        actual_utterance = actual_utterance[len(prefix):].lstrip(': ').strip()
                    # Check if it appears later (model might generate "Sure! Assistant: Yes I can.")
                    idx = actual_utterance.find(prefix)
                    if idx > 0 and actual_utterance[idx-1] in [' ', '\n']:
                         actual_utterance = actual_utterance[:idx].strip()
                
                if actual_utterance:
                    current_conversation_turns.append(f"{current_speaker_prefix} {actual_utterance}") # Store with prefix for context
                    full_conversation_history_prompt += f"\n{current_speaker_prefix} {actual_utterance}"
                else:
                    # If utterance became empty after stripping, stop this conversation turn
                    if self.config.get("generation_settings.conversation.ensure_alternating_speakers", True) and i > 0:
                        break # Avoid empty turns unless it's the very first one potentially
            
            if current_conversation_turns:
                # The turns already include speaker prefixes
                conversations.append([turn.strip() for turn in current_conversation_turns])
        
        # Save to file if requested
        if save_to_file:
            if output_path is None:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"generated_conversations_{domain}_{self.target_lang}_{self.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                output_path = os.path.join(output_dir, filename)
            else:
                # Ensure output directory exists only if there's actually a directory in the path
                output_dirname = os.path.dirname(output_path)
                if output_dirname:  # Only create directory if path contains a directory
                    os.makedirs(output_dirname, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for conversation in conversations:
                    # Save each conversation as a structured object
                    f.write(json.dumps({"conversation": conversation}, ensure_ascii=False) + '\n')
            
            print(f"Conversation data saved to: {output_path}")
        
        return conversations
    
    def generate_batch(self, prompts, batch_size=32, batch_job_description="batch generation"):
        """
        Generate samples for multiple prompts using batch processing.
        
        Args:
            prompts (list): List of prompts to generate from.
            batch_size (int): Batch size for processing.
            batch_job_description (str): Description for batch jobs (OpenAI).
            
        Returns:
            For HuggingFace: List of generated texts
            For OpenAI: Batch job object (use retrieve_batch to get results)
        """
        if self.backend == "huggingface":
            return self._generate_batch_huggingface(prompts, batch_size)
        elif self.backend == "openai":
            return self._generate_batch_openai(prompts, batch_job_description)
        else:
            raise NotImplementedError(f"Batch generation for backend '{self.backend}' is not implemented.")
    
    def _generate_batch_huggingface(self, prompts, batch_size):
        """Generate batch using HuggingFace model."""
        all_generated_texts = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Get generation parameters from config
            gen_params = {
                "num_return_sequences": 1,  # 1 per prompt in batch
                "temperature": self.config.get("generation_settings.default_temperature", 1.0),
                "top_k": self.config.get("generation_settings.default_top_k", 50),
                "top_p": self.config.get("generation_settings.default_top_p", 1.0),
                "do_sample": self.config.get("generation_settings.default_do_sample", True),
                "return_full_text": self.config.get("generation_settings.return_full_text", True),
                "max_new_tokens": self.config.get("generation_settings.default_max_new_tokens", 100),
                "pad_token_id": self.generator_pipeline.tokenizer.eos_token_id if self.generator_pipeline.tokenizer and hasattr(self.generator_pipeline.tokenizer, 'eos_token_id') else None
            }
            
            # Apply specific generation params from config
            hf_specific_params = self.config.get("llm_generator.generation_params", {})
            gen_params.update(hf_specific_params)
            
            # Generate for this batch
            batch_outputs = self.generator_pipeline(batch_prompts, **gen_params)
            
            # Process outputs
            for prompt, output in zip(batch_prompts, batch_outputs):
                text = output['generated_text']
                if gen_params.get("return_full_text", True) and prompt and text.startswith(prompt):
                    # Strip prompt if return_full_text is true and prompt was part of output
                    generated_text = text[len(prompt):].strip()
                else:
                    generated_text = text.strip()
                all_generated_texts.append(generated_text)
        
        return all_generated_texts
    
    def _generate_batch_openai(self, prompts, batch_job_description):
        """Generate batch using OpenAI API."""
        temp_file_path = None
        try:
            # Create temporary JSONL file with batch requests
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as batchfile:
                temp_file_path = batchfile.name
                for i, prompt in enumerate(prompts):
                    request = {
                        "custom_id": f"request-{i+1}",
                        "method": "POST", 
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model_name,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "max_completion_tokens": self.max_gen_tokens,
                            "temperature": self.config.get("generation_settings.default_temperature", 0.7)
                        }
                    }
                    batchfile.write(json.dumps(request) + '\n')
                
                batchfile.flush()
            
            # Upload file to OpenAI
            with open(temp_file_path, "rb") as file_to_upload:
                batch_input_file = self.client.files.create(
                    file=file_to_upload,
                    purpose="batch"
                )

            # Create batch job
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
            
        except Exception as e:
            raise RuntimeError(f"OpenAI batch creation failed: {e}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    logging.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
                    pass
    
    def retrieve_batch(self, saved_batch):
        """
        Retrieve batch generation results (OpenAI only).
        
        Args:
            saved_batch: Batch job object returned from generate_batch
            
        Returns:
            File content if completed, None if still in progress
        """
        if self.backend == "openai":
            return retrieve_batch(self.client, saved_batch, save_results=False, batch_type="generation")
        else:
            raise NotImplementedError("retrieve_batch is only available for OpenAI backend.")
    
    def _filter_generated_content(self, generated_texts, min_quality_score=0.5):
        """Filter generated content based on quality metrics."""
        return filter_generated_content(
            generated_texts, 
            min_quality_score=min_quality_score,
            min_length=self.config.get("generation_settings.filtering.min_length", 10),
            max_length=self.config.get("generation_settings.filtering.max_length", None),
            remove_duplicates=self.config.get("generation_settings.filtering.remove_duplicates", True),
            similarity_threshold=self.config.get("generation_settings.filtering.similarity_threshold", 0.9)
        ) 