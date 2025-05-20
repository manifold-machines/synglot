import random
from transformers import pipeline, set_seed
from .base import Generator
from synglot.utils.config import Config # Ensure Config is imported

class HFGenerator(Generator):
    """Generator using HuggingFace models."""
    
    def __init__(self, target_lang, model_name=None, config=None):
        """
        Initialize HuggingFace generator.
        
        Args:
            target_lang (str): Target language code
            model_name (str, optional): HuggingFace model name. Can also be set in config.
            config (Config | dict, optional): Configuration object or dictionary.
        """
        super().__init__(target_lang, config)
        
        # Determine the model name with a clear priority:
        # 1. Direct argument `model_name`
        # 2. `hf_generator.model_name` from config
        # 3. Hardcoded default for this text generator (e.g., "gpt2")

        resolved_model_name = model_name  # Priority 1: argument from __init__
        
        if resolved_model_name is None:
            # Priority 2: model name from config specific to hf_generator
            resolved_model_name = self.config.get("hf_generator.model_name")

        if resolved_model_name is None:
            # Priority 3: Fallback to a generic text generation model if not specified anywhere
            default_text_gen_model = "Qwen/Qwen2.5-1.5B-Instruct"
            resolved_model_name = default_text_gen_model
            print(
                f"Warning: No model_name provided for HFGenerator via argument or 'hf_generator.model_name' in config. "
                f"Falling back to default text-generation model '{resolved_model_name}'."
            )
        
        self.model_name = resolved_model_name

        if not self.model_name: # Final check, though unlikely if default_text_gen_model is set
            raise ValueError("model_name must be specified for HFGenerator, either as an argument "
                             "or in config as 'hf_generator.model_name'.")

        try:
            self.generator_pipeline = pipeline("text-generation", model=self.model_name, tokenizer=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{self.model_name}': {e}")

        # Set seed from config if present
        seed = self.config.get("seed")
        if seed is not None:
            set_seed(seed)
            
    def generate(self, prompt=None, n_samples=1, **kwargs):
        """
        Generate samples using HuggingFace model.

        Args:
            prompt (str, optional): The prompt to generate from. Defaults to None (empty string).
            n_samples (int, optional): Number of samples to generate. Defaults to 1.
            **kwargs: Additional generation parameters to override config or pass to the pipeline.
                      (e.g., max_new_tokens, max_length, min_length, temperature, return_full_text).

        Returns:
            list[str]: A list of generated text samples, with prompt text stripped if return_full_text is True and prompt is present.
        """
        prompt_text = prompt if prompt is not None else ""

        gen_params = {
            "num_return_sequences": n_samples,
            "temperature": self.config.get("generation_settings.default_temperature", 1.0),
            "top_k": self.config.get("generation_settings.default_top_k", 50),
            "top_p": self.config.get("generation_settings.default_top_p", 1.0),
            "do_sample": self.config.get("generation_settings.default_do_sample", True),
            "return_full_text": self.config.get("generation_settings.return_full_text", True),
            "pad_token_id": self.generator_pipeline.tokenizer.eos_token_id if self.generator_pipeline.tokenizer and hasattr(self.generator_pipeline.tokenizer, 'eos_token_id') else None
        }
        
        # Apply specific generation_params from config if they exist
        # E.g. self.config can have a "hf_generator.generation_params" dictionary
        hf_specific_params = self.config.get(f"hf_generator.generation_params", {})
        gen_params.update(hf_specific_params)

        # Length parameters: prioritize kwargs, then config, then defaults
        default_max_new_tokens = self.config.get("generation_settings.default_max_new_tokens", 100)
        # min_length in HF pipeline is total length, not new tokens
        default_min_length_total = self.config.get("generation_settings.default_min_length_total", None) 

        final_gen_params = {}
        if "max_new_tokens" in kwargs:
            final_gen_params["max_new_tokens"] = kwargs.pop("max_new_tokens")
        elif "max_length" in kwargs: # Max total length
            final_gen_params["max_length"] = kwargs.pop("max_length")
        else:
            final_gen_params["max_new_tokens"] = default_max_new_tokens
        
        if "min_length" in kwargs: # Min total length
            final_gen_params["min_length"] = kwargs.pop("min_length")
        elif default_min_length_total is not None:
            final_gen_params["min_length"] = default_min_length_total

        # Merge general pipeline params and explicit kwargs
        # Kwargs override anything previously set from config
        gen_params.update(kwargs) 
        final_gen_params.update(gen_params)
        
        # Determine if prompt should be stripped based on final return_full_text setting
        # The generate method's kwargs can override the config's return_full_text
        should_return_full_text = final_gen_params.get("return_full_text")

        outputs_raw = self.generator_pipeline(prompt_text, **final_gen_params)
        
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
        
    def generate_pretraining(self, domain="general", n_samples=100, 
                            min_length=50, max_length=200):
        """
        Generate pretraining data.
        Uses diversity settings from the configuration.
        Args:
            domain (str): The domain for generation. "general" uses topic list.
            n_samples (int): Number of samples to generate.
            min_length (int): Minimum length of generated text (total length).
            max_length (int): Maximum length of generated text (total length).
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

            generated_texts = self.generate(
                prompt=current_prompt, 
                n_samples=1, 
                min_length=min_length, 
                max_length=max_length,
                return_full_text=return_prompt # Use config setting for pretrain output
            )
            if generated_texts:
                # If current_prompt was used and return_prompt is False, self.generate already stripped it.
                # If return_prompt is True, it's included.
                pretraining_data.append(generated_texts[0])
        return pretraining_data
        
    def generate_conversations(self, domain="general", n_samples=50,
                              n_turns_min=2, n_turns_max=5):
        """Generate multi-turn conversation data using settings from config."""
        conversations = []
        
        speaker_A = self.config.get("generation_settings.conversation.speaker_A", "User:")
        speaker_B = self.config.get("generation_settings.conversation.speaker_B", "Assistant:")
        turn_max_new_tokens = self.config.get("generation_settings.conversation.turn_max_new_tokens", 80)
        domain_template = self.config.get("generation_settings.conversation.domain_context_template", "This is a conversation about {domain}.")
        # return_full_text for individual turns in conversation is typically False, we build history manually
        # The main self.generate will handle stripping if its own return_full_text is True (default)
        # so we let it be for now, or explicitly pass False if needed for conversation turns.

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

                generated_utterances = self.generate(
                    prompt=prompt_for_model_this_turn.strip(),
                    n_samples=1,
                    max_new_tokens=turn_max_new_tokens,
                    return_full_text=False # We don't want the full history + current prompt in the utterance
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
        return conversations 