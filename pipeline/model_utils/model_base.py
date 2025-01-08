from abc import ABC, abstractmethod
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float
import torch

from src.refusal_direction.pipeline.utils.hook_utils import add_hooks


class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path.lower()
        self.model: PreTrainedModel = self._load_model(model_name_or_path)
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

        self._post_init_validations()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    def _post_init_validations(self):
        assert self.model.dtype in [torch.float16, torch.bfloat16], f"Model dtype is (probably) too big: {self.model.dtype}"
        assert self.tokenizer.chat_template is not None, "Tokenizer does not have a chat template"

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions

    def generate_batch(
            self,
            messages: List[str],
            # prefix_fillers: Optional[List[str]] = None,  # TODO add prefix-force
            fwd_pre_hooks=[], fwd_hooks=[],
            batch_size=8, max_new_tokens=512):  # TODO what limits the max_new_toks here here?
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)  # greedy sampling
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        # outputs = prefix_fillers or [None] * len(messages)  # TODO add prefix-force

        for i in tqdm(range(0, len(messages), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=messages[i:i + batch_size],
                                                                   # outputs=None)  # TODO add prefix-force
                                                                   )

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'message': messages[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions

    def get_residuals(self, messages):
        # TODO convert to transformer-lens
        inputs = self.tokenize_instructions_fn(messages)

        hs = self.model(**inputs, output_hidden_states=True)['hidden_states']
        hs = torch.stack([h[0, :] for h in hs])
        hs = hs.detach().cpu()
        return hs  # n_layers, hidden_dim
