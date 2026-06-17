from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.utils import auto_docstring
from ..cache import DynamicCache

@auto_docstring
class InnerqLlamaForCausalLM(LlamaForCausalLM):
    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
    ):
        dynamic_cache_kwargs = {
            'config':self.config.get_text_config(decoder=True)
        }
        model_kwargs["past_key_values"] = DynamicCache(**dynamic_cache_kwargs)
