# coding=utf-8
# Copyright 2026 Sayed Mohammadreza Tayaranian Hosseini. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Custom Hugging Face LM wrapper that integrates InnerQ key/value cache quantization
into the lm-eval framework. This class extends HFLM to enable configurable low-bit
KV-cache compression during evaluation for supported architectures (e.g., LLaMA, Mistral).

Requires applying the patches provided in the "lm_eval_changes" directory to the
lm-eval framework before use.
"""

from lm_eval.models.huggingface import HFLM
import torch
from transformers import AutoConfig

class HFLMINNERQ(HFLM):
    def __init__(
            self,
            k_bits=2,
            v_bits=2,
            key_group_size=32,
            value_group_size=32,
            key_w_recent=32,
            value_w_recent=32,
            key_w_sink=0,
            value_w_sink=0,
            key_flush_size=32,
            value_flush_size=32,
            key_normalize=False,
            key_quantize_over_token=False,
            value_quantize_over_token=True,
            key_quantize_mode = 'symmetric',
            value_quantize_mode = 'symmetric',
            *args, **kwargs
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.key_group_size = key_group_size
        self.value_group_size = value_group_size
        self.key_w_recent = key_w_recent
        self.value_w_recent = value_w_recent
        self.value_w_sink = value_w_sink
        self.key_w_sink = key_w_sink
        self.key_flush_size = key_flush_size
        self.value_flush_size = value_flush_size
        self.key_normalize = key_normalize
        self.key_quantize_over_token = key_quantize_over_token
        self.value_quantize_over_token = value_quantize_over_token
        self.key_quantize_mode = key_quantize_mode
        self.value_quantize_mode = value_quantize_mode
        super().__init__(*args, **kwargs)

    def _create_model(self,*args, **kwargs) -> None:
        pretrained = kwargs['pretrained']
        revision = kwargs.get('revision','main')
        parallelize = kwargs.get('parallelize',False)
        max_memory_per_gpu = kwargs.get('max_memory_per_gpu', None)
        max_cpu_memory = kwargs.get('max_cpu_memory', None)
        offload_folder = kwargs.get('offload_folder', None)
        gpus = kwargs.get('gpus', None)
        gguf_file = kwargs.get('gguf_file', None)
        quantization_config = kwargs.get('quantization_config', None)
        subfolder = kwargs.get('subfolder', None)
        dtype = torch.float16

        model_type = AutoConfig.from_pretrained(pretrained).model_type
        if model_type == 'llama':
            from innerq.models.llama_innerq import InnerqLlamaForCausalLM
            model_class = InnerqLlamaForCausalLM
        elif model_type == 'mistral':
            from innerq.models.mistral_innerq import InnerqMistralForCausalLM
            model_class = InnerqMistralForCausalLM
        else:
            raise NotImplementedError(f"Quantized cache implementation for {pretrained} is not supported yet.")

        model_kwargs = {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self.config.k_bits = self.k_bits
        self.config.v_bits = self.v_bits
        self.config.key_group_size = self.key_group_size
        self.config.value_group_size = self.value_group_size
        self.config.key_w_recent = self.key_w_recent
        self.config.value_w_recent = self.value_w_recent
        self.config.value_w_sink = self.value_w_sink
        self.config.key_w_sink = self.key_w_sink
        self.config.key_flush_size = self.key_flush_size
        self.config.value_flush_size = self.value_flush_size
        self.config.key_normalize = self.key_normalize
        self.config.key_quantize_over_token = self.key_quantize_over_token
        self.config.value_quantize_over_token = self.value_quantize_over_token
        self.config.key_quantize_mode = self.key_quantize_mode
        self.config.value_quantize_mode = self.value_quantize_mode

        self._model = model_class.from_pretrained(
            pretrained_model_name_or_path=pretrained,
            config=self.config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs,
        )

        return None

