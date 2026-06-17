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
Quantized dynamic cache implementation for Transformer key/value states.

This module defines cache layers used during autoregressive decoding to store,
quantize, dequantize, and update attention key/value tensors. It supports both
standard dynamic caching and sliding-window caching, with configurable key/value
quantization settings such as bit width, group size, quantization axis, symmetric,
asymmetric, and hybrid quantization modes.

The main classes are:
    - InnerqDynamicLayer:
        Maintains a dynamic key/value cache with optional quantization, recent
        unquantized token retention, sink token preservation, and optional key
        normalization.

    - InnerqDynamicSlidingWindowLayer:
        Extends the dynamic layer with sliding-window behavior for attention
        mechanisms that only attend to a fixed-size recent context.

    - DynamicCache:
        Builds a collection of cache layers from a Hugging Face model
        configuration and integrates them with the Transformers Cache interface.
"""

from collections.abc import Iterable
from typing import Any
import torch
from transformers.configuration_utils import PreTrainedConfig
from transformers.cache_utils import (
    CacheLayerMixin,
    Cache,
)
from .quant import (
    quantize_tokendim_asym,
    dequantize_channeldim_asym,
    dequantize_tokendim_asym,
    quantize_channeldim_asym,
    quantize_channeldim_sym,
    dequantize_channeldim_sym,
    dequantize_tokendim_sym,
    quantize_tokendim_sym,
    quantize_hybrid,
    dequantize_hybrid,
    QMode,
)

class InnerqDynamicLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, config, layer_idx):
        self.parse_config(config)
        self.layer_idx = layer_idx
        self.cumulative_length = 0
        super().__init__()

    def parse_config(self, config):
        self.key_bits = getattr(config, 'k_bits', 16)
        self.value_bits = getattr(config, 'v_bits', 16)
        self.key_quantize = False if self.key_bits >= 16 else True
        self.value_quantize = False if self.value_bits >= 16 else True

        self.key_group_size = getattr(config, 'key_group_size', 32)
        self.value_group_size = getattr(config, 'value_group_size', 32)
        self.key_w_recent = getattr(config, 'key_w_recent', 32)
        self.value_w_recent = getattr(config, 'value_w_recent', 32)
        self.key_w_sink = getattr(config, 'key_w_sink', 0)
        self.value_w_sink = getattr(config, 'value_w_sink', 0)
        self.key_flush_size = getattr(config, 'key_flush_size', 32)
        self.value_flush_size = getattr(config, 'value_flush_size', 32)
        self.key_normalize = getattr(config, 'key_normalize', False)
        self.key_normalization_factor = None

        self.key_quantize_over_token = getattr(config, 'key_quantize_over_token', False)
        self.value_quantize_over_token = getattr(config, 'value_quantize_over_token', True)

        self.key_quantize_mode = QMode.from_str(config.key_quantize_mode)
        self.value_quantize_mode = QMode.from_str(config.value_quantize_mode)

        if self.key_quantize_mode == QMode.HYBRID:
            key_asym_qfunc = quantize_tokendim_asym if self.key_quantize_over_token else quantize_channeldim_asym
            key_sym_qfunc = quantize_tokendim_sym if self.key_quantize_over_token else quantize_channeldim_sym
            key_asym_deqfunc = dequantize_tokendim_asym if self.key_quantize_over_token else dequantize_channeldim_asym
            key_sym_deqfunc = dequantize_tokendim_sym if self.key_quantize_over_token else dequantize_channeldim_sym
            self.key_q_func = quantize_hybrid(key_asym_qfunc, key_sym_qfunc, key_asym_deqfunc, key_sym_deqfunc)
            self.key_deq_func = dequantize_hybrid(key_asym_deqfunc, key_sym_deqfunc)
        elif self.key_quantize_mode == QMode.SYM:
            self.key_q_func = quantize_tokendim_sym if self.key_quantize_over_token else quantize_channeldim_sym
            self.key_deq_func = dequantize_tokendim_sym if self.key_quantize_over_token else dequantize_channeldim_sym
        elif self.key_quantize_mode == QMode.ASYM:
            self.key_q_func = quantize_tokendim_asym if self.key_quantize_over_token else quantize_channeldim_asym
            self.key_deq_func = dequantize_tokendim_asym if self.key_quantize_over_token else dequantize_channeldim_asym

        if self.value_quantize_mode == QMode.HYBRID:
            value_asym_qfunc = quantize_tokendim_asym if self.value_quantize_over_token else quantize_channeldim_asym
            value_sym_qfunc = quantize_tokendim_sym if self.value_quantize_over_token else quantize_channeldim_sym
            value_asym_deqfunc = dequantize_tokendim_asym if self.value_quantize_over_token else dequantize_channeldim_asym
            value_sym_deqfunc = dequantize_tokendim_sym if self.value_quantize_over_token else dequantize_channeldim_sym
            self.value_q_func = quantize_hybrid(value_asym_qfunc, value_sym_qfunc, value_asym_deqfunc, value_sym_deqfunc)
            self.value_deq_func = dequantize_hybrid(value_asym_deqfunc, value_sym_deqfunc)
        elif self.value_quantize_mode == QMode.SYM:
            self.value_q_func = quantize_tokendim_sym if self.value_quantize_over_token else quantize_channeldim_sym
            self.value_deq_func = dequantize_tokendim_sym if self.value_quantize_over_token else dequantize_channeldim_sym
        elif self.value_quantize_mode == QMode.ASYM:
            self.value_q_func = quantize_tokendim_asym if self.value_quantize_over_token else quantize_channeldim_asym
            self.value_deq_func = dequantize_tokendim_asym if self.value_quantize_over_token else dequantize_channeldim_asym

    def normalize_keys(self, key_states : torch.Tensor) -> torch.Tensor:
        if self.key_normalize:
            if self.key_normalization_factor == None:
                self.key_normalization_factor = key_states.abs().max(dim=-2, keepdim=True).values.sqrt()
            key_states = key_states / self.key_normalization_factor
        return key_states

    def unnormalize_keys(self, key_states : torch.Tensor) -> torch.Tensor:
        if self.key_normalize:
            if self.key_normalization_factor == None:
                raise Exception('key_normalize is set to True but key_normalization_factor is not set')
            key_states = key_states * self.key_normalization_factor
        return key_states

    def initialize_keys_sink(self, key_states: torch.Tensor) -> torch.Tensor:
        viable_w_sink_key = min(self.key_w_sink, key_states.shape[-2])
        if self.key_quantize:
            self.keys_sink = key_states[:,:,:viable_w_sink_key]
            key_states = key_states[:,:,viable_w_sink_key:]
        else:
            null_shape = torch.as_tensor(key_states.shape).tolist()
            null_shape[-2] = 0
            self.keys_sink = torch.zeros(size=(null_shape), dtype=self.dtype, device=self.device)
        return key_states

    def initialize_values_sink(self, value_states: torch.Tensor) -> torch.Tensor:
        viable_w_sink_value = min(self.value_w_sink, value_states.shape[-2])
        if self.value_quantize:
            self.values_sink = value_states[:,:,:viable_w_sink_value]
            value_states = value_states[:,:,viable_w_sink_value:]
        else:
            null_shape = torch.as_tensor(value_states.shape).tolist()
            null_shape[-2] = 0
            self.values_sink = torch.zeros(size=(null_shape), dtype=self.dtype, device=self.device)
        return value_states

    def initialize_keys_recent(self, key_states: torch.Tensor) -> torch.Tensor:
        if self.key_quantize:
            viable_recent_size_key = min(self.key_w_recent, key_states.shape[-2])
            if self.key_quantize_over_token:
                while (key_states.shape[-2] - viable_recent_size_key) % self.key_flush_size != 0:
                    viable_recent_size_key -= 1
        else:
            viable_recent_size_key = key_states.shape[-2]

        nb_quantized_tokens_key = key_states.shape[-2] - viable_recent_size_key
        self.keys_recent = key_states[:,:,nb_quantized_tokens_key:,:].contiguous()
        key_states = key_states[:,:,:nb_quantized_tokens_key,:].contiguous()

        return key_states

    def initialize_values_recent(self, value_states: torch.Tensor) -> torch.Tensor:
        if self.value_quantize:
            viable_recent_size_value = min(self.value_w_recent, value_states.shape[-2])
            if self.value_quantize_over_token:
                while (value_states.shape[-2] - viable_recent_size_value) % self.value_flush_size != 0:
                    viable_recent_size_value -= 1
        else:
            viable_recent_size_value = value_states.shape[-2]

        nb_quantized_tokens_value = value_states.shape[-2] - viable_recent_size_value
        self.values_recent = value_states[:,:,nb_quantized_tokens_value:,:].contiguous()
        value_states = value_states[:,:,:nb_quantized_tokens_value,:].contiguous()

        return value_states

    def initialize_keys_q(self, key_states : torch.Tensor) -> None:
        if self.key_quantize and key_states.numel() != 0:
            key_states = self.normalize_keys(key_states)
            self.keys_q, self.keys_scale_factor, self.keys_zp_sign = self.key_q_func(
                key_states,
                self.key_group_size,
                self.key_bits,
            )
        else:
            self.keys_q = None
            self.keys_scale_factor = None
            self.keys_zp_sign = None

    def initialize_values_q(self, value_states : torch.Tensor) -> None:
        if self.value_quantize and value_states.numel() != 0:
            self.values_q, self.values_scale_factor, self.values_zp_sign = self.value_q_func(
                value_states,
                self.value_group_size,
                self.value_bits,
            )
        else:
            self.values_q = None
            self.values_scale_factor = None
            self.values_zp_sign = None

    def initialize_sinks(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self.initialize_keys_sink(key_states)
        value_states = self.initialize_values_sink(value_states)
        return key_states, value_states

    def initialize_recents(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self.initialize_keys_recent(key_states)
        value_states = self.initialize_values_recent(value_states)
        return key_states, value_states

    def initial_quantization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.initialize_keys_q(key_states)
        self.initialize_values_q(value_states)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device

        key_states, value_states = self.initialize_sinks(key_states, value_states)
        key_states, value_states = self.initialize_recents(key_states, value_states)
        self.initial_quantization(key_states, value_states)

        self.is_initialized = True

    def get_keys_deq(self) -> torch.Tensor:
        keys_deq = torch.tensor([], dtype=self.dtype, device=self.device)

        if self.keys_sink.numel() != 0:
            keys_deq = torch.cat([keys_deq, self.keys_sink],dim=-2)

        if self.keys_q != None:
            tmp = self.key_deq_func(
                self.keys_q,
                self.keys_scale_factor,
                self.keys_zp_sign,
                self.key_group_size,
                op_dtype = torch.float32
            )
            tmp = self.unnormalize_keys(tmp)
            keys_deq = torch.cat([keys_deq, tmp],dim=-2)

        if self.keys_recent.numel() != 0:
            keys_deq = torch.cat([keys_deq, self.keys_recent],dim=-2)

        return keys_deq

    def get_values_deq(self) -> torch.Tensor:
        values_deq = torch.tensor([], dtype=self.dtype, device=self.device)

        if self.values_sink.numel() != 0:
            values_deq = torch.cat([values_deq, self.values_sink],dim=-2)

        if self.values_q != None:
            tmp = self.value_deq_func(
                self.values_q,
                self.values_scale_factor,
                self.values_zp_sign,
                self.value_group_size,
                op_dtype = torch.float32
            )
            values_deq = torch.cat([values_deq, tmp],dim=-2)

        if self.values_recent.numel() != 0:
            values_deq = torch.cat([values_deq, self.values_recent],dim=-2)

        return values_deq

    def get_deq_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys_deq = self.get_keys_deq().to(self.dtype)
        values_deq = self.get_values_deq().to(self.dtype)
        return keys_deq, values_deq

    def update_keys_recent(self, key_states : torch.Tensor) -> torch.Tensor:
        self.keys_recent = torch.cat([self.keys_recent, key_states],dim=-2)

        if self.key_quantize and self.keys_recent.shape[-2] >= self.key_flush_size:
            viable_key_flush_size = self.key_flush_size
        else: # otherwise, do not flush
            viable_key_flush_size = 0

        key_states = self.keys_recent[:,:,:viable_key_flush_size]
        self.keys_recent = self.keys_recent[:,:,viable_key_flush_size:]

        return key_states

    def update_values_recent(self, value_states : torch.Tensor) -> torch.Tensor:
        self.values_recent = torch.cat([self.values_recent, value_states],dim=-2)

        if self.value_quantize and self.values_recent.shape[-2] >= self.value_flush_size:
            viable_value_flush_size = self.value_flush_size
        else: # otherwise, do not flush
            viable_value_flush_size = 0

        value_states = self.values_recent[:,:,:viable_value_flush_size]
        self.values_recent = self.values_recent[:,:,viable_value_flush_size:]

        return value_states

    def update_recents(self, key_states : torch.Tensor, value_states : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self.update_keys_recent(key_states)
        value_states = self.update_values_recent(value_states)
        return key_states, value_states

    def quantize_new_keys(self, key_states : torch.Tensor) -> None:
        key_states = self.normalize_keys(key_states)
        new_keys_q, new_keys_scale_factor, new_keys_zp_sign = self.key_q_func(
            key_states,
            self.key_group_size,
            self.key_bits,
        )
        if self.key_quantize_mode == QMode.HYBRID:
            new_mask = new_keys_scale_factor['mask']
            self.keys_q = {
                'zp':torch.cat([self.keys_q['zp'], new_keys_q['zp']], dim=2),
                'nozp':torch.cat([self.keys_q['nozp'], new_keys_q['nozp']], dim=2)
            }
            self.keys_scale_factor = {
                'zp':torch.cat([self.keys_scale_factor['zp'], new_keys_scale_factor['zp']], dim=2),
                'nozp':torch.cat([self.keys_scale_factor['nozp'], new_keys_scale_factor['nozp']], dim=2),
                'mask':torch.cat([self.keys_scale_factor['mask'], new_mask], dim=2),
            }
            self.keys_zp_sign = {
                'zp':torch.cat([self.keys_zp_sign['zp'], new_keys_zp_sign['zp']], dim=2),
                'nozp':torch.cat([self.keys_zp_sign['nozp'], new_keys_zp_sign['nozp']], dim=2),
            }
        else:
            self.keys_q = torch.cat([self.keys_q, new_keys_q], dim=2)
            self.keys_scale_factor = torch.cat([self.keys_scale_factor, new_keys_scale_factor], dim=2)
            self.keys_zp_sign = torch.cat([self.keys_zp_sign, new_keys_zp_sign], dim=2)

    def quantize_new_values(self, value_states : torch.Tensor) -> None:
        new_values_q, new_values_scale_factor, new_values_zp_sign = self.value_q_func(
            value_states,
            self.value_group_size,
            self.value_bits,
        )
        if self.value_quantize_mode == QMode.HYBRID:
            new_mask = new_values_scale_factor['mask']
            self.values_q = {
                'zp':torch.cat([self.values_q['zp'], new_values_q['zp']], dim=2),
                'nozp':torch.cat([self.values_q['nozp'], new_values_q['nozp']], dim=2)
            }
            self.values_scale_factor = {
                'zp':torch.cat([self.values_scale_factor['zp'], new_values_scale_factor['zp']], dim=2),
                'nozp':torch.cat([self.values_scale_factor['nozp'], new_values_scale_factor['nozp']], dim=2),
                'mask':torch.cat([self.values_scale_factor['mask'], new_mask], dim=2),
            }
            self.values_zp_sign = {
                'zp':torch.cat([self.values_zp_sign['zp'], new_values_zp_sign['zp']], dim=2),
                'nozp':torch.cat([self.values_zp_sign['nozp'], new_values_zp_sign['nozp']], dim=2),
            }
        else:
            self.values_q = torch.cat([self.values_q, new_values_q], dim=2)
            self.values_scale_factor = torch.cat([self.values_scale_factor, new_values_scale_factor], dim=2)
            self.values_zp_sign = torch.cat([self.values_zp_sign, new_values_zp_sign], dim=2)

    def add_to_cache(self, key_states : torch.Tensor, value_states : torch.Tensor) -> None:
        key_states, value_states = self.update_recents(key_states, value_states)

        if key_states.numel() != 0: # if there is something to be quantized
            if self.keys_q == None: # and there is nothing that is already quantized
                self.initialize_keys_q(key_states)
            else:
                self.quantize_new_keys(key_states)

        if value_states.numel() != 0: # if there is something to be quantized
            if self.values_q == None: # and there is nothing that is already quantized
                self.initialize_values_q(value_states)
            else:
                self.quantize_new_values(value_states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.cumulative_length += key_states.shape[-2]

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            return key_states, value_states
        else:
            self.add_to_cache(key_states, value_states)
            keys_deq, values_deq = self.get_deq_tensors()
            return keys_deq, values_deq

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        raise ValueError('This function is not implementable for quantized cache. It leads to the loss of accuracy on the recent tokens')

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        raise Exception('This function is not implemented yet.')

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        raise Exception('This function is not implemented yet.')

class InnerqDynamicSlidingWindowLayer(InnerqDynamicLayer):
    is_sliding = True

    def __init__(self, config, layer_idx: int, sliding_window: int):
        super().__init__(config, layer_idx)
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_key_states, full_value_states = super().update(key_states=key_states, value_states=value_states, cache_kwargs=cache_kwargs)

        if self.cumulative_length - key_states.shape[-2] > self.sliding_window:
            full_key_states = full_key_states[:,:,-self.sliding_window,:]
            full_value_states = full_value_states[:,:,-self.sliding_window,:]

        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.sliding_window

class DynamicCache(Cache):
    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers = []
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_idx, layer_type in enumerate(layer_types):
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(InnerqDynamicSlidingWindowLayer(config=config, layer_idx=layer_idx, sliding_window=sliding_window))
                else:
                    layers.append(InnerqDynamicLayer(config=config, layer_idx=layer_idx))

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)

