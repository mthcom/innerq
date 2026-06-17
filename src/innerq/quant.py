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
Tensor quantization utilities for Transformer key/value cache compression.

This module provides low-level quantization and dequantization functions used to
compress attention key/value tensors along either the channel dimension or the
token dimension. It supports symmetric, asymmetric, and hybrid quantization modes,
with configurable group size and bit width.

The main components are:
    - QMode:
        Enum defining the supported quantization modes: symmetric, asymmetric,
        and hybrid.

    - Symmetric quantization:
        Stores magnitudes together with sign information, using a shared scale
        per quantization group.

    - Asymmetric quantization:
        Stores quantized values with per-group scale and zero-point offsets,
        allowing better handling of non-zero-centered tensor distributions.

    - Hybrid quantization:
        Computes both symmetric and asymmetric quantizations for each group and
        selects the lower-error representation during dequantization.
"""
import torch
from enum import Enum

class QMode(Enum):
    SYM = 'symmetric'
    ASYM = 'asymmetric'
    HYBRID = 'hybrid'

    @classmethod
    def from_str(cls, value: str):
        normalized = value.strip().lower()

        for member in cls:
            if member.value == normalized:
                return member

        raise ValueError('Value of Enum should be form the choices \'symmetric\', \'asymmetric\', and \'hybrid\'.')

def quantize_asym(tensor, group_size, bit, expanded_dim):
    expanded_shape = tensor.shape[expanded_dim]
    shape = tensor.shape
    assert expanded_shape % group_size == 0
    num_groups = expanded_shape // group_size
    new_shape = shape[:expanded_dim]+(num_groups, group_size)+shape[expanded_dim+1:]
    scale_mn_shape = shape[:expanded_dim]+(num_groups,)+shape[expanded_dim+1:]
    tensor = tensor.reshape(new_shape)
    mn = torch.min(tensor, dim=expanded_dim+1, keepdim=True)[0]
    mx = torch.max(tensor, dim=expanded_dim+1, keepdim=True)[0]
    scale = (mx - mn) / (2 ** bit - 1)
    tensor = tensor - mn
    tensor.div_(scale)
    tensor = tensor.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
    tensor = tensor.reshape(shape)
    return tensor, scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def quantize_channeldim_asym(tensor, group_size, bit):
    expanded_dim = tensor.dim()-1
    return quantize_asym(tensor, group_size, bit, expanded_dim)

def quantize_tokendim_asym(tensor, group_size, bit):
    expanded_dim = tensor.dim()-2
    return quantize_asym(tensor, group_size, bit, expanded_dim)

def dequantize_channeldim_asym(tensor, scale, zero_points, group_size, op_dtype = torch.float32):
    result = tensor.view(tensor.shape[:-1]+(scale.shape[-1],group_size)).to(op_dtype) * scale.unsqueeze(-1).to(op_dtype)
    result = result + zero_points.unsqueeze(-1).to(op_dtype)
    result = result.view(tensor.shape)
    return result

def dequantize_tokendim_asym(tensor, scale, zero_points, group_size, op_dtype = torch.float32):
    result = tensor.view(tensor.shape[:-2]+(scale.shape[-2],group_size, tensor.shape[-1])).to(op_dtype) * scale.unsqueeze(-2).to(op_dtype)
    result = result + zero_points.unsqueeze(-2).to(op_dtype)
    result = result.view(tensor.shape)
    return result

def quantize_sym(tensor, group_size, bit, expanded_dim):
    bit = bit-1 # NOTE: one bit is used up by the sign
    expanded_shape = tensor.shape[expanded_dim]
    shape = tensor.shape
    assert expanded_shape % group_size == 0
    num_groups = expanded_shape // group_size
    new_shape = shape[:expanded_dim]+(num_groups, group_size)+shape[expanded_dim+1:]
    scale_mn_shape = shape[:expanded_dim]+(num_groups,)+shape[expanded_dim+1:]
    signs = tensor.sign()
    tensor = tensor.reshape(new_shape)
    tensor = tensor.abs()
    mx = torch.max(tensor, dim=expanded_dim+1, keepdim=True)[0]
    scale = mx / (2 ** bit - 1)
    tensor.div_(scale)
    tensor = tensor.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
    tensor = tensor.reshape(shape)
    return tensor, scale.reshape(scale_mn_shape), signs

def quantize_channeldim_sym(tensor, group_size, bit):
    expanded_dim = tensor.dim()-1
    return quantize_sym(tensor, group_size, bit, expanded_dim)

def quantize_tokendim_sym(tensor, group_size, bit):
    expanded_dim = tensor.dim()-2
    return quantize_sym(tensor, group_size, bit, expanded_dim)

def dequantize_channeldim_sym(tensor, scale, signs, group_size, op_dtype = torch.float32):
    result = tensor.view(tensor.shape[:-1]+(scale.shape[-1],group_size)).to(op_dtype) * scale.unsqueeze(-1).to(op_dtype)
    result = result.view(tensor.shape)
    result = result * signs
    return result

def dequantize_tokendim_sym(tensor, scale, signs, group_size, op_dtype = torch.float32):
    result = tensor.view(tensor.shape[:-2]+(scale.shape[-2],group_size, tensor.shape[-1])).to(op_dtype) * scale.unsqueeze(-2).to(op_dtype)
    result = result.view(tensor.shape)
    result = result * signs
    return result

def quantize_hybrid(zp_func, nozp_func, zp_deq_func, nozp_deq_func):
    def hybrid_func(tensor, group_size, bit):
        tensor_q_zp = zp_func(tensor, group_size, bit-1) # NOTE:the asym function has no sign bit
        tensor_q_nozp = nozp_func(tensor, group_size, bit)
        tensor_deq_zp = zp_deq_func(*tensor_q_zp, group_size)
        tensor_deq_nozp = nozp_deq_func(*tensor_q_nozp, group_size)

        tensor_shape = torch.as_tensor(tensor.shape)
        scale_shape = torch.as_tensor(tensor_q_zp[1].shape)
        q_dim = (scale_shape!=tensor_shape).nonzero().item()
        reshaped_shape = scale_shape[:q_dim].tolist() + [scale_shape[q_dim].item(),group_size] + scale_shape[q_dim+1:].tolist()
        mask = ((tensor_deq_zp-tensor)/1).view(reshaped_shape).abs().sum(q_dim+1) > ((tensor_deq_nozp-tensor)/1).view(reshaped_shape).abs().sum(q_dim+1)

        q_tensor_combined = {'zp':tensor_q_zp[0],'nozp':tensor_q_nozp[0]}
        scale_combined = {'zp':tensor_q_zp[1],'nozp':tensor_q_nozp[1],'mask':mask}
        zp_combined = {'zp':tensor_q_zp[2],'nozp':tensor_q_nozp[2]}

        return q_tensor_combined, scale_combined, zp_combined
    return hybrid_func

def dequantize_hybrid(zp_func, nozp_func):
    def hybrid_func(tensor, scale, sign_or_zp, group_size, op_dtype = torch.float32):
        zp_tensor = tensor['zp']
        nozp_tensor = tensor['nozp']
        zp_scale = scale['zp']
        nozp_scale = scale['nozp']
        zero_point = sign_or_zp['zp']
        sign = sign_or_zp['nozp']

        tensor_shape = torch.as_tensor(zp_tensor.shape)
        scale_shape = torch.as_tensor(zp_scale.shape)
        q_dim = (scale_shape!=tensor_shape).nonzero().item()
        reshaped_shape = scale_shape[:q_dim].tolist() + [scale_shape[q_dim].item(),group_size] + scale_shape[q_dim+1:].tolist()

        repeat_shape = torch.ones(len(reshaped_shape),dtype=int)
        repeat_shape[q_dim+1] = group_size
        mask = scale['mask'].unsqueeze(q_dim+1).repeat(repeat_shape.tolist())

        zp_deq = zp_func(zp_tensor, zp_scale, zero_point, group_size, op_dtype).reshape(reshaped_shape)
        nozp_deq = nozp_func(nozp_tensor, nozp_scale, sign, group_size, op_dtype).reshape(reshaped_shape)

        combined_deq = torch.where(mask, nozp_deq, zp_deq).view(tensor_shape.tolist())
        return combined_deq
    return hybrid_func

