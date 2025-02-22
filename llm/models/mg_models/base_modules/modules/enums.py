# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import enum


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2  # Overrides `attention_mask` to be a lower triangular matrix
    prefix = 3
    # Forces one to pass an `attention_mask` that's 1 if we need to mask.
    # Tensor that can be broadcast to [micro_batch_size, n_head, seq_length,
    # seq_length]
    custom = 4


class PositionEmbeddingType(enum.Enum):
    rotary = 1
    absolute = 2
    alibi = 3
    flash = 4
    dynamicntk = 5
    yarn = 6
