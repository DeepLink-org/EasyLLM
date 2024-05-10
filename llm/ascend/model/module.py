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
"""AscendSpeed Module"""
import os
import abc
import json
import logging
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import deepspeed
from deepspeed.accelerator import get_accelerator

from ..global_vars import get_args
from ascend.core import parallel_state, tensor_parallel
from ascend.model.lora_utils import is_enable_lora, get_lora_model_classes
from llm.utils.general.error_utils import ensure_valid

_FLOAT_TYPES = (torch.FloatTensor, get_accelerator().FloatTensor)
_HALF_TYPES = (torch.HalfTensor, get_accelerator().HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """
    Megatron specific extensions of torch Module with support
    for pipelining.
    """

    def __init__(self, config=None, share_embeddings_and_output_weights=True):
        super(MegatronModule, self).__init__()
        self.config = config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """
        Use this function to override the state dict for
        saving checkpoints.
        """
        return self.state_dict(destination, prefix, keep_vars)

    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_embeddings_and_output_weights:
                raise Exception('shared_embedding_or_output_weight() called for last '
                                'stage, but share_embeddings_and_output_weights is false')
            return self.word_embeddings.weight

    def initialize_word_embeddings(self):
        args = get_args()
        if not self.share_embeddings_and_output_weights:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_embeddings_and_output_weights is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            ensure_valid(not parallel_state.is_pipeline_first_stage())
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.padded_vocab_size, self.config.hidden_size,
                config=self.config, init_method=self.config.init_method)
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print("WARNING! Distributed processes aren't initialized, so "
                      "word embeddings in the last layer are not initialized. "
                      "If you are just manipulating a model this is fine, but "
                      "this needs to be handled manually. If you are training "
                      "something is definitely wrong.")
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if parallel_state.is_rank_in_embedding_group():
            torch.distributed.all_reduce(self.shared_embedding_or_output_weight().data,
                                         group=parallel_state.get_embedding_group())

        # Ensure that encoder(first stage) and decoder(split stage) position
        # embeddings have the same initial parameter values
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if parallel_state.is_rank_in_position_embedding_group() and \
                args.pipeline_model_parallel_split_rank is not None:
            self.language_model.embedding.cuda()
            position_embeddings = self.language_model.embedding.position_embeddings
            torch.distributed.all_reduce(position_embeddings.weight.data,
                                         group=parallel_state.get_position_embedding_group())


def conversion_helper(val, conversion):
    """
    Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure.
    """
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        if val is None:
            return val

        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor

    def forward(self, *inputs, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)