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

"""ascendspeed arguments."""

import argparse
import dataclasses
import json
import os
import types

import deepspeed
import torch
import torch.nn.functional as F

from ascend.global_vars import set_retro_args, get_retro_args
from ascend.core.transformer import TransformerConfig
from ascend.enums import PositionEmbeddingType
from llm.utils.general.error_utils import (
    check_divisible,
    check_equal,
    ensure_var_is_not_none,
    ensure_var_is_none,
    ensure_valid
)

def validate_args(args, defaults={}):
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    check_divisible(args.world_size, args.tensor_model_parallel_size, error_info='world size' \
                       ' ({}) is not divisible by tensor model parallel size ({})'.format(
        args.world_size, args.tensor_model_parallel_size))
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )
    # Checks.
    if args.no_pipeline_parallel:
        check_equal(args.pipeline_model_parallel_size, 1, error_info="pipeline_model_parallel_size"\
                    " must be 1 if pipeline parallel is disabled")
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    check_divisible(args.world_size, model_parallel_size, error_info='world size is not'\
                    ' divisible by tensor parallel size ({}) times pipeline parallel ' \
                    'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                                       args.pipeline_model_parallel_size))
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)

    # Deprecated arguments
    ensure_var_is_none(args.batch_size, error_message='--batch-size argument is no longer '\
                                        'valid, use --micro-batch-size instead.')
    del args.batch_size
    ensure_var_is_none(args.warmup, error_message='--warmup argument is no longer valid, use ' \
                                    '--lr-warmup-fraction instead.')
    del args.warmup
    ensure_var_is_none(args.model_parallel_size, error_message='--model-parallel-size is no '\
                                                 'longer valid, use --tensor-model-parallel-size instead.')
    del args.model_parallel_size

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    ensure_var_is_not_none(args.micro_batch_size)
    ensure_valid(args.micro_batch_size > 0)
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    ensure_valid(args.global_batch_size > 0)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        error_message = 'pipeline-model-parallel size should be greater than 2 with ' \
                        'interleaved schedule'
        ensure_valid(args.pipeline_model_parallel_size > 2, error_message)
        check_divisible(args.num_layers, args.num_layers_per_virtual_pipeline_stage, error_info='number of layers'\
                        ' is not divisible by number of layers per virtual pipeline stage')
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        ensure_valid(not args.bf16)
        args.params_dtype = torch.half
    if args.bf16:
        ensure_valid(not args.fp16)
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is not off.
    if args.accumulate_allreduce_grads_in_fp32:
        check_equal(args.DDP_impl, 'local')
        ensure_valid(args.use_contiguous_buffers_in_local_ddp)

    # If we use the distributed optimizer, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is on.
    if args.use_distributed_optimizer:
        check_equal(args.DDP_impl, 'local')
        ensure_valid(args.use_contiguous_buffers_in_local_ddp)

    # For torch DDP, we do not use contiguous buffer
    if args.DDP_impl == 'torch':
        args.use_contiguous_buffers_in_local_ddp = False

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.consumed_train_tokens = 0
    args.custom_token_counting = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        ensure_var_is_none(args.train_samples, error_message='expected iteration-based training')
        ensure_var_is_none(args.lr_decay_samples, 'expected iteration-based learning rate decay')
        check_equal(args.lr_warmup_samples, 0, error_info='expected iteration-based learning rate warmup')
        ensure_var_is_none(args.rampup_batch_size, 'expected no batch-size rampup for iteration-based training')
        if args.lr_warmup_fraction is not None:
            check_equal(args.lr_warmup_iters, 0, error_info='can only specify one of lr-warmup-fraction '\
                                                            'and lr-warmup-iters')

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        ensure_var_is_none(args.train_iters, error_message='expected sample-based training')
        ensure_var_is_none(args.lr_decay_iters, error_message='expected sample-based learning rate decay')
        check_equal(args.lr_warmup_iters, 0, error_info='expected sample-based learnig rate warmup')
        if args.lr_warmup_fraction is not None:
            check_equal(args.lr_warmup_samples, 0, error_info='can only specify one of lr-warmup-fraction ' \
                                                              'and lr-warmup-samples')
    if args.num_layers is not None:
        ensure_valid(args.encoder_num_layers is None,
                     'cannot have both num-layers and encoder-num-layers specified')
        args.encoder_num_layers = args.num_layers
    else:
        ensure_valid(args.encoder_num_layers is not None,
                     'either num-layers or encoder-num-layers should be specified')
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        check_divisible(args.hidden_size, args.num_attention_heads)
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    if args.seq_length is not None:
        ensure_var_is_none(args.encoder_seq_length)
        args.encoder_seq_length = args.seq_length
    else:
        ensure_var_is_not_none(args.encoder_seq_length)
        args.seq_length = args.encoder_seq_length

    if args.variable_seq_lengths:
        ensure_valid(args.is_instruction_dataset, 'Dynamic padding based on instruction dataset.')
    
    if args.release_fp32_grad:
        if args.optimizer != "adam":
            raise ValueError(
                "`release_fp32_grad` only support for `Adam` optimizer now.")

    # Retro checks.
    if args.retro_add_retriever:
        # Sequence parallelism unsupported.
        ensure_valid(not args.sequence_parallel, "retro currently does not support sequence parallelism.")

        # Pipeline parallelism unsupported.
        ensure_valid(args.pipeline_model_parallel_size == 1, "retro currently does not support pipeline parallelism.")

        # Load retro
        retro_args_path = os.path.join(args.retro_workdir, "args.json")
        ensure_valid(os.path.exists(retro_args_path), "retro workdir missing args.json")
        with open(retro_args_path) as f:
            retro_args = types.SimpleNamespace(**json.load(f))
            retro_args.retro_return_doc_ids = args.retro_return_doc_ids
            retro_args.retro_gpt_retrieved_length = \
                args.retro_num_retrieved_chunks * \
                retro_args.retro_gpt_chunk_length
            set_retro_args(retro_args)
    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False
    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = PositionEmbeddingType.rope

    if (args.position_embedding_type == PositionEmbeddingType.absolute or
            args.position_embedding_type == PositionEmbeddingType.alibi or
            args.position_embedding_type == PositionEmbeddingType.rope):
        ensure_var_is_not_none(args.max_position_embeddings)
        if not args.seq_length:
            ensure_valid(args.max_position_embeddings >= args.seq_length)
        if args.decoder_seq_length is not None:
            ensure_valid(args.max_position_embeddings >= args.decoder_seq_length)
    else:
        ensure_var_is_none(args.max_position_embeddings)

    if args.seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.seq_length)
    if args.decoder_seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.decoder_seq_length)
    if args.lr is not None:
        ensure_valid(args.min_lr <= args.lr)
    if args.save is not None:
        ensure_var_is_not_none(args.save_interval)
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        error_message = 'lm cross entropy in fp16 only support in fp16 mode.'
        ensure_valid(args.fp16, error_message)
    if args.fp32_residual_connection:
        error_message = 'residual connection in fp32 only supported when using fp16 or bf16.'
        ensure_valid(args.fp16 or args.bf16, error_message)
    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        error_message = 'for distribute-checkpointed-activations to work you '\
                        'need to enable checkpoint-activations'
        ensure_valid(args.checkpoint_activations, error_message)
    torch_major = int(torch.__version__.split('.')[0])
    torch_minor = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if torch_major < 1 or (torch_major == 1 and torch_minor < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')
    else:
        args.no_persist_layer_norm = False
    # Check Alibi-Mask.
    if args.position_embedding_type == PositionEmbeddingType.alibi and args.use_flash_attn:
        ensure_valid(not args.padding_attention_mask, 'FlashAttention of Alibi do not support' +
                        'padding attention mask!')
        ensure_valid(not args.is_instruction_dataset, 'FlashAttention of Alibi do not support' +
                        'is_instruction_dataset!')
    if args.fill_neg_inf:
        ensure_valid(args.square_alibi_mask, 'square-alibi-mask must be passed when you' +
                        'set fill-neg-inf to be true!')
    # Activation recomputing.
    if args.distribute_saved_activations:
        ensure_valid(args.tensor_model_parallel_size > 1, 'can distribute ' +
                     'recomputed activations only across tensor model ' +
                     'parallel groups')
        ensure_valid(args.recompute_granularity == 'full',
                     'distributed recompute activations is only ' +
                     'application to full recompute granularity')
        ensure_valid(args.recompute_method is not None,
                     'for distributed recompute activations to work you ' +
                     'need to use a recompute method ')
        ensure_valid((torch_major, torch_minor) >= (1, 10),
                     'distributed recompute activations are supported for pytorch ' +
                     'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' +
                     'pytorch version is v%s.%s.' % (torch_major, torch_minor))
    if args.recompute_granularity == 'selective':
        ensure_valid(args.recompute_method is None,
                     'recompute method is not yet supported for ' +
                     'selective recomputing granularity')
    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False
    args.curriculum_learning_legacy = False
    args.compression_training = False
    args.fp8_e4m3 = False
    args.fp8_hybrid = False

    if args.group_query_attention and args.position_embedding_type != PositionEmbeddingType.rope:
        raise NotImplementedError(
                    'Currently the group query attention only '
                    'support rotary position embedding.')

    error_message = 'Triangle attn and flash attention should not be used at the same time.'
    ensure_valid(not (args.triangle_attn and args.use_flash_attn), error_message)
    # manually layer distribute
    _get_manual_layer_allocation(args)

    _print_args(args)
    return args

def _get_manual_layer_allocation(args=None):
    if args is not None and args.use_manual_layer_allocation:
        ensure_var_is_not_none(args.manual_layers)
        manual_layers = list(map(int, args.manual_layers.split(',')))
        check_equal(len(manual_layers), args.pipeline_model_parallel_size)
        args.manual_layers = manual_layers


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    ensure_var_is_not_none(getattr(args, arg), error_message='{} argument is None'.format(arg))


def core_transformer_config_from_args(args):
    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['variable_seq_lengths'] = args.variable_seq_lengths
    kw_args['deallocate_pipeline_outputs'] = False
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    kw_args['fp8'] = args.fp8_e4m3 or args.fp8_hybrid
    kw_args['fp8_e4m3'] = args.fp8_e4m3
    kw_args['fp8_margin'] = args.fp8_hybrid
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    return TransformerConfig(**kw_args)
