# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from llm.utils.env import dist_env
from dataclasses import dataclass
import dataclasses
import os
from importlib.metadata import version

import torch
import transformer_engine as te
from pkg_resources import packaging
from torch import Tensor

from .transformer_config import TransformerConfig
from ..base_modules.modules.enums import AttnMaskType
from .random import get_cuda_rng_tracker

_te_version = packaging.version.Version(version("transformer-engine"))


@dataclass
class PackedSeqParams:
    # parameters to TEDotProductAttention and fused rope kernels for the `thd` (packed) sequence format,
    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    max_seqlen_q: Tensor = None
    max_seqlen_kv: Tensor = None


def condition_init_method(config, init_method):
    return init_method if config.perform_initialization else (lambda w: None)


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = 'sbhd'

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv('NVTE_APPLY_QK_LAYER_SCALING', '0'))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs = {}
        if _te_version >= packaging.version.Version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{_te_version} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if _te_version >= packaging.version.Version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if _te_version > packaging.version.Version("0.12.0"):
            self.te_forward_mask_type = True

        # Only Transformer-Engine version >= 1.0.0 supports context parallelism
        if _te_version >= packaging.version.Version("1.0.0"):
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            # extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            extra_kwargs["cp_group"] = dist_env.get_context_parallel_group()
            # extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(
            #     check_initialized=False
            # )
            extra_kwargs["cp_global_ranks"] = dist_env.get_context_parallel_global_ranks()
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
        else:
            assert (
                self.config.context_parallel_size == 1
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if config.window_size is not None:
            # Check version
            assert _te_version >= packaging.version.Version(
                "1.2.0"
            ), f"Transformer-Engine version ({str(_te_version)}) must be >= 1.2.0 to support sliding window attention."
            extra_kwargs['window_size'] = config.window_size

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=self.config.attention_dropout
            if attention_dropout is None
            else attention_dropout,
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker
            if get_cuda_rng_tracker().is_initialized()
            else None,
            # tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_group=dist_env.get_tensor_model_parallel_group(),
            layer_number=layer_number,
            **extra_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: PackedSeqParams = None,
    ):
        packed_seq_kwargs = (
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set after init
        if self.config.apply_rope_fusion and _te_version > packaging.version.Version("0.13.0"):
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        if _te_version < packaging.version.Version("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H copies (#555)
            # These two arguments did not exist prior to 1.3.0
            packed_seq_kwargs.pop("max_seqlen_q", None)
            packed_seq_kwargs.pop("max_seqlen_kv", None)

        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[0] == 1 and value.stride() != key.stride():
                value = value.as_strided(value.shape, key.stride())

        if self.te_forward_mask_type:
            core_attn_out = super().forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **packed_seq_kwargs,
            )
        else:
            core_attn_out = super().forward(query, key, value, attention_mask, **packed_seq_kwargs,)

        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            return core_attn_out.transpose(0, 1)
        else:
            return core_attn_out
