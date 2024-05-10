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
from deepspeed.accelerator.real_accelerator import get_accelerator

if get_accelerator().device_name() == 'cuda':
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
else:
    from torch.nn import LayerNorm
from .distributed import DistributedDataParallel
from .gpt_model import GPTModel, GPTModelPipe
from .language_model import get_language_model
from .module import Float16Module
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from deepspeed.accelerator import get_accelerator
from ascend import get_args
from ascend.core import tensor_parallel, parallel_state
from ascend.core.enums import ModelType
from ascend.model import DistributedDataParallel as LocalDDP
from ascend.model.gpt_model import GPTModel


def model_provider(config, model_type):
    args = get_args()
    args.model_type = model_type
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = GPTModel(
        config=config,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    model.model_type = model_type
    if not isinstance(model, list):
        model = [model]

    # GPU allocation.
    for model_module in model:
        device_name = get_accelerator().current_device_name()
        model_module.to(device_name)

    model = wrap_model(model)

    return model

def wrap_model(model, wrap_with_ddp=True):
    args = get_args()
    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]
    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = get_accelerator().current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=parallel_state.get_data_parallel_group())
                     for model_module in model]
            return model

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            return model
        else:
            raise NotImplementedError('Unknown DDP implementation specified: {}. '
                                      'Exiting.'.format(args.DDP_impl))
    return model


def get_model(config):
    model = model_provider(config, ModelType.encoder_or_decoder)
    return model