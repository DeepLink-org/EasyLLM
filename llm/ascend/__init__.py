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
import sys
import logging

import torch
import deepspeed
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import deepspeed_npu
except Exception as e:
    logging.warning("Warning: You did not install torch_npu or deepspeed_npu")

if 'amp_C' in sys.modules:
    del sys.modules['amp_C']
from . import amp_C
sys.modules['amp_C'] = amp_C

from .global_vars import get_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .global_vars import get_retro_args
from .utils import print_rank_0
from .utils import print_rank_last
from .utils import is_last_rank
from .utils import is_rank_0

import deepspeed
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed import initialize

def adaptor_deepspeed_initialize(*largs, **kwargs):
    return_items = initialize(*largs, **kwargs)
    args = kwargs.get('args')
    if args is not None:
        if isinstance(return_items[0], PipelineEngine):
            return_items[0].is_pipe_partitioned = return_items[0].is_pipe_partitioned and not args.sequence_parallel
            return_items[0].is_grad_partitioned = return_items[0].is_grad_partitioned and not args.sequence_parallel

    return tuple(return_items)


# Fixed an bug where deepspeed PipelineEngine is incompatible with Sequence Parallel.
# When Sequence Parallel is used, PipelineEngine cannot be used, Otherwise, precision problems occur.
deepspeed.initialize = adaptor_deepspeed_initialize

