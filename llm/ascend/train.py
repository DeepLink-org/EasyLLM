from datetime import datetime
import math
import sys
import time
import json

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import deepspeed

from ascend import get_args
from ascend import get_timers
from ascend import get_num_microbatches
from ascend.core import tensor_parallel, parallel_state
from ascend.core.pipeline_parallel.schedules import forward_backward_pipelining_with_foldx_fifo
from ascend.core.pipeline_parallel.schedules import forward_backward_pipelining_with_foldx_aiao
from ascend.core.pipeline_parallel.schedules import get_forward_backward_func, get_forward_func
from ascend.core.memory.auto_recomputing.autorecompute import autorecompute_profile
from llm.utils.general.error_utils import (
    check_equal, 
    check_type, 
    ensure_var_is_not_none, 
    ensure_var_is_none,
    ensure_valid
)
from ascend.core.core_utils import get_model_config
from typing import List
from functools import partial
from ascend.utils import average_losses_across_data_parallel_group

def loss_func(loss_mask, output_tensor):
    args = get_args()

    if args.keep_last_token:
        losses = output_tensor
        loss_mask = loss_mask[..., 1:].contiguous().view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    else:
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm_loss': averaged_loss[0]}

def forward_step_func(data_iterator, model):
    """Forward step."""
    args = get_args()

    timers = get_timers()
    # Get the batch.
    if args.foldx_mode is None:
        timers('batch-generator').start()
    data = None
    if hasattr(data_iterator, '__next__'):
        data = next(data_iterator)
    (tokens, position_ids, attention_mask), (labels, loss_mask) = model.batch_func(data)
    
    if args.foldx_mode is None:
        timers('batch-generator').stop()

    output_tensor, other_losses = model(tokens, position_ids=position_ids, attention_mask=attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_step(model, optimizer, lr_scheduler, data_iterator, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()
    if not isinstance(model, List):
        model = [model]
    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    else:
        optimizer.zero_grad()

    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()

    if forward_backward_func == forward_backward_pipelining_with_foldx_fifo or\
            forward_backward_func == forward_backward_pipelining_with_foldx_aiao:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length)
    else:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    if args.mos or args.kd:
        args.teacher_forward = False
    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers('forward-backward').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
    optimizer.reduce_model_grads(args, timers)
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    if update_successful:
        optimizer.gather_model_params(args, timers)
    timers('optimizer').stop()

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


