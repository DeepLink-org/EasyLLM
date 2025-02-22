import os
import sys
import random

import torch
import numpy as np

from llm.utils.env import dist_env
from llm.utils.general.log_helper import default_logger as logger


def get_checkpoint_name(checkpoints_path, iteration, release=False):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if dist_env.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                dist_env.get_tensor_model_parallel_rank()),
                            'model_optim_rng.pt')
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            dist_env.get_tensor_model_parallel_rank(),
                            dist_env.get_pipeline_model_parallel_rank()),
                        'model_optim_rng.pt')


def load_checkpoint(model, optimizer, cfg_loader, iteration,
                    lora_mode=False, cfg_lora=None):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    if cfg_loader.get("debug", False):
        logger.info('[Debug mode] Model will be initialized randomly...')
        return
    load_mode = cfg_loader.get("load_mode", "deepspeed")
    logger.info('Load Checkpoint in the {} Mode...'.format(load_mode))
    if (load_mode != 'deepspeed'):
        assert hasattr(model, 'load_ckpt_pretrained'), 'if the model do not load by the deepspeed mode, it must provide a load_ckpt_pretrained fn'      # noqa
        model.load_ckpt_pretrained(cfg_loader, model, optimizer)
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if lora_mode and cfg_lora and cfg_lora.get('loader', None):
            assert hasattr(model, 'load_lora_ckpt_pretrained'), 'model has no load_lora_ckpt_pretrained func!'
            model.load_lora_ckpt_pretrained(cfg_lora, model, optimizer)
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        return

    # using the default deepspeed load_checkpoint func
    tag = cfg_loader.get('load_tag', None)
    try:
        # we will update lr in the base runner.
        loaded_dir, state_dict = model.load_checkpoint(cfg_loader['load_path'], tag=tag,
                                                       load_optimizer_states=cfg_loader.get('load_optim', False),
                                                       load_zero=cfg_loader.get('load_zero', False),
                                                       load_lr_scheduler_states=False)
    except TypeError:     # noqa
        logger.warning('The version of deepspeed is not support the option of load_zero, and load_optim')      # noqa
        loaded_dir, state_dict = model.load_checkpoint(cfg_loader['load_path'], tag=tag)
    if loaded_dir is None:
        logger.info('WARNING: could not find the metadata file {} '.format(cfg_loader['load_path']))
        logger.info('Fail to load any checkpoints and will start from random')
        return

    # rng states.
    if cfg_loader.get('load_rng_state', False):
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            # Check for empty states array
            if not state_dict['rng_tracker_states']:
                raise KeyError
            dist_env.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            logger.info(f"Unable to load rng state from checkpoint {cfg_loader['load_path']}. "
                        "Specify --no-load-rng to prevent "
                        "attempting to load the rng state, exiting ...")
            sys.exit()

    logger.info(f"Successfully loaded checkpoint from {cfg_loader['load_path']} at iteration {iteration}")
    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if lora_mode and cfg_lora and cfg_lora.get('loader', None):
        assert hasattr(model, 'load_lora_ckpt_pretrained'), 'model has no load_lora_ckpt_pretrained func!'
        model.load_lora_ckpt_pretrained(cfg_lora, model, optimizer)
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    return


def load_base_state(cfg_loader):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    load_path = cfg_loader['load_path']
    tag = cfg_loader.get('load_tag', None)
    if tag is None:
        latest_path = os.path.join(load_path, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as f:
                tag = f.read().strip()

    assert tag is not None, "Please set the load tag"
    base_state_dict = torch.load(os.path.join(load_path, tag, "base_state.pt"))
    iteration = base_state_dict['iteration']
    consumed_train_samples = base_state_dict['samples']
    consumed_train_tokens = base_state_dict['tokens']

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return iteration, consumed_train_samples, consumed_train_tokens


def save_checkpoint(iteration, consumed_train_samples, consumed_train_tokens, model,
                    cfg_saver, lora_mode=False, cfg_lora=None):
    """Save a model checkpoint."""
    # Only rank zero of the data parallel writes to the disk.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info('saving checkpoint at iteration {:7d} to {}'.format(iteration, cfg_saver['save_path']))      # noqa

    if lora_mode and cfg_lora and cfg_lora.get('saver', None):
        assert hasattr(model, 'save_lora_ckpt_pretrained'), 'model has no save_lora_ckpt_pretrained func!'
        model.save_lora_ckpt_pretrained(cfg_lora, model, iteration=iteration)
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if cfg_lora['saver'].get('only_save_trainable', False):
            return

    save_mode = cfg_saver.get("save_mode", "deepspeed")
    logger.info('Save Checkpoint in the {} Mode...'.format(save_mode))
    if (save_mode != 'deepspeed'):
        assert hasattr(model, 'save_ckpt_pretrained'), 'if the model do not save by the deepspeed mode, it must provide a save_ckpt_pretrained fn'      # noqa
        model.save_ckpt_pretrained(cfg_saver, model)
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # Arguments, iteration, and model.
    state_dict = {}
    # state_dict['args'] = args
    # state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = iteration
    state_dict['samples'] = consumed_train_samples
    state_dict['tokens'] = consumed_train_tokens
    # state_dict['checkpoint_info'] = _checkpoint_info(args, tokenizer)

    # RNG states.
    if cfg_saver.get('save_rng_state', False):
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
        state_dict['rng_tracker_states'] = dist_env.get_cuda_rng_tracker().get_states()      # noqa

    # Saving is a collective communication
    checkpoint_name = get_checkpoint_name(cfg_saver['save_path'], iteration)
    # Trim off the filename and mp_rank_* directory.
    for _ in range(3):
        checkpoint_name = os.path.dirname(checkpoint_name)

    # if not only save lora.
    if (not lora_mode) or (not cfg_lora) or (not cfg_lora.get('only_save_trainable', False)):
        if hasattr(model, 'save_ckpt_pretrained'):
            model.save_ckpt_pretrained(cfg_saver, model, checkpoint_name)
        else:
            save_tag = cfg_saver.get('save_tag', None)
            assert save_tag != 'latest', "latest is the name reserved by deepspeed, you can save the latest with the name like latest_ckpt..."      # noqa
            try:
                model.save_checkpoint(checkpoint_name, tag=save_tag, client_state=state_dict,
                                      save_base_state=cfg_saver.get('save_base_state', True),
                                      save_zero=cfg_saver.get('save_zero', False),
                                      save_optim=cfg_saver.get('save_optim', False))
            except TypeError:
                # for low version of deepspeed
                logger.warning('The version of deepspeed is not support the option of save_base_state, save_zero, and save_optim')      # noqa
                model.save_checkpoint(checkpoint_name, tag=save_tag, client_state=state_dict)

    if torch.distributed.get_rank() == 0:
        base_state_dict = {}
        base_state_dict['iteration'] = iteration
        base_state_dict['samples'] = consumed_train_samples
        base_state_dict['tokens'] = consumed_train_tokens

        save_tag = cfg_saver.get('save_tag', None)
        if save_tag:
            save_path = os.path.join(cfg_saver['save_path'], save_tag)
        else:
            tag = "global_step{}".format(iteration)
            save_path = os.path.join(cfg_saver['save_path'], tag)
        torch.save(base_state_dict, os.path.join(save_path, "base_state.pt"))
        logger.info('Successfully saved base_state at iteration {:7d} to {}'.format(iteration, cfg_saver['save_path']))      # noqa

    logger.info('Successfully saved checkpoint at iteration {:7d} to {}'.format(iteration, cfg_saver['save_path']))      # noqa

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
