import torch

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import BATCH_FN_REGISTRY


@BATCH_FN_REGISTRY.register('flash_batch_pipe')
class FlashBatchFunction(object):
    def __init__(self,
                 tokenizer,
                 eod_mask_loss=True,
                 pretrain=False):
        self.tokenizer = tokenizer
        self.eod_mask_loss = eod_mask_loss
        self.pad_token_id = len(self.tokenizer) - 1
        self.pretrain = pretrain
        self.keys = ['labels', 'input_ids', "cu_seqlens", 'position_ids']

    def __call__(self, data):
        datatype = torch.int64
        # Broadcast data.
        data_b = dist_env.broadcast_data(self.keys, data, datatype)

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()

        attention_mask = tokens.ne(self.pad_token_id)
        loss_mask = attention_mask.clone()
        if len(data_b['cu_seqlens']) == 1:
            _, seq_length = tokens.size()
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
            return (tokens, position_ids, attention_mask), (labels, loss_mask)
        else:
            cu_seqlens = data_b['cu_seqlens']
            position_ids = data_b['position_ids']
            return (tokens, position_ids, attention_mask, cu_seqlens), (labels, loss_mask, cu_seqlens)


@BATCH_FN_REGISTRY.register('mini_rlhf_json_batch_pipe')
class MiniRLHFJsonBatchFunction(object):
    def __init__(self, tokenizer, reset_position_ids, reset_attention_mask,
                 eod_mask_loss=True, prefix_indices=None, loss_on_targets_only=True):
        self.tokenizer = tokenizer
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.prefix_indices = prefix_indices
        self.loss_on_targets_only = loss_on_targets_only
        self.pad_token_id = len(self.tokenizer) - 1

    def __call__(self, data):
        # Items and their type.
        keys = ['labels', 'input_ids']
        datatype = torch.int64
        # Broadcast data.
        data_b = dist_env.broadcast_data(keys, data, datatype)
        scores = dist_env.broadcast_data(['scores'], data, torch.float32)['scores'].float()

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()

        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.pad_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            prefix_indices=self.prefix_indices,
            loss_on_targets_only=self.loss_on_targets_only
        )

        return (tokens, position_ids, attention_mask), (labels, loss_mask, scores)


@BATCH_FN_REGISTRY.register('token_batch_pipe')
class TokenBatchFunction(object):
    def __init__(self, tokenizer, reset_position_ids, reset_attention_mask,
                 eod_mask_loss=True, prefix_indices=None, loss_on_targets_only=True,
                 micro_batch_size=1):
        self.tokenizer = tokenizer
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.prefix_indices = prefix_indices
        self.loss_on_targets_only = loss_on_targets_only
        self.micro_batch_size = micro_batch_size
        self.pad_token_id = len(self.tokenizer) - 1

    def __call__(self, data):
        """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""

        # Move to GPU.
        tokens = data.view(self.micro_batch_size, -1).contiguous().cuda()

        labels = torch.cat([tokens[:, 1:], tokens.new_ones(tokens.shape[0], 1) * self.tokenizer.eos_token_id], dim=-1)
        # Get the attention mask and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eos_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            prefix_indices=self.prefix_indices,
            loss_on_targets_only=self.loss_on_targets_only
        )

        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def build_batch_pipe_fn(cfg_batch_pipe, tokenizer):
    if 'kwargs' not in cfg_batch_pipe:
        cfg_batch_pipe['kwargs'] = {}
    cfg_batch_pipe['kwargs']['tokenizer'] = tokenizer
    return BATCH_FN_REGISTRY.build(cfg_batch_pipe)
