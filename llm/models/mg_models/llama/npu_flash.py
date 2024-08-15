import torch_npu
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def flash_attn_rms_norm(input, weight, eps):
    return torch_npu.npu_rms_norm(input, weight, eps)[0]


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    head_num = q.shape[-2]
    
    if seqlen_q == seqlen_k and seqlen_q < 2048 and seqlen_k < 2048:
        sparse_mode = 0
    else:
        sparse_mode = 2

    seqlen_q = min(seqlen_q, 2048)
    seqlen_k = min(seqlen_k, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q, k, v, head_num, "BSND", 
        atten_mask = attention_mask,
        scale = softmax_scale,
        keep_prob = 1 - dropout_p,
        pre_tockens = seqlen_q,
        next_tockens = 0,
        sparse_mode = sparse_mode,
    )[0]

    return out


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]

    seqlen_qkv = qkv.shape[1]
    head_num = q.shape[-2]

    if seqlen_qkv < 2048:
        sparse_mode = 0
    else:
        sparse_mode = 2

    seqlen_qkv = min(qkv.shape[1], 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen_qkv, seqlen_qkv], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=seqlen_qkv,
        next_tockens=0,
        sparse_mode=sparse_mode,
    )[0]

    return out


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    k = kv[:, :, 0]
    v = kv[:, :, 1]

    s0 = q.shape[1]
    s1 = kv.shape[1]
    head_num = q.shape[-2]

    if s0 == s1 and s0 < 2048 and s1 < 2048:
        sparse_mode = 0
    else:
        sparse_mode = 2

    seqlen_q = min(s0, 2048)
    seqlen_k = min(s1, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=seqlen_k,
        next_tockens=0,
        sparse_mode=sparse_mode,
    )[0]

    return out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None
):
    if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
    head_num = q.shape[-2]
    
    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_k = cu_seqlens_k[1:].tolist()
    seqlen_q = min(max_seqlen_q, 2048)
    seqlen_k = min(max_seqlen_k, 2048)

    if max_seqlen_q < 2048:
        sparse_mode = 0
    else:
        sparse_mode = 2

    attention_mask = (
        torch.triu(
            torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q, k, v, head_num, "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],  # seq_len
        next_tockens=0,  # 0
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q = qkv[:, 0]
    k = qkv[:, 1]
    v = qkv[:, 2]
    n = q.shape[1]
    if max_seqlen > 2048:
        sparse_mode = 2
    else:
        sparse_mode = 0
    cu_seqlens_q = cu_seqlens[1:].tolist()
    cu_seqlens_k = cu_seqlens[1:].tolist()
    seqlen = min(max_seqlen, 2048)
    attention_mask = (
        torch.triu(
            torch.ones([seqlen, seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )
    out = torch_npu.npu_fusion_attention(
        q, k, v, n, "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],  # seq_len
        next_tockens=0,  # 0
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    k = kv[:, 0]
    v = kv[:, 1]
    n = q.shape[1]
    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_k = cu_seqlens_k[1:].tolist()
    seqlen_q = min(max_seqlen_q, 2048)
    seqlen_k = min(max_seqlen_k, 2048)

    if max_seqlen_q > 2048:
        sparse_mode = 2
    else:
        sparse_mode = 0

    attention_mask = (
        torch.triu(
            torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )
    out = torch_npu.npu_fusion_attention(
        q, k, v, n, "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],  # seq_len
        next_tockens=0,  # 0
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,

    )[0]
    return out


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # noqa
        # second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """
    Supports concatenating short samples in one sequence. The attention_mask_in_length is utilized to mask other short samples. It helps efficient training of variant lengths-based samples (e.g., the supervised fine-tuning task in large language model).    # noqa
    The motivation for this function is explained [here](https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).

    For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        ```
        [
          [2, 3, 0, 0, 0, 0],
          [3, 2, 0, 0, 0, 0],
          [6, 0, 0, 0, 0, 0]
        ]
        ```
    , which refers to the 3D-attention mask:
        ```
        [
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
          ]
        ]
        ```.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(seqlen, device=length.device, dtype=length.dtype).expand(len(length), seqlen) < length.unsqueeze(1)
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
