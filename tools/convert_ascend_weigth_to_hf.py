# coding=utf-8
"""Convert weight from ascend to huggingface"""
import argparse
import os
import glob
import sys
import torch

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir", help="llama native model dir")
    parser.add_argument("--ascend-model-dir", type=str, default="./ascend_model_dir", help="ascend model dir")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir", help="ascend model dir")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--num_layers", type=int, default=32,
                        help="num layers")
    parser.add_argument("--num_heads", type=int, default=32,
                        help="num heads")
    parser.add_argument("--num_kv_heads", type=int, default=None,
                        help="num kv heads")
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="hidden size")
    return parser.parse_args()


def get_checkpoint_name(pp_size, tp_rank, pp_rank, checkpoints_path, iteration,
                        release=False, model_name='model_optim_rng.pt'):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    if pp_size == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                tp_rank),
                            model_name)
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            tp_rank,
                            pp_rank),
                        model_name)

def get_checkpoint_tracker_filename(checkpoints_path):
    """
    Tracker file rescords the latest chckpoint during
    training to restart from.
    """
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')

def read_tracker(load_dir):
    iteration = 0
    release = False
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                sys.exit()

    return True, iteration, release

def get_qkv_func(w, n_head=None, hidden_size=None, tp=None, kv_heads=None, i=None):
    if kv_heads is None:
        kv_heads = n_head
    assert n_head % tp == 0
    assert hidden_size % n_head == 0
    assert kv_heads % tp == 0
    assert n_head % kv_heads == 0
    np = n_head // tp
    gp = kv_heads // tp
    repeats = np // gp
    hn = hidden_size // n_head
    w = w.reshape((gp, repeats * hn + hn + hn, -1))
    w = list(w.split([repeats * hn, hn, hn], 1))
    w[0] = w[0].reshape(gp * repeats * hn, -1)
    w[1] = w[1].reshape(gp * hn, -1)
    w[2] = w[2].reshape(gp * hn, -1)
    return w[i]

def update_dt_func(filename, state_dict, models, save_path, n_layer, pp_size, n_head, hidden_size, tp_size, kv_heads):
    pp_num_layer = n_layer // pp_size
    from functools import partial
    qkv_func = partial(get_qkv_func, n_head=n_head, hidden_size=hidden_size, tp=tp_size, kv_heads=kv_heads)
    for name in state_dict:
        if name == "model.embed_tokens.weight":
            ws = [m['model']['language_model']['embedding']['word_embeddings']['weight'] for m in models[0]]
            ws = torch.concat(ws, 0)
            w0 = state_dict[name].shape[0]
            state_dict[name].copy_(ws[:w0, ...])
        if name == "model.norm.weight":
            ws = [m['model']['language_model']['encoder']['final_layernorm.weight'] for m in models[-1]][0]
            state_dict[name].copy_(ws)
        if name == "lm_head.weight":
            ws = [m['model']['language_model']['output_layer']['weight'] for m in models[-1]]
            ws = torch.concat(ws, 0)
            w0 = state_dict[name].shape[0]
            state_dict[name].copy_(ws[:w0, ...])
        if name.startswith('model.layers'):
            ori_id = int(name.split('.')[2])
            pp = ori_id // pp_num_layer
            dst_id = ori_id - pp_num_layer * pp
            if name.endswith('self_attn.q_proj.weight'): 
                ws = [qkv_func(m['model']['language_model']['encoder'][f"layers.{dst_id}.self_attention.query_key_value.weight"], i=0) for m in models[pp]]
                ws = torch.concat(ws, 0)
            if name.endswith('self_attn.k_proj.weight'):
                ws = [qkv_func(m['model']['language_model']['encoder'][f"layers.{dst_id}.self_attention.query_key_value.weight"], i=1) for m in models[pp]]
                ws = torch.concat(ws, 0)
            if name.endswith('self_attn.v_proj.weight'):
                ws = [qkv_func(m['model']['language_model']['encoder'][f"layers.{dst_id}.self_attention.query_key_value.weight"], i=2) for m in models[pp]]
                ws = torch.concat(ws, 0)
            if name.endswith("self_attn.o_proj.weight"):
                ws = [m['model']['language_model']['encoder'][f"layers.{dst_id}.self_attention.dense.weight"] for m in models[pp]]
                ws = torch.concat(ws, 1)
            if name.endswith("mlp.gate_proj.weight"):
                ws = [torch.chunk(m['model']['language_model']['encoder'][f"layers.{dst_id}.mlp.proj.weight"], 2, 0)[0] for m in models[pp]]
                ws = torch.concat(ws, 0)
            if name.endswith("mlp.up_proj.weight"):
                ws = [torch.chunk(m['model']['language_model']['encoder'][f"layers.{dst_id}.mlp.proj.weight"], 2, 0)[1] for m in models[pp]]
                ws = torch.concat(ws, 0)
            if name.endswith("mlp.down_proj.weight"):
                ws = [m['model']['language_model']['encoder'][f"layers.{dst_id}.mlp.dense_4h_to_h.weight"] for m in models[pp]]
                ws = torch.concat(ws, 1)
            if name.endswith("input_layernorm.weight"):
                ws = [m['model']['language_model']['encoder'][f"layers.{dst_id}.input_layernorm.weight"] for m in models[pp]][0]
            if name.endswith("post_attention_layernorm.weight"):
                ws = [m['model']['language_model']['encoder'][f"layers.{dst_id}.post_attention_layernorm.weight"] for m in models[pp]][0]
            if name.endswith("self_attn.rotary_emb.inv_freq"):
                continue
            state_dict[name].copy_(ws)

    save_path = os.path.join(save_path, os.path.basename(filename))
    print (f'save model to {save_path}')
    torch.save(state_dict, save_path)

def main():
    args = get_args()

    load_dir = args.ascend_model_dir
    save_dir = args.output_model_dir
    n_layer, pp_size, n_head, hidden_size, tp_size, kv_heads = args.num_layers, args.pipeline_model_parallel_size, args.num_heads, \
        args.hidden_size, args.tensor_model_parallel_size, args.num_kv_heads

    _, iteration, release = read_tracker(load_dir)

    models = []
    for pp_rank in range(pp_size):
        pp_m = []
        for tp_rank in range(tp_size):
            model_checkpoint_name = get_checkpoint_name(pp_size, tp_rank, pp_rank, load_dir, iteration, release)
            st = torch.load(model_checkpoint_name, map_location='cpu')
            pp_m.append(st)
        models.append(pp_m)

    filenames = glob.glob(os.path.join(args.input_model_dir, '*.bin'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for filename in filenames:
        dt = torch.load(filename, map_location='cpu')
        update_dt_func(filename, dt, models, save_dir, n_layer, pp_size, n_head, hidden_size, tp_size, kv_heads)

if __name__ == '__main__':
    main()
