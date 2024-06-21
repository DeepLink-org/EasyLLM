import numpy as np
from multiprocessing import Pool
from llm.data.nlp_dataset import build_dataset
from llm.data.tokenizer import build_tokenizer
from llm.utils.general.yaml_loader import load_yaml
import argparse
from multiprocessing import Manager


def process(idx):
    meta, out_idx = dataset.get_meta(idx)
    dic[idx] = (out_idx, len(meta['input_ids']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="data root path",
    )
    parser.add_argument(
        "--worker",
        default=32, type=int,
        help="worker num",
    )
    parser.add_argument(
        "--length_path",
        default='./length_path.npy', type=str,
        help="length_path",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    length_path = args.length_path
    dataset_cfg = cfg['data']['train']['dataset']['kwargs']['dataset']
    tokenizer_cfg = cfg['tokenizer']
    tokenizer = build_tokenizer(tokenizer_cfg)
    dataset = build_dataset(dataset_cfg, tokenizer)
    dataset_size = len(dataset)
    indexes = list(range(dataset_size))
    dic = Manager().dict()
    with Pool(args.worker) as p:
        p.map(process, indexes[:])
    lengths = [dic[idx] for idx in indexes[:]]
    np.save(length_path, lengths)
