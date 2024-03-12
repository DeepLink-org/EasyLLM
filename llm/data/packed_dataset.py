from torch.utils.data import Dataset
import itertools as it
import operator
import copy
import os
import math
import numpy as np
import torch
from llm.utils.general.registry_factory import DATASET_REGISTRY
from multiprocessing.pool import ThreadPool as Pool
from llm.data.nlp_dataset import build_dataset
from llm.utils.general.log_helper import default_logger as logger


IGNORE_INDEX = -100
DEFAULT_SEED = 1024


@DATASET_REGISTRY.register("packed")
class PackedDataset(Dataset):
    def __init__(self,
                 dataset={},
                 tokenizer=None,
                 length_path=None,
                 packed_length=4096,
                 worker=8,
                 cache_dir='./cache',
                 epoch=1,
                 ignore_idx=-100,
                 offset_label=False):
        self.dataset = build_dataset(dataset, copy.deepcopy(tokenizer))
        self.dataset_cfg = dataset
        self.epoch = epoch
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        if length_path is None:
            self.length_path = os.path.join(cache_dir, "lengths.npy")
        else:
            self.length_path = length_path
        self.packed_length = packed_length
        self.lengths = []
        self.worker = worker
        self.pad_token_id = len(self.tokenizer) - 1
        self.ignore_idx = ignore_idx
        self.offset_label = offset_label
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Begin preprocess dataset")
        self.preprocess()
        logger.info("Preprocess dataset successed")
        self.seed = DEFAULT_SEED
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples, self.pack_group = self.accu_sample_len(seed=self.seed) # noqa
        self.num_tokens = sum(self.lengths)

    def preprocess(self):

        if os.path.exists(self.length_path):
            logger.info("Load from cache")
            self.lengths = np.load(self.length_path)
        else:
            logger.info("Generate length.npy")
            origin_indexs = list(range(len(self.dataset)))
            lengths_dict = dict()

            def decode_text(idx):
                meta = self.dataset.__getitem__(idx)
                lengths_dict[idx] = len(meta['input_ids'])
            with Pool(self.worker) as p:
                _ = p.map(decode_text, origin_indexs[:])
            for idx in range(len(self.dataset)):
                self.lengths.append(lengths_dict[idx])
            from llm.utils.env import dist_env
            if dist_env.get_data_parallel_rank() == 0 and dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_rank() == 0: # noqa
                np.save(self.length_path, self.lengths)

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        pack_group_epoch = []
        acm_len_samples_epoch = []
        len_samples_shuffled_epoch = []
        sample_indices_epoch = []
        for epoch_idx in range(math.ceil(self.epoch)):
            sample_indices = np.arange(len(self.lengths))
            if seed is not None:
                rng = np.random.RandomState(seed + epoch_idx + 1)
            else:
                rng = np.random.RandomState(self.seed + epoch_idx + 1)
            rng.shuffle(sample_indices)
            len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))

            pack_group = []
            token_length_sum = 0
            each_group = []
            for sample_id, sample_length in enumerate(len_samples_shuffled):
                token_length_sum += sample_length
                if token_length_sum > self.packed_length:
                    pack_group.append(each_group)
                    token_length_sum = sample_length
                    each_group = [sample_id]
                else:
                    each_group.append(sample_id)
            acm_len_samples = list(it.accumulate(len_samples_shuffled, operator.add))
            pack_group_epoch.extend(pack_group)
            acm_len_samples_epoch.extend(acm_len_samples)
            len_samples_shuffled_epoch.extend(len_samples_shuffled)
            sample_indices_epoch.extend(sample_indices)
        if isinstance(self.epoch, float):
            exclued_num = (math.ceil(self.epoch) - int(self.epoch)) * len(self.lengths)
            if exclued_num > 0:
                sample_indices_epoch = sample_indices_epoch[:-exclued_num]
                len_samples_shuffled_epoch = len_samples_shuffled_epoch[:-exclued_num]
                acm_len_samples_epoch = acm_len_samples_epoch[:-exclued_num]
                pack_group_epoch = pack_group_epoch[:-exclued_num]
        return sample_indices_epoch, len_samples_shuffled_epoch, acm_len_samples_epoch, pack_group_epoch

    def __getitem__(self, item: int):
        selected_inds = self.pack_group[item]

        input_ids = []
        labels = []
        cu_seqlens = [0]
        position_ids = []
        for selected_ind in selected_inds:
            index = self.sample_indices[selected_ind]
            meta = self.dataset.__getitem__(index)
            input_ids.append(meta['input_ids'])
            labels.append(meta['labels'])
            cu_seqlens.append(len(meta['input_ids']))
            position_ids.extend(list(range(len(meta['input_ids']))))

        cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
        input_ids = torch.cat(input_ids)[:self.packed_length]
        labels = torch.cat(labels)[:self.packed_length]
        cu_seqlens = torch.clamp(torch.LongTensor(cu_seqlens), max=self.packed_length)
        position_ids = torch.LongTensor(position_ids)[:self.packed_length]

        return {"input_ids": input_ids, "labels": labels, "cu_seqlens": cu_seqlens, "position_ids": position_ids}

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = len(self.pack_group)
        return n_packs

