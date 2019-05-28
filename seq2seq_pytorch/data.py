import os.path as osp
from pathlib import Path
from typing import List, Dict, Callable
from collections import Counter
import pickle

import lineflow as lf
import lineflow.datasets as lfds

import torch
from torch.utils.data import DataLoader


START_TOKEN = '<s>'
END_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def preprocess(x: List):
    src_str: str = x[0]
    tgt_str: str = x[1]
    return (
            [START_TOKEN] + src_str.split() + [END_TOKEN],
            [START_TOKEN] + tgt_str.split() + [END_TOKEN],
            )


def build_vocab(tokens: List, cache: str, max_size: int = 500000) -> (Dict, List):
    if not osp.isfile(cache):
        counter = Counter(tokens)
        words, _ = zip(*counter.most_common(max_size))
        words = [PAD_TOKEN, UNK_TOKEN] + list(words)
        token_to_index = dict(zip(words, range(len(words))))
        if START_TOKEN not in token_to_index:
            token_to_index[START_TOKEN] = len(token_to_index)
            words += [START_TOKEN]
        if END_TOKEN not in token_to_index:
            token_to_index[END_TOKEN] = len(token_to_index)
            words += [END_TOKEN]
        with open(cache, 'wb') as f:
            pickle.dump((token_to_index, words), f)
    else:
        with open(cache, 'rb') as f:
            token_to_index, words = pickle.load(f)

    return token_to_index, words


def postprocess(src_t2i, src_unk_idx, tgt_t2i, tgt_unk_idx):
    def _f(x):
        src_ids = [src_t2i.get(token, src_unk_idx) for token in x[0]]
        tgt_ids = [tgt_t2i.get(token, tgt_unk_idx) for token in x[1]]
        return src_ids, tgt_ids
    return _f


def get_collate_fn(src_pad_idx: int, tgt_pad_idx: int) -> Callable:
    def _f(batch):
        src, tgt = zip(*batch)
        src_lens = [len(x) for x in src]
        src_max_len = max(src_lens)
        tgt_lens = [len(x) for x in tgt]
        tgt_max_len = max(tgt_lens)

        padded_src = [x + [src_pad_idx] * (src_max_len - len(x)) for x in src]
        padded_tgt = [x + [tgt_pad_idx] * (tgt_max_len - len(x)) for x in tgt]
        return (
                torch.LongTensor(padded_src),
                torch.LongTensor(src_lens),
                torch.LongTensor(padded_tgt),
                torch.LongTensor(tgt_lens),
                tgt_max_len
                )
    return _f


def get(savepath: str, bsize: int = 32) -> (DataLoader, DataLoader, Dict, Dict):
    savepath = Path(savepath)

    print('Reading...')
    train = lfds.SmallParallelEnJa('train')
    validation = lfds.SmallParallelEnJa('dev')

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    src_tokens: List = lf.flat_map(lambda x: x[0], train + validation, lazy=True)  # en
    tgt_tokens: List = lf.flat_map(lambda x: x[1], train + validation, lazy=True)  # ja

    print('Building vocabulary...')
    src_t2i, _ = build_vocab(src_tokens, savepath / 'src.voacb')
    tgt_t2i, _ = build_vocab(tgt_tokens, savepath / 'tgt.voacb')

    print(f'Source vocab size: {len(src_t2i)}')
    print(f'Target Vocab size: {len(tgt_t2i)}')

    src_pad_idx = src_t2i[PAD_TOKEN]
    tgt_pad_idx = tgt_t2i[PAD_TOKEN]
    src_unk_idx = src_t2i[UNK_TOKEN]
    tgt_unk_idx = tgt_t2i[UNK_TOKEN]

    print('Postprocessing...')
    train_loader = DataLoader(
            train
            .map(postprocess(src_t2i, src_unk_idx, tgt_t2i, tgt_unk_idx)).save(savepath / 'enja.train.cache'),
            batch_size=bsize,
            shuffle=True,
            num_workers=4,
            collate_fn=get_collate_fn(src_pad_idx, tgt_pad_idx)
            )

    validation_loader = DataLoader(
            validation
            .map(postprocess(src_t2i, src_unk_idx, tgt_t2i, tgt_unk_idx)).save(savepath / 'enja.validation.cache'),
            batch_size=bsize,
            shuffle=False,
            num_workers=4,
            collate_fn=get_collate_fn(src_pad_idx, tgt_pad_idx)
            )

    return train_loader, validation_loader, src_t2i, tgt_t2i
