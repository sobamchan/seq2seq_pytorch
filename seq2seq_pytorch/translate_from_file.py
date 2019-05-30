from pathlib import Path
from typing import List, Callable, Dict
import pickle

import lineflow as lf
import fire

import torch
from torch.utils.data import DataLoader

from seq2seq_pytorch.data import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN
from seq2seq_pytorch.translators import Translator


def preprocess(x: str) -> List:
    return [START_TOKEN] + x.split() + [END_TOKEN]


def postprocess(src_t2i, src_unk_idx, tgt_t2i, tgt_unk_idx):
    def _f(x):
        src_ids = [src_t2i.get(token, src_unk_idx) for token in x]
        return src_ids
    return _f


def get_collate_fn(src_pad_idx: int) -> Callable:
    def _f(src: List):
        src_lens = [len(x) for x in src]
        src_max_len = max(src_lens)

        padded_src = [x + [src_pad_idx] * (src_max_len - len(x)) for x in src]

        return (
                torch.LongTensor(padded_src),
                torch.LongTensor(src_lens)
                )
    return _f


def trim_special_tokens(tokens):
    return [token for token in tokens if token not in [PAD_TOKEN, END_TOKEN, START_TOKEN]]


def run(fpath: str, translator_path: str, src_vocab_path: str, tgt_vocab_path: str,
        bsize: int = 32, savedir: str = './test'):
    fpath = Path(fpath)
    fname = fpath.stem
    savedir = Path(savedir)

    dataset = lf.TextDataset(str(fpath)).map(preprocess)

    src_t2i: Dict
    tgt_t2i: Dict
    with open(src_vocab_path, 'rb') as f:
        src_t2i, _ = pickle.load(f)
    with open(tgt_vocab_path, 'rb') as f:
        tgt_t2i, _ = pickle.load(f)
    translator: Translator = torch.load(translator_path)

    src_unk_idx = src_t2i[UNK_TOKEN]
    tgt_unk_idx = tgt_t2i[UNK_TOKEN]
    src_pad_idx = src_t2i[PAD_TOKEN]

    dataloader = DataLoader(
            dataset.map(postprocess(src_t2i, src_unk_idx, tgt_t2i, tgt_unk_idx))
            .save((savedir / fname).with_suffix('.pred.cache')),
            batch_size=bsize,
            shuffle=False,
            num_workers=4,
            collate_fn=get_collate_fn(src_pad_idx)
            )

    tgt_i2t = {v: k for k, v in tgt_t2i.items()}

    pred_sents = []
    for batch in dataloader:
        src, src_lens = batch
        pred_seqs = translator.translate(src, src_lens, max_target_len=30)
        for pids in pred_seqs:
            ptokens = [tgt_i2t.get(int(pid)) for pid in pids]
            ptokens = trim_special_tokens(ptokens)
            pred_sents.append(' '.join(ptokens))

    with open(savedir / 'pred.txt', 'w') as f:
        f.write('\n'.join(pred_sents))


if __name__ == '__main__':
    fire.Fire()
