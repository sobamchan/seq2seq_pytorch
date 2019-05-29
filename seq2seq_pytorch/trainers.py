from typing import List

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

import torch.nn as nn

from seq2seq_pytorch.data import PAD_TOKEN


class TrainerBase(object):

    def __init__(self, corpus, optimizers, translator, clip):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError


class Trainer(TrainerBase):

    def __init__(self, dataloader, optimizers, translator, clip):
        self.dataloader = dataloader
        self.optimizers = optimizers
        self.translator = translator
        self.clip = clip

    def step(self, batch):
        src, src_lens, tgt, tgt_lens, mask, max_target_len = batch

        # Reset gradients
        for o in self.optimizers:
            o.zero_grad()

        # Compute loss
        translator = self.translator
        loss, print_loss = translator.score(
                src,
                src_lens,
                tgt,
                tgt_lens,
                mask,
                max_target_len,
                train=True
                )

        # Clipping, backpropagate, optimize
        loss.backward()

        nn.utils.clip_grad_norm_(translator.encoder.parameters(), self.clip)
        nn.utils.clip_grad_norm_(translator.decoder.parameters(), self.clip)
        nn.utils.clip_grad_norm_(
                translator.encoder_embedding.parameters(), self.clip
                )
        nn.utils.clip_grad_norm_(
                translator.decoder_embedding.parameters(), self.clip
                )
        nn.utils.clip_grad_norm_(translator.generator.parameters(), self.clip)

        for o in self.optimizers:
            o.step()

        return print_loss

    def train_one_epoch(self):

        losses = []
        for batch in self.dataloader:
            loss = self.step(batch)
            losses.append(loss)

        return np.mean(losses)

    def test(self):
        for batch in self.dataloader:
            src, src_lens, tgt, _, _, max_target_len = batch
            pred_seqs = self.translator.translate(src, src_lens, max_target_len)
            print(tgt[0])
            print(pred_seqs[0])
            break


class BaseValidator:

    def __init__(self, dataloader, translator):
        raise NotImplementedError

    def calc_score(self):
        raise NotImplementedError


class BleuValidator:

    def __init__(self, dataloader, translator):
        self.dataloader = dataloader
        self.translator = translator

    def trim_pad(self, tokens):
        return [token for token in tokens if token != PAD_TOKEN]

    def calc_score(self) -> (float, List):

        scores = []
        src_sents = []
        pred_sents = []
        tgt_sents = []

        for batch in self.dataloader:
            src, src_lens, tgt, _, _, max_target_len = batch
            pred_seqs = self.translator.translate(src, src_lens, max_target_len)

            st2i = self.translator.src_voc
            tt2i = self.translator.tgt_voc

            ti2t = {v: k for k, v in tt2i.items()}
            si2t = {v: k for k, v in st2i.items()}

            for sids, pids, tids in zip(src, pred_seqs, tgt):
                stokens = [si2t.get(int(sid)) for sid in sids]
                ptokens = [ti2t.get(int(pid)) for pid in pids]
                ttokens = [ti2t.get(int(tid)) for tid in tids]

                stokens = self.trim_pad(stokens)
                ptokens = self.trim_pad(ptokens)
                ttokens = self.trim_pad(ttokens)

                ssent = ' '.join(stokens)
                psent = ' '.join(ptokens)
                tsent = ' '.join(ttokens)

                bleu = sentence_bleu([tsent], psent)
                scores.append(bleu)

                src_sents.append(ssent)
                pred_sents.append(psent)
                tgt_sents.append(tsent)

        return np.mean(scores), src_sents, pred_sents, tgt_sents
