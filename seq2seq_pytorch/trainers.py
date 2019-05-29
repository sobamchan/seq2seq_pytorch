import numpy as np

import torch.nn as nn


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
        src, src_lens, tgt, tgt_lens, max_target_len = batch

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
            src, src_lens, tgt, tgt_lens, max_target_len = batch
            pred_seqs = self.translator.translate(src, src_lens, max_target_len)
            print(tgt[0])
            print(pred_seqs[0])
            break
