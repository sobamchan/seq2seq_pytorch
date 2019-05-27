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
        src, lens, tgt, max_target_len = batch

        # Reset gradients
        for o in self.optimizers:
            o.zero_grad()

        # Compute loss
        translator = self.translator
        # loss, print_loss = translator.score(
        #         pair_batch,
        #         train=True
        #         )
        loss = translator.score(
                src,
                lens,
                tgt,
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

        # return print_loss
        return loss.item()

    def train_one_epoch(self):

        losses = []
        for batch in self.dataloader:
            loss = self.step(batch)
            losses.append(loss)

        return np.mean(losses)
