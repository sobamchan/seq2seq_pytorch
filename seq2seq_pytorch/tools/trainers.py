import numpy as np

import torch.nn as nn


class Trainer:

    def __init__(self, corpus, optimizers, translator, batch_size, clip):
        self.corpus = corpus
        self.optimizers = optimizers
        self.translator = translator
        self.batch_size = batch_size
        self.clip = clip

    def step(self):
        # Reset gradients
        for o in self.optimizers:
            o.zero_grad()

        # Prepare dataset
        pair_batch = self.corpus.next_batch()

        # Compute loss
        translator = self.translator
        loss, print_loss = translator.score(
                pair_batch,
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
        self.corpus.initialize()
        while not self.corpus.is_final_batch:
            loss = self.step()
            losses.append(loss)

        return np.mean(losses)
