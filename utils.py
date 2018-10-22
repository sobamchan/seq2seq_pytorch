import random

import torch
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
        src, lens, tgt, mask, max_target_len, is_final =\
            self.corpus.next_batch()

        # Compute loss
        translator = self.translator
        loss = translator.score(
                src=src,
                lens=lens,
                tgt=tgt,
                mask=mask,
                max_target_len=max_target_len,
                train=True
                )

        # Clipping, backpropagate, optimize
        loss.backward()

        nn.utils.clip_grad_norm_(translator.encoder, self.clip)
        nn.utils.clip_grad_norm_(translator.decoder, self.clip)
        nn.utils.clip_grad_norm_(translator.encoder_embedding, self.clip)
        nn.utils.clip_grad_norm_(translator.decoder_embedding, self.clip)
        nn.utils.clip_grad_norm_(translator.generator, self.clip)

        for o in self.optimizers:
            o.step()


class Translator:

    def __init__(self, encoder_embedding, decoder_embedding, generator,
                 src_voc, tgt_voc, encoder, decoder, teacher_forcing_ratio,
                 device):
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.generator = generator
        self.src_voc = src_voc
        self.tgt_voc = tgt_voc
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.teacher_forcing_ratio = teacher_forcing_ratio

    def __mask_nllloss(self, inp, target, mask):
        device = self.device

        n_total = mask.sum()
        cross_entropy = - torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, n_total.item()

    def __train(self, train=True):
        if train:
            self.encoder_embedding.train()
            self.decoder_embedding.train()
            self.encoder.train()
            self.decoder.train()
            self.generator.train()
        else:
            self.encoder_embedding.eval()
            self.decoder_embedding.eval()
            self.encoder.eval()
            self.decoder.eval()
            self.generator.eval()

    def score(self, src, lens, tgt, mask=None, max_target_len=20, train=True):
        self.__train(train)
        batch_size = src.size(1)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src, lens)

        # Prepare to decode
        decoder_input = torch.LongTensor([[self.tgt_voc.w2i['SOS']
                                           for _ in range(batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden[:self.decoder.layers_n]

        use_teacher_forcing =\
            True if random.random() < self.teacher_forcing_ratio else False

        loss = 0
        print_losses = []
        n_totals = 0

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator
                        )
                decoder_input = tgt[t].view(1, -1)
                mask_loss, n_total = self.__mask_nllloss(
                        decoder_output, tgt[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator
                        )
                _, decoder_max = decoder_output.max(dim=1)
                decoder_input = torch.LongTensor([[decoder_max[i]
                                                  for i in range(batch_size)]])
                decoder_input = decoder_input.to(self.device)
                mask_loss, n_total = self.__mask_nllloss(
                        decoder_output, tgt[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        return loss

    def greedy(self, sentences):
        pass


class Corpus:
    pass
