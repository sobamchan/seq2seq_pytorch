import random
import itertools

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn

from seq2seq_pytorch.tools.evaluator import GreedySearchDecoder
from seq2seq_pytorch.tools.evaluator import evaluate_line


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


class Validator:

    def __init__(self, corpus, translator, batch_size):
        self.corpus = corpus
        self.translator = translator
        self.batch_size = batch_size

    def __calc_loss(self):

        # Get batch
        pair_batch = self.corpus.next_batch()

        # Compute loss
        _, print_loss = self.translator.score(
                pair_batch,
                train=False
                )

        return print_loss

    def calc_score(self):

        losses = []
        self.corpus.initialize()
        while not self.corpus.is_final_batch:
            loss = self.__calc_loss()
            losses.append(loss)

        return np.mean(losses)


class BleuValidator:

    def __init__(self, corpus, translator, batch_size):
        self.corpus = corpus
        self.translator = translator
        self.batch_size = batch_size

    def __calc_loss(self):

        # Get batch
        pair_batch = self.corpus.next_batch()
        src, tgt = zip(*pair_batch)

        # Compute loss
        pred_sents = self.translator.greedy(src)

        bleus = []
        for s, t in zip(pred_sents, tgt):
            bleu = sentence_bleu([t], s)
            bleus.append(bleu)

        return np.mean(bleus)

    def calc_score(self):

        bleus = []
        self.corpus.initialize()
        while not self.corpus.is_final_batch:
            bleu = self.__calc_loss()
            bleus.append(bleu)

        return np.mean(bleus)


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

    def score(self, pair_batch, train=True):
        self.__train(train)

        src, lens, tgt, mask, max_target_len =\
            batch_to_train_data(self.src_voc, self.tgt_voc, pair_batch)

        src = src.to(self.device)
        lens = lens.to(self.device)
        tgt = tgt.to(self.device)
        mask = mask.to(self.device)

        batch_size = src.size(1)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(
                src, lens, embedding=self.encoder_embedding
                )

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

        if train and use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator, self.decoder_embedding
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
                        self.generator, self.decoder_embedding
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

        return loss, sum(print_losses) / n_totals

    def greedy(self, sentences):
        searcher = GreedySearchDecoder(
                self.device,
                self.encoder_embedding,
                self.decoder_embedding,
                self.generator,
                self.encoder,
                self.decoder
                )
        pred_sentences = [
                evaluate_line(
                    self.device,
                    searcher,
                    self.src_voc,
                    self.tgt_voc,
                    sentence
                    )
                for sentence in sentences]
        return pred_sentences


def zero_padding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def input_var(l, voc):
    indexes_batch = [voc.sent2idx(sentence)
                     for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch, voc.w2i['PAD'])
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def output_var(l, voc):
    indexes_batch = [voc.sent2idx(sentence)
                     for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch, voc.w2i['PAD'])
    mask = binary_matrix(pad_list, voc.w2i['PAD'])
    mask = torch.ByteTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


def batch_to_train_data(src_voc, tgt_voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, src_voc)
    output, mask, max_target_len = output_var(output_batch, tgt_voc)
    return inp, lengths, output, mask, max_target_len
