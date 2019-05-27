import random

import torch
import torch.nn as nn

from seq2seq_pytorch.tools.evaluator import GreedySearchDecoder
from seq2seq_pytorch.tools.evaluator import evaluate_line


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
        self.criterion = nn.NLLLoss()

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

    def score(self, src, lens, tgt, max_target_len, train=True):
        self.__train(train)

        src = src.to(self.device)
        lens = lens.to(self.device)
        tgt = tgt.to(self.device)

        batch_size = src.size(0)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(
                src, lens, embedding=self.encoder_embedding
                )

        # Prepare to decode
        decoder_input = torch.LongTensor([[self.tgt_voc['<s>']
                                           for _ in range(batch_size)]])
        decoder_input = decoder_input.to(self.device)
        print(encoder_hidden.size())
        print(encoder_outputs.size())
        input()
        decoder_hidden = encoder_hidden[:self.decoder.layers_n]

        use_teacher_forcing =\
            True if random.random() < self.teacher_forcing_ratio else False

        loss = 0
        print_losses = []
        # n_totals = 0

        if train and use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator, self.decoder_embedding
                        )
                decoder_input = tgt[t].view(1, -1)
                # mask_loss, n_total = self.__mask_nllloss(
                #         decoder_output, tgt[t], mask[t])
                _loss = self.criterion(decoder_output, tgt[t])
                loss += _loss
                print_losses.append(_loss.item())
                # n_totals += n_total
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
                # mask_loss, n_total = self.__mask_nllloss(
                #         decoder_output, tgt[t], mask[t])
                loss += self.criterion(decoder_output, tgt[t])
                # print_losses.append(mask_loss.item() * n_total)
                # n_totals += n_total

        # return loss, sum(print_losses) / n_totals
        return loss

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
