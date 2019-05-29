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

    def __mask_nllloss(self, inp, tgt, mask):
        device = self.device

        n_total = mask.sum()
        # cross_entropy = - torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        cross_entropy = self.criterion(inp, tgt)
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

    def score(self, src, src_lens, tgt, tgt_lens, mask, max_target_len, train=True):
        '''
        tgt: [B, S]
        '''
        self.__train(train)

        src = src.to(self.device)
        src_lens = src_lens.to(self.device)
        tgt = tgt.to(self.device)
        tgt_lens = tgt_lens.to(self.device)
        mask = mask.to(self.device)

        bsize = src.size(0)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(
                src, src_lens, embedding=self.encoder_embedding
                )

        # Prepare to decode
        decoder_input = torch.LongTensor([self.tgt_voc['<s>']
                                          for _ in range(bsize)])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden[:self.decoder.layers_n]
        # decoder_hidden: [nlayers, B, H]

        use_teacher_forcing =\
            True if random.random() < self.teacher_forcing_ratio else False

        loss = 0
        print_losses = []
        n_totals = 0

        if train and use_teacher_forcing:
            for t in range(max_target_len):

                # Decode
                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator, self.decoder_embedding
                        )

                decoder_input = tgt[:, t]
                # tgt: [B, S]
                # decoder_input: [B]

                # Calculate loss
                # _loss = self.criterion(decoder_output, tgt[:, t])
                _loss, ntotal = self.__mask_nllloss(decoder_output, tgt[:, t], mask[:, t])
                loss += _loss
                print_losses.append(_loss.item() * ntotal)
                n_totals += ntotal
        else:
            for t in range(max_target_len):

                decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        self.generator, self.decoder_embedding
                        )

                _, decoder_max = decoder_output.max(dim=1)
                decoder_input = torch.LongTensor([decoder_max[i]
                                                 for i in range(bsize)])
                # decoder_input: B
                decoder_input = decoder_input.to(self.device)
                # loss += self.criterion(decoder_output, tgt[:, t])
                _loss, ntotal = self.__mask_nllloss(decoder_output, tgt[:, t], mask[:, t])

                print_losses.append(loss.item() * ntotal)
                n_totals += ntotal

        return loss, sum(print_losses) / torch.sum(tgt_lens).item()

    def translate(self, src, src_lens, max_target_len):
        self.__train(False)
        src = src.to(self.device)
        src_lens = src_lens.to(self.device)
        bsize = src.size(0)

        encoder_outputs, encoder_hidden = self.encoder(
                src, src_lens, embedding=self.encoder_embedding
                )

        decoder_input = torch.LongTensor([
            self.tgt_voc['<s>'] for _ in range(bsize)
            ])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden[:self.decoder.layers_n]

        tgt_seqs = torch.zeros(bsize, max_target_len)
        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs,
                    self.generator, self.decoder_embedding
                    )
            _, decoder_max = decoder_output.max(dim=1)  # [B]
            decoder_input = torch.LongTensor([
                decoder_max[i] for i in range(bsize)
                ])
            decoder_input = decoder_input.to(self.device)

            tgt_seqs[:, t] = decoder_max
        return tgt_seqs
