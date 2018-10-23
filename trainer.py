import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

import data
import models
from corpus import CorpusReader
from utils import Translator
from utils import Trainer
from utils import Validator

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

MAX_LENGTH = 10


class Runner:

    def __init__(self, args):
        self.args = args
        load_filename = args.load_filename
        self.corpus_name = 'corpus_name'

        src_train_sents, tgt_train_sents, src_valid_sents, tgt_valid_sents,\
            src_voc, tgt_voc =\
            data.main(
                    args.train_src,
                    args.train_tgt,
                    args.valid_src,
                    args.valid_tgt
                    )

        if args.min_count is not None:
            src_voc.trim(args.min_count)
            tgt_voc.trim(args.min_count)
        else:
            if args.src_voc_size is not None:
                src_voc.extract_topk(args.src_voc_size)
            if args.tgt_voc_size is not None:
                tgt_voc.extract_topk(args.tgt_voc_size)

        self.save_dir = args.save_dir

        self.src_voc = src_voc
        self.tgt_voc = tgt_voc

        device = torch.device('cuda' if args.use_cuda else 'cpu')
        self.device = device

        self.model_name = args.model_name
        self.attn_model = args.attn_model
        self.hid_n = args.hid_n
        self.encoder_layers_n = args.encoder_layers_n
        self.decoder_layers_n = args.decoder_layers_n
        self.dropout = args.dropout
        self.batch_size = args.batch_size

        self.clip = args.clip
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.learning_rate = args.lr
        self.decoder_learning_ratio = args.decoder_learning_ratio
        self.iteration_n = args.epoch
        self.print_every = args.print_every
        self.save_every = args.save_every

        self.best_loss = 1e+10

        if load_filename is not None:
            checkpoint = torch.load(load_filename)
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            generator_sd = checkpoint['generator']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            src_embedding_sd = checkpoint['src_embedding']
            tgt_embedding_sd = checkpoint['tgt_embedding']
            src_voc.__dict__ = checkpoint['src_voc_dict']
            tgt_voc.__dict__ = checkpoint['tgt_voc_dict']

        print('Building encoder and decoder...')
        src_embedding = nn.Embedding(self.src_voc.num_words, self.hid_n)
        tgt_embedding = nn.Embedding(self.tgt_voc.num_words, self.hid_n)
        if load_filename:
            src_embedding.load_state_dict(src_embedding_sd)
            tgt_embedding.load_state_dict(tgt_embedding_sd)
        self.src_embedding = src_embedding.to(device)
        self.tgt_embedding = tgt_embedding.to(device)

        encoder = models.EncoderRNN(
                self.hid_n, self.hid_n, self.encoder_layers_n, self.dropout
                )
        decoder = models.LuongAttnDecoderRNN(
                self.attn_model, self.hid_n, self.hid_n,
                self.tgt_voc.num_words, self.decoder_layers_n, self.dropout
                )
        generator = models.LinearGenerator(self.hid_n, self.tgt_voc.num_words)

        if load_filename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
            generator.load_state_dict(generator_sd)

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.generator = generator.to(device)

        encoder_optimizer = optim.Adam(
                encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(
                decoder.parameters(),
                lr=self.learning_rate * self.decoder_learning_ratio
                )
        generator_optimizer = optim.Adam(
                generator.parameters(),
                lr=self.learning_rate * self.decoder_learning_ratio
                )
        src_embedding_optimizer = optim.Adam(
                src_embedding.parameters(),
                lr=self.learning_rate
                )
        tgt_embedding_optimizer = optim.Adam(
                tgt_embedding.parameters(),
                lr=self.learning_rate * self.decoder_learning_ratio
                )

        if load_filename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.generator_optimizer = generator_optimizer
        self.src_embedding_optimizer = src_embedding_optimizer
        self.tgt_embedding_optimizer = tgt_embedding_optimizer

        self.train_corpus = CorpusReader(
                src_train_sents,
                tgt_train_sents,
                src_voc,
                tgt_voc,
                args.batch_size
                )
        self.train_translator = Translator(
                src_embedding,
                tgt_embedding,
                generator,
                src_voc,
                tgt_voc,
                self.encoder,
                self.decoder,
                args.teacher_forcing_ratio,
                device
                )
        self.train_trainer = Trainer(
                self.train_corpus,
                (
                    encoder_optimizer,
                    decoder_optimizer,
                    generator_optimizer,
                    src_embedding_optimizer,
                    tgt_embedding_optimizer,
                    ),
                self.train_translator,
                self.batch_size,
                self.clip
                )

        self.valid_corpus = CorpusReader(
                src_valid_sents,
                tgt_valid_sents,
                src_voc,
                tgt_voc,
                args.batch_size
                )
        self.validator = Validator(
                self.valid_corpus,
                self.train_translator,
                self.batch_size
                )

        if load_filename is not None:
            self.start_iteration = checkpoint['iteration'] + 1
        else:
            self.start_iteration = 1

    def train_one_batch(self, input_variable, lengths, target_variable,
                        mask, max_target_len, max_length=MAX_LENGTH):

        device = self.device
        encoder = self.encoder
        decoder = self.decoder
        generator = self.generator

        encoder.train()
        decoder.train()
        generator.train()

        encoder_optimizer = self.encoder_optimizer
        decoder_optimizer = self.decoder_optimizer
        generator_optimizer = self.generator_optimizer

        batch_size = input_variable.size(1)
        clip = self.clip

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        decoder_input = torch.LongTensor([[SOS_token
                                           for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        decoder_hidden = encoder_hidden[:decoder.layers_n]

        use_teacher_forcing =\
            True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        generator
                        )
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, n_total = self.__mask_nllloss(
                        decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs,
                        generator
                        )
                _, decoder_max = decoder_output.max(dim=1)
                decoder_input = torch.LongTensor([[decoder_max[i]
                                                  for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                mask_loss, n_total = self.__mask_nllloss(
                        decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        loss.backward()

        nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        nn.utils.clip_grad_norm_(generator.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()
        generator_optimizer.step()

        return sum(print_losses) / n_totals

    def calc_valid_loss(self, input_variable, lengths, target_variable,
                        mask, max_target_len, max_length=MAX_LENGTH):

        device = self.device
        encoder = self.encoder
        decoder = self.decoder
        generator = self.generator

        encoder.eval()
        decoder.eval()
        generator.eval()

        batch_size = input_variable.size(1)

        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        decoder_input = torch.LongTensor([[SOS_token
                                           for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        decoder_hidden = encoder_hidden[:decoder.layers_n]

        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs,
                    generator
                    )
            _, decoder_max = decoder_output.max(dim=1)
            decoder_input = torch.LongTensor([[decoder_max[i]
                                              for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, n_total = self.__mask_nllloss(
                    decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

        return sum(print_losses) / n_totals

    def train_iters(self):
        for i_epoch in range(self.iteration_n):
            train_loss = self.train_trainer.train_one_epoch()
            valid_loss = self.validator.calc_losses()
            print('%dth epoch: loss -> %f, valid loss -> %f' % (
                i_epoch, train_loss, valid_loss
                ))

    def dump_checkpoint(self, iteration, best=False):
        save_dir = self.save_dir
        model_name = self.model_name
        corpus_name = self.corpus_name

        directory = os.path.join(
                save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                    self.encoder_layers_n,
                    self.decoder_layers_n,
                    self.hid_n
                    ))

        if not os.path.exists(directory):
            os.makedirs(directory)

        fname_format = '{}_{}_best.tar' if best else '{}_{}.tar'

        torch.save({
            'iteration': iteration,
            'en': self.encoder.state_dict(),
            'de': self.decoder.state_dict(),
            'generator': self.generator.state_dict(),
            'en_opt': self.encoder_optimizer.state_dict(),
            'de_opt': self.decoder_optimizer.state_dict(),
            'src_voc_dict': self.src_voc.__dict__,
            'tgt_voc_dict': self.tgt_voc.__dict__,
            'src_embedding': self.src_embedding.state_dict(),
            'tgt_embedding': self.tgt_embedding.state_dict(),
            'args': self.args
            },
            os.path.join(directory,
                         fname_format.format(iteration, 'checkpoint')))
