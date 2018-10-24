import os

import torch
import torch.nn as nn
import torch.optim as optim

import data
import models
from corpus import CorpusReader
from utils import Translator
from utils import Trainer
from utils import Validator
from utils import BleuValidator


class Runner:

    def __init__(self, args):
        self.args = args
        load_filename = args.load_filename
        self.corpus_name = args.corpus_name

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
        self.bleu_validator = BleuValidator(
                self.valid_corpus,
                self.train_translator,
                self.batch_size
                )

        if load_filename is not None:
            self.start_iteration = checkpoint['iteration'] + 1
        else:
            self.start_iteration = 1

    def train_iters(self):
        '''Main training loop'''

        for i_epoch in range(1, self.iteration_n + 1):
            train_loss = self.train_trainer.train_one_epoch()

            if i_epoch % self.print_every == 0:
                _valid_loss = self.validator.calc_score()
                _bleu = self.bleu_validator.calc_score()
                print(
                        '%dth epoch: loss -> %f, valid loss -> %f, bleu -> %f'
                        % (i_epoch, train_loss, _valid_loss, _bleu)
                        )

            if i_epoch % self.save_every == 0:
                dumped_to = self.dump(i_epoch)
                print('Dumped model to %s' % dumped_to)

    def dump(self, iteration, best=False):
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

        fname_format = '{}_{}_best.pth' if best else '{}_{}.pth'
        fname = fname_format.format(iteration, 'translator')

        torch.save(
                self.train_translator,
                os.path.join(
                    directory, fname
                    )
                )

        return fname
