import os
from datetime import datetime

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from seq2seq_pytorch.dataset import data
from seq2seq_pytorch.models import models
from seq2seq_pytorch.dataset.corpus import CorpusReader
from seq2seq_pytorch.tools.translators import Translator
from seq2seq_pytorch.tools.trainers import Trainer
from seq2seq_pytorch.tools.validators import Validator
from seq2seq_pytorch.tools.validators import BleuValidator


class RunnerBase(object):

    def __init__(self, args):
        raise NotImplementedError

    def train_iters(self):
        raise NotImplementedError

    def dump(self, iteration, best=False):
        raise NotImplementedError


class Runner(RunnerBase):
    '''The Runner class holds information and instances which necessary for
    experiments and run iterations and dump checkpoints
    '''

    def __init__(self, args):
        self.args = args

        device = torch.device('cuda' if args.use_cuda else 'cpu')
        self.device = device

        # Load datas
        src_train_sents, tgt_train_sents, src_valid_sents, tgt_valid_sents,\
            src_voc, tgt_voc =\
            data.main(
                    args.train_src,
                    args.train_tgt,
                    args.valid_src,
                    args.valid_tgt
                    )

        # Limit vocab
        if args.min_count is not None:
            src_voc.trim(args.min_count)
            tgt_voc.trim(args.min_count)
        else:
            if args.src_voc_size is not None:
                src_voc.extract_topk(args.src_voc_size)
            if args.tgt_voc_size is not None:
                tgt_voc.extract_topk(args.tgt_voc_size)

        self.src_voc = src_voc
        self.tgt_voc = tgt_voc

        # Determine directory to save things
        self.corpus_name = args.corpus_name
        save_dir_base = args.save_dir_base
        model_name = args.model_name
        corpus_name = args.corpus_name

        self.save_dir = os.path.join(
                save_dir_base, model_name, corpus_name,
                '{}-{}_{}'.format(
                    args.encoder_layers_n,
                    args.decoder_layers_n,
                    args.hid_n
                    ),
                datetime.now().strftime('%Y%m%d_%H%M%S')
                )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Tensorboard logger preparation
        if args.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_dir)
        else:
            self.tb_writer = None

        # Copy variables
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

        # Build models
        print('Building encoder and decoder...')
        src_embedding = nn.Embedding(self.src_voc.num_words, self.hid_n)
        tgt_embedding = nn.Embedding(self.tgt_voc.num_words, self.hid_n)
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

        # Load checkpoint if there is
        load_filename = args.load_filename
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

            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
            generator.load_state_dict(generator_sd)
            src_embedding.load_state_dict(src_embedding_sd)
            tgt_embedding.load_state_dict(tgt_embedding_sd)

        # Transfer models to the designated device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.generator = generator.to(device)

        # Instantiate optimizers
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

        # Prepare CorpusReader, Translator, Trainer, Validator  classes
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

            # Tensorboard logging
            if self.args.use_tensorboard:
                self.tb_writer.add_scalar(
                        'train/loss', train_loss, i_epoch
                        )

            if i_epoch % self.print_every == 0:
                _valid_loss = self.validator.calc_score()
                _bleu = self.bleu_validator.calc_score()
                print(
                        '%dth epoch: loss -> %f, valid loss -> %f, bleu -> %f'
                        % (i_epoch, train_loss, _valid_loss, _bleu)
                        )

                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.tb_writer.add_scalar(
                            'val/loss', _valid_loss, i_epoch
                            )
                    self.tb_writer.add_scalar(
                            'val/bleu', _bleu, i_epoch
                            )

            if i_epoch % self.save_every == 0:
                dumped_to = self.dump(i_epoch)
                print('Dumped model to %s' % dumped_to)

    def dump(self, iteration, best=False):
        fname_format = '{}_{}_best.pth' if best else '{}_{}.pth'
        fname = fname_format.format(iteration, 'translator')

        torch.save(
                self.train_translator,
                os.path.join(
                    self.save_dir, fname
                    )
                )

        return fname
