import os
import random
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

import data
import models

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

MAX_LENGTH = 10


class Trainer:

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

        train_pairs = list(zip(src_train_sents, tgt_train_sents))
        valid_pairs = list(zip(src_valid_sents, tgt_valid_sents))

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
        self.train_pairs = train_pairs
        self.valid_pairs = valid_pairs

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
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding

        encoder = models.EncoderRNN(
                self.hid_n, src_embedding, self.encoder_layers_n, self.dropout
                )
        decoder = models.LuongAttnDecoderRNN(
                self.attn_model, tgt_embedding, self.hid_n,
                self.tgt_voc.num_words, self.decoder_layers_n, self.dropout
                )
        if load_filename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        encoder_optimizer = optim.Adam(
                encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(
                decoder.parameters(),
                lr=self.learning_rate * self.decoder_learning_ratio
                )
        if load_filename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        if load_filename is not None:
            self.start_iteration = checkpoint['iteration'] + 1
        else:
            self.start_iteration = 1

    def __indexes_from_sentence(self, voc, sentence):
        return [voc.w2i.get(word, voc.w2i['UNK'])
                for word in sentence.split(' ')] + [EOS_token]

    def __zero_padding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def __binary_matrix(self, l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def __input_var(self, l, voc):
        indexes_batch = [self.__indexes_from_sentence(voc, sentence)
                         for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = self.__zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    def __output_var(self, l, voc):
        indexes_batch = [self.__indexes_from_sentence(voc, sentence)
                         for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = self.__zero_padding(indexes_batch)
        mask = self.__binary_matrix(pad_list)
        mask = torch.ByteTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_len

    def __batch_to_train_data(self, src_voc, tgt_voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.__input_var(input_batch, src_voc)
        output, mask, max_target_len = self.__output_var(output_batch, tgt_voc)
        return inp, lengths, output, mask, max_target_len

    def __mask_nllloss(self, inp, target, mask):
        device = self.device

        n_total = mask.sum()
        cross_entropy = - torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, n_total.item()

    def train_one_batch(self, input_variable, lengths, target_variable,
                        mask, max_target_len, max_length=MAX_LENGTH):

        device = self.device
        encoder = self.encoder
        decoder = self.decoder

        encoder.train()
        decoder.train()

        encoder_optimizer = self.encoder_optimizer
        decoder_optimizer = self.decoder_optimizer
        batch_size = input_variable.size(1)
        clip = self.clip

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

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
                        decoder_input, decoder_hidden, encoder_outputs
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
                        decoder_input, decoder_hidden, encoder_outputs
                        )
                # _, topi = decoder_output.topk(1)
                _, decoder_max = decoder_output.max(dim=1)
                # decoder_input =\
                #     torch.LongTensor([[topi[i][0]
                #                        for i in range(batch_size)]])
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

        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def calc_valid_loss(self, input_variable, lengths, target_variable,
                        mask, max_target_len, max_length=MAX_LENGTH):

        device = self.device
        encoder = self.encoder
        decoder = self.decoder

        encoder.eval()
        decoder.eval()

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
                    decoder_input, decoder_hidden, encoder_outputs
                    )
            # _, topi = decoder_output.topk(1)
            _, decoder_max = decoder_output.max(dim=1)
            # decoder_input = torch.LongTensor([[topi[i][0]
            #                                   for i in range(batch_size)]])
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
        src_voc = self.src_voc
        tgt_voc = self.tgt_voc
        pairs = self.train_pairs

        batch_size = self.batch_size
        iteration_n = self.iteration_n
        print_every = self.print_every
        save_every = self.save_every

        # training_batches =\
        #     [self.__batch_to_train_data(
        #         src_voc, tgt_voc, [random.choice(pairs)
        #                            for _ in range(batch_size)])
        #      for _ in range(iteration_n)]

        print('Initializing...')
        start_iteration = 1
        train_print_loss = 0
        valid_print_loss = 0

        start_iteration = self.start_iteration

        print('Training...')
        for iteration in range(start_iteration, iteration_n + 1):
            # training_batch = training_batches[iteration - 1]
            # input_variable, lengths, target_variable, mask, max_target_len =\
            #     training_batch

            loss_batch = 0
            batch_n = 0
            random.shuffle(pairs)
            for start_idx in range(0, len(pairs), batch_size):

                # Select samples for batch
                end_idx = start_idx + batch_size
                end_idx = end_idx if end_idx < len(pairs) else len(pairs)
                pairs_batch = pairs[start_idx:end_idx]
                training_batch = self.__batch_to_train_data(
                        src_voc, tgt_voc, pairs_batch
                        )
                input_variable, lengths, target_variable,\
                    mask, max_target_len = training_batch

                loss = self.train_one_batch(
                        input_variable, lengths, target_variable,
                        mask, max_target_len
                        )
                loss_batch += loss
                batch_n += 1

            ave_loss = loss_batch / batch_n
            train_print_loss += ave_loss

            if iteration % print_every == 0:
                valid_batch =\
                    self.__batch_to_train_data(
                        src_voc, tgt_voc, self.valid_pairs)
                input_variable, lengths,\
                    target_variable, mask, max_target_len = valid_batch

                loss = self.calc_valid_loss(
                        input_variable, lengths, target_variable,
                        mask, max_target_len
                        )
                valid_print_loss += loss

                if valid_print_loss < self.best_loss:
                    print('Best loss achived!')
                    self.best_loss = valid_print_loss
                    self.dump_checkpoint(iteration, best=True)

                print_loss_avg = train_print_loss / print_every
                print(
                        'Iteration: {}; Percent complete: {:.1f}%;\
                        Average loss {:.5f}; Valid loss {:.5f}'.format(
                            iteration,
                            iteration / iteration_n * 100,
                            print_loss_avg,
                            valid_print_loss
                            )
                        )
                train_print_loss = 0
                valid_print_loss = 0

            if (iteration % save_every == 0):
                self.dump_checkpoint(iteration)
                # directory = os.path.join(
                #         save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                #             self.encoder_layers_n,
                #             self.decoder_layers_n,
                #             self.hid_n
                #             ))
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # torch.save({
                #     'iteration': iteration,
                #     'en': encoder.state_dict(),
                #     'de': decoder.state_dict(),
                #     'en_opt': encoder_optimizer.state_dict(),
                #     'de_opt': decoder_optimizer.state_dict(),
                #     'loss': loss,
                #     'src_voc_dict': src_voc.__dict__,
                #     'tgt_voc_dict': tgt_voc.__dict__,
                #     'src_embedding': src_embedding.state_dict(),
                #     'tgt_embedding': tgt_embedding.state_dict(),
                #     'args': self.args
                #     },
                #     os.path.join(directory,
                #                  '{}_{}.tar'.format(iteration,
                #                                     'checkpoint')))

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
            'en_opt': self.encoder_optimizer.state_dict(),
            'de_opt': self.decoder_optimizer.state_dict(),
            # 'loss': loss,
            'src_voc_dict': self.src_voc.__dict__,
            'tgt_voc_dict': self.tgt_voc.__dict__,
            'src_embedding': self.src_embedding.state_dict(),
            'tgt_embedding': self.tgt_embedding.state_dict(),
            'args': self.args
            },
            os.path.join(directory,
                         fname_format.format(iteration, 'checkpoint')))
