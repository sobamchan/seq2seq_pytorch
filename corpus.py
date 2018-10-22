import random
import itertools

import torch


class CorpusReader:

    def __init__(self, src_sents, tgt_sents, src_voc, tgt_voc, batch_size):
        self.pairs = list(zip(src_sents, tgt_sents))
        self.pairs_n = len(self.pairs)

        self.src_voc = src_voc
        self.tgt_voc = tgt_voc

        self.current_idx = 0
        self.batch_size = batch_size

    def prepare(self, sents):
        pass

    def initialize(self):
        self.current_idx = 0
        random.shuffle(self.pairs)

    def __zero_padding(self, l, fillvalue):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def __binary_matrix(self, l, value):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == value:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def __input_var(self, l, voc):
        indexes_batch = [voc.sent2idx(sentence)
                         for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = self.__zero_padding(indexes_batch, voc.w2i['PAD'])
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    def __output_var(self, l, voc):
        indexes_batch = [voc.sent2idx(sentence)
                         for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = self.__zero_padding(indexes_batch, voc.w2i['PAD'])
        mask = self.__binary_matrix(pad_list, voc.w2i['PAD'])
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

    def next_batch(self):
        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        self.current_idx += self.batch_size

        is_final_batch = False

        if end_idx > self.pairs_n:
            end_idx = self.pairs_n
            self.initialize()
            is_final_batch = True

        pair_batch = self.pairs[start_idx:end_idx]
        src, lens, tgt, mask, max_target_len =\
            self.__batch_to_train_data(self.src_voc, self.tgt_voc, pair_batch)

        return src, lens, tgt, mask, max_target_len, is_final_batch
