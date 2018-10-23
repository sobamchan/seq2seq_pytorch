import random


class CorpusReader:

    def __init__(self, src_sents, tgt_sents, src_voc, tgt_voc, batch_size):
        self.pairs = list(zip(src_sents, tgt_sents))
        self.pairs_n = len(self.pairs)

        self.src_voc = src_voc
        self.tgt_voc = tgt_voc

        self.batch_size = batch_size

        self.initialize()

    def initialize(self):
        self.current_idx = 0
        self.is_final_batch = False
        random.shuffle(self.pairs)

    def next_batch(self):
        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        self.current_idx += self.batch_size

        if end_idx > self.pairs_n:
            end_idx = self.pairs_n
            self.initialize()
            self.is_final_batch = True

        pair_batch = self.pairs[start_idx:end_idx]
        return pair_batch
