import numpy as np
from nltk.translate.bleu_score import sentence_bleu


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
