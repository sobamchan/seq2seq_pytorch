import re
import unicodedata

PAD_token = 0
SOS_token = 1
EOS_token = 2
MIN_COUNT = 3
MAX_LENGTH = 10


class Voc:

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.w2i = {}
        self.w2c = {}
        self.i2w = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.num_words
            self.w2c[word] = 1
            self.i2w[self.num_words] = word
            self.num_words += 1
        else:
            self.w2c[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.w2c.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.w2i), len(keep_words) / len(self.w2i)
            ))

        self.w2i = {}
        self.w2c = {}
        self.i2w = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)


def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # Comment out this line when you use Japanese
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def read_vocs(datafile, voc_name='voc'):
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    sents = [normalize_string(l) for l in lines]
    voc = Voc(voc_name)
    return voc, sents


def filter_pair(p):
    return (len(p[0].split(' ')) < MAX_LENGTH)\
            and\
           (len(p[1].split(' ')) < MAX_LENGTH)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name, datafile):
    print('Start preparing training data...')

    voc, sents = read_vocs(datafile, corpus_name)
    print('Read {!s} sentences'.format(len(sents)))

    print('Counting words...')
    for sent in sents:
        voc.add_sentence(sent)
    print('Counted words: ', voc.num_words)
    return voc, sents


def main(train_src, train_tgt, valid_src, valid_tgt):
    src_voc, src_train_sents = load_prepare_data('train_src', train_src)
    tgt_voc, tgt_train_sents = load_prepare_data('train_tgt', train_tgt)
    _, src_valid_sents = load_prepare_data('valid_src', valid_src)
    _, tgt_valid_sents = load_prepare_data('valid_tgt', valid_tgt)

    return (
            src_train_sents, tgt_train_sents, src_valid_sents, tgt_valid_sents,
            src_voc, tgt_voc
            )