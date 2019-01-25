import itertools

import torch


def zero_padding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def input_var(l, voc):
    indexes_batch = [voc.sent2idx(sentence)
                     for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch, voc.w2i['PAD'])
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def output_var(l, voc):
    indexes_batch = [voc.sent2idx(sentence)
                     for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch, voc.w2i['PAD'])
    mask = binary_matrix(pad_list, voc.w2i['PAD'])
    mask = torch.ByteTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


def batch_to_train_data(src_voc, tgt_voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, src_voc)
    output, mask, max_target_len = output_var(output_batch, tgt_voc)
    return inp, lengths, output, mask, max_target_len
