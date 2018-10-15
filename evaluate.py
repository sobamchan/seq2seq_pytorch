import argparse

import torch
import torch.nn as nn

import models
from data import Voc
from evaluator import evaluate_input
from evaluator import GreedySearchDecoder


def get_args():
    p = argparse.ArgumentParser('Interaction mode')
    p.add_argument(
            '-checkpoint',
            type=str,
            help='Path to checkpoint',
            required=True
            )
    p.add_argument('-use_cuda', action='store_true')
    return p.parse_args()


def main(args):
    checkpoint = torch.load(args.checkpoint)

    src_voc = Voc('eval')
    tgt_voc = Voc('eval')
    src_voc.__dict__ = checkpoint['src_voc_dict']
    tgt_voc.__dict__ = checkpoint['tgt_voc_dict']

    src_embedding = nn.Embedding(src_voc.num_words, checkpoint['args'].hid_n)
    tgt_embedding = nn.Embedding(tgt_voc.num_words, checkpoint['args'].hid_n)
    encoder = models.EncoderRNN(
            checkpoint['args'].hid_n,
            src_embedding,
            checkpoint['args'].encoder_layers_n,
            checkpoint['args'].dropout
            )
    decoder = models.LuongAttnDecoderRNN(
            checkpoint['args'].attn_model,
            tgt_embedding,
            checkpoint['args'].hid_n,
            tgt_voc.num_words,
            checkpoint['args'].decoder_layers_n,
            checkpoint['args'].dropout
            )

    src_embedding.load_state_dict(checkpoint['src_embedding'])
    tgt_embedding.load_state_dict(checkpoint['tgt_embedding'])
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    src_embedding = src_embedding.to(device)
    tgt_embedding = tgt_embedding.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    searcher = GreedySearchDecoder(device, encoder, decoder)
    evaluate_input(device, encoder, decoder, searcher, src_voc, tgt_voc)


if __name__ == '__main__':
    args = get_args()
    main(args)
