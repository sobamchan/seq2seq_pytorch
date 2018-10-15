import re
import unicodedata

import torch
import torch.nn as nn

SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10


class GreedySearchDecoder(nn.Module):

    def __init__(self, device, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inp_seq, inp_lengths, max_length):
        device = self.device

        encoder_outputs, encoder_hidden = self.encoder(inp_seq, inp_lengths)
        decoder_hidden = encoder_hidden[:self.decoder.layers_n]
        decoder_input =\
            torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden =\
                self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores


def indexes_from_sentence(voc, sentence):
    print(sentence)
    return [voc.w2i.get(word, 'UNK')
            for word in sentence.split(' ')] + [EOS_token]


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # Comment this line when you use Japanese
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            )


def evaluate(device, encoder, decoder, searcher,
             src_voc, tgt_voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexes_from_sentence(src_voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [tgt_voc.i2w.get(token.item(), 'UNK') for token in tokens]
    return decoded_words


def evaluate_input(device, encoder, decoder, searcher, src_voc, tgt_voc):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            input_sentence = normalize_string(input_sentence)
            output_words = evaluate(
                    device, encoder, decoder, searcher, src_voc,
                    tgt_voc, input_sentence
                    )
            output_words[:] = [x
                               for x in output_words
                               if not (x == 'EOS' or x == 'PAD')]
            output_words[:] = [x
                               for x in output_words
                               if not (x == 'EOS' or x == 'PAD')]
            print('BOT: ', ' '.join(output_words))
        except KeyError:
            print('ERROR: Encountered unknown word.')
