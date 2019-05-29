import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

PAD_token = 0
SOS_token = 1
EOS_token = 2
MIN_COUNT = 3
MAX_LENGTH = 10


class EncoderRNN(nn.Module):

    def __init__(self, hid_n, emb_size, layers_n=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.layers_n = layers_n
        self.hid_n = hid_n
        self.emb_size = emb_size

        self.gru = nn.GRU(
                hid_n,
                hid_n,
                layers_n,
                dropout=(0 if layers_n == 1 else dropout),
                bidirectional=True,
                batch_first=True
                )

    def forward(self, inp_seq, inp_lengths, embedding, hid=None):
        '''
        IN:
          - inp_seq: [B, S]
          - inp_lengths: [B]
        '''
        embedded = embedding(inp_seq)  # [B, S, H]
        packed = pack_padded_sequence(embedded, inp_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hid)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # outputs: [B, S, bidirectional * H]
        # hidden: [bidirectional * nlayers, B, H]

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hid_n] + outputs[:, :, self.hid_n:]
        # outputs: [B, S, H]

        return outputs, hidden


class Attn(nn.Module):

    def __init__(self, method, hid_n):
        super(Attn, self).__init__()
        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             ' is not an appropriate attention method.')

        self.hid_n = hid_n
        if self.method == 'general':
            self.attn = nn.Linear(hid_n, hid_n)
        elif self.method == 'concat':
            self.attn = nn.Linear(hid_n * 2, hid_n)
            self.v = nn.Parameter(torch.FloatTensor(hid_n))

    def dot_score(self, hid, encoder_outputs):
        '''
        hid: [1, B, H]
        encoder_outputs: [B, S, H]
        '''
        _mid = hid * encoder_outputs  # [B, S, H]
        return torch.sum(_mid, dim=2)

    def general_score(self, hid, encoder_outputs):
        '''
        hid: [B, 1, bidirectional * H]
        encoder_outputs: [B, S, H]
        '''
        energy = self.attn(encoder_outputs)  # [B, S, bidirectional * H]
        _mid = hid * energy  # [B, S, bidirectional * H]
        return torch.sum(_mid, dim=2)  # [B, S]

    def concat_score(self, hid, encoder_output):
        energy = self.attn(torch.cat((hid.expand(encoder_output.size(0),
                                                 -1, -1),
                                      encoder_output),
                                     2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hid, encoder_outputs):

        if self.method == 'general':
            attn_energies = self.general_score(hid, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hid, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hid, encoder_outputs)

        # attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LinearGenerator(nn.Module):

    def __init__(self, hid_n, output_size):
        super(LinearGenerator, self).__init__()
        self.out = nn.Linear(hid_n, output_size)
        self.logsoftmax = nn.Softmax(dim=1)

    def forward(self, hid):
        return self.out(hid)
        # return self.logsoftmax(self.out(hid))


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, emb_size, hid_n,
                 output_size, layers_n=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hid_n = hid_n
        self.output_size = output_size
        self.layers_n = layers_n
        self.dropout = dropout

        self.emb_size = emb_size
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
                hid_n,
                hid_n,
                layers_n,
                dropout=(0 if layers_n == 1 else dropout),
                batch_first=True
                )
        self.concat = nn.Linear(hid_n * 2, hid_n)
        self.attn = Attn(attn_model, hid_n)

    def forward(self, input_step, last_hid, encoder_outputs,
                generator, embedding):
        '''
        Note: One step (word) at a time.

        IN:
          - input_step: [B]
          - last_hid: [nlayers_dec, B, H]
          - encoder_outputs: [B, S, H]
        OUT:
          - output: [B, V_tgt]
          - hid: [1, B, H]
        '''
        embedded = embedding(input_step)  # [B, H]
        embedded = embedded.unsqueeze(1)  # [B, 1, H]
        embedded = self.embedding_dropout(embedded)

        rnn_output, hid = self.gru(embedded, last_hid)  # [B, 1, bidirectional * H], [nlayers * bidirectional, B, H]
        attn_weights = self.attn(rnn_output, encoder_outputs)  # [B, 1, bidirectional * H] * [B, S, H] -> [B, 1, S]

        context = attn_weights.bmm(encoder_outputs)  # [B, 1, S] * [B, S, H] -> [B, 1, H]

        rnn_output = rnn_output.squeeze(1)  # [B, H]
        context = context.squeeze(1)  # [B, H]
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = generator(concat_output)  # [B, V_tgt]
        output = F.log_softmax(output, dim=1)
        # hid: [1, B, H]
        return output, hid
