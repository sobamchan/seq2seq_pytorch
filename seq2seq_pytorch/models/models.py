import torch
import torch.nn.functional as F
import torch.nn as nn

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
                bidirectional=True
                )

    def forward(self, inp_seq, inp_lengths, embedding, hid=None):
        embedded = embedding(inp_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inp_lengths)
        outputs, hidden = self.gru(packed, hid)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hid_n] + outputs[:, :, self.hid_n:]

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

    def dot_score(self, hid, encoder_output):
        return torch.sum(hid * encoder_output, dim=2)

    def general_score(self, hid, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hid * energy, dim=2)

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

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LinearGenerator(nn.Module):

    def __init__(self, hid_n, output_size):
        super(LinearGenerator, self).__init__()
        self.out = nn.Linear(hid_n, output_size)
        self.logsoftmax = nn.Softmax(dim=1)

    def forward(self, hid):
        return self.logsoftmax(self.out(hid))


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
                dropout=(0 if layers_n == 1 else dropout)
                )
        self.concat = nn.Linear(hid_n * 2, hid_n)
        self.attn = Attn(attn_model, hid_n)

    def forward(self, input_step, last_hid, encoder_outputs,
                generator, embedding):
        # Note: we run this one step (word) at a time
        embedded = embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hid = self.gru(embedded, last_hid)
        attn_weights = self.attn(rnn_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = generator(concat_output)

        return output, hid
