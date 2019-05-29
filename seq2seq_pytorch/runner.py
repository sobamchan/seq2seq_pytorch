import torch
import torch.nn as nn
import torch.optim as optim


from seq2seq_pytorch.trainers import Trainer, BleuValidator
from seq2seq_pytorch.translators import Translator
from seq2seq_pytorch.models import models
from seq2seq_pytorch import data


def run():
    hid_n = 300
    encoder_layers_n = 3
    decoder_layers_n = 3
    lr = 0.001
    dropout = 0.5
    attn_model = 'general'
    teacher_forcing_ratio = 1.0
    clip = 50.0
    bsize = 512
    use_cuda = True

    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader, src_t2i, tgt_t2i = data.get('./test', bsize)

    src_embedding = nn.Embedding(len(src_t2i), hid_n).to(device)
    tgt_embedding = nn.Embedding(len(tgt_t2i), hid_n).to(device)

    encoder = models.EncoderRNN(
            hid_n, hid_n, encoder_layers_n, dropout
            ).to(device)
    decoder = models.LuongAttnDecoderRNN(
            attn_model, hid_n, hid_n, len(tgt_t2i),  decoder_layers_n, dropout
            ).to(device)
    generator = models.LinearGenerator(hid_n, len(tgt_t2i)).to(device)

    src_embedding_opt = optim.Adam(src_embedding.parameters(), lr=lr)
    tgt_embedding_opt = optim.Adam(tgt_embedding.parameters(), lr=lr)
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)
    generator_opt = optim.Adam(generator.parameters(), lr=lr)

    train_translator = Translator(
            src_embedding, tgt_embedding, generator, src_t2i, tgt_t2i,
            encoder, decoder, teacher_forcing_ratio, device
            )
    train_trainer = Trainer(
            train_loader,
            (
                src_embedding_opt,
                tgt_embedding_opt,
                encoder_opt,
                decoder_opt,
                generator_opt
                ),
            train_translator,
            clip
            )

    bleu_validator = BleuValidator(
            valid_loader,
            train_translator
            )

    for _ in range(30):
        loss = train_trainer.train_one_epoch()
        bleu, src_sents, pred_sents, tgt_sents = bleu_validator.calc_score()
        print(f'Loss: {loss}')
        print(f'Bleu: {bleu}')

        print(f'Source sentence: {src_sents[1]}')
        print(f'Generated sentence: {pred_sents[1]}')
        print(f'Target sentence: {tgt_sents[1]}')


if __name__ == '__main__':
    run()
