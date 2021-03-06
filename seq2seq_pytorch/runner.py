from pathlib import Path

import fire

import torch
import torch.nn as nn
import torch.optim as optim


from seq2seq_pytorch.trainers import Trainer, BleuValidator, LossValidator
from seq2seq_pytorch.translators import Translator
from seq2seq_pytorch import models
from seq2seq_pytorch import data


def run(hid_n: int = 300, encoder_layers_n: int = 3, decoder_layers_n: int = 3,
        lr: float = 0.001, dropout: float = 0.5, attn_model: str = 'general', teacher_forcing_ratio: float = 1.0,
        clip: float = 50.0, vocab_size: int = 5000, bsize: int = 512, epoch: int = 50,
        use_cuda: bool = True, data_cache_path: str = './test', savedir: str = None, save_every: int = 5):

    savedir = Path(savedir) if savedir else None

    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, valid_loader, src_t2i, tgt_t2i = data.get(data_cache_path, bsize, vocab_size)

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

    loss_validator = LossValidator(
            valid_loader,
            train_translator
            )

    best_bleu = 0

    for iepoch in range(1, epoch + 1):
        loss = train_trainer.train_one_epoch()
        bleu, src_sents, pred_sents, tgt_sents = bleu_validator.calc_score()
        valid_loss = loss_validator.calc_score()

        print(f'Loss: {loss}')
        print(f'Bleu: {bleu}')
        print(f'Valid loss: {valid_loss}')

        print(f'Source sentence: {src_sents[1]}')
        print(f'Generated sentence: {pred_sents[1]}')
        print(f'Target sentence: {tgt_sents[1]}')

        if best_bleu < bleu:
            best_bleu = bleu
            print('Achived best bleu!')
            # Dump
            dump(train_translator, savedir, f'translator.best.pth')

        if iepoch % save_every == 0:
            # Dump
            dump(train_translator, savedir, f'translator.{iepoch}.pth')


def dump(translator: Translator, savedir: Path, fname: str):
    if savedir:
        savepath = savedir / fname
        print(f'Dumping translator to {savepath}')
        torch.save(translator, savepath)
    else:
        print('No savedir is gave. Skip dumping...')


if __name__ == '__main__':
    fire.Fire()
