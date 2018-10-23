import argparse

from trainer import Runner


def get_args():
    p = argparse.ArgumentParser('Pytorch chatbot')

    # Basic config
    p.add_argument('-use_cuda', action='store_true')

    # Data config
    p.add_argument('-corpus_name', type=str, default='corpus_name')
    p.add_argument('-train_src', type=str)
    p.add_argument('-train_tgt', type=str)
    p.add_argument('-valid_src', type=str)
    p.add_argument('-valid_tgt', type=str)
    p.add_argument('-load_filename', type=str, default=None)
    p.add_argument('-save_dir', type=str, required=True)
    p.add_argument('-min_count', type=int, default=None)
    p.add_argument('-src_voc_size', type=int, default=None)
    p.add_argument('-tgt_voc_size', type=int, default=None)

    # Model config
    p.add_argument('-model_name', type=str, default='cb_model')
    p.add_argument(
            '-attn_model', type=str, default='dot',
            help='Attention type: [dot, general, concat]'
            )
    p.add_argument('-hid_n', type=int, default=500)
    p.add_argument('-encoder_layers_n', type=int, default=2)
    p.add_argument('-decoder_layers_n', type=int, default=2)
    p.add_argument('-dropout', type=float, default=0.1)

    # Training config
    p.add_argument('-batch_size', type=int, default=64)
    p.add_argument('-clip', type=float, default=50.0)
    p.add_argument('-teacher_forcing_ratio', type=float, default=1.0)
    p.add_argument('-lr', type=float, default=0.0001)
    p.add_argument('-decoder_learning_ratio', type=float, default=5.0)
    p.add_argument('-epoch', type=int, default=4000)

    # Visualize config
    p.add_argument('-print_every', type=int, default=1)
    p.add_argument('-save_every', type=int, default=5)

    return p.parse_args()


def main(args):
    r = Runner(args)
    r.train_iters()


if __name__ == '__main__':
    args = get_args()
    main(args)
