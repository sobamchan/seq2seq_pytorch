import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
            '-translator', type=str, required=True,
            help='Path to translator pth path'
            )
    p.add_argument('-use_cuda', action='store_true')
    p.add_argument('-input_file', type=str, default=None)
    p.add_argument('-output_file', type=str, default=None)
    args = p.parse_args()

    translator = torch.load(args.translator)

    pred_sents = []
    lines = open(args.input_file, 'r').readlines()
    pred_sents = translator.greedy(lines)

    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            f.write('\n'.join(pred_sents))
    else:
        print('\n'.join(pred_sents))


if __name__ == '__main__':
    main()
