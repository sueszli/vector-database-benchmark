from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sentencepiece as spm

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='sentencepiece model to use for decoding')
    parser.add_argument('--input', required=True, help='input file to decode')
    parser.add_argument('--input_format', choices=['piece', 'id'], default='piece')
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)
    if args.input_format == 'piece':

        def decode(input):
            if False:
                return 10
            return ''.join(sp.DecodePieces(input))
    elif args.input_format == 'id':

        def decode(input):
            if False:
                for i in range(10):
                    print('nop')
            return ''.join(sp.DecodeIds(input))
    else:
        raise NotImplementedError

    def tok2int(tok):
        if False:
            i = 10
            return i + 15
        return int(tok) if tok != '<<unk>>' else 0
    with open(args.input, 'r', encoding='utf-8') as h:
        for line in h:
            if args.input_format == 'id':
                print(decode(list(map(tok2int, line.rstrip().split()))))
            elif args.input_format == 'piece':
                print(decode(line.rstrip().split()))
if __name__ == '__main__':
    main()