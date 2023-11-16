import argparse
import sys
from fairseq.data import Dictionary

def get_parser():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='filters a lexicon given a unit dictionary')
    parser.add_argument('-d', '--unit-dict', help='unit dictionary', required=True)
    return parser

def main():
    if False:
        while True:
            i = 10
    parser = get_parser()
    args = parser.parse_args()
    d = Dictionary.load(args.unit_dict)
    symbols = set(d.symbols)
    for line in sys.stdin:
        items = line.rstrip().split()
        skip = len(items) < 2
        for x in items[1:]:
            if x not in symbols:
                skip = True
                break
        if not skip:
            print(line, end='')
if __name__ == '__main__':
    main()