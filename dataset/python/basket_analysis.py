__author__ = 'nastra'

from collections import defaultdict
from itertools import chain


def load_transactions():
    dataset = [[int(tok) for tok in line.strip().split()] for line in open('data/retail.dat')]
    counts = defaultdict(int)
    for elem in chain(*dataset):
        counts[elem] += 1

    return dataset, counts


if __name__ == '__main__':
    dataset, counts = load_transactions()

    