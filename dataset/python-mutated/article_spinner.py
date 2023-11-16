from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import nltk
import random
import numpy as np
from bs4 import BeautifulSoup
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i + 2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i + 1])
trigram_probabilities = {}
for (k, words) in iteritems(trigrams):
    if len(set(words)) > 1:
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for (w, c) in iteritems(d):
            d[w] = float(c) / n
        trigram_probabilities[k] = d

def random_sample(d):
    if False:
        for i in range(10):
            print('nop')
    r = random.random()
    cumulative = 0
    for (w, p) in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w

def test_spinner():
    if False:
        i = 10
        return i + 15
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print('Original:', s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i + 2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i + 1] = w
    print('Spun:')
    print(' '.join(tokens).replace(' .', '.').replace(" '", "'").replace(' ,', ',').replace('$ ', '$').replace(' !', '!'))
if __name__ == '__main__':
    test_spinner()