from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from builtins import str, bytes, dict, int
from builtins import map, zip
from builtins import range
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import random
from collections import defaultdict
from pattern.text import Model
from pattern.vector import shuffled, SLP
from pattern.en import lexicon, parsetree
from random import seed
from io import open
print('load training data...')

def corpus(path, encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    ' Yields sentences of (word, tag)-tuples from the given corpus,\n        which is a .txt file with a sentence on each line, \n        with slash-encoded tokens (e.g., the/DT cat/NN).\n    '
    for s in open(path, encoding=encoding):
        s = list(map(lambda w: w.split('/'), s.strip().split(' ')))
        s = list(map(lambda w: (w[0].replace('&slash;', '/'), w[1]), s))
        yield s
path = os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'corpora', 'tagged-en-oanc.txt')
data = list(corpus(path))
print('load training lexicon...')
f = defaultdict(lambda : defaultdict(int))
for s in data:
    for (w, tag) in s:
        f[w][tag] += 1
(known, unknown) = (set(), set())
for (w, tags) in f.items():
    n = sum(tags.values())
    m = sorted(tags, key=tags.__getitem__, reverse=True)[0]
    if float(tags[m]) / n >= 0.97 and n > 1:
        known.add(w)
    if float(tags[m]) / n < 0.92 and w in lexicon:
        unknown.add(w)
print('training model...')
seed(0)
m = Model(known=known, unknown=unknown, classifier=SLP())
for iteration in range(5):
    for s in shuffled(data[:20000]):
        prev = None
        next = None
        for (i, (w, tag)) in enumerate(s):
            if i < len(s) - 1:
                next = s[i + 1]
            m.train(w, tag, prev, next)
            prev = (w, tag)
            next = None
f = os.path.join(os.path.dirname(__file__), 'en-model.slp')
m.save(f, final=True)
print('loading model...')
f = os.path.join(os.path.dirname(__file__), 'en-model.slp')
lexicon.model = Model.load(f, lexicon)
print('testing...')
(i, n) = (0, 0)
for s1 in data[-5000:]:
    s2 = ' '.join((w for (w, tag) in s1))
    s2 = parsetree(s2, tokenize=False)
    s2 = ((w.string, w.tag or '') for w in s2[0])
    for ((w1, tag1), (w2, tag2)) in zip(s1, s2):
        if tag1 == tag2.split('-')[0]:
            i += 1
        n += 1
print(float(i) / n)