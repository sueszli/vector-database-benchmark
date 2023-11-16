from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from builtins import str, bytes, dict, int
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import random
from pattern.db import Datasheet
from pattern.nl import tag, predicative
from pattern.vector import SVM, KNN, NB, count, shuffled
classifier = SVM()
print('loading data...')
data = os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'corpora', 'polarity-nl-bol.com.csv')
data = Datasheet.load(data)
data = shuffled(data)

def instance(review):
    if False:
        while True:
            i = 10
    v = tag(review)
    v = [word for (word, pos) in v if pos in ('JJ', 'RB') or word in '!']
    v = [predicative(word) for word in v]
    v = count(v)
    return v
print('training...')
for (score, review) in data[:1000]:
    classifier.train(instance(review), type=int(score) > 0)
print('testing...')
i = n = 0
for (score, review) in data[1000:1500]:
    if classifier.classify(instance(review)) == (int(score) > 0):
        i += 1
    n += 1
print(float(i) / n)