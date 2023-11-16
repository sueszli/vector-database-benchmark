from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import numpy as np
import string
import sys
initial = {}
second_word = {}
transitions = {}

def remove_punctuation_2(s):
    if False:
        while True:
            i = 10
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    if False:
        while True:
            i = 10
    return s.translate(str.maketrans('', '', string.punctuation))
if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def add2dict(d, k, v):
    if False:
        return 10
    if k not in d:
        d[k] = []
    d[k].append(v)
for line in open('robert_frost.txt'):
    tokens = remove_punctuation(line.rstrip().lower()).split()
    T = len(tokens)
    for i in range(T):
        t = tokens[i]
        if i == 0:
            initial[t] = initial.get(t, 0.0) + 1
        else:
            t_1 = tokens[i - 1]
            if i == T - 1:
                add2dict(transitions, (t_1, t), 'END')
            if i == 1:
                add2dict(second_word, t_1, t)
            else:
                t_2 = tokens[i - 2]
                add2dict(transitions, (t_2, t_1), t)
initial_total = sum(initial.values())
for (t, c) in iteritems(initial):
    initial[t] = c / initial_total

def list2pdict(ts):
    if False:
        print('Hello World!')
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.0) + 1
    for (t, c) in iteritems(d):
        d[t] = c / n
    return d
for (t_1, ts) in iteritems(second_word):
    second_word[t_1] = list2pdict(ts)
for (k, ts) in iteritems(transitions):
    transitions[k] = list2pdict(ts)

def sample_word(d):
    if False:
        for i in range(10):
            print('nop')
    p0 = np.random.random()
    cumulative = 0
    for (t, p) in iteritems(d):
        cumulative += p
        if p0 < cumulative:
            return t
    assert False

def generate():
    if False:
        print('Hello World!')
    for i in range(4):
        sentence = []
        w0 = sample_word(initial)
        sentence.append(w0)
        w1 = sample_word(second_word[w0])
        sentence.append(w1)
        while True:
            w2 = sample_word(transitions[w0, w1])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sentence))
generate()