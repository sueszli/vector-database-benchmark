__author__ = 'nastra'

from gensim import corpora, models, similarities
from os import path
import numpy as np
from matplotlib import pyplot as plt

if not path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')


def plot_topics(thetas, xlabel, ylabel, filename):
    plt.hist([len(t) for t in thetas], np.arange(42))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)


def print_topics(mod):
    #global ti, words, f, w, tf
    for ti in xrange(84):
        words = mod.show_topic(ti, 20)
        tf = sum(f for f, w in words)
        print('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for f, w in words))
        print(120 * "=")


corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=None)

print_topics(model)

topics = [model[c] for c in corpus]
plot_topics(topics, 'Nr of topics', 'Nr of documents', 'data/histogram.png')

# larger alpha leads to more documents per topic
model2 = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)
topics2 = [model2[c] for c in corpus]
plot_topics(topics2, 'Nr of topics', 'Nr of documents', 'data/histogram_with_alpha.png')