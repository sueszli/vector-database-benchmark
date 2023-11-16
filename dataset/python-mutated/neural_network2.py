from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from markov import get_bigram_probs
if __name__ == '__main__':
    (sentences, word2idx) = get_sentences_with_word2idx_limit_vocab(2000)
    V = len(word2idx)
    print('Vocab size:', V)
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)
    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)
    losses = []
    epochs = 1
    lr = 0.01

    def softmax(a):
        if False:
            return 10
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)
    W_bigram = np.log(bigram_probs)
    bigram_losses = []
    t0 = datetime.now()
    for epoch in range(epochs):
        random.shuffle(sentences)
        j = 0
        for sentence in sentences:
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = sentence[:n - 1]
            targets = sentence[1:]
            hidden = np.tanh(W1[inputs])
            predictions = softmax(hidden.dot(W2))
            loss = -np.sum(np.log(predictions[np.arange(n - 1), targets])) / (n - 1)
            losses.append(loss)
            doutput = predictions
            doutput[np.arange(n - 1), targets] -= 1
            W2 = W2 - lr * hidden.T.dot(doutput)
            dhidden = doutput.dot(W2.T) * (1 - hidden * hidden)
            W1_copy = W1.copy()
            np.subtract.at(W1, inputs, lr * dhidden)
            if epoch == 0:
                bigram_predictions = softmax(W_bigram[inputs])
                bigram_loss = -np.sum(np.log(bigram_predictions[np.arange(n - 1), targets])) / (n - 1)
                bigram_losses.append(bigram_loss)
            if j % 100 == 0:
                print('epoch:', epoch, 'sentence: %s/%s' % (j, len(sentences)), 'loss:', loss)
            j += 1
    print('Elapsed time training:', datetime.now() - t0)
    plt.plot(losses)
    avg_bigram_loss = np.mean(bigram_losses)
    print('avg_bigram_loss:', avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    def smoothed_loss(x, decay=0.99):
        if False:
            return 10
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y
    plt.plot(smoothed_loss(losses))
    plt.show()
    plt.subplot(1, 2, 1)
    plt.title('Neural Network Model')
    plt.imshow(np.tanh(W1).dot(W2))
    plt.subplot(1, 2, 2)
    plt.title('Bigram Probs')
    plt.imshow(W_bigram)
    plt.show()