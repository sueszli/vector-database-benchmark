from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    if False:
        print('Hello World!')
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                bigram_probs[sentence[i - 1], sentence[i]] += 1
            if i == len(sentence) - 1:
                bigram_probs[sentence[i], end_idx] += 1
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs
if __name__ == '__main__':
    (sentences, word2idx) = get_sentences_with_word2idx_limit_vocab(10000)
    V = len(word2idx)
    print('Vocab size:', V)
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    def get_score(sentence):
        if False:
            for i in range(10):
                print('nop')
        score = 0
        for i in range(len(sentence)):
            if i == 0:
                score += np.log(bigram_probs[start_idx, sentence[i]])
            else:
                score += np.log(bigram_probs[sentence[i - 1], sentence[i]])
        score += np.log(bigram_probs[sentence[-1], end_idx])
        return score / (len(sentence) + 1)
    idx2word = dict(((v, k) for (k, v) in iteritems(word2idx)))

    def get_words(sentence):
        if False:
            return 10
        return ' '.join((idx2word[i] for i in sentence))
    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()
    while True:
        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]
        fake = np.random.choice(V, size=len(real), p=sample_probs)
        print('REAL:', get_words(real), 'SCORE:', get_score(real))
        print('FAKE:', get_words(fake), 'SCORE:', get_score(fake))
        custom = input('Enter your own sentence:\n')
        custom = custom.lower().split()
        bad_sentence = False
        for token in custom:
            if token not in word2idx:
                bad_sentence = True
        if bad_sentence:
            print('Sorry, you entered words that are not in the vocabulary')
        else:
            custom = [word2idx[token] for token in custom]
            print('SCORE:', get_score(custom))
        cont = input('Continue? [Y/n]')
        if cont and cont.lower() in ('N', 'n'):
            break