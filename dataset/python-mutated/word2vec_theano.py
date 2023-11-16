from __future__ import print_function, division
from builtins import range
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from glob import glob
import os
import sys
import string
import theano
import theano.tensor as T

def remove_punctuation_2(s):
    if False:
        i = 10
        return i + 15
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    if False:
        print('Hello World!')
    return s.translate(str.maketrans('', '', string.punctuation))
if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
    if False:
        while True:
            i = 10
    V = 20000
    files = glob('../large_files/enwiki*.txt')
    all_word_counts = {}
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\\{\\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1
    print('finished counting')
    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [w for (w, count) in all_word_counts[:V - 1]] + ['<UNK>']
    word2idx = {w: i for (i, w) in enumerate(top_words)}
    unk = word2idx['<UNK>']
    sents = []
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\\{\\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sent = [word2idx[w] if w in word2idx else unk for w in s]
                    sents.append(sent)
    return (sents, word2idx)

def train_model(savedir):
    if False:
        while True:
            i = 10
    (sentences, word2idx) = get_wiki()
    vocab_size = len(word2idx)
    window_size = 5
    learning_rate = 0.025 * 128
    final_learning_rate = 0.0001 * 128
    num_negatives = 5
    samples_per_epoch = int(100000.0)
    epochs = 1
    D = 50
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs
    W = np.random.randn(vocab_size, D) / np.sqrt(D + vocab_size)
    V = np.random.randn(D, vocab_size) / np.sqrt(D + vocab_size)
    thW = theano.shared(W)
    thV = theano.shared(V)
    th_pos_word = T.ivector('pos_word')
    th_neg_word = T.ivector('neg_word')
    th_context = T.ivector('context')
    th_lr = T.scalar('learning_rate')
    input_words = T.concatenate([th_pos_word, th_neg_word])
    W_subset = thW[input_words]
    dbl_context = T.concatenate([th_context, th_context])
    V_subset = thV[:, dbl_context]
    logits = W_subset.dot(V_subset)
    out = T.nnet.sigmoid(logits)
    n = th_pos_word.shape[0]
    th_cost = -T.log(out[:n]).mean() - T.log(1 - out[n:]).mean()
    gW = T.grad(th_cost, W_subset)
    gV = T.grad(th_cost, V_subset)
    W_update = T.inc_subtensor(W_subset, -th_lr * gW)
    V_update = T.inc_subtensor(V_subset, -th_lr * gV)
    updates = [(thW, W_update), (thV, V_update)]
    cost_op = theano.function(inputs=[th_pos_word, th_neg_word, th_context], outputs=th_cost)
    cost_train_op = theano.function(inputs=[th_pos_word, th_neg_word, th_context, th_lr], outputs=th_cost, updates=updates)
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)
    costs = []
    total_words = sum((len(sentence) for sentence in sentences))
    print('total number of words in corpus:', total_words)
    threshold = 1e-05
    p_drop = 1 - np.sqrt(threshold / p_neg)
    for epoch in range(epochs):
        np.random.shuffle(sentences)
        cost = 0
        counter = 0
        inputs = []
        targets = []
        negwords = []
        t0 = datetime.now()
        for sentence in sentences:
            sentence = [w for w in sentence if np.random.random() < 1 - p_drop[w]]
            if len(sentence) < 2:
                continue
            randomly_ordered_positions = np.random.choice(len(sentence), size=len(sentence), replace=False)
            for pos in randomly_ordered_positions:
                word = sentence[pos]
                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)
                n = len(context_words)
                inputs += [word] * n
                negwords += [neg_word] * n
                targets += context_words
                if len(inputs) >= 128:
                    c = cost_train_op(inputs, negwords, targets, learning_rate)
                    cost += c
                    if np.isnan(c):
                        print('c is nan:', c)
                        exit()
                    inputs = []
                    targets = []
                    negwords = []
            counter += 1
            if counter % 100 == 0:
                sys.stdout.write('processed %s / %s, cost: %s\r' % (counter, len(sentences), c))
                sys.stdout.flush()
        dt = datetime.now() - t0
        print('epoch complete:', epoch, 'cost:', cost, 'dt:', dt)
        costs.append(cost)
        learning_rate -= learning_rate_delta
    plt.plot(costs)
    plt.show()
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)
    (W, V) = (thW.get_value(), thV.get_value())
    np.savez('%s/weights.npz' % savedir, W, V)
    return (word2idx, W, V)

def get_negative_sampling_distribution(sentences, vocab_size):
    if False:
        print('Hello World!')
    word_freq = np.zeros(vocab_size)
    word_count = sum((len(sentence) for sentence in sentences))
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    p_neg = word_freq ** 0.75
    p_neg = p_neg / p_neg.sum()
    assert np.all(p_neg > 0)
    return p_neg

def get_context(pos, sentence, window_size):
    if False:
        while True:
            i = 10
    start = max(0, pos - window_size)
    end_ = min(len(sentence), pos + window_size)
    context = []
    for (ctx_pos, ctx_word_idx) in enumerate(sentence[start:end_], start=start):
        if ctx_pos != pos:
            context.append(ctx_word_idx)
    return context

def load_model(savedir):
    if False:
        while True:
            i = 10
    with open('%s/word2idx.json' % savedir) as f:
        word2idx = json.load(f)
    npz = np.load('%s/weights.npz' % savedir)
    W = npz['arr_0']
    V = npz['arr_1']
    return (word2idx, W, V)

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    if False:
        i = 10
        return i + 15
    (V, D) = W.shape
    print('testing: %s - %s = %s - %s' % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print('Sorry, %s not in word2idx' % w)
            return
    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]
    vec = p1 - n1 + n2
    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break
    print('got: %s - %s = %s - %s' % (pos1, neg1, idx2word[best_idx], neg2))
    print('closest 10:')
    for i in idx:
        print(idx2word[i], distances[i])
    print('dist to %s:' % pos2, cos_dist(p2, vec))

def test_model(word2idx, W, V):
    if False:
        i = 10
        return i + 15
    idx2word = {i: w for (w, i) in word2idx.items()}
    for We in (W, (W + V.T) / 2):
        print('**********')
        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)
if __name__ == '__main__':
    (word2idx, W, V) = train_model('w2v_model')
    test_model(word2idx, W, V)