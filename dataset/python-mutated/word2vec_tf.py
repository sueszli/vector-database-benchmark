from __future__ import print_function, division
from builtins import range
import json
import tensorflow as tf
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
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def remove_punctuation_2(s):
    if False:
        for i in range(10):
            print('nop')
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    if False:
        print('Hello World!')
    return s.translate(str.maketrans('', '', string.punctuation))
if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def download_text8(dst):
    if False:
        while True:
            i = 10
    pass

def get_text8():
    if False:
        i = 10
        return i + 15
    path = '../large_files/text8'
    if not os.path.exists(path):
        download_text8(path)
    words = open(path).read()
    word2idx = {}
    sents = [[]]
    count = 0
    for word in words.split():
        if word not in word2idx:
            word2idx[word] = count
            count += 1
        sents[0].append(word2idx[word])
    print('count:', count)
    return (sents, word2idx)

def get_wiki():
    if False:
        return 10
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
        return 10
    (sentences, word2idx) = get_wiki()
    vocab_size = len(word2idx)
    window_size = 10
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5
    samples_per_epoch = int(100000.0)
    epochs = 20
    D = 50
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs
    p_neg = get_negative_sampling_distribution(sentences)
    W = np.random.randn(vocab_size, D).astype(np.float32)
    V = np.random.randn(D, vocab_size).astype(np.float32)
    tf_input = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    tf_context = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    tfW = tf.Variable(W)
    tfV = tf.Variable(V.T)

    def dot(A, B):
        if False:
            for i in range(10):
                print('nop')
        C = A * B
        return tf.reduce_sum(input_tensor=C, axis=1)
    emb_input = tf.nn.embedding_lookup(params=tfW, ids=tf_input)
    emb_output = tf.nn.embedding_lookup(params=tfV, ids=tf_context)
    correct_output = dot(emb_input, emb_output)
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(input=correct_output)), logits=correct_output)
    emb_input = tf.nn.embedding_lookup(params=tfW, ids=tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(input=incorrect_output)), logits=incorrect_output)
    loss = tf.reduce_mean(input_tensor=pos_loss) + tf.reduce_mean(input_tensor=neg_loss)
    train_op = tf.compat.v1.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
    session = tf.compat.v1.Session()
    init_op = tf.compat.v1.global_variables_initializer()
    session.run(init_op)
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
            for (j, pos) in enumerate(randomly_ordered_positions):
                word = sentence[pos]
                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)
                n = len(context_words)
                inputs += [word] * n
                negwords += [neg_word] * n
                targets += context_words
            if len(inputs) >= 128:
                (_, c) = session.run((train_op, loss), feed_dict={tf_input: inputs, tf_negword: negwords, tf_context: targets})
                cost += c
                inputs = []
                targets = []
                negwords = []
            counter += 1
            if counter % 100 == 0:
                sys.stdout.write('processed %s / %s\r' % (counter, len(sentences)))
                sys.stdout.flush()
        dt = datetime.now() - t0
        print('epoch complete:', epoch, 'cost:', cost, 'dt:', dt)
        costs.append(cost)
        learning_rate -= learning_rate_delta
    plt.plot(costs)
    plt.show()
    (W, VT) = session.run((tfW, tfV))
    V = VT.T
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)
    np.savez('%s/weights.npz' % savedir, W, V)
    return (word2idx, W, V)

def get_negative_sampling_distribution(sentences):
    if False:
        while True:
            i = 10
    word_freq = {}
    word_count = sum((len(sentence) for sentence in sentences))
    for sentence in sentences:
        for word in sentence:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    V = len(word_freq)
    p_neg = np.zeros(V)
    for j in range(V):
        p_neg[j] = word_freq[j] ** 0.75
    p_neg = p_neg / p_neg.sum()
    assert np.all(p_neg > 0)
    return p_neg

def get_context(pos, sentence, window_size):
    if False:
        i = 10
        return i + 15
    start = max(0, pos - window_size)
    end_ = min(len(sentence), pos + window_size)
    context = []
    for (ctx_pos, ctx_word_idx) in enumerate(sentence[start:end_], start=start):
        if ctx_pos != pos:
            context.append(ctx_word_idx)
    return context

def load_model(savedir):
    if False:
        print('Hello World!')
    with open('%s/word2idx.json' % savedir) as f:
        word2idx = json.load(f)
    npz = np.load('%s/weights.npz' % savedir)
    W = npz['arr_0']
    V = npz['arr_1']
    return (word2idx, W, V)

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    if False:
        print('Hello World!')
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
    print('got: %s - %s = %s - %s' % (pos1, neg1, idx2word[idx[0]], neg2))
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
        analogy('japan', 'sushi', 'england', 'bread', word2idx, idx2word, We)
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
    (word2idx, W, V) = train_model('w2v_tf')
    test_model(word2idx, W, V)