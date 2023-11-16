from __future__ import print_function, division
from builtins import range
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from util import init_weight, get_ptb_data, display_tree
from datetime import datetime
from sklearn.metrics import f1_score
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

class RecursiveNN:

    def __init__(self, V, D, K, activation=tf.tanh):
        if False:
            for i in range(10):
                print('nop')
        self.V = V
        self.D = D
        self.K = K
        self.f = activation

    def fit(self, trees, test_trees, reg=0.001, epochs=8, train_inner_nodes=False):
        if False:
            return 10
        D = self.D
        V = self.V
        K = self.K
        N = len(trees)
        We = init_weight(V, D)
        W11 = np.random.randn(D, D, D) / np.sqrt(3 * D)
        W22 = np.random.randn(D, D, D) / np.sqrt(3 * D)
        W12 = np.random.randn(D, D, D) / np.sqrt(3 * D)
        W1 = init_weight(D, D)
        W2 = init_weight(D, D)
        bh = np.zeros(D)
        Wo = init_weight(D, K)
        bo = np.zeros(K)
        self.We = tf.Variable(We.astype(np.float32))
        self.W11 = tf.Variable(W11.astype(np.float32))
        self.W22 = tf.Variable(W22.astype(np.float32))
        self.W12 = tf.Variable(W12.astype(np.float32))
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.bh = tf.Variable(bh.astype(np.float32))
        self.Wo = tf.Variable(Wo.astype(np.float32))
        self.bo = tf.Variable(bo.astype(np.float32))
        self.weights = [self.We, self.W11, self.W22, self.W12, self.W1, self.W2, self.Wo]
        words = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='words')
        left_children = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='left_children')
        right_children = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='right_children')
        labels = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='labels')
        self.words = words
        self.left = left_children
        self.right = right_children
        self.labels = labels

        def dot1(a, B):
            if False:
                i = 10
                return i + 15
            return tf.tensordot(a, B, axes=[[0], [1]])

        def dot2(B, a):
            if False:
                while True:
                    i = 10
            return tf.tensordot(B, a, axes=[[1], [0]])

        def recursive_net_transform(hiddens, n):
            if False:
                print('Hello World!')
            h_left = hiddens.read(left_children[n])
            h_right = hiddens.read(right_children[n])
            return self.f(dot1(h_left, dot2(self.W11, h_left)) + dot1(h_right, dot2(self.W22, h_right)) + dot1(h_left, dot2(self.W12, h_right)) + dot1(h_left, self.W1) + dot1(h_right, self.W2) + self.bh)

        def recurrence(hiddens, n):
            if False:
                i = 10
                return i + 15
            w = words[n]
            h_n = tf.cond(pred=w >= 0, true_fn=lambda : tf.nn.embedding_lookup(params=self.We, ids=w), false_fn=lambda : recursive_net_transform(hiddens, n))
            hiddens = hiddens.write(n, h_n)
            n = tf.add(n, 1)
            return (hiddens, n)

        def condition(hiddens, n):
            if False:
                return 10
            return tf.less(n, tf.shape(input=words)[0])
        hiddens = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False)
        (hiddens, _) = tf.while_loop(cond=condition, body=recurrence, loop_vars=[hiddens, tf.constant(0)], parallel_iterations=1)
        h = hiddens.stack()
        logits = tf.matmul(h, self.Wo) + self.bo
        prediction_op = tf.argmax(input=logits, axis=1)
        self.prediction_op = prediction_op
        rcost = reg * sum((tf.nn.l2_loss(p) for p in self.weights))
        if train_inner_nodes:
            labeled_indices = tf.compat.v1.where(labels >= 0)
            cost_op = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather(logits, labeled_indices), labels=tf.gather(labels, labeled_indices))) + rcost
        else:
            cost_op = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[-1], labels=labels[-1])) + rcost
        train_op = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.008).minimize(cost_op)
        self.session = tf.compat.v1.Session()
        init_op = tf.compat.v1.global_variables_initializer()
        self.session.run(init_op)
        costs = []
        sequence_indexes = range(N)
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            n_total = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                (words_, left, right, lab) = trees[j]
                (c, p, _) = self.session.run((cost_op, prediction_op, train_op), feed_dict={words: words_, left_children: left, right_children: right, labels: lab})
                if np.isnan(c):
                    print("Cost is nan! Let's stop here.                         Why don't you try decreasing the learning rate?")
                    for p in self.params:
                        print(p.get_value().sum())
                    exit()
                cost += c
                n_correct += p[-1] == lab[-1]
                n_total += 1
                it += 1
                if it % 10 == 0:
                    sys.stdout.write('j/N: %d/%d correct rate so far: %f, cost so far: %f\r' % (it, N, float(n_correct) / n_total, cost))
                    sys.stdout.flush()
            n_test_correct = 0
            n_test_total = 0
            for (words_, left, right, lab) in test_trees:
                p = self.session.run(prediction_op, feed_dict={words: words_, left_children: left, right_children: right, labels: lab})
                n_test_correct += p[-1] == lab[-1]
                n_test_total += 1
            print('i:', i, 'cost:', cost, 'train acc:', float(n_correct) / n_total, 'test acc:', float(n_test_correct) / n_test_total, 'time for epoch:', datetime.now() - t0)
            costs.append(cost)
        plt.plot(costs)
        plt.show()

    def predict(self, words, left, right, lab):
        if False:
            return 10
        return self.session.run(self.prediction_op, feed_dict={self.words: words, self.left: left, self.right: right, self.labels: lab})

    def score(self, trees):
        if False:
            for i in range(10):
                print('nop')
        n_total = len(trees)
        n_correct = 0
        for (words, left, right, lab) in trees:
            p = self.predict(words, left, right, lab)
            n_correct += p[-1] == lab[-1]
        return float(n_correct) / n_total

    def f1_score(self, trees):
        if False:
            i = 10
            return i + 15
        Y = []
        P = []
        for (words, left, right, lab) in trees:
            p = self.predict(words, left, right, lab)
            Y.append(lab[-1])
            P.append(p[-1])
        return f1_score(Y, P, average=None).mean()

def add_idx_to_tree(tree, current_idx):
    if False:
        while True:
            i = 10
    if tree is None:
        return current_idx
    current_idx = add_idx_to_tree(tree.left, current_idx)
    current_idx = add_idx_to_tree(tree.right, current_idx)
    tree.idx = current_idx
    current_idx += 1
    return current_idx

def tree2list(tree, parent_idx, is_binary=False):
    if False:
        for i in range(10):
            print('nop')
    if tree is None:
        return ([], [], [], [])
    (words_left, left_child_left, right_child_left, labels_left) = tree2list(tree.left, tree.idx, is_binary)
    (words_right, left_child_right, right_child_right, labels_right) = tree2list(tree.right, tree.idx, is_binary)
    if tree.word is None:
        w = -1
        left = tree.left.idx
        right = tree.right.idx
    else:
        w = tree.word
        left = -1
        right = -1
    words = words_left + words_right + [w]
    left_child = left_child_left + left_child_right + [left]
    right_child = right_child_left + right_child_right + [right]
    if is_binary:
        if tree.label > 2:
            label = 1
        elif tree.label < 2:
            label = 0
        else:
            label = -1
    else:
        label = tree.label
    labels = labels_left + labels_right + [label]
    return (words, left_child, right_child, labels)

def main(is_binary=True):
    if False:
        print('Hello World!')
    (train, test, word2idx) = get_ptb_data()
    for t in train:
        add_idx_to_tree(t, 0)
    train = [tree2list(t, -1, is_binary) for t in train]
    if is_binary:
        train = [t for t in train if t[3][-1] >= 0]
    for t in test:
        add_idx_to_tree(t, 0)
    test = [tree2list(t, -1, is_binary) for t in test]
    if is_binary:
        test = [t for t in test if t[3][-1] >= 0]
    train = shuffle(train)
    test = shuffle(test)
    smalltest = test[:1000]
    V = len(word2idx)
    print('vocab size:', V)
    D = 10
    K = 2 if is_binary else 5
    model = RecursiveNN(V, D, K)
    model.fit(train, smalltest, reg=0.001, epochs=20, train_inner_nodes=True)
    print('train accuracy:', model.score(train))
    print('test accuracy:', model.score(test))
    print('train f1:', model.f1_score(train))
    print('test f1:', model.f1_score(test))
if __name__ == '__main__':
    main()