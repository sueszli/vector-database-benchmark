from __future__ import print_function, division
from builtins import range
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from util import init_weight, get_ptb_data, display_tree
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def get_labels(tree):
    if False:
        for i in range(10):
            print('nop')
    if tree is None:
        return []
    return get_labels(tree.left) + get_labels(tree.right) + [tree.label]

class TNN:

    def __init__(self, V, D, K, activation):
        if False:
            for i in range(10):
                print('nop')
        self.D = D
        self.f = activation
        We = init_weight(V, D)
        W1 = init_weight(D, D)
        W2 = init_weight(D, D)
        bh = np.zeros(D)
        Wo = init_weight(D, K)
        bo = np.zeros(K)
        self.We = tf.Variable(We.astype(np.float32))
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.bh = tf.Variable(bh.astype(np.float32))
        self.Wo = tf.Variable(Wo.astype(np.float32))
        self.bo = tf.Variable(bo.astype(np.float32))
        self.params = [self.We, self.W1, self.W2, self.Wo]

    def fit(self, trees, lr=0.1, mu=0.9, reg=0.1, epochs=5):
        if False:
            while True:
                i = 10
        train_ops = []
        costs = []
        predictions = []
        all_labels = []
        i = 0
        N = len(trees)
        print('Compiling ops')
        for t in trees:
            i += 1
            sys.stdout.write('%d/%d\r' % (i, N))
            sys.stdout.flush()
            logits = self.get_output(t)
            labels = get_labels(t)
            all_labels.append(labels)
            cost = self.get_cost(logits, labels, reg)
            costs.append(cost)
            prediction = tf.argmax(input=logits, axis=1)
            predictions.append(prediction)
            train_op = tf.compat.v1.train.MomentumOptimizer(lr, mu).minimize(cost)
            train_ops.append(train_op)
        self.predictions = predictions
        self.all_labels = all_labels
        self.saver = tf.compat.v1.train.Saver()
        init = tf.compat.v1.initialize_all_variables()
        actual_costs = []
        per_epoch_costs = []
        correct_rates = []
        with tf.compat.v1.Session() as session:
            session.run(init)
            for i in range(epochs):
                t0 = datetime.now()
                (train_ops, costs, predictions, all_labels) = shuffle(train_ops, costs, predictions, all_labels)
                epoch_cost = 0
                n_correct = 0
                n_total = 0
                j = 0
                N = len(train_ops)
                for (train_op, cost, prediction, labels) in zip(train_ops, costs, predictions, all_labels):
                    (_, c, p) = session.run([train_op, cost, prediction])
                    epoch_cost += c
                    actual_costs.append(c)
                    n_correct += np.sum(p == labels)
                    n_total += len(labels)
                    j += 1
                    if j % 10 == 0:
                        sys.stdout.write('j: %d, N: %d, c: %f\r' % (j, N, c))
                        sys.stdout.flush()
                print('epoch:', i, 'cost:', epoch_cost, 'elapsed time:', datetime.now() - t0)
                per_epoch_costs.append(epoch_cost)
                correct_rates.append(n_correct / float(n_total))
            self.saver.save(session, 'recursive.ckpt')
        plt.plot(actual_costs)
        plt.title('cost per train_op call')
        plt.show()
        plt.plot(per_epoch_costs)
        plt.title('per epoch costs')
        plt.show()
        plt.plot(correct_rates)
        plt.title('correct rates')
        plt.show()

    def get_cost(self, logits, labels, reg):
        if False:
            print('Hello World!')
        cost = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        rcost = sum((tf.nn.l2_loss(p) for p in self.params))
        cost += reg * rcost
        return cost

    def get_output_recursive(self, tree, list_of_logits, is_root=True):
        if False:
            print('Hello World!')
        if tree.word is not None:
            x = tf.nn.embedding_lookup(params=self.We, ids=[tree.word])
        else:
            x1 = self.get_output_recursive(tree.left, list_of_logits, is_root=False)
            x2 = self.get_output_recursive(tree.right, list_of_logits, is_root=False)
            x = self.f(tf.matmul(x1, self.W1) + tf.matmul(x2, self.W2) + self.bh)
        logits = tf.matmul(x, self.Wo) + self.bo
        list_of_logits.append(logits)
        return x

    def get_output(self, tree):
        if False:
            for i in range(10):
                print('nop')
        logits = []
        self.get_output_recursive(tree, logits)
        return tf.concat(logits, 0)

    def score(self, trees):
        if False:
            for i in range(10):
                print('nop')
        if trees is None:
            predictions = self.predictions
            all_labels = self.all_labels
        else:
            predictions = []
            all_labels = []
            i = 0
            N = len(trees)
            print('Compiling ops')
            for t in trees:
                i += 1
                sys.stdout.write('%d/%d\r' % (i, N))
                sys.stdout.flush()
                logits = self.get_output(t)
                labels = get_labels(t)
                all_labels.append(labels)
                prediction = tf.argmax(input=logits, axis=1)
                predictions.append(prediction)
        n_correct = 0
        n_total = 0
        with tf.compat.v1.Session() as session:
            self.saver.restore(session, 'recursive.ckpt')
            for (prediction, y) in zip(predictions, all_labels):
                p = session.run(prediction)
                n_correct += p[-1] == y[-1]
                n_total += len(y)
        return float(n_correct) / n_total

def main():
    if False:
        while True:
            i = 10
    (train, test, word2idx) = get_ptb_data()
    train = train[:100]
    test = test[:100]
    V = len(word2idx)
    D = 80
    K = 5
    model = TNN(V, D, K, tf.nn.relu)
    model.fit(train)
    print('train accuracy:', model.score(None))
    print('test accuracy:', model.score(test))
if __name__ == '__main__':
    main()