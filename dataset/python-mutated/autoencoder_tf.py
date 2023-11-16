from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Autoencoder:

    def __init__(self, D, M):
        if False:
            return 10
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.W = tf.Variable(tf.random_normal(shape=(D, M)) * np.sqrt(2.0 / M))
        self.b = tf.Variable(np.zeros(M).astype(np.float32))
        self.V = tf.Variable(tf.random_normal(shape=(M, D)) * np.sqrt(2.0 / D))
        self.c = tf.Variable(np.zeros(D).astype(np.float32))
        self.Z = tf.nn.relu(tf.matmul(self.X, self.W) + self.b)
        logits = tf.matmul(self.Z, self.V) + self.c
        self.X_hat = tf.nn.sigmoid(logits)
        self.cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=logits))
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def fit(self, X, epochs=30, batch_sz=64):
        if False:
            print('Hello World!')
        costs = []
        n_batches = len(X) // batch_sz
        print('n_batches:', n_batches)
        for i in range(epochs):
            print('epoch:', i)
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:(j + 1) * batch_sz]
                (_, c) = self.sess.run((self.train_op, self.cost), feed_dict={self.X: batch})
                c /= batch_sz
                costs.append(c)
                if j % 100 == 0:
                    print('iter: %d, cost: %.3f' % (j, c))
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        return self.sess.run(self.X_hat, feed_dict={self.X: X})

def main():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = util.get_mnist()
    model = Autoencoder(784, 300)
    model.fit(X)
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = model.predict([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title('Reconstruction')
        plt.show()
        ans = input('Generate another?')
        if ans and ans[0] in ('n' or 'N'):
            done = True
if __name__ == '__main__':
    main()