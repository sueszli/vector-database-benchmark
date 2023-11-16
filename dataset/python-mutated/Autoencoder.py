import numpy as np
import tensorflow as tf

class Autoencoder(object):

    def __init__(self, n_layers, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        if False:
            i = 10
            return i + 15
        self.n_layers = n_layers
        self.transfer = transfer_function
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
        self.hidden_encode = []
        h = self.x
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(tf.add(tf.matmul(h, self.weights['encode'][layer]['w']), self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)
        self.hidden_recon = []
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(tf.add(tf.matmul(h, self.weights['recon'][layer]['w']), self.weights['recon'][layer]['b']))
            self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        if False:
            return 10
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        encoder_weights = []
        for layer in range(len(self.n_layers) - 1):
            w = tf.Variable(initializer((self.n_layers[layer], self.n_layers[layer + 1]), dtype=tf.float32))
            b = tf.Variable(tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        recon_weights = []
        for layer in range(len(self.n_layers) - 1, 0, -1):
            w = tf.Variable(initializer((self.n_layers[layer], self.n_layers[layer - 1]), dtype=tf.float32))
            b = tf.Variable(tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights

    def partial_fit(self, X):
        if False:
            for i in range(10):
                print('nop')
        (cost, opt) = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        if False:
            i = 10
            return i + 15
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        if False:
            print('Hello World!')
        return self.sess.run(self.hidden_encode[-1], feed_dict={self.x: X})

    def generate(self, hidden=None):
        if False:
            print('Hello World!')
        if hidden is None:
            hidden = np.random.normal(size=self.weights['encode'][-1]['b'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden_encode[-1]: hidden})

    def reconstruct(self, X):
        if False:
            while True:
                i = 10
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        if False:
            print('Hello World!')
        raise NotImplementedError
        return self.sess.run(self.weights)

    def getBiases(self):
        if False:
            return 10
        raise NotImplementedError
        return self.sess.run(self.weights)