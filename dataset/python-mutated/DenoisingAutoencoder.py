import tensorflow as tf

class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        if False:
            while True:
                i = 10
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        if False:
            print('Hello World!')
        all_weights = dict()
        all_weights['w1'] = tf.get_variable('w1', shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        if False:
            return 10
        (cost, opt) = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        if False:
            print('Hello World!')
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        if False:
            return 10
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if False:
            for i in range(10):
                print('nop')
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        if False:
            while True:
                i = 10
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        if False:
            return 10
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        if False:
            return 10
        return self.sess.run(self.weights['b1'])

class MaskingNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), dropout_probability=0.95):
        if False:
            return 10
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.dropout_probability = dropout_probability
        self.keep_prob = tf.placeholder(tf.float32)
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(tf.nn.dropout(self.x, self.keep_prob), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        if False:
            i = 10
            return i + 15
        all_weights = dict()
        all_weights['w1'] = tf.get_variable('w1', shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        if False:
            while True:
                i = 10
        (cost, opt) = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.keep_prob: self.dropout_probability})
        return cost

    def calc_total_cost(self, X):
        if False:
            print('Hello World!')
        return self.sess.run(self.cost, feed_dict={self.x: X, self.keep_prob: 1.0})

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.keep_prob: 1.0})

    def generate(self, hidden=None):
        if False:
            i = 10
            return i + 15
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        if False:
            while True:
                i = 10
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.keep_prob: 1.0})

    def getWeights(self):
        if False:
            while True:
                i = 10
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        if False:
            return 10
        return self.sess.run(self.weights['b1'])