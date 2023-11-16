import os
import unittest
import tensorflow as tf
import tensorlayer as tl
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Simple_MNIST_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        cls.y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
        network = tl.layers.InputLayer(cls.x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
        network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')
        network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop3')
        cls.network = tl.layers.DenseLayer(network, n_units=10, name='output')
        y = cls.network.outputs
        cls.cost = tl.cost.cross_entropy(y, cls.y_, name='cost')
        correct_prediction = tf.equal(tf.argmax(y, 1), cls.y_)
        cls.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_params = cls.network.trainable_weights
        cls.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cls.cost, var_list=train_params)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()

    def test_reuse_vgg(self):
        if False:
            return 10
        (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))
        with self.assertNotRaises(Exception):
            with tf.Session() as sess:
                tl.layers.initialize_global_variables(sess)
                self.network.print_params()
                self.network.print_layers()
                tl.utils.fit(sess, self.network, self.train_op, self.cost, X_train, y_train, self.x, self.y_, acc=self.acc, batch_size=500, n_epoch=1, print_freq=1, X_val=X_val, y_val=y_val, eval_train=False)
                tl.utils.test(sess, self.network, self.acc, X_test, y_test, self.x, self.y_, batch_size=None, cost=self.cost)
                tl.files.save_npz(self.network.all_params, name='model.npz')
                sess.close()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()