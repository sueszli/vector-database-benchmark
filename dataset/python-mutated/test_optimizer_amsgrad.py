import os
import unittest
import tensorflow as tf
import tensorlayer as tl
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        cls.y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
        cls.network = tl.layers.InputLayer(cls.x, name='input')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.8, name='drop1')
        cls.network = tl.layers.DenseLayer(cls.network, 800, tf.nn.relu, name='relu1')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.5, name='drop2')
        cls.network = tl.layers.DenseLayer(cls.network, 800, tf.nn.relu, name='relu2')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.5, name='drop3')
        cls.network = tl.layers.DenseLayer(cls.network, n_units=10, name='output')
        cls.y = cls.network.outputs
        cls.cost = tl.cost.cross_entropy(cls.y, cls.y_, name='cost')
        correct_prediction = tf.equal(tf.argmax(cls.y, 1), cls.y_)
        cls.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_params = cls.network.all_params
        optimizer = tl.optimizers.AMSGrad(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        cls.train_op = optimizer.minimize(cls.cost, var_list=train_params)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()

    def test_training(self):
        if False:
            print('Hello World!')
        with self.assertNotRaises(Exception):
            (X_train, y_train, X_val, y_val, _, _) = tl.files.load_mnist_dataset(shape=(-1, 784))
            with tf.Session() as sess:
                tl.layers.initialize_global_variables(sess)
                self.network.print_params()
                self.network.print_layers()
                tl.utils.fit(sess, self.network, self.train_op, self.cost, X_train, y_train, self.x, self.y_, acc=self.acc, batch_size=500, n_epoch=1, print_freq=1, X_val=X_val, y_val=y_val, eval_train=False)
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()