import os
import unittest
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3, inception_v3_arg_scope
import tensorlayer as tl
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
slim = tf.contrib.slim
keras = tf.keras

class Layer_Importer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.net_in = dict()
        x = tf.placeholder(tf.float32, shape=[None, 784])
        cls.net_in['lambda'] = tl.layers.InputLayer(x, name='input')
        x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        cls.net_in['slim'] = tl.layers.InputLayer(x, name='input_layer')

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        tf.reset_default_graph()

    def test_lambda_layer(self):
        if False:
            while True:
                i = 10

        def keras_block(x):
            if False:
                print('Hello World!')
            x = keras.layers.Dropout(0.8)(x)
            x = keras.layers.Dense(100, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            logits = keras.layers.Dense(10, activation='linear')(x)
            return logits
        with self.assertNotRaises(Exception):
            tl.layers.LambdaLayer(self.net_in['lambda'], fn=keras_block, name='keras')

    def test_slim_layer(self):
        if False:
            return 10
        with self.assertNotRaises(Exception):
            with slim.arg_scope(inception_v3_arg_scope()):
                tl.layers.SlimNetsLayer(self.net_in['slim'], slim_layer=inception_v3, slim_args={'num_classes': 1001, 'is_training': False}, name='InceptionV3')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()