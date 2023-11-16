import os
import time
import unittest
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.logging.contrib import hyperdash as hd
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TL_Logger_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.apikey = os.getenv('HYPERDASH_APIKEY', 'test_api_key')

    def test_apikey_unset(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            hd.HyperDashHandler.reset_apikey()
            hd.HyperDashHandler.get_apikey()

    def test_apikey_set(self):
        if False:
            return 10
        with self.assertNotRaises(ValueError):
            hd.HyperDashHandler.set_apikey(self.apikey)
            hd.HyperDashHandler.get_apikey()

    def test_monitor(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertNotRaises(Exception):
            hd.HyperDashHandler.set_apikey(self.apikey)

            @hd.monitor('TRAVIS 1 - dogs vs. cats')
            def train_dogs_vs_cats(exp=None):
                if False:
                    for i in range(10):
                        print('nop')
                lr = exp.param('learning rate', 0.005)
                tl.logging.debug('Learning Rate: %f' % lr)
                for (epoch, accuracy) in enumerate([10, 30, 50, 70, 80, 90, 95, 100]):
                    tl.logging.debug('Epoch %d - Accuracy %d%%' % (epoch + 1, accuracy))
                    exp.metric(name='accuracy', value=accuracy)
                    time.sleep(0.1)
            train_dogs_vs_cats()

    def test_monitor_variant(self):
        if False:
            return 10
        with self.assertNotRaises(Exception):

            @hd.monitor('TRAVIS 2 - dogs vs. cats', api_key=self.apikey)
            def train_dogs_vs_cats(exp=None):
                if False:
                    i = 10
                    return i + 15
                lr = exp.param('learning rate', 0.005)
                tl.logging.debug('Learning Rate: %f' % lr)
                for (epoch, accuracy) in enumerate([10, 30, 50, 70, 80, 90, 95, 100]):
                    tl.logging.debug('Epoch %d - Accuracy %d%%' % (epoch + 1, accuracy))
                    exp.metric(name='accuracy', value=accuracy)
                    time.sleep(0.1)
            train_dogs_vs_cats()

    def test_Experiment(self):
        if False:
            return 10
        hd.HyperDashHandler.set_apikey(self.apikey)
        with self.assertNotRaises(Exception):

            def train_dogs_vs_cats():
                if False:
                    i = 10
                    return i + 15
                exp = hd.Experiment('TRAVIS 3 - dogs vs. cats')
                lr = exp.param('learning rate', 0.005)
                tl.logging.debug('Learning Rate: %f' % lr)
                for (epoch, accuracy) in enumerate([10, 30, 50, 70, 80, 90, 95, 100]):
                    tl.logging.debug('Epoch %d - Accuracy %d%%' % (epoch + 1, accuracy))
                    exp.metric(name='accuracy', value=accuracy)
                    time.sleep(0.1)
                exp.end()
            train_dogs_vs_cats()

    def test_Experiment_variant(self):
        if False:
            return 10
        with self.assertNotRaises(Exception):

            def train_dogs_vs_cats():
                if False:
                    while True:
                        i = 10
                exp = hd.Experiment('TRAVIS 4 - dogs vs. cats', api_key=self.apikey)
                lr = exp.param('learning rate', 0.005)
                tl.logging.debug('Learning Rate: %f' % lr)
                for (epoch, accuracy) in enumerate([10, 30, 50, 70, 80, 90, 95, 100]):
                    tl.logging.debug('Epoch %d - Accuracy %d%%' % (epoch + 1, accuracy))
                    exp.metric(name='accuracy', value=accuracy)
                    time.sleep(0.1)
                exp.end()
            train_dogs_vs_cats()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()