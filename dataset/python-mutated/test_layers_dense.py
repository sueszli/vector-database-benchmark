import os
import unittest
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Layer_BinaryDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        print('-' * 20, 'Layer_BinaryDense_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = BinaryDense(n_units=5)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = BinaryDense(n_units=5, in_channels=10)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.ones(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        pass

    def test_layer_n1(self):
        if False:
            return 10
        print(self.n1[0])
        self.assertEqual(tf.reduce_sum(self.n1).numpy() % 1, 0.0)

    def test_layer_n2(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.n2[0])
        self.assertEqual(tf.reduce_sum(self.n2).numpy() % 1, 0.0)

    def test_model_n3(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.n3[0])
        self.assertEqual(tf.reduce_sum(self.n3).numpy() % 1, 0.0)

    def test_exception(self):
        if False:
            return 10
        try:
            layer = BinaryDense(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = BinaryDense(n_units=5, use_gemm=True)
            out = layer(self.ni)
            self.fail('use gemm')
        except Exception as e:
            print(e)

class Layer_DorefaDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('-' * 20, 'Layer_DorefaDense_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = DorefaDense(n_units=5)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = DorefaDense(n_units=5, in_channels=10)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.ones(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_layer_n1(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.n1[0])

    def test_layer_n2(self):
        if False:
            return 10
        print(self.n2[0])

    def test_model_n3(self):
        if False:
            while True:
                i = 10
        print(self.n3[0])

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        try:
            layer = DorefaDense(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = DorefaDense(n_units=5, use_gemm=True)
            out = layer(self.ni)
            self.fail('use gemm')
        except Exception as e:
            print(e)

class Layer_DropconnectDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        print('-' * 20, 'Layer_DropconnectDense_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = DropconnectDense(n_units=5, keep=1.0)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = DropconnectDense(n_units=5, in_channels=10, keep=0.01)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.ones(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        pass

    def test_layer_n1(self):
        if False:
            while True:
                i = 10
        print(self.n1[0])

    def test_layer_n2(self):
        if False:
            i = 10
            return i + 15
        zero_rate = tf.reduce_mean(tf.cast(tf.equal(self.n2, 0.0), tf.float32))
        print(zero_rate)
        self.assertGreater(zero_rate, 0.0)
        print(self.n2[0])

    def test_model_n3(self):
        if False:
            i = 10
            return i + 15
        print(self.n3[0])

    def test_exception(self):
        if False:
            print('Hello World!')
        try:
            layer = DropconnectDense(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = DropconnectDense(n_units=5, keep=0.0)
            self.fail('keep no elements')
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

class Layer_QuanDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        print('-' * 20, 'Layer_QuanDense_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = QuanDense(n_units=5)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = QuanDense(n_units=5, in_channels=10)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.random.uniform(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        pass

    def test_layer_n1(self):
        if False:
            print('Hello World!')
        print(self.n1[0])

    def test_layer_n2(self):
        if False:
            return 10
        print(self.n2[0])

    def test_model_n3(self):
        if False:
            i = 10
            return i + 15
        print(self.n3[0])

    def test_exception(self):
        if False:
            while True:
                i = 10
        try:
            layer = QuanDense(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = QuanDense(n_units=5, use_gemm=True)
            out = layer(self.ni)
            self.fail('use gemm')
        except Exception as e:
            print(e)

class Layer_QuanDenseWithBN_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        print('-' * 20, 'Layer_QuanDenseWithBN_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = QuanDenseWithBN(n_units=5)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = QuanDenseWithBN(n_units=5, in_channels=10)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.random.uniform(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        pass

    def test_layer_n1(self):
        if False:
            i = 10
            return i + 15
        print(self.n1[0])

    def test_layer_n2(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.n2[0])

    def test_model_n3(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.n3[0])

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        try:
            layer = QuanDenseWithBN(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = QuanDenseWithBN(n_units=5, use_gemm=True)
            out = layer(self.ni)
            self.fail('use gemm')
        except Exception as e:
            print(e)

class Layer_TernaryDense_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        print('-' * 20, 'Layer_BinaryDense_Test', '-' * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]
        cls.ni = Input(cls.inputs_shape, name='input_layer')
        cls.layer1 = TernaryDense(n_units=5)
        nn = cls.layer1(cls.ni)
        cls.layer1._nodes_fixed = True
        cls.M = Model(inputs=cls.ni, outputs=nn)
        cls.layer2 = TernaryDense(n_units=5, in_channels=10)
        cls.layer2._nodes_fixed = True
        cls.inputs = tf.ones(cls.inputs_shape)
        cls.n1 = cls.layer1(cls.inputs)
        cls.n2 = cls.layer2(cls.inputs)
        cls.n3 = cls.M(cls.inputs, is_train=True)
        print(cls.layer1)
        print(cls.layer2)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        pass

    def test_layer_n1(self):
        if False:
            for i in range(10):
                print('nop')
        print(np.unique(self.n1.numpy().reshape(-1)))
        print(self.n1[0])

    def test_layer_n2(self):
        if False:
            for i in range(10):
                print('nop')
        print(np.unique(self.n2.numpy().reshape(-1)))
        print(self.n2[0])

    def test_model_n3(self):
        if False:
            print('Hello World!')
        print(np.unique(self.n3.numpy().reshape(-1)))
        print(self.n3[0])

    def test_exception(self):
        if False:
            while True:
                i = 10
        try:
            layer = TernaryDense(n_units=5)
            inputs = Input([4, 10, 5], name='ill_inputs')
            out = layer(inputs)
            self.fail('ill inputs')
        except Exception as e:
            print(e)
        try:
            layer = TernaryDense(n_units=5, use_gemm=True)
            out = layer(self.ni)
            self.fail('use gemm')
        except Exception as e:
            print(e)
if __name__ == '__main__':
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()