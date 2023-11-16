import os
import unittest
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        print('\n#################################')
        cls.batch_size = 8
        cls.inputs_shape = [cls.batch_size, 100, 1]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')
        cls.n1 = tl.layers.Conv1dLayer(shape=(5, 1, 32), stride=2)(cls.input_layer)
        cls.n2 = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2)(cls.n1)
        cls.n3 = tl.layers.DeConv1dLayer(shape=(5, 64, 32), outputs_shape=(cls.batch_size, 50, 64), strides=(1, 2, 1), name='deconv1dlayer')(cls.n2)
        cls.n4 = tl.layers.SeparableConv1d(n_filter=32, filter_size=3, strides=2, padding='SAME', act='relu', name='separable_1d')(cls.n3)
        cls.n5 = tl.layers.SubpixelConv1d(scale=2, act=tf.nn.relu, in_channels=32, name='subpixel_1d')(cls.n4)
        cls.model = Model(inputs=cls.input_layer, outputs=cls.n5)
        print('Testing Conv1d model: \n', cls.model)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        pass

    def test_layer_n1(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [50, 32])

    def test_layer_n2(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [25, 32])

    def test_layer_n3(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 2)
        self.assertEqual(self.n3.get_shape().as_list()[1:], [50, 64])

    def test_layer_n4(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 3)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [25, 32])

    def test_layer_n5(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.n5.get_shape().as_list()[1:], [50, 16])

class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('\n#################################')
        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 400, 400, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')
        cls.n1 = tl.layers.Conv2dLayer(act=tf.nn.relu, shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME', b_init=tf.constant_initializer(value=0.0), name='conv2dlayer')(cls.input_layer)
        cls.n2 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d')(cls.n1)
        cls.n3 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, b_init=None, name='conv2d_no_bias')(cls.n2)
        cls.n4 = tl.layers.DeConv2dLayer(shape=(5, 5, 32, 32), outputs_shape=(cls.batch_size, 100, 100, 32), strides=(1, 2, 2, 1), name='deconv2dlayer')(cls.n3)
        cls.n5 = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d')(cls.n4)
        cls.n6 = tl.layers.DepthwiseConv2d(filter_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise')(cls.n5)
        cls.n7 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, in_channels=64, name='conv2d2')(cls.n6)
        cls.n8 = tl.layers.BinaryConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, in_channels=32, name='binaryconv2d')(cls.n7)
        cls.n9 = tl.layers.SeparableConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='separableconv2d')(cls.n8)
        cls.n10 = tl.layers.GroupConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), n_group=2, name='group')(cls.n9)
        cls.n11 = tl.layers.DorefaConv2d(n_filter=32, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='dorefaconv2d')(cls.n10)
        cls.n12 = tl.layers.TernaryConv2d(n_filter=64, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='ternaryconv2d')(cls.n11)
        cls.n13 = tl.layers.QuanConv2d(n_filter=32, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='quancnn2d')(cls.n12)
        cls.n14 = tl.layers.SubpixelConv2d(scale=2, act=tf.nn.relu, name='subpixelconv2d')(cls.n13)
        cls.n15 = tl.layers.QuanConv2dWithBN(n_filter=64, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='quancnnbn2d')(cls.n14)
        cls.model = Model(cls.input_layer, cls.n15)
        print('Testing Conv2d model: \n', cls.model)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        pass

    def test_layer_n1(self):
        if False:
            return 10
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [200, 200, 32])

    def test_layer_n2(self):
        if False:
            return 10
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n3(self):
        if False:
            return 10
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 1)
        self.assertEqual(self.n3.get_shape().as_list()[1:], [50, 50, 32])

    def test_layer_n4(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 2)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n5(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n5._info[0].layer.all_weights), 2)
        self.assertEqual(self.n5.get_shape().as_list()[1:], [200, 200, 32])

    def test_layer_n6(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.n6._info[0].layer.all_weights), 2)
        self.assertEqual(self.n6.get_shape().as_list()[1:], [200, 200, 64])

    def test_layer_n7(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n7._info[0].layer.all_weights), 2)
        self.assertEqual(self.n7.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n8(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.n8._info[0].layer.all_weights), 2)
        self.assertEqual(self.n8.get_shape().as_list()[1:], [50, 50, 64])

    def test_layer_n9(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.n9._info[0].layer.all_weights), 3)
        self.assertEqual(self.n9.get_shape().as_list()[1:], [24, 24, 32])

    def test_layer_n10(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.n10._info[0].layer.all_weights), 2)
        self.assertEqual(self.n10.get_shape().as_list()[1:], [12, 12, 64])

    def test_layer_n11(self):
        if False:
            return 10
        self.assertEqual(len(self.n11._info[0].layer.all_weights), 2)
        self.assertEqual(self.n11.get_shape().as_list()[1:], [12, 12, 32])

    def test_layer_n12(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.n12._info[0].layer.all_weights), 2)
        self.assertEqual(self.n12.get_shape().as_list()[1:], [12, 12, 64])

    def test_layer_n13(self):
        if False:
            return 10
        self.assertEqual(len(self.n13._info[0].layer.all_weights), 2)
        self.assertEqual(self.n13.get_shape().as_list()[1:], [12, 12, 32])

    def test_layer_n14(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.n14.get_shape().as_list()[1:], [24, 24, 8])

    def test_layer_n15(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.n15._info[0].layer.all_weights), 5)
        self.assertEqual(self.n15.get_shape().as_list()[1:], [24, 24, 64])

class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        print('\n#################################')
        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 20, 20, 20, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')
        cls.n1 = tl.layers.Conv3dLayer(shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))(cls.input_layer)
        cls.n2 = tl.layers.DeConv3dLayer(shape=(2, 2, 2, 128, 32), outputs_shape=(cls.batch_size, 20, 20, 20, 128), strides=(1, 2, 2, 2, 1))(cls.n1)
        cls.n3 = tl.layers.Conv3d(n_filter=64, filter_size=(3, 3, 3), strides=(3, 3, 3), act=tf.nn.relu, b_init=None, in_channels=128, name='conv3d_no_bias')(cls.n2)
        cls.n4 = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))(cls.n3)
        cls.model = Model(inputs=cls.input_layer, outputs=cls.n4)
        print('Testing Conv3d model: \n', cls.model)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        pass

    def test_layer_n1(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [10, 10, 10, 32])

    def test_layer_n2(self):
        if False:
            return 10
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [20, 20, 20, 128])

    def test_layer_n3(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 1)
        self.assertEqual(self.n3.get_shape().as_list()[1:], [7, 7, 7, 64])

    def test_layer_n4(self):
        if False:
            return 10
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 2)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [14, 14, 14, 32])

class Exception_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing exception in activation #####')

    def test_exception(cls):
        if False:
            print('Hello World!')
        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 400, 400, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')
        try:
            cls.n1 = tl.layers.Conv2dLayer(act='activation', shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME', b_init=tf.constant_initializer(value=0.0), name='conv2dlayer')(cls.input_layer)
        except Exception as e:
            cls.assertIsInstance(e, Exception)
            print(e)
if __name__ == '__main__':
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()