import os
import unittest
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *
from tests.utils import CustomTestCase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def RemoveDateInConfig(config):
    if False:
        print('Hello World!')
    config['version_info']['save_date'] = None
    return config

def basic_static_model():
    if False:
        for i in range(10):
            print('nop')
    ni = Input((None, 24, 24, 3))
    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name='conv1')(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name='conv2')(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)
    nn = Flatten(name='flatten')(nn)
    nn = Dense(100, act=None, name='dense1')(nn)
    M = Model(inputs=ni, outputs=nn)
    return M

class Model_Save_and_Load_without_weights(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        print('##### begin testing save_graph, load_graph, without weights #####')

    def test_save(self):
        if False:
            for i in range(10):
                print('nop')
        M1 = basic_static_model()
        print('Model config = \n', M1.config)
        print('Model = \n', M1)
        M1.save(filepath='basic_model_without_weights.hdf5', save_weights=False)
        M2 = Model.load(filepath='basic_model_without_weights.hdf5', load_weights=False)
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual(M1_config, M2_config)

def get_model(inputs_shape):
    if False:
        while True:
            i = 10
    ni = Input(inputs_shape)
    nn = Dropout(keep=0.8)(ni)
    nn = Dense(n_units=800, act=tf.nn.relu, in_channels=784)(nn)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=800, act=tf.nn.relu, in_channels=800)(nn)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=10, act=tf.nn.relu, in_channels=800)(nn)
    M = Model(inputs=ni, outputs=nn)
    return M

class Model_Save_with_weights(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        print('##### begin testing save_graph, after training, with weights #####')

    def test_save(self):
        if False:
            print('Hello World!')
        tl.logging.set_verbosity(tl.logging.DEBUG)
        (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))
        MLP = get_model([None, 784])
        print(MLP)
        n_epoch = 3
        batch_size = 500
        train_weights = MLP.trainable_weights
        optimizer = tf.optimizers.Adam(lr=0.0001)
        for epoch in range(n_epoch):
            print('epoch = ', epoch)
            for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                MLP.train()
                with tf.GradientTape() as tape:
                    _logits = MLP(X_batch)
                    _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))
        MLP.eval()
        (val_loss, val_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   val loss: {}'.format(val_loss / n_iter))
        print('   val acc:  {}'.format(val_acc / n_iter))
        MLP.save('MLP.hdf5')

class Model_Load_with_weights_and_train(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing load_graph, after training, with weights, and train again #####')

    def test_save(self):
        if False:
            print('Hello World!')
        MLP = Model.load('MLP.hdf5')
        MLP.eval()
        n_epoch = 3
        batch_size = 500
        train_weights = MLP.trainable_weights
        optimizer = tf.optimizers.Adam(lr=0.0001)
        (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))
        (val_loss, val_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   val loss: {}'.format(val_loss / n_iter))
        print('   val acc:  {}'.format(val_acc / n_iter))
        assert val_acc > 0.7
        for epoch in range(n_epoch):
            print('epoch = ', epoch)
            for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                MLP.train()
                with tf.GradientTape() as tape:
                    _logits = MLP(X_batch)
                    _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))
        MLP.save('MLP.hdf5')

def create_base_network(input_shape):
    if False:
        print('Hello World!')
    'Base network to be shared (eq. to feature extraction).\n    '
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, act=tf.nn.relu)(x)
    x = Dropout(0.9)(x)
    x = Dense(128, act=tf.nn.relu)(x)
    x = Dropout(0.9)(x)
    x = Dense(128, act=tf.nn.relu)(x)
    return Model(input, x)

def get_siamese_network(input_shape):
    if False:
        while True:
            i = 10
    'Create siamese network with shared base network as layer\n    '
    base_layer = create_base_network(input_shape).as_layer()
    ni_1 = Input(input_shape)
    ni_2 = Input(input_shape)
    nn_1 = base_layer(ni_1)
    nn_2 = base_layer(ni_2)
    return Model(inputs=[ni_1, ni_2], outputs=[nn_1, nn_2])

class Reuse_ModelLayer_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing save_graph, load_graph, including ModelLayer and reuse #####')

    def test_save(self):
        if False:
            i = 10
            return i + 15
        input_shape = (None, 784)
        M1 = get_siamese_network(input_shape)
        print('Model config = \n', M1.config)
        print('Model = \n', M1)
        M1.save(filepath='siamese.hdf5', save_weights=False)
        M2 = Model.load(filepath='siamese.hdf5', load_weights=False)
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual(M1_config, M2_config)

class Vgg_LayerList_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing save_graph, load_graph, including LayerList #####')

    def test_save(self):
        if False:
            for i in range(10):
                print('nop')
        M1 = tl.models.vgg16(mode='static')
        print('Model config = \n', M1.config)
        print('Model = \n', M1)
        M1.save(filepath='vgg.hdf5', save_weights=False)
        M2 = Model.load(filepath='vgg.hdf5', load_weights=False)
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual(M1_config, M2_config)

class List_inputs_outputs_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        print('##### begin testing model with list inputs and outputs #####')

    def test_list_inputs_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        ni_1 = Input(shape=[4, 16])
        ni_2 = Input(shape=[4, 32])
        a_1 = Dense(80)(ni_1)
        b_1 = Dense(160)(ni_2)
        concat = Concat()([a_1, b_1])
        a_2 = Dense(10)(concat)
        b_2 = Dense(20)(concat)
        M1 = Model(inputs=[ni_1, ni_2], outputs=[a_2, b_2])
        print('Model config = \n', M1.config)
        print('Model = \n', M1)
        M1.save(filepath='list.hdf5', save_weights=False)
        M2 = Model.load(filepath='list.hdf5', load_weights=False)
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual(M1_config, M2_config)

class Lambda_layer_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing lambda layer #####')

    def test_lambda_layer_no_para_no_args(self):
        if False:
            return 10
        x = tl.layers.Input([8, 3], name='input')
        y = tl.layers.Lambda(lambda x: 2 * x, name='lambda')(x)
        M1 = tl.models.Model(x, y)
        M1.save('lambda_no_para_no_args.hdf5')
        M2 = tl.models.Model.load('lambda_no_para_no_args.hdf5')
        print(M1)
        print(M2)
        M1.eval()
        M2.eval()
        npInput = np.zeros((8, 3)) + 3
        output1 = M1(npInput).numpy()
        output2 = M1(npInput).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual((output1 == output2).all(), True)
        self.assertEqual(M1_config, M2_config)

    def test_lambda_layer_no_para_with_args(self):
        if False:
            while True:
                i = 10

        def customize_func(x, foo=42):
            if False:
                for i in range(10):
                    print('nop')
            return foo * x
        x = tl.layers.Input([8, 3], name='input')
        y = tl.layers.Lambda(customize_func, fn_args={'foo': 3}, name='lambda')(x)
        M1 = tl.models.Model(x, y)
        M1.save('lambda_no_para_with_args.hdf5')
        M2 = tl.models.Model.load('lambda_no_para_with_args.hdf5')
        print(M1)
        print(M2)
        M1.eval()
        M2.eval()
        npInput = np.zeros((8, 3)) + 3
        output1 = M1(npInput).numpy()
        output2 = M2(npInput).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual((output1 == output2).all(), True)
        self.assertEqual((output1 == np.zeros((8, 3)) + 9).all(), True)
        self.assertEqual(M1_config, M2_config)

    def test_lambda_layer_keras_model(self):
        if False:
            while True:
                i = 10
        input_shape = [100, 5]
        in_2 = tl.layers.Input(input_shape, name='input')
        layers = [tf.keras.layers.Dense(10, activation=tf.nn.relu), tf.keras.layers.Dense(5, activation=tf.nn.sigmoid), tf.keras.layers.Dense(1, activation=tf.nn.relu)]
        perceptron = tf.keras.Sequential(layers)
        _ = perceptron(np.random.random(input_shape).astype(np.float32))
        plambdalayer = tl.layers.Lambda(perceptron, perceptron.trainable_variables)(in_2)
        M2 = tl.models.Model(inputs=in_2, outputs=plambdalayer)
        M2.save('M2_keras.hdf5')
        M4 = Model.load('M2_keras.hdf5')
        M2.eval()
        M4.eval()
        npInput = np.zeros(input_shape) + 3
        output2 = M2(npInput).numpy()
        output4 = M4(npInput).numpy()
        M2_config = RemoveDateInConfig(M2.config)
        M4_config = RemoveDateInConfig(M4.config)
        self.assertEqual((output2 == output4).all(), True)
        self.assertEqual(M2_config, M4_config)
        ori_weights = M4.all_weights
        ori_val = ori_weights[1].numpy()
        modify_val = np.zeros_like(ori_val) + 10
        M4.all_weights[1].assign(modify_val)
        M4 = Model.load('M2_keras.hdf5')
        self.assertLess(np.max(np.abs(ori_val - M4.all_weights[1].numpy())), 1e-07)

    def test_lambda_layer_keras_layer(self):
        if False:
            return 10
        input_shape = [100, 5]
        in_1 = tl.layers.Input(input_shape, name='input')
        denselayer = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        _ = denselayer(np.random.random(input_shape).astype(np.float32))
        dlambdalayer = tl.layers.Lambda(denselayer, denselayer.trainable_variables)(in_1)
        M1 = tl.models.Model(inputs=in_1, outputs=dlambdalayer)
        M1.save('M1_keras.hdf5')
        M3 = Model.load('M1_keras.hdf5')
        M1.eval()
        M3.eval()
        npInput = np.zeros(input_shape) + 3
        output1 = M1(npInput).numpy()
        output3 = M3(npInput).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M3_config = RemoveDateInConfig(M3.config)
        self.assertEqual((output1 == output3).all(), True)
        self.assertEqual(M1_config, M3_config)
        ori_weights = M3.all_weights
        ori_val = ori_weights[1].numpy()
        modify_val = np.zeros_like(ori_val) + 10
        M3.all_weights[1].assign(modify_val)
        M3 = Model.load('M1_keras.hdf5')
        self.assertLess(np.max(np.abs(ori_val - M3.all_weights[1].numpy())), 1e-07)

class ElementWise_lambda_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        print('##### begin testing elementwise lambda layer #####')

    def test_elementwise_no_para_with_args(self):
        if False:
            i = 10
            return i + 15

        def func(noise, mean, std, foo=42):
            if False:
                while True:
                    i = 10
            return mean + noise * tf.exp(std * 0.5) + foo
        noise = tl.layers.Input([100, 1])
        mean = tl.layers.Input([100, 1])
        std = tl.layers.Input([100, 1])
        out = tl.layers.ElementwiseLambda(fn=func, fn_args={'foo': 84}, name='elementwiselambda')([noise, mean, std])
        M1 = Model(inputs=[noise, mean, std], outputs=out)
        M1.save('elementwise_npwa.hdf5')
        M2 = Model.load('elementwise_npwa.hdf5')
        M1.eval()
        M2.eval()
        ipt = [np.zeros((100, 1)) + 11, np.zeros((100, 1)) + 21, np.zeros((100, 1)) + 31]
        output1 = M1(ipt).numpy()
        output2 = M2(ipt).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual((output1 == output2).all(), True)
        self.assertEqual(M1_config, M2_config)

    def test_elementwise_no_para_no_args(self):
        if False:
            print('Hello World!')

        def func(noise, mean, std, foo=42):
            if False:
                i = 10
                return i + 15
            return mean + noise * tf.exp(std * 0.5) + foo
        noise = tl.layers.Input([100, 1])
        mean = tl.layers.Input([100, 1])
        std = tl.layers.Input([100, 1])
        out = tl.layers.ElementwiseLambda(fn=func, name='elementwiselambda')([noise, mean, std])
        M1 = Model(inputs=[noise, mean, std], outputs=out)
        M1.save('elementwise_npna.hdf5')
        M2 = Model.load('elementwise_npna.hdf5')
        M1.eval()
        M2.eval()
        ipt = [np.zeros((100, 1)) + 11, np.zeros((100, 1)) + 21, np.zeros((100, 1)) + 31]
        output1 = M1(ipt).numpy()
        output2 = M2(ipt).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual((output1 == output2).all(), True)
        self.assertEqual(M1_config, M2_config)

    def test_elementwise_lambda_func(self):
        if False:
            i = 10
            return i + 15
        noise = tl.layers.Input([100, 1])
        mean = tl.layers.Input([100, 1])
        std = tl.layers.Input([100, 1])
        out = tl.layers.ElementwiseLambda(fn=lambda x, y, z: x + y * tf.exp(z * 0.5), name='elementwiselambda')([noise, mean, std])
        M1 = Model(inputs=[noise, mean, std], outputs=out)
        M1.save('elementwise_lambda.hdf5')
        M2 = Model.load('elementwise_lambda.hdf5')
        M1.eval()
        M2.eval()
        ipt = [(np.zeros((100, 1)) + 11).astype(np.float32), (np.zeros((100, 1)) + 21).astype(np.float32), (np.zeros((100, 1)) + 31).astype(np.float32)]
        output1 = M1(ipt).numpy()
        output2 = M2(ipt).numpy()
        M1_config = RemoveDateInConfig(M1.config)
        M2_config = RemoveDateInConfig(M2.config)
        self.assertEqual((output1 == output2).all(), True)
        self.assertEqual(M1_config, M2_config)

class basic_dynamic_model(Model):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(basic_dynamic_model, self).__init__()
        self.conv1 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=3, name='conv1')
        self.pool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')
        self.conv2 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=16, name='conv2')
        self.pool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(100, act=None, in_channels=576, name='dense1')
        self.dense2 = Dense(10, act=None, in_channels=100, name='dense2')

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Dynamic_config_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        print('##### begin testing exception in dynamic mode #####')

    def test_dynamic_config(self):
        if False:
            for i in range(10):
                print('nop')
        M1 = basic_dynamic_model()
        print(M1.config)
        for layer in M1.all_layers:
            print(layer.config)

class Exception_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        print('##### begin testing exception in dynamic mode #####')

    def test_exception(self):
        if False:
            return 10
        M1 = basic_dynamic_model()
        try:
            M1.save('dynamic.hdf5', save_weights=False)
        except Exception as e:
            self.assertIsInstance(e, RuntimeError)
            print(e)
        M2 = basic_static_model()
        M2.save('basic_static_mode.hdf5', save_weights=False)
        try:
            M3 = Model.load('basic_static_mode.hdf5')
        except Exception as e:
            self.assertIsInstance(e, RuntimeError)
            print(e)
if __name__ == '__main__':
    tl.logging.set_verbosity(tl.logging.DEBUG)
    unittest.main()