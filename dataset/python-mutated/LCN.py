""" LCN to estimate 3D human poses from 2D poses.

# Reference:
- [pose_lcn](
    https://github.com/rujiewu/pose_lcn)

"""
import numpy as np
import tensorflow as tf
from tensorlayer.layers import Layer, Dropout, Dense, Input, BatchNorm, Reshape, Elementwise
from tensorlayer.models import Model
from tensorlayer import logging
from .common import mask_weight, neighbour_matrix
BATCH_SIZE = 200
M_0 = 17
IN_F = 2
IN_JOINTS = 17
OUT_JOINTS = 17
F = 64
NUM_LAYERS = 3
weights_url = {'link': 'https://pan.baidu.com/s/1HBHWsAfyAlNaavw0iyUmUQ', 'password': 'ec07'}

class Base_layer(Layer):

    def __init__(self, F=F, in_joints=IN_JOINTS, out_joints=OUT_JOINTS, regularization=0.0, max_norm=True, residual=True, mask_type='locally_connected', neighbour_matrix=neighbour_matrix, init_type='ones', in_F=IN_F):
        if False:
            return 10
        super().__init__()
        self.F = F
        self.in_joints = in_joints
        self.regularizers = []
        self.regularization = regularization
        self.max_norm = max_norm
        self.out_joints = out_joints
        self.residual = residual
        self.mask_type = mask_type
        self.init_type = init_type
        self.in_F = in_F
        assert neighbour_matrix.shape[0] == neighbour_matrix.shape[1]
        assert neighbour_matrix.shape[0] == in_joints
        self.neighbour_matrix = neighbour_matrix
        self._initialize_mask()

    def _initialize_mask(self):
        if False:
            while True:
                i = 10
        '\n        Parameter\n            mask_type\n                locally_connected\n                locally_connected_learnable\n            init_type\n                same: use L to init learnable part in mask\n                ones: use 1 to init learnable part in mask\n                random: use random to init learnable part in mask\n        '
        if 'locally_connected' in self.mask_type:
            assert self.neighbour_matrix is not None
            L = self.neighbour_matrix.T
            assert L.shape == (self.in_joints, self.in_joints)
            if 'learnable' not in self.mask_type:
                self.mask = tf.constant(L)
            else:
                if self.init_type == 'same':
                    initializer = L
                elif self.init_type == 'ones':
                    initializer = tf.initializers.ones
                elif self.init_type == 'random':
                    initializer = tf.random.uniform
                var_mask = tf.Variable(name='mask', shape=[self.in_joints, self.out_joints] if self.init_type != 'same' else None, dtype=tf.float32, initial_value=initializer)
                var_mask = tf.nn.softmax(var_mask, axis=0)
                self.mask = var_mask * tf.constant(L != 0, dtype=tf.float32)

    def _get_weights(self, name, initializer, shape, regularization=True, trainable=True):
        if False:
            while True:
                i = 10
        var = tf.Variable(initial_value=initializer(shape=shape, dtype=tf.float32), name=name, trainable=True)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        if trainable is True:
            if self._trainable_weights is None:
                self._trainable_weights = list()
            self._trainable_weights.append(var)
        else:
            if self._nontrainable_weights is None:
                self._nontrainable_weights = list()
            self._nontrainable_weights.append(var)
        return var

    def kaiming(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf\n\n        Args\n            shape: dimensions of the tf array to initialize\n            dtype: data type of the array\n            partition_info: (Optional) info about how the variable is partitioned.\n                See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26\n                Needed to be used as an initializer.\n        Returns\n            Tensorflow array with initial weights\n        '
        return tf.random.truncated_normal(shape, dtype=dtype) * tf.sqrt(2 / float(shape[0]))

    def mask_weights(self, weights):
        if False:
            return 10
        return mask_weight(weights)

class Mask_layer(Base_layer):

    def __init__(self, in_channels=17, out_channels=None, name=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        (self.w_name, self.b_name) = name
        if self.in_channels:
            self.build(None)
            self._built = True

    def build(self, inputs_shape):
        if False:
            i = 10
            return i + 15
        if self.in_channels is None:
            self.in_channels = inputs_shape[1]
        self.weight = self._get_weights(self.w_name, self.kaiming, [self.in_channels, self.out_channels], regularization=self.regularization != 0)
        self.bias = self._get_weights(self.b_name, self.kaiming, [self.out_channels], regularization=self.regularization != 0)
        self.weight = tf.clip_by_norm(self.weight, 1) if self.max_norm else self.weight
        self.weight = self.mask_weights(self.weight)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        outputs = tf.matmul(x, self.weight) + self.bias
        return outputs

class End_layer(Base_layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def build(self, inputs_shape):
        if False:
            while True:
                i = 10
        pass

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = inputs
        x = tf.reshape(x, [-1, self.in_joints, self.in_F])
        y = tf.reshape(y, [-1, self.out_joints, 3])
        y = tf.concat([x[:, :, :2] + y[:, :, :2], tf.expand_dims(y[:, :, 2], axis=-1)], axis=2)
        y = tf.reshape(y, [-1, self.out_joints * 3])
        return y

def batch_normalization_warp(y):
    if False:
        return 10
    (_, output_size) = y.get_shape()
    output_size = int(output_size)
    out_F = int(output_size / IN_JOINTS)
    y = Reshape([-1, IN_JOINTS, out_F])(y)
    y = BatchNorm(act='lrelu', epsilon=0.001)(y)
    y = Reshape([-1, output_size])(y)
    return y

def two_linear_train(inputs, idx):
    if False:
        return 10
    '\n    Make a bi-linear block with optional residual connection\n\n    Args\n        xin: the batch that enters the block\n        idx: integer. Number of layer (for naming/scoping)\n        Returns\n    y: the batch after it leaves the block\n    '
    output_size = IN_JOINTS * F
    input_size1 = int(inputs.get_shape()[1])
    output = Mask_layer(in_channels=input_size1, out_channels=output_size, name=['w2' + str(idx), 'b2' + str(idx)])(inputs)
    output = batch_normalization_warp(output)
    output = Dropout(keep=0.8)(output)
    input_size2 = int(output.get_shape()[1])
    output = Mask_layer(in_channels=input_size2, out_channels=output_size, name=['w3_' + str(idx), 'b3_' + str(idx)])(output)
    output = batch_normalization_warp(output)
    output = Dropout(keep=0.8)(output)
    output = Elementwise(combine_fn=tf.add)([inputs, output])
    return output

def cgcnn_train():
    if False:
        while True:
            i = 10
    input_layer = Input(shape=(BATCH_SIZE, M_0 * IN_F))
    output = Mask_layer(in_channels=IN_JOINTS * IN_F, out_channels=IN_JOINTS * F, name=['w1', 'b1'])(input_layer)
    output = batch_normalization_warp(output)
    output = Dropout(keep=0.8)(output)
    for idx in range(NUM_LAYERS):
        output = two_linear_train(output, idx)
    input_size4 = int(output.get_shape()[1])
    output = Mask_layer(in_channels=input_size4, out_channels=OUT_JOINTS * 3, name=['w4', 'b4'])(output)
    output = End_layer()([input_layer, output])
    network = Model(inputs=input_layer, outputs=output)
    return network

def two_linear_inference(xin):
    if False:
        i = 10
        return i + 15
    '\n    Make a bi-linear block with optional residual connection\n\n    Args\n        xin: the batch that enters the block\n    y: the batch after it leaves the block\n    '
    output_size = IN_JOINTS * F
    output = Dense(n_units=output_size, act=None)(xin)
    output = batch_normalization_warp(output)
    output = Dense(n_units=output_size, act=None)(output)
    output = batch_normalization_warp(output)
    y = Elementwise(tf.add)([xin, output])
    return y

def cgcnn_inference():
    if False:
        while True:
            i = 10
    input_layer = Input(shape=(BATCH_SIZE, M_0 * IN_F))
    output = Dense(n_units=IN_JOINTS * F, act=None)(input_layer)
    output = batch_normalization_warp(output)
    for i in range(3):
        output = two_linear_inference(output)
    output = Dense(n_units=OUT_JOINTS * 3, act=None)(output)
    output = End_layer()([input_layer, output])
    network = Model(inputs=input_layer, outputs=output)
    return network

def restore_params(network, model_path='model.npz'):
    if False:
        for i in range(10):
            print('nop')
    logging.info('Restore pre-trained weights')
    try:
        npz = np.load(model_path, allow_pickle=True)
    except:
        print('Download the model file, placed in the /model ')
        print('Weights download: ', weights_url['link'], 'password:', weights_url['password'])
    txt_path = 'model/pose_weights_config.txt'
    f = open(txt_path, 'r')
    line = f.readlines()
    for i in range(len(line)):
        if len(npz[line[i].strip()].shape) == 2:
            _weight = mask_weight(npz[line[i].strip()])
        else:
            _weight = npz[line[i].strip()]
        network.all_weights[i].assign(_weight)
        logging.info('  Loading weights %s in %s' % (network.all_weights[i].shape, network.all_weights[i].name))

def CGCNN(pretrained=True):
    if False:
        while True:
            i = 10
    'Pre-trained LCN model.\n\n    Parameters\n    ------------\n    pretrained : boolean\n        Whether to load pretrained weights. Default False.\n\n    Examples\n    ---------\n    LCN to estimate 3D human poses from 2D poses, see `computer_vision.py\n    <https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/app/computer_vision.py>`__\n    With TensorLayer\n\n    >>> # get the whole model, without pre-trained LCN parameters\n    >>> lcn = tl.app.CGCNN(pretrained=False)\n    >>> # get the whole model, restore pre-trained LCN parameters\n    >>> lcn = tl.app.CGCNN(pretrained=True)\n    >>> # use for inferencing\n    >>> output = lcn(img, is_train=False)\n\n    '
    if pretrained:
        network = cgcnn_inference()
        restore_params(network, model_path='model/lcn_model.npz')
    else:
        network = cgcnn_train()
    return network