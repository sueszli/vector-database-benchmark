"""Tests for gamma_mapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.platform import flags
from morph_net.op_regularizers import gamma_mapper
FLAGS = flags.FLAGS
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
NUM_CHANNELS = 3

def get_op(name):
    if False:
        return 10
    return tf.get_default_graph().get_operation_by_name(name)
CONV1_GAMMA = [0.1 * x for x in range(13)]
SEP_CONV_GAMMA = [0.07 * x for x in range(23)]
CKPT_FILE_NAME = 'ckpt'

def build_model():
    if False:
        i = 10
        return i + 15
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    conv1 = layers.conv2d(image, 13, (3, 3), padding='SAME', scope='conv1')
    layers.separable_conv2d(conv1, 23, (3, 3), 1, scope='sep_conv')

def setUpModule():
    if False:
        return 10
    "Save a model for later loading it.\n\n  This is the only way we're aware of for assigning values to variables\n  irrespectively of their type (regular or partitioned), since partitioned\n  variables do not support assignment.\n  "
    with tf.Graph().as_default():
        params = {'normalizer_fn': layers.batch_norm, 'normalizer_params': {'scale': True}}
        with tf.contrib.framework.arg_scope([layers.conv2d, layers.separable_conv2d], **params):
            build_model()
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            conv_gamma = tf.get_variable('conv1/BatchNorm/gamma')
            sep_gamma = tf.get_variable('sep_conv/BatchNorm/gamma')
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        s.run([conv_gamma.assign(CONV1_GAMMA), sep_gamma.assign(SEP_CONV_GAMMA)])
        saver = tf.train.Saver()
        saver.save(s, os.path.join(FLAGS.test_tmpdir, CKPT_FILE_NAME))

class ConvGammaMapperTest(parameterized.TestCase, tf.test.TestCase):

    def createMapper(self, connectivity):
        if False:
            while True:
                i = 10
        if connectivity:
            return gamma_mapper.ConvGammaMapperByConnectivity()
        return gamma_mapper.ConvGammaMapperByName()

    def setUp(self):
        if False:
            return 10
        tf.reset_default_graph()

    def TestSuccess(self, connectivity, partitioning, fused, use_resource):
        if False:
            for i in range(10):
                print('nop')
        params = {'trainable': True, 'normalizer_fn': layers.batch_norm, 'normalizer_params': {'scale': True, 'fused': fused}}
        partitioner = tf.fixed_size_partitioner(2) if partitioning else None
        with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner, use_resource=use_resource):
            with tf.contrib.framework.arg_scope([layers.conv2d, layers.separable_conv2d], **params):
                build_model()
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(FLAGS.test_tmpdir, CKPT_FILE_NAME))
        mapper = self.createMapper(connectivity)
        conv = get_op('conv1/Conv2D')
        sep_conv = get_op('sep_conv/separable_conv2d')
        with sess.as_default():
            self.assertAllClose(CONV1_GAMMA, mapper.get_gamma(conv).eval())
            self.assertAllClose(SEP_CONV_GAMMA, mapper.get_gamma(sep_conv).eval())

    def testSuccess(self):
        if False:
            while True:
                i = 10
        for connectivity in (False, True):
            for partitioning in (False, True):
                for fused in (False, True):
                    if connectivity and (not fused):
                        continue
                    for use_resource in (False, True):
                        tf.reset_default_graph()
                        self.TestSuccess(connectivity, partitioning, fused, use_resource)

    @parameterized.named_parameters(('_name_nopart', False, False), ('_name_part', False, True), ('_conn_nopart', True, False), ('_conn_part', True, True))
    def testNoBatchNorm(self, connectivity, partitioning):
        if False:
            while True:
                i = 10
        partitioner = tf.fixed_size_partitioner(2) if partitioning else None
        with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
            build_model()
        mapper = self.createMapper(connectivity)
        conv = get_op('conv1/Conv2D')
        self.assertEqual(None, mapper.get_gamma(conv))

    @parameterized.named_parameters(('_name_nopart', False), ('_conn_nopart', True))
    def testNotAConv(self, connectivity):
        if False:
            print('Hello World!')
        build_model()
        mapper = self.createMapper(connectivity)
        bias_add = get_op('conv1/BiasAdd')
        with self.assertRaises(ValueError):
            mapper.get_gamma(bias_add)

    @parameterized.named_parameters(('_name_nopart', False), ('_conn_nopart', True))
    def testNotAnOpButATensor(self, connectivity):
        if False:
            for i in range(10):
                print('nop')
        build_model()
        mapper = self.createMapper(connectivity)
        conv = get_op('conv1/Conv2D')
        with self.assertRaises(ValueError):
            mapper.get_gamma(conv.outputs[0])

    @parameterized.named_parameters(('_name_nopart', False), ('_conn_nopart', True))
    def testNotInGraph(self, connectivity):
        if False:
            for i in range(10):
                print('nop')
        mapper = self.createMapper(connectivity)
        build_model()
        conv = get_op('conv1/Conv2D')
        with self.assertRaises(KeyError):
            mapper.get_gamma(conv)

def build_resnet(block_fn, resnet_fn):
    if False:
        while True:
            i = 10
    params = {'trainable': True, 'normalizer_fn': layers.batch_norm, 'normalizer_params': {'is_training': True, 'scale': True, 'fused': True}}
    with arg_scope([layers.conv2d], **params):
        with arg_scope([layers.batch_norm], **params['normalizer_params']):
            blocks = [block_fn('block1', base_depth=7, num_units=2, stride=2), block_fn('block2', base_depth=13, num_units=2, stride=2)]
            image = tf.constant(0.0, shape=[1, 2, 2, NUM_CHANNELS])
            return resnet_fn(image, blocks, include_root_block=False, is_training=False)[0]

class ConvGammaMapperByConnectivityResnetTest(parameterized.TestCase, tf.test.TestCase):

    def assertGammaMatchesConv(self, mapper, prefix):
        if False:
            while True:
                i = 10
        conv = get_op(prefix + '/Conv2D')
        gamma = mapper.get_gamma(conv)
        self.assertTrue(gamma.op.name.startswith(prefix + '/BatchNorm/gamma'))

    def assertConvsConnectedToGammas(self, conv_names, gamma_prefixes, mapper):
        if False:
            i = 10
            return i + 15
        'Asserts that each convolution is connected to each gamma.\n\n    Args:\n      conv_names: A list of strings representing names of Conv2D operations.\n      gamma_prefixes: A list of strings representing name prefixes of gamma\n        variables (we only verify prefixes because suffixes may depend on\n        whether we have partitioning or no).\n      mapper: a ConvGammaMapperByConnectivity object\n    '

        def make_set(item):
            if False:
                print('Hello World!')
            return item if isinstance(item, set) else set([item])
        convs = [get_op(conv_name) for conv_name in conv_names]
        gamma_sets = [make_set(mapper.get_gamma(conv)) for conv in convs]
        if len(gamma_sets) > 1:
            for i in range(1, len(gamma_sets)):
                self.assertEqual(gamma_sets[i], gamma_sets[0])
        actual_gamma_names = sorted([g.op.name for g in gamma_sets[0]])
        gamma_prefixes = sorted(gamma_prefixes)
        for (expected, actual) in zip(gamma_prefixes, actual_gamma_names):
            self.assertTrue(actual.startswith(expected))

    def testSuccessResnetV2(self):
        if False:
            for i in range(10):
                print('nop')
        build_resnet(resnet_v2.resnet_v2_block, resnet_v2.resnet_v2)
        mapper = gamma_mapper.ConvGammaMapperByConnectivity()
        for block in (1, 2):
            for unit in (1, 2):
                for conv in (1, 2):
                    self.assertGammaMatchesConv(mapper, 'resnet_v2/block%d/unit_%d/bottleneck_v2/conv%d' % (block, unit, conv))
        self.assertConvsConnectedToGammas(['resnet_v2/block1/unit_1/bottleneck_v2/shortcut/Conv2D', 'resnet_v2/block1/unit_1/bottleneck_v2/conv3/Conv2D'], ['resnet_v2/block1/unit_2/bottleneck_v2/preact/gamma', 'resnet_v2/block2/unit_1/bottleneck_v2/preact/gamma'], mapper)
        self.assertConvsConnectedToGammas(['resnet_v2/block1/unit_2/bottleneck_v2/conv3/Conv2D'], ['resnet_v2/block2/unit_1/bottleneck_v2/preact/gamma'], mapper)
        self.assertConvsConnectedToGammas(['resnet_v2/block2/unit_1/bottleneck_v2/shortcut/Conv2D', 'resnet_v2/block2/unit_1/bottleneck_v2/conv3/Conv2D'], ['resnet_v2/block2/unit_2/bottleneck_v2/preact/gamma', 'resnet_v2/postnorm/gamma'], mapper)
        self.assertConvsConnectedToGammas(['resnet_v2/block2/unit_2/bottleneck_v2/conv3/Conv2D'], ['resnet_v2/postnorm/gamma'], mapper)

    def testSuccessResnetV1(self):
        if False:
            for i in range(10):
                print('nop')
        build_resnet(resnet_v1.resnet_v1_block, resnet_v1.resnet_v1)
        mapper = gamma_mapper.ConvGammaMapperByConnectivity()
        for block in (1, 2):
            self.assertGammaMatchesConv(mapper, 'resnet_v1/block%d/unit_1/bottleneck_v1/shortcut' % block)
            for unit in (1, 2):
                for conv in (1, 2, 3):
                    self.assertGammaMatchesConv(mapper, 'resnet_v1/block%d/unit_%d/bottleneck_v1/conv%d' % (block, unit, conv))
if __name__ == '__main__':
    tf.test.main()