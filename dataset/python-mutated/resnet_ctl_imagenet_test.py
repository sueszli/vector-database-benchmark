"""Test the ResNet model with ImageNet data using CTL."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import tensorflow.compat.v2 as tf
from tensorflow.python.eager import context
from official.utils.testing import integration
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_ctl_imagenet_main

class CtlImagenetTest(tf.test.TestCase):
    """Unit tests for Keras ResNet with ImageNet using CTL."""
    _extra_flags = ['-batch_size', '4', '-train_steps', '4', '-use_synthetic_data', 'true']
    _tempdir = None

    def get_temp_dir(self):
        if False:
            i = 10
            return i + 15
        if not self._tempdir:
            self._tempdir = tempfile.mkdtemp(dir=super(CtlImagenetTest, self).get_temp_dir())
        return self._tempdir

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(CtlImagenetTest, cls).setUpClass()
        common.define_keras_flags()

    def setUp(self):
        if False:
            return 10
        super(CtlImagenetTest, self).setUp()
        imagenet_preprocessing.NUM_IMAGES['validation'] = 4

    def tearDown(self):
        if False:
            return 10
        super(CtlImagenetTest, self).tearDown()
        tf.io.gfile.rmtree(self.get_temp_dir())

    def test_end_to_end_no_dist_strat(self):
        if False:
            print('Hello World!')
        'Test Keras model with 1 GPU, no distribution strategy.'
        extra_flags = ['-distribution_strategy', 'off', '-model_dir', 'ctl_imagenet_no_dist_strat', '-data_format', 'channels_last']
        extra_flags = extra_flags + self._extra_flags
        integration.run_synthetic(main=resnet_ctl_imagenet_main.run, tmp_root=self.get_temp_dir(), extra_flags=extra_flags)

    def test_end_to_end_2_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Keras model with 2 GPUs.'
        num_gpus = '2'
        if context.num_gpus() < 2:
            num_gpus = '0'
        extra_flags = ['-num_gpus', num_gpus, '-distribution_strategy', 'default', '-model_dir', 'ctl_imagenet_2_gpu', '-data_format', 'channels_last']
        extra_flags = extra_flags + self._extra_flags
        integration.run_synthetic(main=resnet_ctl_imagenet_main.run, tmp_root=self.get_temp_dir(), extra_flags=extra_flags)
if __name__ == '__main__':
    assert tf.version.VERSION.startswith('2.')
    tf.test.main()