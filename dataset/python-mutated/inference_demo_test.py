"""Tests for CycleGAN inference demo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import numpy as np
import PIL
import tensorflow as tf
import inference_demo
import train
FLAGS = tf.flags.FLAGS
mock = tf.test.mock
tfgan = tf.contrib.gan

def _basenames_from_glob(file_glob):
    if False:
        return 10
    return [os.path.basename(file_path) for file_path in tf.gfile.Glob(file_glob)]

class InferenceDemoTest(tf.test.TestCase):

    def setUp(self):
        if False:
            return 10
        self._export_dir = os.path.join(FLAGS.test_tmpdir, 'export')
        self._ckpt_path = os.path.join(self._export_dir, 'model.ckpt')
        self._image_glob = os.path.join(FLAGS.test_srcdir, 'google3/third_party/tensorflow_models/gan/cyclegan/testdata', '*.jpg')
        self._genx_dir = os.path.join(FLAGS.test_tmpdir, 'genx')
        self._geny_dir = os.path.join(FLAGS.test_tmpdir, 'geny')

    @mock.patch.object(tfgan, 'gan_train', autospec=True)
    @mock.patch.object(train.data_provider, 'provide_custom_data', autospec=True)
    def testTrainingAndInferenceGraphsAreCompatible(self, mock_provide_custom_data, unused_mock_gan_train):
        if False:
            print('Hello World!')
        train_sess = tf.Session()
        FLAGS.image_set_x_file_pattern = '/tmp/x/*.jpg'
        FLAGS.image_set_y_file_pattern = '/tmp/y/*.jpg'
        FLAGS.batch_size = 3
        FLAGS.patch_size = 128
        FLAGS.generator_lr = 0.02
        FLAGS.discriminator_lr = 0.3
        FLAGS.train_log_dir = self._export_dir
        FLAGS.master = 'master'
        FLAGS.task = 0
        FLAGS.cycle_consistency_loss_weight = 2.0
        FLAGS.max_number_of_steps = 1
        mock_provide_custom_data.return_value = (tf.zeros([3, 4, 4, 3]), tf.zeros([3, 4, 4, 3]))
        train.main(None)
        init_op = tf.global_variables_initializer()
        train_sess.run(init_op)
        train_saver = tf.train.Saver()
        train_saver.save(train_sess, save_path=self._ckpt_path)
        tf.reset_default_graph()
        FLAGS.patch_dim = FLAGS.patch_size
        logging.info('dir_path: %s', os.listdir(self._export_dir))
        FLAGS.checkpoint_path = self._ckpt_path
        FLAGS.image_set_x_glob = self._image_glob
        FLAGS.image_set_y_glob = self._image_glob
        FLAGS.generated_x_dir = self._genx_dir
        FLAGS.generated_y_dir = self._geny_dir
        inference_demo.main(None)
        logging.info('gen x: %s', os.listdir(self._genx_dir))
        self.assertSetEqual(set(_basenames_from_glob(FLAGS.image_set_x_glob)), set(os.listdir(FLAGS.generated_y_dir)))
        self.assertSetEqual(set(_basenames_from_glob(FLAGS.image_set_y_glob)), set(os.listdir(FLAGS.generated_x_dir)))
        for directory in [FLAGS.generated_x_dir, FLAGS.generated_x_dir]:
            for base_name in os.listdir(directory):
                image_path = os.path.join(directory, base_name)
                self.assertRealisticImage(image_path)

    def assertRealisticImage(self, image_path):
        if False:
            for i in range(10):
                print('nop')
        logging.info('Testing %s for realism.', image_path)
        input_np = np.asarray(PIL.Image.open(image_path))
        self.assertEqual(len(input_np.shape), 3)
        self.assertGreaterEqual(input_np.shape[0], 50)
        self.assertGreaterEqual(input_np.shape[1], 50)
        self.assertGreater(np.mean(input_np), 20)
        self.assertGreater(np.var(input_np), 100)
if __name__ == '__main__':
    tf.test.main()