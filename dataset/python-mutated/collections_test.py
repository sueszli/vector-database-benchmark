"""Tests for inception."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception.slim import slim

def get_variables(scope=None):
    if False:
        print('Hello World!')
    return slim.variables.get_variables(scope)

def get_variables_by_name(name):
    if False:
        for i in range(10):
            print('nop')
    return slim.variables.get_variables_by_name(name)

class CollectionsTest(tf.test.TestCase):

    def testVariables(self):
        if False:
            i = 10
            return i + 15
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d], batch_norm_params={'decay': 0.9997}):
                slim.inception.inception_v3(inputs)
            self.assertEqual(len(get_variables()), 388)
            self.assertEqual(len(get_variables_by_name('weights')), 98)
            self.assertEqual(len(get_variables_by_name('biases')), 2)
            self.assertEqual(len(get_variables_by_name('beta')), 96)
            self.assertEqual(len(get_variables_by_name('gamma')), 0)
            self.assertEqual(len(get_variables_by_name('moving_mean')), 96)
            self.assertEqual(len(get_variables_by_name('moving_variance')), 96)

    def testVariablesWithoutBatchNorm(self):
        if False:
            while True:
                i = 10
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d], batch_norm_params=None):
                slim.inception.inception_v3(inputs)
            self.assertEqual(len(get_variables()), 196)
            self.assertEqual(len(get_variables_by_name('weights')), 98)
            self.assertEqual(len(get_variables_by_name('biases')), 98)
            self.assertEqual(len(get_variables_by_name('beta')), 0)
            self.assertEqual(len(get_variables_by_name('gamma')), 0)
            self.assertEqual(len(get_variables_by_name('moving_mean')), 0)
            self.assertEqual(len(get_variables_by_name('moving_variance')), 0)

    def testVariablesByLayer(self):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d], batch_norm_params={'decay': 0.9997}):
                slim.inception.inception_v3(inputs)
            self.assertEqual(len(get_variables()), 388)
            self.assertEqual(len(get_variables('conv0')), 4)
            self.assertEqual(len(get_variables('conv1')), 4)
            self.assertEqual(len(get_variables('conv2')), 4)
            self.assertEqual(len(get_variables('conv3')), 4)
            self.assertEqual(len(get_variables('conv4')), 4)
            self.assertEqual(len(get_variables('mixed_35x35x256a')), 28)
            self.assertEqual(len(get_variables('mixed_35x35x288a')), 28)
            self.assertEqual(len(get_variables('mixed_35x35x288b')), 28)
            self.assertEqual(len(get_variables('mixed_17x17x768a')), 16)
            self.assertEqual(len(get_variables('mixed_17x17x768b')), 40)
            self.assertEqual(len(get_variables('mixed_17x17x768c')), 40)
            self.assertEqual(len(get_variables('mixed_17x17x768d')), 40)
            self.assertEqual(len(get_variables('mixed_17x17x768e')), 40)
            self.assertEqual(len(get_variables('mixed_8x8x2048a')), 36)
            self.assertEqual(len(get_variables('mixed_8x8x2048b')), 36)
            self.assertEqual(len(get_variables('logits')), 2)
            self.assertEqual(len(get_variables('aux_logits')), 10)

    def testVariablesToRestore(self):
        if False:
            print('Hello World!')
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d], batch_norm_params={'decay': 0.9997}):
                slim.inception.inception_v3(inputs)
            variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            self.assertEqual(len(variables_to_restore), 388)
            self.assertListEqual(variables_to_restore, get_variables())

    def testVariablesToRestoreWithoutLogits(self):
        if False:
            i = 10
            return i + 15
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d], batch_norm_params={'decay': 0.9997}):
                slim.inception.inception_v3(inputs, restore_logits=False)
            variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            self.assertEqual(len(variables_to_restore), 384)

    def testRegularizationLosses(self):
        if False:
            return 10
        batch_size = 5
        (height, width) = (299, 299)
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=4e-05):
                slim.inception.inception_v3(inputs)
            losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.assertEqual(len(losses), len(get_variables_by_name('weights')))

    def testTotalLossWithoutRegularization(self):
        if False:
            i = 10
            return i + 15
        batch_size = 5
        (height, width) = (299, 299)
        num_classes = 1001
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            dense_labels = tf.random_uniform((batch_size, num_classes))
            with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0):
                (logits, end_points) = slim.inception.inception_v3(inputs, num_classes=num_classes)
                slim.losses.cross_entropy_loss(logits, dense_labels, label_smoothing=0.1, weight=1.0)
                slim.losses.cross_entropy_loss(end_points['aux_logits'], dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')
            losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
            self.assertEqual(len(losses), 2)

    def testTotalLossWithRegularization(self):
        if False:
            return 10
        batch_size = 5
        (height, width) = (299, 299)
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            dense_labels = tf.random_uniform((batch_size, num_classes))
            with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=4e-05):
                (logits, end_points) = slim.inception.inception_v3(inputs, num_classes)
                slim.losses.cross_entropy_loss(logits, dense_labels, label_smoothing=0.1, weight=1.0)
                slim.losses.cross_entropy_loss(end_points['aux_logits'], dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')
            losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
            self.assertEqual(len(losses), 2)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.assertEqual(len(reg_losses), 98)
if __name__ == '__main__':
    tf.test.main()