"""Tests for DSN model assembly functions."""
import numpy as np
import tensorflow as tf
import dsn

class HelperFunctionsTest(tf.test.TestCase):

    def testBasicDomainSeparationStartPoint(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session() as sess:
            step = tf.contrib.slim.get_or_create_global_step()
            sess.run(tf.global_variables_initializer())
            params = {'domain_separation_startpoint': 2}
            weight = dsn.dsn_loss_coefficient(params)
            weight_np = sess.run(weight)
            self.assertAlmostEqual(weight_np, 1e-10)
            step_op = tf.assign_add(step, 1)
            step_np = sess.run(step_op)
            weight = dsn.dsn_loss_coefficient(params)
            weight_np = sess.run(weight)
            self.assertAlmostEqual(weight_np, 1e-10)
            step_np = sess.run(step_op)
            tf.logging.info(step_np)
            weight = dsn.dsn_loss_coefficient(params)
            weight_np = sess.run(weight)
            self.assertAlmostEqual(weight_np, 1.0)

class DsnModelAssemblyTest(tf.test.TestCase):

    def _testBuildDefaultModel(self):
        if False:
            for i in range(10):
                print('nop')
        images = tf.to_float(np.random.rand(32, 28, 28, 1))
        labels = {}
        labels['classes'] = tf.one_hot(tf.to_int32(np.random.randint(0, 9, 32)), 10)
        params = {'use_separation': True, 'layers_to_regularize': 'fc3', 'weight_decay': 0.0, 'ps_tasks': 1, 'domain_separation_startpoint': 1, 'alpha_weight': 1, 'beta_weight': 1, 'gamma_weight': 1, 'recon_loss_name': 'sum_of_squares', 'decoder_name': 'small_decoder', 'encoder_name': 'default_encoder'}
        return (images, labels, params)

    def testBuildModelDann(self):
        if False:
            i = 10
            return i + 15
        (images, labels, params) = self._testBuildDefaultModel()
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'dann_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 6)

    def testBuildModelDannSumOfPairwiseSquares(self):
        if False:
            return 10
        (images, labels, params) = self._testBuildDefaultModel()
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'dann_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 6)

    def testBuildModelDannMultiPSTasks(self):
        if False:
            while True:
                i = 10
        (images, labels, params) = self._testBuildDefaultModel()
        params['ps_tasks'] = 10
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'dann_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 6)

    def testBuildModelMmd(self):
        if False:
            print('Hello World!')
        (images, labels, params) = self._testBuildDefaultModel()
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'mmd_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 6)

    def testBuildModelCorr(self):
        if False:
            print('Hello World!')
        (images, labels, params) = self._testBuildDefaultModel()
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'correlation_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 6)

    def testBuildModelNoDomainAdaptation(self):
        if False:
            i = 10
            return i + 15
        (images, labels, params) = self._testBuildDefaultModel()
        params['use_separation'] = False
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'none', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
            self.assertEqual(len(loss_tensors), 1)
            self.assertEqual(len(tf.contrib.losses.get_regularization_losses()), 0)

    def testBuildModelNoAdaptationWeightDecay(self):
        if False:
            for i in range(10):
                print('nop')
        (images, labels, params) = self._testBuildDefaultModel()
        params['use_separation'] = False
        params['weight_decay'] = 1e-05
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'none', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
            self.assertEqual(len(loss_tensors), 1)
            self.assertTrue(len(tf.contrib.losses.get_regularization_losses()) >= 1)

    def testBuildModelNoSeparation(self):
        if False:
            while True:
                i = 10
        (images, labels, params) = self._testBuildDefaultModel()
        params['use_separation'] = False
        with self.test_session():
            dsn.create_model(images, labels, tf.cast(tf.ones([32]), tf.bool), images, labels, 'dann_loss', params, 'dann_mnist')
            loss_tensors = tf.contrib.losses.get_losses()
        self.assertEqual(len(loss_tensors), 2)
if __name__ == '__main__':
    tf.test.main()