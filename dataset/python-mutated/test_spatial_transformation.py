from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import keras.backend as k
import numpy as np
from art.attacks.evasion.spatial_transformation import SpatialTransformation
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr
from tests.utils import get_image_classifier_pt, get_tabular_classifier_kr
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestSpatialTransformation(TestBase):
    """
    A unittest class for testing Spatial attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.n_train = 100
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_2_tensorflow_classifier(self):
        if False:
            i = 10
            return i + 15
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        attack_st = SpatialTransformation(tfc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False)
        x_train_adv = attack_st.generate(self.x_train_mnist)
        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.71, delta=0.02)
        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)
        x_test_adv = attack_st.generate(self.x_test_mnist)
        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        if sess is not None:
            sess.close()

    def test_4_keras_classifier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        attack_st = SpatialTransformation(krc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False)
        x_train_adv = attack_st.generate(self.x_train_mnist)
        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.71, delta=0.02)
        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)
        x_test_adv = attack_st.generate(self.x_test_mnist)
        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        k.clear_session()

    def test_3_pytorch_classifier(self):
        if False:
            print('Hello World!')
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test_mnist.copy()
        ptc = get_image_classifier_pt(from_logits=True)
        attack_st = SpatialTransformation(ptc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False)
        x_train__mnistadv = attack_st.generate(x_train_mnist)
        self.assertAlmostEqual(x_train__mnistadv[0, 0, 13, 18], 0.627451, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.57, delta=0.03)
        self.assertEqual(attack_st.attack_trans_x, 0)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 0.0)
        x_test_adv = attack_st.generate(x_test_mnist)
        self.assertLessEqual(abs(x_test_adv[0, 0, 14, 14] - 0.008591662), 0.01)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=1e-05)

    def test_5_failure_feature_vectors(self):
        if False:
            i = 10
            return i + 15
        attack_params = {'max_translation': 10.0, 'num_translations': 3, 'max_rotation': 30.0, 'num_rotations': 3}
        classifier = get_tabular_classifier_kr()
        attack = SpatialTransformation(classifier=classifier, verbose=False)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)
        with self.assertRaises(ValueError) as context:
            attack.generate(data)
        self.assertIn('Feature vectors detected.', str(context.exception))

    def test_check_params(self):
        if False:
            while True:
                i = 10
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = SpatialTransformation(ptc, max_translation=-1)
        with self.assertRaises(ValueError):
            _ = SpatialTransformation(ptc, num_translations=-1)
        with self.assertRaises(ValueError):
            _ = SpatialTransformation(ptc, max_rotation=-1)
        with self.assertRaises(ValueError):
            _ = SpatialTransformation(ptc, verbose='False')

    def test_1_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(SpatialTransformation, [BaseEstimator, NeuralNetworkMixin])
if __name__ == '__main__':
    unittest.main()