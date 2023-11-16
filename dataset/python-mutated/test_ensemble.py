from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import keras.backend as k
import numpy as np
from art.estimators.classification.ensemble import EnsembleClassifier
from tests.utils import TestBase, get_image_classifier_kr
logger = logging.getLogger(__name__)

class TestEnsembleClassifier(TestBase):
    """
    This class tests the ensemble classifier.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        classifier_1 = get_image_classifier_kr()
        classifier_2 = get_image_classifier_kr()
        cls.ensemble = EnsembleClassifier(classifiers=[classifier_1, classifier_2], clip_values=(0, 1))

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        k.clear_session()

    def test_fit(self):
        if False:
            return 10
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit(self.x_train_mnist, self.y_train_mnist)

    def test_fit_generator(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit_generator(None)

    def test_layers(self):
        if False:
            print('Hello World!')
        with self.assertRaises(NotImplementedError):
            self.ensemble.get_activations(self.x_test_mnist, layer=2)

    def test_predict(self):
        if False:
            i = 10
            return i + 15
        predictions = self.ensemble.predict(self.x_test_mnist, raw=False)
        self.assertTrue(predictions.shape, (self.n_test, 10))
        expected_predictions_1 = np.asarray([0.12109935, 0.0498215, 0.0993958, 0.06410097, 0.11366927, 0.04645343, 0.06419807, 0.30685693, 0.07616713, 0.05823759])
        np.testing.assert_array_almost_equal(predictions[0, :], expected_predictions_1, decimal=4)
        predictions_raw = self.ensemble.predict(self.x_test_mnist, raw=True)
        self.assertEqual(predictions_raw.shape, (2, self.n_test, 10))
        expected_predictions_2 = np.asarray([0.06054967, 0.02491075, 0.0496979, 0.03205048, 0.05683463, 0.02322672, 0.03209903, 0.15342847, 0.03808356, 0.02911879])
        np.testing.assert_array_almost_equal(predictions_raw[0, 0, :], expected_predictions_2, decimal=4)

    def test_loss_gradient(self):
        if False:
            while True:
                i = 10
        gradients = self.ensemble.loss_gradient(self.x_test_mnist, self.y_test_mnist, raw=False)
        self.assertEqual(gradients.shape, (self.n_test, 28, 28, 1))
        expected_predictions_1 = np.asarray([0.0559206, 0.05338925, 0.0648919, 0.07925165, -0.04029291, -0.11281465, 0.01850601, 0.00325054, 0.08163195, 0.03333949, 0.031766, -0.02420463, -0.07815556, -0.04698735, 0.10711591, 0.04086434, -0.03441073, 0.01071284, -0.04229195, -0.01386157, 0.02827487, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_predictions_1, decimal=4)
        gradients_2 = self.ensemble.loss_gradient(self.x_test_mnist, self.y_test_mnist, raw=True)
        self.assertEqual(gradients_2.shape, (2, self.n_test, 28, 28, 1))
        expected_predictions_2 = np.asarray([-0.02444103, -0.06092717, -0.0449727, 0.00737736, -0.0462507, -0.06225448, -0.08359106, -0.00270847, -0.009243, -0.00214317, -0.04728884, 0.00369186, 0.02211389, 0.02094269, 0.00219593, -0.02638348, 0.00148741, -0.004582, -0.00621604, 0.01604268, 0.0174383, -0.01077293, -0.00548703, -0.01247547, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients_2[0, 5, 14, :, 0], expected_predictions_2, decimal=4)

    def test_class_gradient(self):
        if False:
            print('Hello World!')
        gradients = self.ensemble.class_gradient(self.x_test_mnist, None, raw=False)
        self.assertEqual(gradients.shape, (self.n_test, 10, 28, 28, 1))
        expected_predictions_1 = np.asarray([-0.0010557447, -0.0010079544, -0.00077426434, 0.0017387432, 0.0021773507, 5.0880699e-05, 0.0016497371, 0.00261131, 0.006090431, 0.00041080985, 0.0025268078, -0.00036661502, -0.0030568996, -0.0011665225, 0.003890431, 0.00031726385, 0.001320326, -0.0001172093, -0.0014315104, -0.00047676818, 0.00097251288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_predictions_1, decimal=4)
        gradients_2 = self.ensemble.class_gradient(self.x_test_mnist, raw=True)
        self.assertEqual(gradients_2.shape, (2, self.n_test, 10, 28, 28, 1))
        expected_predictions_2 = np.asarray([-0.00052787235, -0.00050397718, -0.00038713217, 0.00086937158, 0.0010886753, 2.5440349e-05, 0.00082486856, 0.001305655, 0.0030452155, 0.00020540493, 0.0012634039, -0.00018330751, -0.0015284498, -0.00058326125, 0.0019452155, 0.00015863193, 0.000660163, -5.8604652e-05, -0.00071575522, -0.00023838409, 0.00048625644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients_2[0, 0, 5, 14, :, 0], expected_predictions_2, decimal=4)

    def test_repr(self):
        if False:
            print('Hello World!')
        repr_ = repr(self.ensemble)
        self.assertIn('art.estimators.classification.ensemble.EnsembleClassifier', repr_)
        self.assertIn('classifier_weights=array([0.5, 0.5])', repr_)
        self.assertIn('channels_first=False, clip_values=array([0., 1.], dtype=float32), preprocessing_defences=None, postprocessing_defences=None, preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)', repr_)
if __name__ == '__main__':
    unittest.main()