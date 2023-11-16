from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
import GPy
from art.estimators.classification.GPy import GPyGaussianProcessClassifier
from tests.utils import TestBase, master_seed
logger = logging.getLogger(__name__)

class TestGPyGaussianProcessClassifier(TestBase):
    """
    This class tests the GPy Gaussian Process classifier.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)
        super().setUpClass()
        cls.y_train_iris_binary = cls.y_train_iris[:, 1]
        cls.y_test_iris_binary = cls.y_test_iris[:, 1]
        gpkern = GPy.kern.RBF(np.shape(cls.x_train_iris)[1])
        m = GPy.models.GPClassification(cls.x_train_iris, cls.y_train_iris_binary.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer='lbfgs')
        cls.classifier = GPyGaussianProcessClassifier(m)

    def setUp(self):
        if False:
            return 10
        master_seed(seed=1234)
        super().setUp()

    def test_predict(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(np.mean((self.classifier.predict(self.x_test_iris[:3])[:, 0] > 0.5) == self.y_test_iris_binary[:3]) > 0.6)
        outlier = np.ones(np.shape(self.x_test_iris[:3])) * 10.0
        self.assertTrue(np.sum(self.classifier.predict(outlier).flatten() == 0.5) == 6.0)

    def test_predict_unc(self):
        if False:
            while True:
                i = 10
        outlier = np.ones(np.shape(self.x_test_iris[:3])) * (np.max(self.x_test_iris.flatten()) * 10.0)
        self.assertTrue(np.mean(self.classifier.predict_uncertainty(outlier) > self.classifier.predict_uncertainty(self.x_test_iris[:3])) == 1.0)

    def test_loss_gradient(self):
        if False:
            print('Hello World!')
        grads = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris_binary[0:1])
        self.assertTrue(np.sum(grads < 0.0) == 3.0)
        self.assertTrue(np.sum(grads > 0.0) == 1.0)
        self.assertTrue(np.argmax(grads) == 2)

    def test_class_gradient(self):
        if False:
            while True:
                i = 10
        grads = self.classifier.class_gradient(self.x_test_iris[0:1], int(self.y_test_iris_binary[0:1]))
        self.assertTrue(np.sum(grads < 0.0) == 1.0)
        self.assertTrue(np.sum(grads > 0.0) == 3.0)
        self.assertTrue(np.argmax(grads) == 1)
if __name__ == '__main__':
    unittest.main()