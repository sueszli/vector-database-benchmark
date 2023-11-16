from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from torch import nn, optim
from art.estimators.regression.pytorch import PyTorchRegressor
from tests.utils import TestBase, master_seed
logger = logging.getLogger(__name__)

class TestPytorchRegressor(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        master_seed(seed=1234, set_torch=True)
        super().setUpClass()

        class TestModel(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.features = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 10), nn.ReLU())
                self.output = nn.Linear(10, 1)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.output(self.features(x))
        cls.pytorch_model = TestModel()
        cls.art_model = PyTorchRegressor(model=cls.pytorch_model, loss=nn.modules.loss.MSELoss(), input_shape=(10,), optimizer=optim.Adam(cls.pytorch_model.parameters(), lr=0.01))
        cls.art_model.fit(cls.x_train_diabetes.astype(np.float32), cls.y_train_diabetes.astype(np.float32))

    def test_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(self.art_model, type(PyTorchRegressor(model=self.pytorch_model, loss=nn.modules.loss.MSELoss(), input_shape=(10,), optimizer=optim.Adam(self.pytorch_model.parameters(), lr=0.01))))
        with self.assertRaises(TypeError):
            PyTorchRegressor(model='model', loss=nn.modules.loss.MSELoss, input_shape=(10,), optimizer=optim.Adam(self.pytorch_model.parameters(), lr=0.01))

    def test_predict(self):
        if False:
            return 10
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4].astype(np.float32))
        y_expected = np.array([[19.2], [31.8], [13.8], [42.1]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        if False:
            i = 10
            return i + 15
        self.art_model.save(filename='test.file', path=None)
        self.art_model.save(filename='test.file', path='./')

    def test_input_shape(self):
        if False:
            return 10
        np.testing.assert_equal(self.art_model.input_shape, (10,))

    def test_compute_loss(self):
        if False:
            while True:
                i = 10
        test_loss = self.art_model.compute_loss(self.x_test_diabetes[:4].astype(np.float32), self.y_test_diabetes[:4].astype(np.float32))
        loss_expected = [3461.6, 5214.4, 3994.9, 9003.6]
        np.testing.assert_array_almost_equal(test_loss, loss_expected, decimal=1)

    def test_loss_gradient(self):
        if False:
            while True:
                i = 10
        grad = self.art_model.loss_gradient(self.x_test_diabetes[:4].astype(np.float32), self.y_test_diabetes[:4].astype(np.float32))
        grad_expected = [-49.4, 129.9, -170.1, -116.6, -225.2, -171.9, 174.6, -166.8, -223.9, -154.4]
        np.testing.assert_array_almost_equal(grad[0], grad_expected, decimal=1)

    def test_get_activations(self):
        if False:
            i = 10
            return i + 15
        act = self.art_model.get_activations(self.x_test_diabetes[:4].astype(np.float32), 1)
        act_expected = np.array([[19.2], [31.8], [13.8], [42.1]])
        np.testing.assert_array_almost_equal(act, act_expected, decimal=1)
if __name__ == '__main__':
    unittest.main()