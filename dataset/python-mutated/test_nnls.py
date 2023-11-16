import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.optimize import nnls

class TestNNLS:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.rng = np.random.default_rng(1685225766635251)

    def test_nnls(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.arange(25.0).reshape(-1, 5)
        x = np.arange(5.0)
        y = a @ x
        (x, res) = nnls(a, y)
        assert res < 1e-07
        assert np.linalg.norm(a @ x - y) < 1e-07

    def test_nnls_tall(self):
        if False:
            while True:
                i = 10
        a = self.rng.uniform(low=-10, high=10, size=[50, 10])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[10]))
        x[::2] = 0
        b = a @ x
        (xact, rnorm) = nnls(a, b, atol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
        assert_allclose(xact, x, rtol=0.0, atol=1e-10)
        assert rnorm < 1e-12

    def test_nnls_wide(self):
        if False:
            print('Hello World!')
        a = self.rng.uniform(low=-10, high=10, size=[100, 120])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[120]))
        x[::2] = 0
        b = a @ x
        (xact, rnorm) = nnls(a, b, atol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
        assert_allclose(xact, x, rtol=0.0, atol=1e-10)
        assert rnorm < 1e-12

    def test_maxiter(self):
        if False:
            i = 10
            return i + 15
        a = self.rng.uniform(size=(5, 10))
        b = self.rng.uniform(size=5)
        with assert_raises(RuntimeError):
            nnls(a, b, maxiter=1)