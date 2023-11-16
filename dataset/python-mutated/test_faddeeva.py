import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
from scipy.special._testutils import FuncData

class TestVoigtProfile:

    @pytest.mark.parametrize('x, sigma, gamma', [(np.nan, 1, 1), (0, np.nan, 1), (0, 1, np.nan), (1, np.nan, 0), (np.nan, 1, 0), (1, 0, np.nan), (np.nan, 0, 1), (np.nan, 0, 0)])
    def test_nan(self, x, sigma, gamma):
        if False:
            for i in range(10):
                print('nop')
        assert np.isnan(sc.voigt_profile(x, sigma, gamma))

    @pytest.mark.parametrize('x, desired', [(-np.inf, 0), (np.inf, 0)])
    def test_inf(self, x, desired):
        if False:
            print('Hello World!')
        assert sc.voigt_profile(x, 1, 1) == desired

    def test_against_mathematica(self):
        if False:
            for i in range(10):
                print('nop')
        points = np.array([[-7.89, 45.06, 6.66, 0.007792107366038881], [-0.05, 7.98, 24.13, 0.012068223646769913], [-13.98, 16.83, 42.37, 0.006244223636213236], [-12.66, 0.21, 6.32, 0.010052516161087379], [11.34, 4.25, 21.96, 0.011369892362727892], [-11.56, 20.4, 30.53, 0.007633276043209746], [-9.17, 25.61, 8.32, 0.011646345779083005], [16.59, 18.05, 2.5, 0.01363776883752681], [9.11, 2.12, 39.33, 0.007664404080727768], [-43.33, 0.3, 45.68, 0.003668046387533015]])
        FuncData(sc.voigt_profile, points, (0, 1, 2), 3, atol=0, rtol=1e-15).check()

    def test_symmetry(self):
        if False:
            return 10
        x = np.linspace(0, 10, 20)
        assert_allclose(sc.voigt_profile(x, 1, 1), sc.voigt_profile(-x, 1, 1), rtol=1e-15, atol=0)

    @pytest.mark.parametrize('x, sigma, gamma, desired', [(0, 0, 0, np.inf), (1, 0, 0, 0)])
    def test_corner_cases(self, x, sigma, gamma, desired):
        if False:
            return 10
        assert sc.voigt_profile(x, sigma, gamma) == desired

    @pytest.mark.parametrize('sigma1, gamma1, sigma2, gamma2', [(0, 1, 1e-16, 1), (1, 0, 1, 1e-16), (0, 0, 1e-16, 1e-16)])
    def test_continuity(self, sigma1, gamma1, sigma2, gamma2):
        if False:
            return 10
        x = np.linspace(1, 10, 20)
        assert_allclose(sc.voigt_profile(x, sigma1, gamma1), sc.voigt_profile(x, sigma2, gamma2), rtol=1e-16, atol=1e-16)