import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning

class TestDistDependenceMeasures:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        '\n        Values were obtained via the R `energy` package.\n\n        R code:\n        ------\n        > dcov.test(x, y, R=200)\n\n            dCov independence test (permutation test)\n\n        data:  index 1, replicates 200\n        nV^2 = 45829, p-value = 0.004975\n        sample estimates:\n            dCov\n        47.86925\n\n        > DCOR(x, y)\n        $dCov\n        [1] 47.86925\n\n        $dCor\n        [1] 0.9999704\n\n        $dVarX\n        [1] 47.28702\n\n        $dVarY\n        [1] 48.46151\n        '
        np.random.seed(3)
        cls.x = np.array(range(1, 101)).reshape((20, 5))
        cls.y = cls.x + np.log(cls.x)
        cls.dcor_exp = 0.9999704
        cls.dcov_exp = 47.86925
        cls.dvar_x_exp = 47.28702
        cls.dvar_y_exp = 48.46151
        cls.pval_emp_exp = 0.004975
        cls.test_stat_emp_exp = 45829
        cls.S_exp = 5686.03162
        cls.test_stat_asym_exp = 2.8390102
        cls.pval_asym_exp = 0.00452

    def test_input_validation_nobs(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='same number of observations'):
            ddm.distance_covariance_test(self.x[:2, :], self.y)

    def test_input_validation_unknown_method(self):
        if False:
            return 10
        with pytest.raises(ValueError, match="Unknown 'method' parameter"):
            ddm.distance_covariance_test(self.x, self.y, method='wrong_name')

    def test_statistic_value_asym_method(self):
        if False:
            for i in range(10):
                print('nop')
        (statistic, pval, method) = ddm.distance_covariance_test(self.x, self.y, method='asym')
        assert method == 'asym'
        assert_almost_equal(statistic, self.test_stat_asym_exp, 4)
        assert_almost_equal(pval, self.pval_asym_exp, 3)

    def test_statistic_value_emp_method(self):
        if False:
            return 10
        with pytest.warns(HypothesisTestWarning):
            (statistic, pval, method) = ddm.distance_covariance_test(self.x, self.y, method='emp')
        assert method == 'emp'
        assert_almost_equal(statistic, self.test_stat_emp_exp, 0)
        assert_almost_equal(pval, self.pval_emp_exp, 3)

    def test_fallback_to_asym_method(self):
        if False:
            for i in range(10):
                print('nop')
        match_text = 'The asymptotic approximation will be used'
        with pytest.warns(UserWarning, match=match_text):
            (statistic, pval, _) = ddm.distance_covariance_test(self.x, self.y, method='emp', B=200)
            assert_almost_equal(statistic, self.test_stat_emp_exp, 0)
            assert_almost_equal(pval, self.pval_asym_exp, 3)

    def test_statistics_for_2d_input(self):
        if False:
            for i in range(10):
                print('nop')
        stats = ddm.distance_statistics(np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float))
        assert_almost_equal(stats.test_statistic, self.test_stat_emp_exp, 0)
        assert_almost_equal(stats.distance_correlation, self.dcor_exp, 4)
        assert_almost_equal(stats.distance_covariance, self.dcov_exp, 4)
        assert_almost_equal(stats.dvar_x, self.dvar_x_exp, 4)
        assert_almost_equal(stats.dvar_y, self.dvar_y_exp, 4)
        assert_almost_equal(stats.S, self.S_exp, 4)

    def test_statistics_for_1d_input(self):
        if False:
            print('Hello World!')
        x = np.array(range(1, 21), dtype=float)
        y = x + np.log(x)
        stats = ddm.distance_statistics(x, y)
        assert_almost_equal(stats.test_statistic, 398.94623, 5)
        assert_almost_equal(stats.distance_correlation, 0.9996107, 4)
        assert_almost_equal(stats.distance_covariance, 4.4662414, 4)
        assert_almost_equal(stats.dvar_x, 4.2294799, 4)
        assert_almost_equal(stats.dvar_y, 4.7199304, 4)
        assert_almost_equal(stats.S, 49.8802, 4)

    def test_results_on_the_iris_dataset(self):
        if False:
            return 10
        '\n        R code example from the `energy` package documentation for\n        `energy::distance_covariance.test`:\n\n        > x <- iris[1:50, 1:4]\n        > y <- iris[51:100, 1:4]\n        > set.seed(1)\n        > dcov.test(x, y, R=200)\n\n            dCov independence test (permutation test)\n\n        data:  index 1, replicates 200\n        nV^2 = 0.5254, p-value = 0.9552\n        sample estimates:\n             dCov\n        0.1025087\n        '
        try:
            iris = get_rdataset('iris').data.values[:, :4]
        except IGNORED_EXCEPTIONS:
            pytest.skip('Failed with HTTPError or URLError, these are random')
        x = np.asarray(iris[:50], dtype=float)
        y = np.asarray(iris[50:100], dtype=float)
        stats = ddm.distance_statistics(x, y)
        assert_almost_equal(stats.test_statistic, 0.5254, 4)
        assert_almost_equal(stats.distance_correlation, 0.3060479, 4)
        assert_almost_equal(stats.distance_covariance, 0.1025087, 4)
        assert_almost_equal(stats.dvar_x, 0.2712927, 4)
        assert_almost_equal(stats.dvar_y, 0.4135274, 4)
        assert_almost_equal(stats.S, 0.667456, 4)
        (test_statistic, _, method) = ddm.distance_covariance_test(x, y, B=199)
        assert_almost_equal(test_statistic, 0.5254, 4)
        assert method == 'emp'

    def test_results_on_the_quakes_dataset(self):
        if False:
            i = 10
            return i + 15
        '\n        R code:\n        ------\n\n        > data("quakes")\n        > x = quakes[1:50, 1:3]\n        > y = quakes[51:100, 1:3]\n        > dcov.test(x, y, R=200)\n\n            dCov independence test (permutation test)\n\n        data:  index 1, replicates 200\n        nV^2 = 45046, p-value = 0.4577\n        sample estimates:\n            dCov\n        30.01526\n        '
        try:
            quakes = get_rdataset('quakes').data.values[:, :3]
        except IGNORED_EXCEPTIONS:
            pytest.skip('Failed with HTTPError or URLError, these are random')
        x = np.asarray(quakes[:50], dtype=float)
        y = np.asarray(quakes[50:100], dtype=float)
        stats = ddm.distance_statistics(x, y)
        assert_almost_equal(np.round(stats.test_statistic), 45046, 0)
        assert_almost_equal(stats.distance_correlation, 0.1894193, 4)
        assert_almost_equal(stats.distance_covariance, 30.01526, 4)
        assert_almost_equal(stats.dvar_x, 170.1702, 4)
        assert_almost_equal(stats.dvar_y, 147.5545, 4)
        assert_almost_equal(stats.S, 52265, 0)
        (test_statistic, _, method) = ddm.distance_covariance_test(x, y, B=199)
        assert_almost_equal(np.round(test_statistic), 45046, 0)
        assert method == 'emp'

    def test_dcor(self):
        if False:
            print('Hello World!')
        assert_almost_equal(ddm.distance_correlation(self.x, self.y), self.dcor_exp, 4)

    def test_dcov(self):
        if False:
            for i in range(10):
                print('nop')
        assert_almost_equal(ddm.distance_covariance(self.x, self.y), self.dcov_exp, 4)

    def test_dvar(self):
        if False:
            while True:
                i = 10
        assert_almost_equal(ddm.distance_variance(self.x), self.dvar_x_exp, 4)