from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
kidney_table = StringIO('Days      Duration Weight ID\n    0.0      1      1      1\n    2.0      1      1      2\n    1.0      1      1      3\n    3.0      1      1      4\n    0.0      1      1      5\n    2.0      1      1      6\n    0.0      1      1      7\n    5.0      1      1      8\n    6.0      1      1      9\n    8.0      1      1     10\n    2.0      1      2      1\n    4.0      1      2      2\n    7.0      1      2      3\n   12.0      1      2      4\n   15.0      1      2      5\n    4.0      1      2      6\n    3.0      1      2      7\n    1.0      1      2      8\n    5.0      1      2      9\n   20.0      1      2     10\n   15.0      1      3      1\n   10.0      1      3      2\n    8.0      1      3      3\n    5.0      1      3      4\n   25.0      1      3      5\n   16.0      1      3      6\n    7.0      1      3      7\n   30.0      1      3      8\n    3.0      1      3      9\n   27.0      1      3     10\n    0.0      2      1      1\n    1.0      2      1      2\n    1.0      2      1      3\n    0.0      2      1      4\n    4.0      2      1      5\n    2.0      2      1      6\n    7.0      2      1      7\n    4.0      2      1      8\n    0.0      2      1      9\n    3.0      2      1     10\n    5.0      2      2      1\n    3.0      2      2      2\n    2.0      2      2      3\n    0.0      2      2      4\n    1.0      2      2      5\n    1.0      2      2      6\n    3.0      2      2      7\n    6.0      2      2      8\n    7.0      2      2      9\n    9.0      2      2     10\n   10.0      2      3      1\n    8.0      2      3      2\n   12.0      2      3      3\n    3.0      2      3      4\n    7.0      2      3      5\n   15.0      2      3      6\n    4.0      2      3      7\n    9.0      2      3      8\n    6.0      2      3      9\n    1.0      2      3     10\n')
kidney_table.seek(0)
kidney_table = read_csv(kidney_table, sep='\\s+', engine='python').astype(int)

class TestAnovaLM:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.data = kidney_table
        cls.kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)', data=cls.data).fit()

    def test_results(self):
        if False:
            i = 10
            return i + 15
        Df = np.array([1, 2, 2, 54])
        sum_sq = np.array([2.339693, 16.97129, 0.6356584, 28.9892])
        mean_sq = np.array([2.339693, 8.485645, 0.3178292, 0.536837])
        f_value = np.array([4.358293, 15.80674, 0.5920404, np.nan])
        pr_f = np.array([0.0415617, 3.944502e-06, 0.5567479, np.nan])
        results = anova_lm(self.kidney_lm)
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, sum_sq, 4)
        np.testing.assert_almost_equal(results['F'].values, f_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, pr_f)

class TestAnovaLMNoconstant:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.data = kidney_table
        cls.kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight) - 1', data=cls.data).fit()

    def test_results(self):
        if False:
            return 10
        Df = np.array([2, 2, 2, 54])
        sum_sq = np.array([158.6415227, 16.97129, 0.6356584, 28.9892])
        mean_sq = np.array([79.3207613, 8.485645, 0.3178292, 0.536837])
        f_value = np.array([147.7557648, 15.80674, 0.5920404, np.nan])
        pr_f = np.array([1.262324e-22, 3.944502e-06, 0.5567479, np.nan])
        results = anova_lm(self.kidney_lm)
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, sum_sq, 4)
        np.testing.assert_almost_equal(results['F'].values, f_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, pr_f)

class TestAnovaLMCompare(TestAnovaLM):

    def test_results(self):
        if False:
            print('Hello World!')
        new_model = ols('np.log(Days+1) ~ C(Duration) + C(Weight)', self.data).fit()
        results = anova_lm(new_model, self.kidney_lm)
        Res_Df = np.array([56, 54])
        RSS = np.array([29.62486, 28.9892])
        Df = np.array([0, 2])
        Sum_of_Sq = np.array([np.nan, 0.6356584])
        F = np.array([np.nan, 0.5920404])
        PrF = np.array([np.nan, 0.5567479])
        np.testing.assert_equal(results['df_resid'].values, Res_Df)
        np.testing.assert_almost_equal(results['ssr'].values, RSS, 4)
        np.testing.assert_almost_equal(results['df_diff'].values, Df)
        np.testing.assert_almost_equal(results['ss_diff'].values, Sum_of_Sq)
        np.testing.assert_almost_equal(results['F'].values, F)
        np.testing.assert_almost_equal(results['Pr(>F)'].values, PrF)

class TestAnovaLMCompareNoconstant(TestAnovaLM):

    def test_results(self):
        if False:
            while True:
                i = 10
        new_model = ols('np.log(Days+1) ~ C(Duration) + C(Weight) - 1', self.data).fit()
        results = anova_lm(new_model, self.kidney_lm)
        Res_Df = np.array([56, 54])
        RSS = np.array([29.62486, 28.9892])
        Df = np.array([0, 2])
        Sum_of_Sq = np.array([np.nan, 0.6356584])
        F = np.array([np.nan, 0.5920404])
        PrF = np.array([np.nan, 0.5567479])
        np.testing.assert_equal(results['df_resid'].values, Res_Df)
        np.testing.assert_almost_equal(results['ssr'].values, RSS, 4)
        np.testing.assert_almost_equal(results['df_diff'].values, Df)
        np.testing.assert_almost_equal(results['ss_diff'].values, Sum_of_Sq)
        np.testing.assert_almost_equal(results['F'].values, F)
        np.testing.assert_almost_equal(results['Pr(>F)'].values, PrF)

class TestAnova2(TestAnovaLM):

    def test_results(self):
        if False:
            return 10
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([3.067066, 13.27205, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F_value = np.array([5.667033, 12.26141, 0.1760025, np.nan])
        PrF = np.array([0.02106078, 4.487909e-05, 0.8391231, np.nan])
        results = anova_lm(anova_ii, typ='II')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova2Noconstant(TestAnovaLM):

    def test_results(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum) - 1', data).fit()
        Sum_Sq = np.array([154.7131692, 13.27205, 0.1905093, 27.60181])
        Df = np.array([2, 2, 2, 51])
        F_value = np.array([142.9321191, 12.26141, 0.1760025, np.nan])
        PrF = np.array([1.238624e-21, 4.487909e-05, 0.8391231, np.nan])
        results = anova_lm(anova_ii, typ='II')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova2HC0(TestAnovaLM):

    def test_results(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F = np.array([6.972744, 13.7804, 0.1709936, np.nan])
        PrF = np.array([0.01095599, 1.641682e-05, 0.8433081, np.nan])
        results = anova_lm(anova_ii, typ='II', robust='hc0')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova2HC1(TestAnovaLM):

    def test_results(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F = np.array([6.238771, 12.32983, 0.1529943, np.nan])
        PrF = np.array([0.01576555, 4.285456e-05, 0.858527, np.nan])
        results = anova_lm(anova_ii, typ='II', robust='hc1')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova2HC2(TestAnovaLM):

    def test_results(self):
        if False:
            print('Hello World!')
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F = np.array([6.267499, 12.25354, 0.1501224, np.nan])
        PrF = np.array([0.01554009, 4.511826e-05, 0.8609815, np.nan])
        results = anova_lm(anova_ii, typ='II', robust='hc2')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova2HC3(TestAnovaLM):

    def test_results(self):
        if False:
            return 10
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F = np.array([5.633786, 10.89842, 0.1317223, np.nan])
        PrF = np.array([0.02142223, 0.0001145965, 0.8768817, np.nan])
        results = anova_lm(anova_ii, typ='II', robust='hc3')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova3(TestAnovaLM):

    def test_results(self):
        if False:
            i = 10
            return i + 15
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F_value = np.array([279.7545, 5.367071, 12.43245, 0.1760025, np.nan])
        PrF = np.array([2.379855e-22, 0.02457384, 3.999431e-05, 0.8391231, np.nan])
        results = anova_lm(anova_iii, typ='III')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova3HC0(TestAnovaLM):

    def test_results(self):
        if False:
            i = 10
            return i + 15
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([298.3404, 5.723638, 13.76069, 0.1709936, np.nan])
        PrF = np.array([5.876255e-23, 0.02046031, 1.662826e-05, 0.8433081, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc0')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova3HC1(TestAnovaLM):

    def test_results(self):
        if False:
            return 10
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([266.9361, 5.12115, 12.3122, 0.1529943, np.nan])
        PrF = np.array([6.54355e-22, 0.02792296, 4.336712e-05, 0.858527, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc1')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova3HC2(TestAnovaLM):

    def test_results(self):
        if False:
            return 10
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([264.5137, 5.074677, 12.19158, 0.1501224, np.nan])
        PrF = np.array([7.958286e-22, 0.02860926, 4.704831e-05, 0.8609815, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc2')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)

class TestAnova3HC3(TestAnovaLM):

    def test_results(self):
        if False:
            return 10
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([234.4026, 4.496996, 10.79903, 0.1317223, np.nan])
        PrF = np.array([1.037224e-20, 0.03883841, 0.0001228716, 0.8768817, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc3')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)