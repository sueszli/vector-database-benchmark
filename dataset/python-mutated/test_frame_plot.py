import pandas as pd
import numpy as np
from pyspark import pandas as ps
from pyspark.pandas.config import set_option, reset_option, option_context
from pyspark.pandas.plot import TopNPlotBase, SampledPlotBase, HistogramPlotBase, BoxPlotBase
from pyspark.pandas.exceptions import PandasNotImplementedError
from pyspark.testing.pandasutils import PandasOnSparkTestCase

class DataFramePlotTestsMixin:

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        set_option('plotting.max_rows', 2000)
        set_option('plotting.sample_ratio', None)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        reset_option('plotting.max_rows')
        reset_option('plotting.sample_ratio')
        super().tearDownClass()

    def test_missing(self):
        if False:
            i = 10
            return i + 15
        psdf = ps.DataFrame(np.random.rand(2500, 4), columns=['a', 'b', 'c', 'd'])
        unsupported_functions = ['hexbin']
        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError, 'method.*DataFrame.*{}.*not implemented'.format(name)):
                getattr(psdf.plot, name)()

    def test_topn_max_rows(self):
        if False:
            i = 10
            return i + 15
        pdf = pd.DataFrame(np.random.rand(2500, 4), columns=['a', 'b', 'c', 'd'])
        psdf = ps.from_pandas(pdf)
        data = TopNPlotBase().get_top_n(psdf)
        self.assertEqual(len(data), 2000)

    def test_sampled_plot_with_ratio(self):
        if False:
            while True:
                i = 10
        with option_context('plotting.sample_ratio', 0.5):
            pdf = pd.DataFrame(np.random.rand(2500, 4), columns=['a', 'b', 'c', 'd'])
            psdf = ps.from_pandas(pdf)
            data = SampledPlotBase().get_sampled(psdf)
            self.assertEqual(round(len(data) / 2500, 1), 0.5)

    def test_sampled_plot_with_max_rows(self):
        if False:
            while True:
                i = 10
        pdf = pd.DataFrame(np.random.rand(2000, 4), columns=['a', 'b', 'c', 'd'])
        psdf = ps.from_pandas(pdf)
        data = SampledPlotBase().get_sampled(psdf)
        self.assertEqual(round(len(data) / 2000, 1), 1)

    def test_compute_hist_single_column(self):
        if False:
            for i in range(10):
                print('nop')
        psdf = ps.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])
        expected_bins = np.linspace(1, 50, 11)
        bins = HistogramPlotBase.get_bins(psdf[['a']].to_spark(), 10)
        expected_histogram = np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1])
        histogram = HistogramPlotBase.compute_hist(psdf[['a']], bins)[0]
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))
        self.assert_eq(pd.Series(expected_histogram, name='a'), histogram, almost=True)

    def test_compute_hist_multi_columns(self):
        if False:
            print('Hello World!')
        expected_bins = np.linspace(1, 50, 11)
        psdf = ps.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50], 'b': [50, 50, 30, 30, 30, 24, 10, 5, 4, 3, 1]})
        bins = HistogramPlotBase.get_bins(psdf.to_spark(), 10)
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))
        expected_histograms = [np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1]), np.array([4, 1, 0, 0, 1, 3, 0, 0, 0, 2])]
        histograms = HistogramPlotBase.compute_hist(psdf, bins)
        expected_names = ['a', 'b']
        for (histogram, expected_histogram, expected_name) in zip(histograms, expected_histograms, expected_names):
            self.assert_eq(pd.Series(expected_histogram, name=expected_name), histogram, almost=True)

    def test_compute_box_multi_columns(self):
        if False:
            for i in range(10):
                print('nop')

        def check_box_multi_columns(psdf):
            if False:
                return 10
            k = 1.5
            multicol_stats = BoxPlotBase.compute_multicol_stats(psdf, ['a', 'b', 'c'], whis=k, precision=0.01)
            multicol_outliers = BoxPlotBase.multicol_outliers(psdf, multicol_stats)
            multicol_whiskers = BoxPlotBase.calc_multicol_whiskers(['a', 'b', 'c'], multicol_outliers)
            for col in ['a', 'b', 'c']:
                col_stats = multicol_stats[col]
                col_whiskers = multicol_whiskers[col]
                (stats, fences) = BoxPlotBase.compute_stats(psdf[col], col, whis=k, precision=0.01)
                outliers = BoxPlotBase.outliers(psdf[col], col, *fences)
                whiskers = BoxPlotBase.calc_whiskers(col, outliers)
                self.assertEqual(stats['mean'], col_stats['mean'])
                self.assertEqual(stats['med'], col_stats['med'])
                self.assertEqual(stats['q1'], col_stats['q1'])
                self.assertEqual(stats['q3'], col_stats['q3'])
                self.assertEqual(fences[0], col_stats['lfence'])
                self.assertEqual(fences[1], col_stats['ufence'])
                self.assertEqual(whiskers[0], col_whiskers['min'])
                self.assertEqual(whiskers[1], col_whiskers['max'])
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50], 'b': [3, 2, 5, 4, 5, 6, 8, 8, 11, 60, 90], 'c': [-30, -2, 5, 4, 5, 6, -8, 8, 11, 12, 18]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])
        psdf = ps.from_pandas(pdf)
        check_box_multi_columns(psdf)
        check_box_multi_columns(-psdf)

class DataFramePlotTests(DataFramePlotTestsMixin, PandasOnSparkTestCase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.plot.test_frame_plot import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)