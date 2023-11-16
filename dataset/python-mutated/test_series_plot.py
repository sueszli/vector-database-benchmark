import unittest
import pandas as pd
import numpy as np
from pyspark import pandas as ps
from pyspark.pandas.plot import PandasOnSparkPlotAccessor, BoxPlotBase
from pyspark.testing.pandasutils import have_plotly, plotly_requirement_message

class SeriesPlotTestsMixin:

    @property
    def pdf1(self):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

    @property
    def psdf1(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf1)

    @unittest.skipIf(not have_plotly, plotly_requirement_message)
    def test_plot_backends(self):
        if False:
            return 10
        plot_backend = 'plotly'
        with ps.option_context('plotting.backend', plot_backend):
            self.assertEqual(ps.options.plotting.backend, plot_backend)
            module = PandasOnSparkPlotAccessor._get_plot_backend(plot_backend)
            self.assertEqual(module.__name__, 'pyspark.pandas.plot.plotly')

    def test_plot_backends_incorrect(self):
        if False:
            for i in range(10):
                print('nop')
        fake_plot_backend = 'none_plotting_module'
        with ps.option_context('plotting.backend', fake_plot_backend):
            self.assertEqual(ps.options.plotting.backend, fake_plot_backend)
            with self.assertRaises(ValueError):
                PandasOnSparkPlotAccessor._get_plot_backend(fake_plot_backend)

    def test_box_summary(self):
        if False:
            return 10

        def check_box_summary(psdf, pdf):
            if False:
                print('Hello World!')
            k = 1.5
            (stats, fences) = BoxPlotBase.compute_stats(psdf['a'], 'a', whis=k, precision=0.01)
            outliers = BoxPlotBase.outliers(psdf['a'], 'a', *fences)
            whiskers = BoxPlotBase.calc_whiskers('a', outliers)
            fliers = BoxPlotBase.get_fliers('a', outliers, whiskers[0])
            expected_mean = pdf['a'].mean()
            expected_median = pdf['a'].median()
            expected_q1 = np.percentile(pdf['a'], 25)
            expected_q3 = np.percentile(pdf['a'], 75)
            iqr = expected_q3 - expected_q1
            expected_fences = (expected_q1 - k * iqr, expected_q3 + k * iqr)
            pdf['outlier'] = ~pdf['a'].between(fences[0], fences[1])
            expected_whiskers = (pdf.query('not outlier')['a'].min(), pdf.query('not outlier')['a'].max())
            expected_fliers = pdf.query('outlier')['a'].values
            self.assertEqual(expected_mean, stats['mean'])
            self.assertEqual(expected_median, stats['med'])
            self.assertEqual(expected_q1, stats['q1'] + 0.5)
            self.assertEqual(expected_q3, stats['q3'] - 0.5)
            self.assertEqual(expected_fences[0], fences[0] + 2.0)
            self.assertEqual(expected_fences[1], fences[1] - 2.0)
            self.assertEqual(expected_whiskers[0], whiskers[0])
            self.assertEqual(expected_whiskers[1], whiskers[1])
            self.assertEqual(expected_fliers, fliers)
        check_box_summary(self.psdf1, self.pdf1)
        check_box_summary(-self.psdf1, -self.pdf1)

class SeriesPlotTests(SeriesPlotTestsMixin, unittest.TestCase):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.plot.test_series_plot import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)