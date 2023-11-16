import unittest
import warnings
import pytest
from chainer import testing
from chainer.training import extensions
try:
    import matplotlib
    _available = True
except ImportError:
    _available = False

class TestPlotReport(unittest.TestCase):

    def test_available(self):
        if False:
            i = 10
            return i + 15
        if _available:
            self.assertTrue(extensions.PlotReport.available())
        else:
            with pytest.warns(UserWarning):
                self.assertFalse(extensions.PlotReport.available())

    @unittest.skipUnless(_available, 'matplotlib is not installed')
    def test_lazy_import(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            matplotlib.use('Agg')
            matplotlib.use('PS')
        self.assertEqual(len(w), 0)
testing.run_module(__name__, __file__)