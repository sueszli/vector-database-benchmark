import os
import unittest
import numpy
import six
import chainer
from chainer import testing
from chainer.training import extensions
try:
    import matplotlib
    _available = True
except ImportError:
    _available = False

class TestVariableStatisticsPlot(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        stop_trigger = (2, 'iteration')
        extension_trigger = (1, 'iteration')
        self.filename = 'variable_statistics_plot_test.png'
        self.trainer = testing.get_trainer_with_mock_updater(stop_trigger=stop_trigger)
        x = numpy.random.rand(1, 2, 3)
        self.extension = extensions.VariableStatisticsPlot(chainer.variable.Variable(x), trigger=extension_trigger, filename=self.filename)
        self.trainer.extend(self.extension, extension_trigger)

    @unittest.skipUnless(_available, 'matplotlib is not installed')
    def test_run_and_save_plot(self):
        if False:
            while True:
                i = 10
        matplotlib.use('Agg')
        try:
            self.trainer.run()
        finally:
            os.remove(os.path.join(self.trainer.out, self.filename))

@testing.parameterize({'shape': (2, 7, 3), 'n': 5, 'reservoir_size': 3})
class TestReservoir(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.xs = [numpy.random.uniform(-1, 1, self.shape) for i in range(self.n)]

    def test_reservoir_size(self):
        if False:
            for i in range(10):
                print('nop')
        self.reservoir = extensions.variable_statistics_plot.Reservoir(size=self.reservoir_size, data_shape=self.shape)
        for x in self.xs:
            self.reservoir.add(x)
        (idxs, data) = self.reservoir.get_data()
        assert len(idxs) == self.reservoir_size
        assert len(data) == self.reservoir_size
        assert idxs.ndim == 1
        assert data[0].shape == self.xs[0].shape
        testing.assert_allclose(idxs, numpy.sort(idxs))

@testing.parameterize({'shape': (2, 7, 3)})
class TestStatistician(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = numpy.random.uniform(-1, 1, self.shape)

    def test_statistician_percentile(self):
        if False:
            print('Hello World!')
        self.percentile_sigmas = (0.0, 50.0, 100.0)
        self.statistician = extensions.variable_statistics_plot.Statistician(collect_mean=True, collect_std=True, percentile_sigmas=self.percentile_sigmas)
        stat = self.statistician(self.x, axis=None, dtype=self.x.dtype)
        for s in six.itervalues(stat):
            assert s.dtype == self.x.dtype
        testing.assert_allclose(stat['mean'], numpy.mean(self.x))
        testing.assert_allclose(stat['std'], numpy.std(self.x))
        percentile = stat['percentile']
        assert len(percentile) == 3
        testing.assert_allclose(percentile[0], numpy.min(self.x))
        testing.assert_allclose(percentile[1], numpy.median(self.x))
        testing.assert_allclose(percentile[2], numpy.max(self.x))
testing.run_module(__name__, __file__)