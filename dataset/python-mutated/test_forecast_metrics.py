import numpy as np
import pytest
import time
from unittest import TestCase
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from bigdl.chronos.metric.forecast_metrics import Evaluator
from .. import op_torch, op_tf2, op_distributed, op_diff_set_all

@op_torch
@op_tf2
class TestChronosForecastMetrics(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_forecast_metric(self):
        if False:
            print('Hello World!')
        n_samples = 50
        y_true = np.arange(n_samples) + 1
        y_pred = y_true + 1
        assert_almost_equal(Evaluator.evaluate('mse', y_true, y_pred, aggregate='mean')[0], 1.0)
        assert_almost_equal(Evaluator.evaluate('mae', y_true, y_pred, aggregate='mean')[0], 1.0)
        assert_almost_equal(Evaluator.evaluate('r2', y_true, y_pred, aggregate='mean')[0], 0.995, 2)
        assert_almost_equal(Evaluator.evaluate('smape', y_true, y_pred, aggregate='mean')[0], 3.895, 3)
        assert_almost_equal(Evaluator.evaluate('r2', y_true.reshape(5, 5, 2), y_pred.reshape(5, 5, 2), aggregate='mean')[0], 0.995, 2)
        assert_almost_equal(np.mean(Evaluator.evaluate('r2', y_true.reshape(25, 2), y_pred.reshape(25, 2), aggregate=None)[0]), 0.995, 2)
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, -0.3, 2, 8])
        assert_almost_equal(Evaluator.evaluate('mape', y_true, y_pred, aggregate='mean')[0], 17.74 / 100, 2)
        assert_almost_equal(Evaluator.evaluate('RMSE', y_true, y_pred, aggregate='mean')[0], 0.57, 2)

    def test_highdim_array_metrics(self):
        if False:
            i = 10
            return i + 15
        y_true = np.array([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = np.array([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])
        assert_almost_equal(Evaluator.evaluate('smape', y_true, y_pred, aggregate=None)[0], [[9.09, 25.0], [0.0, 6.67]], 2)
        assert_almost_equal(Evaluator.evaluate('mape', y_true, y_pred, aggregate=None)[0], [[16.67 / 100, 40.0 / 100], [0 / 100, 14.29 / 100]], 2)
        assert_almost_equal(Evaluator.evaluate('rmse', y_true, y_pred, aggregate=None)[0], [[0.5, 0.2], [0, 1]], 2)
        assert_almost_equal(Evaluator.evaluate('mse', y_true, y_pred, aggregate=None)[0], [[0.25, 0.04], [0, 1]], 2)
        y_true = np.array([[1, 2], [0.4, 5], [1, 2], [0.4, 5]])
        y_pred = np.array([[2, 1], [0.2, 3], [2, 1], [0.2, 3]])
        assert_almost_equal(Evaluator.evaluate('mse', y_true, y_pred, aggregate=None)[0], [0.52, 2.5], 2)
        assert_almost_equal(Evaluator.evaluate('smape', y_true, y_pred, aggregate=None)[0], [33.33, 29.17], 2)
        y_true = np.array([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = np.array([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])
        (mse, rmse, mape, smape) = Evaluator.evaluate(['mse', 'rmse', 'mape', 'smape'], y_true, y_pred, aggregate=None)
        assert_almost_equal(mse, [[0.25, 0.04], [0, 1]], 2)
        assert_almost_equal(rmse, [[0.5, 0.2], [0, 1]], 2)
        assert_almost_equal(mape, [[16.67 / 100, 40.0 / 100], [0 / 100, 14.29 / 100]], 2)
        assert_almost_equal(smape, [[9.09, 25.0], [0.0, 6.67]], 2)

    def test_standard_input(self):
        if False:
            while True:
                i = 10
        y_true = np.random.randn(100, 2, 2)
        y_pred = np.random.randn(100, 2, 2)
        with pytest.raises(RuntimeError):
            Evaluator.evaluate('test_smape', y_true, y_pred, aggregate=None)
        with pytest.raises(RuntimeError):
            Evaluator.evaluate('mse', y_true, y_pred.reshape(100, 4))
        y_true = [10, 2, 5]
        with pytest.raises(RuntimeError):
            Evaluator.evaluate('mse', y_true, y_true)

    @op_distributed
    def test_smape_equal_orca(self):
        if False:
            i = 10
            return i + 15
        from bigdl.orca.automl.metrics import sMAPE
        y_true = np.random.randn(100, 4)
        y_pred = np.random.randn(100, 4)
        smape = Evaluator.evaluate('smape', y_true, y_pred, aggregate='mean')[0]
        orca_smape = sMAPE(y_true, y_pred, multioutput='uniform_average')
        assert_almost_equal(smape, orca_smape, 6)

    def test_get_latency(self):
        if False:
            print('Hello World!')

        def test_func(count):
            if False:
                return 10
            time.sleep(0.001 * count)
        with pytest.raises(RuntimeError):
            Evaluator.get_latency(test_func, 5, num_running='10')
        with pytest.raises(RuntimeError):
            Evaluator.get_latency(test_func, 5, num_running=-10)
        latency_list = Evaluator.get_latency(test_func, 5)
        assert isinstance(latency_list, dict)
        for info in ['p50', 'p90', 'p95', 'p99']:
            assert info in latency_list
            assert isinstance(latency_list[info], float)

    @op_diff_set_all
    def test_plot(self):
        if False:
            i = 10
            return i + 15
        y = np.random.randn(100, 24, 1)
        pred = np.random.randn(100, 24, 1)
        x = np.random.randn(100, 48, 1)
        std = np.random.randn(100, 24, 1)
        Evaluator.plot(pred, x=x, ground_truth=y, std=std, prediction_interval=0.95, layout=(3, 4), figsize=(16, 8))