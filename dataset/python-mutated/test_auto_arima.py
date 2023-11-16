from bigdl.chronos.utils import LazyImport
AutoARIMA = LazyImport('bigdl.chronos.autots.model.auto_arima.AutoARIMA')
hp = LazyImport('bigdl.orca.automl.hp')
import numpy as np
from unittest import TestCase
from ... import op_distributed, op_diff_set_all

def get_data():
    if False:
        return 10
    np.random.seed(0)
    seq_len = 400
    data = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    validation_data = np.random.rand(horizon)
    return (data, validation_data)

@op_distributed
@op_diff_set_all
class TestAutoARIMA(TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        if False:
            for i in range(10):
                print('nop')
        (data, validation_data) = get_data()
        auto_arima = AutoARIMA(metric='mse', p=hp.randint(0, 4), q=hp.randint(0, 4), seasonality_mode=hp.choice([True, False]), P=hp.randint(5, 12), Q=hp.randint(5, 12), m=hp.choice([4, 7]))
        auto_arima.fit(data=data, validation_data=validation_data, epochs=1, n_sampling=1)
        best_model = auto_arima.get_best_model()

    def test_fit_metric(self):
        if False:
            while True:
                i = 10
        (data, validation_data) = get_data()
        from torchmetrics.functional import mean_squared_error
        import torch

        def customized_metric(y_true, y_pred):
            if False:
                while True:
                    i = 10
            return mean_squared_error(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy()
        auto_arima = AutoARIMA(metric=customized_metric, metric_mode='min', p=hp.randint(0, 4), q=hp.randint(0, 4), seasonality_mode=hp.choice([True, False]), P=hp.randint(5, 12), Q=hp.randint(5, 12), m=hp.choice([4, 7]))
        auto_arima.fit(data=data, validation_data=validation_data, epochs=1, n_sampling=1)
        best_model = auto_arima.get_best_model()