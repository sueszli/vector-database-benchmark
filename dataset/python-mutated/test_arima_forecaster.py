import numpy as np
import tempfile
import os
from bigdl.chronos.utils import LazyImport
ARIMAForecaster = LazyImport('bigdl.chronos.forecaster.arima_forecaster.ARIMAForecaster')
from unittest import TestCase
import pytest
from .. import op_diff_set_all

def create_data():
    if False:
        while True:
            i = 10
    seq_len = 400
    data = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    validation_data = np.random.rand(horizon)
    return (data, validation_data)

@op_diff_set_all
class TestChronosModelARIMAForecaster(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_arima_forecaster_fit_eval_pred(self):
        if False:
            while True:
                i = 10
        (data, validation_data) = create_data()
        forecaster = ARIMAForecaster(p=2, q=2, seasonality_mode=True, P=1, Q=1, m=7)
        train_loss = forecaster.fit(data, validation_data)
        test_pred = forecaster.predict(len(validation_data))
        assert len(test_pred) == len(validation_data)
        test_rolling_pred = forecaster.predict(len(validation_data), rolling=True)
        assert len(test_rolling_pred) == len(validation_data)
        test_mse = forecaster.evaluate(validation_data)

    def test_arima_forecaster_save_restore(self):
        if False:
            while True:
                i = 10
        (data, validation_data) = create_data()
        forecaster = ARIMAForecaster(p=2, q=2, seasonality_mode=True, P=1, Q=1, m=7)
        train_loss = forecaster.fit(data, validation_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, 'pkl')
            test_pred_save = forecaster.predict(len(validation_data))
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(len(validation_data))
        np.testing.assert_almost_equal(test_pred_save, test_pred_restore)

    def test_arima_forecaster_runtime_error(self):
        if False:
            return 10
        (data, validation_data) = create_data()
        forecaster = ARIMAForecaster(p=2, q=2, seasonality_mode=True, P=1, Q=1, m=7)
        with pytest.raises(Exception, match='You must call fit or restore first before calling predict!'):
            forecaster.predict(horizon=len(validation_data))
        with pytest.raises(Exception, match='Input invalid validation_data of None'):
            forecaster.evaluate(validation_data=None)
        with pytest.raises(Exception, match='You must call fit or restore first before calling evaluate!'):
            forecaster.evaluate(validation_data=validation_data)
        with pytest.raises(Exception, match='You must call fit or restore first before calling save!'):
            model_file = 'tmp.pkl'
            forecaster.save(model_file)

    def test_arima_forecaster_shape_error(self):
        if False:
            return 10
        (data, validation_data) = create_data()
        forecaster = ARIMAForecaster(p=2, q=2, seasonality_mode=True, P=1, Q=1, m=7)
        with pytest.raises(RuntimeError):
            forecaster.fit(data.reshape(-1, 1), validation_data)
        with pytest.raises(RuntimeError):
            forecaster.fit(data, validation_data.reshape(-1, 1))